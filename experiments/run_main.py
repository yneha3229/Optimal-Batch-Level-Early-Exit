"""Main experiment: every method x cost regime x batch size, tuned on the evolve split (max speedup with accuracy
within 1 point of the full model), reported on the held-out split, plus the slowdown-attack test.

Methods
  full                 no early exit
  threshold            per-sample confidence threshold, exited inputs leave the batch (DeeBERT-style)
  threshold_nocompact  same, exited inputs stay in the batch until it ends (wait-for-all; most research code)
  consensus/majority/greedy   whole-batch rules evaluated by DREX
  unified              one data-independent exit for the whole batch (SkipDecode-style)
  lanes                difficulty-aware lanes: regroup by exit-0 confidence, threshold inside (Recall/DART-style)
  ceebert              per-sample UCB over thresholds, label-free confidence-minus-cost reward (CeeBERT-style)
  random_collective    150 random hand-parameterised collective rules (equal-budget search baseline)
  theory[_indiv_only]  optimal-stopping rule from c(n) = a + b n (calibration from evolve labels)
  theory+rescue_*      cut-off inputs get a combined prediction over the exits they passed
  bandit[+rescue_conf] label-free partial-feedback bandit, trained online on the evolve stream; lambda set by its own
                       1%-disagreement target, never tuned with labels
  funsearch[+rescue]   best evolved program (if evolve/best/<task>_best.py exists)

Usage: ~/venv/bin/python experiments/run_main.py --task mnli [--regimes enc_eager_meas dec_eager real_mnli]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bandit.bandit import CeeBertStyle, LabelFreeStopRule  # noqa: E402
from simulator import policies as P  # noqa: E402
from simulator.attacks import hardest_pool, victim_experiment  # noqa: E402
from simulator.rescue import fit_learned_weights, make_rescue  # noqa: E402
from simulator.run_sim import load_costs  # noqa: E402
from simulator.simulator import Cost, Traces, load_traces, run_batch, simulate, simulate_lanes  # noqa: E402

ACC_TOL = 0.01
TASK_EXITS = {"wikitext": [8, 12, 16, 20, 24, 28], "wikitext_llama": [12, 16, 20, 24, 28, 32]}
DECODER_TASKS = {"wikitext", "wikitext_llama"}      # batched token decoding: per-sample exits stay in the batch (no rebatching)
LABEL_FREE = {"bandit", "bandit+rescue_conf", "bandit_t3pct"}
G: dict = {}   # per-worker globals


# ----------------------------------------------------------------------------------------------- setup
def all_costs(task: str) -> dict[str, Cost]:
    cs = {c.name: c for c in load_costs()}
    rc = ROOT / f"results/runner_cost_{task}.json"
    if rc.exists():
        d = json.loads(rc.read_text())
        cs[f"real_{task}"] = Cost(f"real_{task}", d["a"], d["b"], d["head"], d.get("skip", 0.0), d.get("exit_pass", 0.0))
    return cs


def init_worker(task: str) -> None:
    import pyarrow.parquet as pq
    path = ROOT / f"traces/{task}_val.parquet"
    probs = "p_0_0" in pq.read_schema(path).names
    ex = TASK_EXITS.get(task)
    G["ev"] = load_traces(path, "evolve", exit_layers=ex, with_probs=probs)
    G["ho"] = load_traces(path, "heldout", exit_layers=ex, with_probs=probs)
    G["cal"] = P.Calibration(G["ev"])
    G["cal2"] = P.Calibration2D(G["ev"])
    G["W"] = fit_learned_weights(G["ev"]) if probs else None
    G["decoder"] = task in DECODER_TASKS
    G["task"] = task
    G["costs"] = all_costs(task)
    att_path = ROOT / f"traces/{task}_attack.parquet"
    ho = G["ho"]
    if att_path.exists():                        # real slowdown inputs (DeepSloth on ViT)
        G["attack"] = load_traces(att_path, None, with_probs=True)
        G["clean"] = ho
        G["attack_kind"] = "deepsloth"
    else:                                        # label-free stand-in: inputs that stay least confident
        pool = hardest_pool(ho, 0.02)
        keep = np.setdiff1d(np.arange(len(ho)), pool)
        G["attack"], G["clean"] = ho.subset(pool), ho.subset(keep)
        G["attack_kind"] = "hardest_2pct"
    fc = ho.conf[:, -1]
    G["hard10"] = fc <= np.quantile(fc, 0.10)    # held-out rows: hardest 10% by full-model confidence (no labels)
    fev = None
    fpath = ROOT / f"evolve/best/{task}_best.py"
    if fpath.exists():
        spec = importlib.util.spec_from_file_location("best", fpath)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fev = mod.decide
    G["funsearch"] = fev


def n_batches(B: int) -> int:
    return max(150, 8000 // B)


def train_bandit(tr: Traces, cost: Cost, B: int, seed: int, target: float = 0.01) -> LabelFreeStopRule:
    C = int(tr.label.max() + 1) if tr.probs is None else tr.probs.shape[-1]
    bd = LabelFreeStopRule(tr.exit_layers, cost, num_classes=C, lam=0.05, target_disagree=target, seed=seed)
    rng = np.random.default_rng(seed)
    for _ in range(max(1500, 40000 // B)):
        rows = rng.choice(len(tr), B, replace=False)
        if bd.is_probe():
            bd.observe_probe(tr.conf[rows], tr.pred[rows])
        else:
            run_batch(bd.decide, tr, rows, cost)
    return bd


def train_ceebert(tr: Traces, cost: Cost, B: int, mu: float, seed: int) -> float:
    taus = np.round(np.linspace(0.5, 0.99, 15), 3)
    cb = CeeBertStyle(taus, mu, tr.exit_layers)
    rng = np.random.default_rng(seed)
    for _ in range(max(600, 20000 // B)):
        cb.choose()
        rows = rng.choice(len(tr), B, replace=False)
        o = run_batch(cb.decide, tr, rows, cost)
        cb.update(tr.conf[rows, o.exit_of], o.exit_of)
    return float(taus[int(cb.n.argmax())])


def candidates(fam: str, cost: Cost, B: int) -> list[tuple[dict, object, dict]]:
    """(params, decide, sim kwargs) for one family. Bandit/ceebert candidates are trained here (label-free)."""
    ev, cal, L = G["ev"], G["cal"], G["ev"].exit_layers
    taus = np.round(np.linspace(0.40, 0.995, 36), 4)
    lams = np.round(np.logspace(-3.5, 1.5, 60), 7)
    has_probs = G["W"] is not None
    conf_r = make_rescue("conf") if has_probs else None
    learn_r = make_rescue("learned", G["W"]) if has_probs else None
    th = lambda lam, **k: P.theory_rule(lam, cal, L, cost, **k)
    dec = G["decoder"]
    if fam.startswith(("theory+rescue", "bandit+rescue", "funsearch+rescue")) and not has_probs:
        return []
    if fam == "lanes" and dec:
        return []
    if fam == "full":
        return [({}, P.never_exit(), {"exits_enabled": False})]
    if fam == "threshold":
        return [({"tau": t}, P.threshold(t), {"compact": not dec}) for t in taus]
    if fam == "threshold_nocompact":
        return [({"tau": t}, P.threshold(t), {"compact": False}) for t in taus]
    if fam in ("consensus", "majority", "greedy"):
        return [({"tau": t}, getattr(P, fam)(t), {}) for t in taus]
    if fam == "unified":
        return [({"exit_idx": e}, P.unified_exit(e), {}) for e in range(len(L))]
    if fam == "lanes":
        return [({"tau": t}, P.threshold(t), {"lanes": True}) for t in taus]
    if fam == "random_collective":
        rng = np.random.default_rng(0)
        return [(r, P.param_collective(**r), {}) for r in
                (dict(tau=float(rng.uniform(0.5, 1.0)), frac_cut=float(rng.uniform(0, 0.5)),
                      conf_floor=float(rng.uniform(0.3, 0.95))) for _ in range(150))]
    if fam == "theory":
        return [({"lam": l, "table": "theory"}, th(l), {}) for l in lams]
    if fam == "theory_hist":
        return [({"lam": l, "hist": True}, P.theory_rule(l, G["cal2"], L, cost), {}) for l in lams]
    if fam == "theory_robust":
        return [({"lam": l, "trim_frac": 1 / 16}, th(l, trim_frac=1 / 16), {}) for l in lams]
    if fam == "theory_robust8":
        return [({"lam": l, "trim_frac": 1 / 8}, th(l, trim_frac=1 / 8), {}) for l in lams]
    if fam == "theory_indiv_only":
        return [({"lam": l, "table": "theory", "collective": False}, th(l, collective=False), {}) for l in lams]
    if fam == "theory+rescue_conf":
        return [({"lam": l, "table": "theory", "rescue": "conf"}, th(l), {"rescue": conf_r}) for l in lams]
    if fam == "theory+rescue_learned":
        return [({"lam": l, "table": "theory", "rescue": "learned"}, th(l), {"rescue": learn_r}) for l in lams]
    if fam in ("bandit", "bandit+rescue_conf", "bandit_t3pct"):
        target = 0.03 if fam == "bandit_t3pct" else 0.01
        bd = train_bandit(ev, cost, B, seed=B, target=target)
        p = {"lam": bd.lam, "table": f"bandit{'_t3pct' if target == 0.03 else ''}_{cost.name}_B{B}",
             "probes": bd.n_probes, "target": target}
        kw = {"rescue": conf_r} if fam.endswith("conf") else {}
        if fam.endswith("conf"):
            p["rescue"] = "conf"
        return [(p, bd.decide, kw)]
    if fam == "ceebert":
        out = []
        for mu in np.round(np.logspace(-2, 0.5, 10), 4):
            tau = train_ceebert(ev, cost, B, mu, seed=B)
            out.append(({"mu": float(mu), "tau": tau}, P.threshold(tau), {"compact": not dec}))
        return out
    if fam in ("funsearch", "funsearch+rescue_conf"):
        if G["funsearch"] is None:
            return []
        kw = {"rescue": conf_r} if fam.endswith("conf") else {}
        return [({"program": f"evolve/best/{G['task']}_best.py", **({"rescue": "conf"} if kw else {})}, G["funsearch"], kw)]
    raise KeyError(fam)


FAMILIES = ["full", "threshold", "threshold_nocompact", "consensus", "majority", "greedy", "unified", "lanes",
            "ceebert", "random_collective", "theory_indiv_only", "theory", "theory_hist", "theory_robust",
            "theory_robust8", "theory+rescue_conf",
            "theory+rescue_learned", "bandit", "bandit+rescue_conf", "bandit_t3pct", "funsearch", "funsearch+rescue_conf"]


def sim(decide, tr: Traces, cost: Cost, B: int, seed: int, kw: dict, per_sample=False) -> dict:
    kw = dict(kw)
    if kw.pop("lanes", False):
        return simulate_lanes(decide, tr, cost, B, n_batches(B), seed=seed)
    return simulate(decide, tr, cost, B, n_batches(B), seed=seed, per_sample=per_sample, **kw)


def job(args) -> list[dict]:
    reg, B, fam = args
    cost = G["costs"][reg]
    rows = []
    for i, (p, decide, kw) in enumerate(candidates(fam, cost, B)):
        r = sim(decide, G["ev"], cost, B, 1000 * B, kw)
        rows.append({"regime": reg, "B": B, "rho": cost.a / (cost.b * B), "family": fam, "param_id": i,
                     "params": json.dumps(p), "split": "evolve", **{k: v for k, v in r.items()}})
        if B == 64:   # full Pareto curve on held-out at the headline batch size
            r = sim(decide, G["ho"], cost, B, 1000 * B + 1, kw)
            rows.append({**rows[-1], "split": "heldout", **{k: v for k, v in r.items()}})
    return rows


def report_job(args) -> tuple[dict, list[dict]]:
    """Held-out evaluation of one selected policy: accuracy, speedup, hardest-10% accuracy, attack test."""
    reg, B, fam, pid, params = args
    cost = G["costs"][reg]
    p, decide, kw = candidates(fam, cost, B)[pid]
    ho = G["ho"]
    r = sim(decide, ho, cost, B, 1000 * B + 1, kw, per_sample=fam != "lanes")
    full = simulate(P.never_exit(), ho, cost, B, n_batches(B), seed=1000 * B + 1, exits_enabled=False)
    out = {"regime": reg, "B": B, "rho": cost.a / (cost.b * B), "family": fam, "param_id": pid, "params": params,
           "heldout_acc": r["accuracy"], "heldout_acc_drop": full["accuracy"] - r["accuracy"],
           "heldout_speedup": r["speedup"], "heldout_flops_frac": r["flops_frac"], "frac_cut": r["frac_cut"]}
    if "rows" in r:
        hard = G["hard10"][r["rows"]]
        out["hard10_acc"] = float(r["correct"][hard].mean())
        out["hard10_full_acc"] = float(ho.correct[r["rows"][hard], -1].mean())
    att = []
    if fam not in ("lanes", "threshold_nocompact") and (B in (16, 64)):
        for a in victim_experiment(decide, G["clean"], G["attack"], cost, B, ks=(0, 1, 2, 4, 8), n_batches=200,
                                   seed=7, rescue=kw.get("rescue"), compact=kw.get("compact", True)):
            att.append({"regime": reg, "B": B, "family": fam, "attack": G["attack_kind"], **a})
        out["victim_slowdown_k1"] = att[1]["victim_slowdown"]
    return out, att


def make_tables(task: str, regs: list[str], Bs: list[int]) -> dict:
    """Gain tables for the GPU runner's TableRule (theory rule + bandit snapshots)."""
    ev, cal, costs = G["ev"], G["cal"], G["costs"]
    E = len(ev.exit_layers)
    tabs = {}
    for reg in regs:
        c = costs[reg]
        base = {"a": c.a, "b": c.b, "head": c.head, "exit_layers": ev.exit_layers}
        tabs.setdefault("theory", {**base, "edges": [e[1:-1] for e in cal.edges],
                                   "G": [np.maximum(t - t[:, [e]], 0) for e, t in enumerate(cal.table)]})
        tabs[f"theory_{reg}"] = {**base, "edges": [e[1:-1] for e in cal.edges],
                                 "G": [np.maximum(t - t[:, [e]], 0) for e, t in enumerate(cal.table)]}
        for B in Bs:
            for tgt, name in [(0.01, "bandit"), (0.03, "bandit_t3pct")]:
                bd = train_bandit(ev, c, B, seed=B, target=tgt)
                Gs = [bd.gain_table(e, np.arange(bd.nb)) for e in range(E)]
                tabs[f"{name}_{reg}_B{B}"] = {**base, "edges": [bd.edges] * E, "G": Gs}
    return tabs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--regimes", nargs="+", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 16, 32, 64, 128])
    ap.add_argument("--families", nargs="+", default=FAMILIES)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    init_worker(args.task)
    default = ("dec_eager", f"real_{args.task}") if G["decoder"] else ("enc_eager_meas", "dec_eager", f"real_{args.task}")
    regs = args.regimes or [r for r in default if r in G["costs"]]
    print("regimes:", {r: (round(G["costs"][r].a, 4), round(G["costs"][r].b, 5), round(G["costs"][r].head, 4)) for r in regs})
    print("attack pool:", G["attack_kind"], len(G["attack"]), "| funsearch program:", G["funsearch"] is not None, flush=True)
    jobs = list(itertools.product(regs, args.batch_sizes, args.families))
    with Pool(args.workers, initializer=init_worker, initargs=(args.task,)) as pool:
        rows = [r for rs in pool.imap_unordered(job, jobs, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"results/main_{args.task}_all.csv", index=False)

    # selection on evolve
    picks = []
    for (reg, B), g in df[df.split == "evolve"].groupby(["regime", "B"]):
        full_ev = g[g.family == "full"].accuracy.iloc[0]
        for fam, gf in g.groupby("family"):
            ok = gf if fam in LABEL_FREE else gf[gf.accuracy >= full_ev - ACC_TOL]
            if ok.empty:
                continue
            best = ok.loc[ok.speedup.idxmax()]
            picks.append((reg, int(B), fam, int(best.param_id), best.params))
    with Pool(args.workers, initializer=init_worker, initargs=(args.task,)) as pool:
        res = pool.map(report_job, picks, chunksize=1)
    sel = pd.DataFrame([r for r, _ in res])
    att = pd.DataFrame([a for _, al in res for a in al])
    sel.to_csv(ROOT / f"results/main_{args.task}_selected.csv", index=False)
    att.to_csv(ROOT / f"results/main_{args.task}_attack.csv", index=False)
    for reg in regs:
        sel[sel.regime == reg].to_csv(ROOT / f"results/main_{args.task}_{reg}_selected.csv", index=False)
    pd.set_option("display.width", 250)
    for reg in regs:
        s = sel[sel.regime == reg]
        print(f"\n=== {reg}: held-out speedup at matched accuracy (sim) ===")
        print(s.pivot_table(index="family", columns="B", values="heldout_speedup").round(2).to_string())
        print(f"--- {reg}, B=64 detail ---")
        print(s[s.B == 64][["family", "heldout_acc", "heldout_acc_drop", "heldout_speedup", "heldout_flops_frac",
                            "hard10_acc", "victim_slowdown_k1"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
