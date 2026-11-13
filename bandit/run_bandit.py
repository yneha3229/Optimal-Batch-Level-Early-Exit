"""Bandit experiment (H3): online, label-free learning of the batch-stopping rule on the held-out stream.

Per batch t (B inputs drawn from the held-out split):
  score_t = accuracy_t - lam * rel_latency_t                 (labels used only to *evaluate*)
  fidelity score = agreement-with-full-model_t - lam * rel_latency_t    (the bandit's own label-free objective)
Regret against two oracles that know the true tables in advance:
  acc oracle       theory rule calibrated on held-out *labels*
  fidelity oracle  theory rule calibrated on held-out agreement with the final exit (the bandit's target)
Also: CeeBERT-style per-sample bandit, a probe-rate ablation, and the learned stopping threshold vs n_active.

Usage: ~/venv/bin/python bandit/run_bandit.py --task mnli --regime dec_eager --B 64 --T 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bandit.bandit import CeeBertStyle, LabelFreeStopRule  # noqa: E402
from experiments.run_main import TASK_EXITS, all_costs  # noqa: E402
from simulator import policies as P  # noqa: E402
from simulator.simulator import Traces, full_latency, load_traces, run_batch  # noqa: E402


def agree_traces(tr: Traces) -> Traces:
    """Same traces with the final-exit prediction as the label (pseudo-labels)."""
    return Traces(tr.conf, tr.pred, tr.pred[:, -1].copy(), tr.exit_layers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--regime", default="dec_eager")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--T", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=None, help="default: lambda the theory rule picked on evolve")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    path = ROOT / f"traces/{args.task}_val.parquet"
    ex = TASK_EXITS.get(args.task)
    ev, ho = load_traces(path, "evolve", exit_layers=ex), load_traces(path, "heldout", exit_layers=ex)
    cost = all_costs(args.task)[args.regime]
    B, L = args.B, ho.exit_layers
    lam = args.lam
    if lam is None:
        sel = ROOT / f"results/main_{args.task}_selected.csv"
        s = pd.read_csv(sel) if sel.exists() else pd.DataFrame()
        s = s[(s.regime == args.regime) & (s.B == B) & (s.family == "theory")] if len(s) else s
        lam = json.loads(s.params.iloc[0])["lam"] if len(s) else 0.1
    F = full_latency(L, cost, B)
    C = int(max(ev.label.max(), ho.label.max()) + 1)
    oracle_acc = P.theory_rule(lam, P.Calibration(ho), L, cost)
    oracle_fid = P.theory_rule(lam, P.Calibration(agree_traces(ho)), L, cost)
    offline = P.theory_rule(lam, P.Calibration(ev), L, cost)

    def score(o, rows):
        acc = float((o.pred == ho.label[rows]).mean())
        fid = float((o.pred == ho.pred[rows, -1]).mean())
        rel = o.latency / F
        return acc - lam * rel, fid - lam * rel, acc, rel

    rows_out, thr_out = [], []
    variants = [("bandit", dict(probe_c=1.0)), ("bandit_probe0.3", dict(probe_c=0.3)),
                ("bandit_probe3", dict(probe_c=3.0)), ("bandit_dual1pct", dict(probe_c=1.0, target_disagree=0.01))]
    for seed in args.seeds:
        rng = np.random.default_rng(100 + seed)
        stream = [rng.choice(len(ho), B, replace=False) for _ in range(args.T)]
        bds = {name: LabelFreeStopRule(L, cost, C, lam=lam, seed=seed, **kw) for name, kw in variants}
        cb = CeeBertStyle(np.round(np.linspace(0.5, 0.99, 15), 3), mu=lam, exit_layers=L)
        for t, r in enumerate(stream):
            rec = {"seed": seed, "t": t}
            for name, pol in [("oracle_acc", oracle_acc), ("oracle_fid", oracle_fid), ("offline_theory", offline)]:
                s, sf, a, rel = score(run_batch(pol, ho, r, cost), r)
                rec.update({f"{name}_score": s, f"{name}_fid_score": sf, f"{name}_acc": a, f"{name}_rel": rel})
            for name, bd in bds.items():
                probe = bd.is_probe()
                o = run_batch(P.never_exit(), ho, r, cost) if probe else run_batch(bd.decide, ho, r, cost)
                if probe:
                    bd.observe_probe(ho.conf[r], ho.pred[r])
                s, sf, a, rel = score(o, r)
                rec.update({f"{name}_score": s, f"{name}_fid_score": sf, f"{name}_acc": a, f"{name}_rel": rel,
                            f"{name}_probe": probe, f"{name}_lam": bd.lam})
            cb.choose()
            o = run_batch(cb.decide, ho, r, cost)
            cb.update(ho.conf[r, o.exit_of], o.exit_of)
            s, sf, a, rel = score(o, r)
            rec.update({"ceebert_score": s, "ceebert_fid_score": sf, "ceebert_acc": a, "ceebert_rel": rel})
            rows_out.append(rec)
            if t + 1 in (50, 300, 1000, args.T):
                bd = bds["bandit"]
                for e in range(len(L) - 1):
                    for n in sorted({1, 2, 4, 8, 16, 32, 48, B} & set(range(1, B + 1))):
                        thr_out.append({"seed": seed, "t": t + 1, "exit": e, "layer": L[e], "n_active": n,
                                        "threshold": bd.implied_threshold(e, n, B)})
        print(f"seed {seed} done; probes: " + ", ".join(f"{k}={b.n_probes}" for k, b in bds.items()), flush=True)
    df = pd.DataFrame(rows_out)
    tag = f"{args.task}_{args.regime}_B{B}"
    df.to_csv(ROOT / f"results/bandit_{tag}.csv", index=False)
    pd.DataFrame(thr_out).to_csv(ROOT / f"results/bandit_threshold_{tag}.csv", index=False)
    last = df[df.t >= args.T - 500]
    print(f"lambda = {lam}; mean over the last 500 batches (all seeds):")
    for m in ["oracle_acc", "oracle_fid", "offline_theory", "bandit", "bandit_probe0.3", "bandit_probe3",
              "bandit_dual1pct", "ceebert"]:
        print(f"  {m:16s} acc {last[f'{m}_acc'].mean():.4f}  rel_latency {last[f'{m}_rel'].mean():.3f}  "
              f"score {last[f'{m}_score'].mean():.4f}  fid_score {last[f'{m}_fid_score'].mean():.4f}")
    for m in ["bandit", "bandit_probe0.3", "bandit_probe3"]:
        reg = (df.groupby("seed").apply(lambda g: (g.oracle_fid_fid_score - g[f"{m}_fid_score"]).cumsum().iloc[-1])).mean()
        print(f"  cumulative fidelity regret after T={args.T}: {m} {reg:.1f}")


if __name__ == "__main__":
    main()
