"""P6 (simulation): every policy family x every cost regime x batch size, tuned on the evolve split,
reported on the held-out split.

Cost regimes come from the measured P1 fits:
  enc_eager   RoBERTa-base layer, fp16, L=128, plain PyTorch           (A6000)
  enc_graph   same, CUDA graphs (launch overhead removed)               (A6000)
  enc_3080ti  same as enc_eager on an RTX 3080 Ti                        (hardware transfer)
  dec_eager   Qwen2.5-1.5B decode step per layer, context 256            (what-if: decoder-like rho on MNLI traces)

Usage: ~/venv/bin/python simulator/run_sim.py
"""
from __future__ import annotations

import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import policies as P  # noqa: E402
from simulator.simulator import Cost, load_traces, simulate  # noqa: E402

TRACES = ROOT / "traces/mnli_val.parquet"
BATCH_SIZES = [1, 8, 16, 32, 64, 128]
ACC_TOL = 0.01  # "within 1% of the full model" (absolute), checked on the evolve split


def _fit(path: str, **match) -> dict:
    fits = json.loads((ROOT / path).read_text())["fits"]
    return next(f for f in fits if all(f.get(k) == v for k, v in match.items()))


def _head(gpu: str, mode: str) -> float:
    d = pd.read_csv(ROOT / f"results/p1_timing_{gpu}.csv")
    g = d[(d.component == "exit_head") & (d["mode"] == mode) & (d.dtype == "float16") & (d.seq_len == 128)]
    return float(g.median_ms.median())


def load_costs() -> list[Cost]:
    costs = []
    for name, gpu, mode in [("enc_eager", "A6000", "eager"), ("enc_graph", "A6000", "cuda_graph"),
                            ("enc_3080ti", "3080Ti", "eager")]:
        p = Path(f"results/p1_fit_{gpu}.json")
        if (ROOT / p).exists():
            f = _fit(str(p), mode=mode, dtype="float16", seq_len=128, component="layer")
            costs.append(Cost(name, f["a_ms"], f["b_ms"], _head(gpu, mode)))
    ovh = ROOT / "results/p6_exit_overhead_enc_eager.json"
    if ovh.exists() and costs:  # per-exit overhead measured on the real GPU (head + sync + host decision)
        enc = costs[0]
        costs.append(Cost("enc_eager_meas", enc.a, enc.b, float(json.loads(ovh.read_text())["64"])))
    dec = ROOT / "results/p1_decoder_fit_Qwen2.5-1.5B_A6000.json"
    if dec.exists():
        f = _fit(str(dec.relative_to(ROOT)), context=256)
        costs.append(Cost("dec_eager", f["a_ms"], max(f["b_ms"], 1e-5), _head("A6000", "eager")))
    return costs


def families(cost: Cost, cal: P.Calibration, exit_layers: list[int]) -> dict[str, list[tuple[dict, object, bool]]]:
    taus = np.round(np.linspace(0.40, 0.995, 36), 4)
    lams = np.round(np.logspace(-3, 1.5, 36), 6)
    rng = np.random.default_rng(0)
    rand = [dict(tau=float(rng.uniform(0.5, 1.0)), frac_cut=float(rng.uniform(0, 0.5)),
                 conf_floor=float(rng.uniform(0.3, 0.95))) for _ in range(150)]
    return {
        "full": [({}, P.never_exit(), True)],
        "threshold": [({"tau": t}, P.threshold(t), True) for t in taus],
        "threshold_nocompact": [({"tau": t}, P.threshold(t), False) for t in taus],
        "consensus": [({"tau": t}, P.consensus(t), True) for t in taus],
        "majority": [({"tau": t}, P.majority(t), True) for t in taus],
        "greedy": [({"tau": t}, P.greedy(t), True) for t in taus],
        "random_collective": [(r, P.param_collective(**r), True) for r in rand],
        "theory": [({"lam": l}, P.theory_rule(l, cal, exit_layers, cost), True) for l in lams],
        "theory_indiv_only": [({"lam": l}, P.theory_rule(l, cal, exit_layers, cost, collective=False), True) for l in lams],
    }


def _job(args):
    cost, B, fam, params, compact, seed_base = args
    ev, ho = load_traces(TRACES, "evolve"), load_traces(TRACES, "heldout")
    cal = P.Calibration(ev)
    fam_list = families(cost, cal, ev.exit_layers)[fam]
    n_batches = max(150, 8000 // B)
    out = []
    for i in params:
        p, decide, compact = fam_list[i]
        for split, tr, seed in [("evolve", ev, seed_base), ("heldout", ho, seed_base + 1)]:
            r = simulate(decide, tr, cost, B, n_batches, seed=seed, compact=compact, exits_enabled=fam != "full")
            out.append({"regime": cost.name, "a": cost.a, "b": cost.b, "head": cost.head, "B": B,
                        "rho": cost.a / (cost.b * B), "family": fam, "param_id": i,
                        "params": json.dumps(p), "split": split, **r})
    return out


def main() -> None:
    costs = load_costs()
    print("cost regimes:", *[f"{c.name}: a={c.a:.4f} b={c.b:.5f} head={c.head:.4f}" for c in costs], sep="\n  ")
    ev = load_traces(TRACES, "evolve")
    cal = P.Calibration(ev)
    jobs = []
    for cost, B in itertools.product(costs, BATCH_SIZES):
        for fam, lst in families(cost, cal, ev.exit_layers).items():
            ids = list(range(len(lst)))
            for chunk in np.array_split(ids, max(1, len(ids) // 6)):
                jobs.append((cost, B, fam, [int(i) for i in chunk], None, 1000 * B))
    print(f"{len(jobs)} jobs", flush=True)
    with Pool(48) as pool:
        rows = [r for rs in pool.imap_unordered(_job, jobs, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/p6_sim_all.csv", index=False)

    # Select per (regime, B, family) on evolve: max speedup with acc >= full_acc - ACC_TOL; report held-out.
    sel = []
    for (reg, B), g in df.groupby(["regime", "B"]):
        ev_g, ho_g = g[g.split == "evolve"], g[g.split == "heldout"]
        full_ev = ev_g[ev_g.family == "full"].accuracy.iloc[0]
        full_ho = ho_g[ho_g.family == "full"].accuracy.iloc[0]
        for fam, gf in ev_g.groupby("family"):
            ok = gf[gf.accuracy >= full_ev - ACC_TOL]
            if ok.empty:
                continue
            best = ok.loc[ok.speedup.idxmax()]
            h = ho_g[(ho_g.family == fam) & (ho_g.param_id == best.param_id)].iloc[0]
            sel.append({"regime": reg, "B": B, "rho": h.rho, "family": fam, "params": best.params,
                        "evolve_acc": best.accuracy, "evolve_speedup": best.speedup,
                        "heldout_acc": h.accuracy, "heldout_acc_drop": full_ho - h.accuracy,
                        "heldout_speedup": h.speedup, "heldout_flops_frac": h.flops_frac})
    sel = pd.DataFrame(sel)
    sel.to_csv(ROOT / "results/p6_sim_selected.csv", index=False)

    fams = ["threshold", "threshold_nocompact", "consensus", "majority", "greedy", "random_collective",
            "theory_indiv_only", "theory"]
    regs = [c.name for c in costs]
    # Figure 1: speedup vs batch size at matched accuracy
    fig, axes = plt.subplots(1, len(regs), figsize=(4.6 * len(regs), 4), squeeze=False)
    for ax, reg in zip(axes[0], regs):
        for fam in fams:
            g = sel[(sel.regime == reg) & (sel.family == fam)].sort_values("B")
            ax.plot(g.B, g.heldout_speedup, "o-" if fam.startswith("theory") else ".--", label=fam,
                    lw=2.5 if fam == "theory" else 1)
        ax.axhline(1, color="k", lw=0.8)
        ax.set(xscale="log", xticks=BATCH_SIZES, xticklabels=BATCH_SIZES, title=reg,
               xlabel="batch size", ylabel="held-out speedup (acc within 1%)")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/p6_sim_speedup_vs_batch.png", dpi=150)

    # Figure 2: accuracy vs latency Pareto at B=64, held-out
    fig, axes = plt.subplots(1, len(regs), figsize=(4.6 * len(regs), 4), squeeze=False)
    for ax, reg in zip(axes[0], regs):
        g = df[(df.regime == reg) & (df.B == 64) & (df.split == "heldout")]
        for fam in fams:
            gf = g[g.family == fam].sort_values("rel_latency")
            ax.plot(gf.rel_latency, gf.accuracy, "-" if fam.startswith("theory") else ".", label=fam,
                    lw=2.5 if fam == "theory" else 1.2, ms=3)
        full = g[g.family == "full"].iloc[0]
        ax.plot(1, full.accuracy, "k*", ms=12, label="full")
        ax.axhline(full.accuracy - ACC_TOL, color="k", lw=0.6, ls=":")
        ax.set(title=f"{reg}, B=64", xlabel="relative latency", ylabel="held-out accuracy",
               ylim=(full.accuracy - 0.08, full.accuracy + 0.01), xlim=(0, 1.25))
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/p6_sim_pareto_B64.png", dpi=150)

    show = sel[sel.B.isin([1, 16, 64, 128])].pivot_table(index=["regime", "B"], columns="family", values="heldout_speedup")
    print("\nheld-out wall-clock speedup at matched accuracy (sim):")
    print(show[[f for f in fams if f in show.columns]].round(2).to_string())
    drop = sel[sel.B == 64].pivot_table(index="regime", columns="family", values="heldout_acc_drop")
    print("\nheld-out accuracy drop vs full at B=64:")
    print((drop[[f for f in fams if f in drop.columns]] * 100).round(2).to_string())


if __name__ == "__main__":
    main()
