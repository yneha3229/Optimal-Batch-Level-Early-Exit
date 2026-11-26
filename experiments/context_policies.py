"""Which exit rule wins in LLM decoding as the prompt gets longer? Simulation with measured costs.

Traces: traces/wikitext_<ckpt>ctx<lo>_val.parquet (experiments/context_scaling.py acc --save-traces), one per context range.
Costs:  results/context_cost_<cost-ckpt>_<mode>_<gpu>.csv (experiments/context_scaling.py cost [--graph]): per-layer time
        at context T and batch B, exit check, lm_head pass, KV propagation. For a batch of size B at context T the
        cost model is the secant through the measured points n = 1 and n = B:  c(n) = a + b n,
        b = (c(B) - c(1)) / (B - 1), a = c(1) - b  (exact at both ends; the measured curve is not linear in n).
Rules:  unified exit, majority vote, per-sample threshold with compaction (finished rows stop paying; needs paged
        attention, which our runner does not have), per-sample threshold without compaction (what our runner does),
        theory rule (per-sample exits + collective stop, compaction), theory rule with collective stops only off.
Protocol: parameters selected on the evolve split within --budget of the full model's accuracy, reported on held-out.
SIMULATED: the costs are measured, the batches are replayed from traces. Not a wall-clock result.

Usage: ~/venv/bin/python experiments/context_policies.py --ckpt qlora --cost results/context_cost_qwen_graph_RTX3080Ti.csv
Output: results/context_policies_<ckpt>_<mode>_<gpu>.csv
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import run_main as RM  # noqa: E402
from simulator import policies as P  # noqa: E402
from simulator.simulator import Cost, simulate  # noqa: E402

FAMS = ["unified", "majority", "threshold", "threshold_nocompact", "theory", "theory_indiv_only"]
BUDGET = 0.01


def init(task: str) -> None:
    RM.init_worker(task)
    RM.G["decoder"] = False          # "threshold" = with compaction; "threshold_nocompact" = without


def job(a) -> dict | None:
    task, T, B, fam, c = a
    cost = Cost(f"ctx{T}", c["a"], c["b"], c["head"], c["skip"], c["exit_pass"])
    ev, ho = RM.G["ev"], RM.G["ho"]
    nb = RM.n_batches(B)
    full_ev = float(ev.correct[:, -1].mean())
    best = None
    for i, (p, decide, kw) in enumerate(RM.candidates(fam, cost, B)):
        r = simulate(decide, ev, cost, B, nb, seed=1000 * B, **kw)
        if r["accuracy"] >= full_ev - BUDGET and (best is None or r["speedup"] > best[0]):
            best = (r["speedup"], i, p, decide, kw)
    if best is None:
        return None
    _, i, p, decide, kw = best
    r = simulate(decide, ho, cost, B, nb, seed=1000 * B + 1, **kw)
    full = simulate(P.never_exit(), ho, cost, B, nb, seed=1000 * B + 1, exits_enabled=False)
    return {"task": task, "T": T, "B": B, "family": fam, "params": json.dumps(p), "a": cost.a, "b": cost.b,
            "rho": cost.a / max(cost.b * B, 1e-12), "evolve_speedup": best[0], "heldout_speedup": r["speedup"],
            "heldout_drop_pts": 100 * (full["accuracy"] - r["accuracy"]), "flops_frac": r["flops_frac"],
            "frac_cut": r["frac_cut"]}


def main() -> None:
    global BUDGET
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="qlora")
    ap.add_argument("--cost", required=True)
    ap.add_argument("--pairs", nargs="+", default=["128:256", "1024:2048", "2048:4096"],
                    help="trace position range lo : context length T of the cost table to use")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])
    ap.add_argument("--budget", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=30)
    args = ap.parse_args()
    BUDGET = args.budget
    ct = pd.read_csv(ROOT / args.cost if not Path(args.cost).is_absolute() else args.cost)
    rows = []
    for pair in args.pairs:
        lo, T = (int(v) for v in pair.split(":"))
        task = f"wikitext_{args.ckpt}ctx{lo}"
        g = ct[ct["T"] == T].set_index("B")
        jobs = []
        for B in args.batch_sizes:
            if B not in g.index or 1 not in g.index:
                continue
            b = (g.per_layer_ms[B] - g.per_layer_ms[1]) / (B - 1)
            c = {"a": float(g.per_layer_ms[1] - b), "b": float(max(b, 1e-7)), "head": float(g.head_ms[B]),
                 "skip": float(g.skip_ms[B]), "exit_pass": float(g.exit_pass_ms[B])}
            jobs += [(task, T, B, fam, c) for fam in FAMS]
        with Pool(args.workers, initializer=init, initargs=(task,)) as pool:
            rows += [r for r in pool.imap_unordered(job, jobs, chunksize=1) if r]
        print("done", task, flush=True)
    df = pd.DataFrame(rows).sort_values(["T", "B", "family"])
    mode, gpu = ct["mode"].iloc[0], ct["gpu"].iloc[0]
    sfx = "" if args.budget == 0.01 else f"_b{round(100 * args.budget)}"
    df.to_csv(ROOT / f"results/context_policies_{args.ckpt}_{mode}_{gpu}{sfx}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"held-out speedup, budget {100 * args.budget:.0f} pt on evolve, costs: {mode} on {gpu} (SIMULATED)")
    print(df.pivot_table(index=["T", "B"], columns="family", values="heldout_speedup").round(3)
          .join(df.groupby(["T", "B"]).rho.first().round(2)).to_string())
    print("held-out accuracy drop (points)")
    print(df.pivot_table(index=["T", "B"], columns="family", values="heldout_drop_pts").round(2).to_string())


if __name__ == "__main__":
    main()
