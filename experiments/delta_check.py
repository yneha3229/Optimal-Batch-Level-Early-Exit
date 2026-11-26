"""Does a stricter certificate (smaller failure probability delta) fix the MNLI held-out overshoot?

Same fixed evolve / held-out split and certified selection as experiments/local_hw.py (front), run at several delta
values; reports the held-out accuracy drop and cost-model speedup of the chosen parameters.
Usage: python experiments/delta_check.py --task mnli --name a6000 --deltas 0.10 0.05 0.02
Output: results/delta_check_<task>.csv
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import local_hw as lh  # noqa: E402
from simulator.simulator import load_traces  # noqa: E402

G: dict = {}
_CERT = lh.certified


def _job(a):
    fam, B, delta = a
    lh.certified = lambda r, key="drop", alpha=lh.ALPHA, delta=delta: _CERT(r, key, alpha, delta)
    r = lh.select_family(fam, G["fit"], G["cert"], G["cost"], B, seed=B, key="drop")
    out = {"family": fam, "B": B, "delta": delta, "n_certified": r["n_certified"]}
    if r["cert_params"] is None:
        return {**out, "ho_speedup": 1.0, "ho_drop_pts": 0.0}
    e = lh.evaluate(fam, r["cert_params"], G["fit"], G["ho"], lh.disjoint_batches(len(G["ho"]), B, 100 + B), G["cost"])
    return {**out, "ho_speedup": e["speedup"], "ho_drop_pts": 100 * e["drop"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--name", default="a6000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--cost-model", default="table")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
    ap.add_argument("--families", nargs="+", default=lh.FAMILIES)
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.10, 0.05, 0.02])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    lh.set_task(args.task)
    path = ROOT / f"traces/{args.task}_val.parquet"
    ev = load_traces(path, "evolve", exit_layers=list(lh.LAYERS))
    G["ho"] = load_traces(path, "heldout", exit_layers=list(lh.LAYERS))
    G["fit"], G["cert"] = lh.split_evolve(ev, 0)
    G["cost"] = lh.load_cost(args)
    jobs = [(f, B, d) for d in args.deltas for B in args.batch_sizes for f in args.families]
    with Pool(args.workers) as pool:
        df = pd.DataFrame(pool.map(_job, jobs, chunksize=1))
    df.to_csv(ROOT / f"results/delta_check_{args.task}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index=["family", "B"], columns="delta", values=["ho_speedup", "ho_drop_pts"]).round(3))


if __name__ == "__main__":
    main()
