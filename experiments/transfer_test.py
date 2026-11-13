"""Step 1 of docs/DIRECTION.md: do tuned exit policies transfer across deployments?

Settings (per task): batch size {8, 16, 64, 128} x cost regime {eager runner, CUDA-graph runner} = 8.
For every method and every source setting, parameters are selected on the evolve split at the source (fastest within
`margin` of the full model), then the SAME parameters are evaluated on the held-out split at every target setting.
For the theory rule "the same parameters" means the same lambda; the rule reads the target's measured (a, b, head, n),
which is the point of the test. Heuristics have no cost input, so their parameters are used unchanged.

Score at a target = held-out speedup if held-out accuracy is within 1 point of the full model, else 1.0 (a deployment
that breaks the accuracy budget falls back to the full model), divided by the best in-domain score at that target
(best method tuned at the target itself). Simulator only: a screen before the real-GPU step.

Usage: ~/venv/bin/python experiments/transfer_test.py [--tasks mnli qqp cifar100] [--workers 32]
Outputs: results/transfer_<task>_all.csv (every candidate, every setting), results/transfer_pairs.csv (every
         source -> target pair), results/transfer_summary.csv
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
from simulator.simulator import Cost  # noqa: E402

BS = [8, 16, 64, 128]
FAMS = ["threshold", "majority", "consensus", "unified", "random_collective", "theory"]
SEEDS = [1, 2, 3]          # held-out batch draws
TOL = 0.01                 # accuracy budget on held-out


def init(task: str) -> None:
    RM.init_worker(task)
    g = json.loads((ROOT / f"results/runner_cost_graph_{task}.json").read_text())
    RM.G["costs"][f"graph_{task}"] = Cost(f"graph_{task}", g["a"], g["b"], max(g["head"], 0.0))


def job(args) -> list[dict]:
    reg, B, fam = args
    cost = RM.G["costs"][reg]
    rows = []
    for i, (p, decide, kw) in enumerate(RM.candidates(fam, cost, B)):
        ev = RM.sim(decide, RM.G["ev"], cost, B, 1000 * B, kw)
        r = {"regime": reg, "B": B, "family": fam, "param_id": i, "params": json.dumps(p),
             "ev_acc": ev["accuracy"], "ev_speedup": ev["speedup"]}
        for s in SEEDS:
            ho = RM.sim(decide, RM.G["ho"], cost, B, 1000 * B + s, kw)
            r[f"ho_acc_{s}"], r[f"ho_lat_{s}"], r[f"ho_full_{s}"] = ho["accuracy"], ho["latency_ms"], ho["full_ms"]
        rows.append(r)
    return rows


def analyse(task: str, df: pd.DataFrame, margin: float) -> pd.DataFrame:
    """One row per (method, source, target): score of the source-tuned parameters at the target."""
    df = df.copy()
    df["setting"] = df.regime.str.split("_").str[0] + "_B" + df.B.astype(str)
    acc = df[[f"ho_acc_{s}" for s in SEEDS]].mean(1)
    df["ho_acc"] = acc
    df["ho_speedup"] = df[[f"ho_full_{s}" for s in SEEDS]].sum(1) / df[[f"ho_lat_{s}" for s in SEEDS]].sum(1)
    df["ho_speedup_sd"] = np.std([df[f"ho_full_{s}"] / df[f"ho_lat_{s}"] for s in SEEDS], 0)
    full = df[df.family == "full"].set_index("setting")
    key = df.set_index(["setting", "family", "param_id"])
    pick = {}
    for (st, fam), g in df[df.family != "full"].groupby(["setting", "family"]):
        ok = g[g.ev_acc >= full.loc[st, "ev_acc"] - margin]
        pick[(st, fam)] = int(ok.loc[ok.ev_speedup.idxmax()].param_id) if len(ok) else None
    rows = []
    settings = sorted(df.setting.unique())
    for fam in FAMS:
        for src, tgt in itertools.product(settings, settings):
            pid = pick.get((src, fam))
            if pid is None:
                sp, valid, drop = 1.0, True, 0.0       # nothing selectable: run the full model
            else:
                r = key.loc[(tgt, fam, pid)]
                drop = full.loc[tgt, "ho_acc"] - r.ho_acc
                valid = drop <= TOL
                sp = r.ho_speedup
            rows.append({"task": task, "margin": margin, "family": fam, "source": src, "target": tgt,
                         "speedup": sp, "acc_drop_pts": 100 * drop, "valid": bool(valid),
                         "score": sp if valid else 1.0})
    out = pd.DataFrame(rows)
    best = out[out.source == out.target].groupby("target").score.max().rename("best_in_domain")
    out = out.join(best, on="target")
    out["rel"] = out.score / out.best_in_domain
    mode = lambda s: s.split("_")[0]
    out["kind"] = np.where(out.source == out.target, "in_domain",
                           np.where(out.source.map(mode) == out.target.map(mode), "other_batch_size", "other_mode"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["mnli", "qqp", "cifar100"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--analyse-only", action="store_true")
    args = ap.parse_args()
    pairs = []
    for task in args.tasks:
        raw = ROOT / f"results/transfer_{task}_all.csv"
        if not args.analyse_only:
            jobs = [(reg, B, fam) for reg in (f"real_{task}", f"graph_{task}") for B in BS for fam in ["full"] + FAMS]
            with Pool(args.workers, initializer=init, initargs=(task,)) as pool:
                rows = [r for rs in pool.imap_unordered(job, jobs, chunksize=1) for r in rs]
            pd.DataFrame(rows).to_csv(raw, index=False)
            print(f"{task}: simulated {len(rows)} candidate-settings", flush=True)
        df = pd.read_csv(raw)
        pairs += [analyse(task, df, m) for m in (0.01, 0.005)]
    P = pd.concat(pairs)
    P.to_csv(ROOT / "results/transfer_pairs.csv", index=False)
    summ = (P.groupby(["margin", "kind", "family"])
             .agg(mean_rel=("rel", "mean"), worst_rel=("rel", "min"), violation_rate=("valid", lambda v: 1 - v.mean()),
                  mean_speedup=("score", "mean"), n=("rel", "size")).reset_index())
    summ.to_csv(ROOT / "results/transfer_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for m in (0.01, 0.005):
        print(f"\n=== selection margin {100 * m:.1f} pt on evolve; budget 1 pt on held-out; all tasks pooled ===")
        s = summ[summ.margin == m]
        for col in ["mean_rel", "worst_rel", "violation_rate"]:
            print(f"-- {col}")
            print(s.pivot(index="family", columns="kind", values=col).round(3).to_string())
    print("\n=== per task, transfer pairs only (margin 0.5 pt): mean / worst relative score ===")
    t = P[(P.margin == 0.005) & (P.kind != "in_domain")].groupby(["task", "family"]).rel.agg(["mean", "min"]).round(3)
    print(t.unstack(0).to_string())


if __name__ == "__main__":
    main()
