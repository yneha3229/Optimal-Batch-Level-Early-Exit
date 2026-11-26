"""Report item 5.3 / docs/DIRECTION.md step 3: fluctuating load.

A server with dynamic batching does not run at one batch size: when it frees up it takes everything queued (up to
B_max). This script builds such a batch-size sequence from an arrival trace and replays every method on it:

  arrival trace   Poisson arrivals with a rate that swings between light and heavy load (a sinusoid plus bursts);
                  a single server takes min(queue, B_max) requests whenever it is free, and serves a batch of size B
                  in the full model's time c(B) from the measured cost table. The resulting sequence of batch sizes
                  (from 1 to 128) is fixed and shared by all methods, so they are compared on identical work.
  methods         for each family of local_hw.py (cost rule, cost rule + exit plan, per-sample threshold, majority,
                  unified exit, random search over a three-parameter collective rule):
                    once@64   certified selection at B = 64, the same parameters used at every batch size
                    once@8    the same at B = 8
                    per_B     re-selected at each batch-size bucket {1, 2, 4, ..., 128}, and each batch uses the
                              parameters of its bucket (largest bucket <= B); the upper reference
  metric          throughput speedup = sum over batches of full-model time / sum of the method's time (cost-model
                  replay), and the accuracy drop over every served input, on the held-out split; also per load level.

Pass (DIRECTION.md): the cost rule tuned once within 5% of the best per-batch-size re-tuned heuristic and ahead of
every heuristic tuned once, all within the one-point budget.

Usage:
  python experiments/load_trace.py --task qqp --name a6000 --workers 12
  python experiments/load_trace.py --task cifar100 --cost-json results/runner_cost_cifar100.json --seq-len 197
Output: results/load_trace_<task>.csv (per method), results/load_trace_<task>_levels.csv (per load level),
        results/load_trace_<task>_batches.csv (the batch-size sequence).
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import local_hw as lh  # noqa: E402
from simulator.simulator import load_traces  # noqa: E402

BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]
G: dict = {}


def load_cost(args) -> lh.TCost:
    if args.cost_json:
        d = json.loads(Path(args.cost_json).read_text())
        pts = [{"B": p["B"], "full_ms": p.get("full_ms", p.get("full_ms_per_batch"))} for p in d["points"]]
        return lh.TCost({"device": Path(args.cost_json).stem, "seq_len": args.seq_len, "a": d["a"], "b": d["b"],
                         "head": d["head"], "points": pts}, "table")
    return lh.load_cost(args)


def batch_sizes(cost, n_batches: int, b_max: int, seed: int, period_s: float) -> np.ndarray:
    """Batch sizes seen by a dynamic-batching server under a fluctuating Poisson arrival rate.
    The rate swings between 2% and 130% of the server's full-batch capacity (b_max / c_full(b_max))."""
    rng = np.random.default_rng(seed)
    cap = b_max / (lh.full_ms(cost, b_max) / 1000.0)              # requests per second at B = b_max
    t, q, out = 0.0, 0, []
    while len(out) < n_batches:
        phase = 0.5 * (1 + np.sin(2 * np.pi * t / period_s))      # 0..1
        burst = 1.0 + (rng.random() < 0.02) * rng.uniform(1, 3)
        rate = cap * (0.02 + 1.28 * phase) * burst
        if q == 0:                                                 # idle: wait for the next arrival
            dt = rng.exponential(1 / rate)
            t += dt
            q = 1
        B = min(q, b_max)
        out.append(B)
        q -= B
        s = lh.full_ms(cost, B) / 1000.0                           # service time of the batch
        q += rng.poisson(rate * s)
        t += s
    return np.array(out)


def _select(a):
    fam, B = a
    return (fam, B), lh.select_family(fam, G["fit"], G["cert"], G["cost"], B, seed=B, key="drop")


def bucket_of(B: int) -> int:
    return max(b for b in BUCKETS if b <= B)


def run_method(fam: str, params_of, Bs: np.ndarray, seed: int) -> dict:
    """Replay one method over the batch-size sequence; params_of(B) -> params or None (= full model)."""
    ho, cost, fit = G["ho"], G["cost"], G["fit"]
    rng = np.random.default_rng(500 + seed)
    t_pol, t_full, n_ok_full, n_ok, n = 0.0, 0.0, 0, 0, 0
    lev = []
    cache = {}
    for B in Bs:
        rows = rng.choice(len(ho), B, replace=False)
        p = params_of(int(B))
        f_ms = lh.full_ms(cost, int(B))
        if p is None:
            lat, pred = f_ms, ho.pred[rows, -1]
        else:
            key = json.dumps(p, sort_keys=True)
            if key not in cache:
                cache[key] = lh.make(fam, p, fit, cost)
            plan, decide = cache[key]
            l, pr = lh.replay(decide, lh.sub_traces(ho, plan), [rows], cost)
            lat, pred = float(l[0]), pr[0]
        ok_full = int((ho.pred[rows, -1] == ho.label[rows]).sum())
        ok = int((pred == ho.label[rows]).sum())
        t_pol += lat
        t_full += f_ms
        n_ok_full += ok_full
        n_ok += ok
        n += B
        lev.append((B, lat, f_ms, ok_full - ok))
    lv = pd.DataFrame(lev, columns=["B", "lat", "full", "lost"])
    lv["level"] = pd.cut(lv.B, [0, 4, 32, 96, 128], labels=["B<=4", "5-32", "33-96", "97-128"])
    per = lv.groupby("level", observed=True).agg(lat=("lat", "sum"), full=("full", "sum"), lost=("lost", "sum"),
                                                  n=("B", "sum"))
    levels = {f"speedup[{k}]": float(r.full / r.lat) for k, r in per.iterrows()}
    levels.update({f"drop_pts[{k}]": float(100 * r.lost / r.n) for k, r in per.iterrows()})
    return {"speedup": t_full / t_pol, "drop_pts": 100 * (n_ok_full - n_ok) / n, "inputs": n, **levels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qqp")
    ap.add_argument("--name", default="a6000")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--cost-json", default=None)
    ap.add_argument("--cost-model", default="table")
    ap.add_argument("--families", nargs="+", default=lh.FAMILIES)
    ap.add_argument("--n-batches", type=int, default=3000)
    ap.add_argument("--b-max", type=int, default=128)
    ap.add_argument("--period", type=float, default=20.0, help="seconds per load cycle")
    ap.add_argument("--seeds", type=int, default=3, help="arrival traces")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delta", type=float, default=lh.DELTA, help="failure probability of the certificate")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    if args.delta != lh.DELTA:                       # select_family calls lh.certified(r, key) with its default delta
        _cert = lh.certified
        lh.certified = lambda r, key="drop", alpha=lh.ALPHA, delta=args.delta: _cert(r, key, alpha, delta)
    lh.set_task(args.task)
    path = ROOT / f"traces/{args.task}_val.parquet"
    ev = load_traces(path, "evolve", exit_layers=list(lh.LAYERS))
    G["ho"] = load_traces(path, "heldout", exit_layers=list(lh.LAYERS))
    G["fit"], G["cert"] = lh.split_evolve(ev, 0)
    G["cost"] = load_cost(args)

    jobs = [(f, B) for f in args.families for B in BUCKETS]
    with Pool(args.workers) as pool:
        sel = dict(pool.map(_select, jobs, chunksize=1))
    sel_rows = [{"family": f, "B": B, "params": json.dumps(r["cert_params"]),
                 "cert_speedup": None if r["cert_ev"] is None else r["cert_ev"]["speedup"]} for (f, B), r in sel.items()]
    pd.DataFrame(sel_rows).to_csv(ROOT / f"results/load_trace_{args.task}{args.tag}_selected.csv", index=False)

    rows, seqs = [], []
    for seed in range(args.seeds):
        Bs = batch_sizes(G["cost"], args.n_batches, args.b_max, seed, args.period)
        seqs.append(pd.DataFrame({"seed": seed, "batch": np.arange(len(Bs)), "B": Bs}))
        print(f"trace {seed}: {len(Bs)} batches, {Bs.sum()} inputs, B quantiles "
              f"{np.quantile(Bs, [0.1, 0.25, 0.5, 0.75, 0.9]).astype(int).tolist()}", flush=True)
        rows.append({"seed": seed, "family": "full", "tuning": "-", **run_method("full", lambda B: None, Bs, seed)})
        for f in args.families:
            for tuning, pf in (("once@64", lambda B, f=f: sel[(f, 64)]["cert_params"]),
                               ("once@8", lambda B, f=f: sel[(f, 8)]["cert_params"]),
                               ("per_B", lambda B, f=f: sel[(f, bucket_of(B))]["cert_params"])):
                rows.append({"seed": seed, "family": f, "tuning": tuning, **run_method(f, pf, Bs, seed)})
    df = pd.DataFrame(rows)
    pd.concat(seqs).to_csv(ROOT / f"results/load_trace_{args.task}{args.tag}_batches.csv", index=False)
    df.to_csv(ROOT / f"results/load_trace_{args.task}{args.tag}.csv", index=False)
    m = df.groupby(["family", "tuning"]).mean(numeric_only=True).drop(columns=["seed", "inputs"]).reset_index()
    m.to_csv(ROOT / f"results/load_trace_{args.task}{args.tag}_levels.csv", index=False)
    pd.set_option("display.width", 250)
    print(m.round(3).to_string(index=False))

    # DIRECTION.md step 3 criterion, on the means over traces; only methods within the 1-point budget count
    ok = m[m.drop_pts <= 100 * lh.ALPHA]
    heur = ["threshold", "threshold+plan", "majority", "unified", "random_collective"]
    best_perB = ok[(ok.tuning == "per_B") & ok.family.isin(heur)].speedup.max()
    best_once = ok[(ok.tuning != "per_B") & ok.family.isin(heur)].speedup.max()
    for f in [f for f in ("theory", "theory+plan") if f in args.families]:
        r = ok[(ok.family == f) & (ok.tuning == "once@64")]
        if len(r):
            s = float(r.speedup.iloc[0])
            print(f"{f} once@64: {s:.3f}x; best per-B heuristic {best_perB:.3f}x (ratio {s / best_perB:.3f}, pass >= 0.95); "
                  f"best tuned-once heuristic {best_once:.3f}x (ahead: {s > best_once})")
        else:
            print(f"{f} once@64: over the 1-point budget")


if __name__ == "__main__":
    main()
