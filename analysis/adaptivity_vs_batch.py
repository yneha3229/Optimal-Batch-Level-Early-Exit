"""How much can a data-dependent whole-batch stop gain over a fixed (unified) exit, as a function of batch size?

Setting: pure fixed cost (b = 0), which is what batched LLM decoding measures (Qwen2.5-1.5B: a = 1.07 ms, b ~ 0).
Then per-input exits save nothing, latency is proportional to the depth the batch runs to, and the only question is
when the whole batch stops. No timing model is involved: depth and accuracy are exact from the traces.

Two policies, both tuned on the evolve split to a budget on the accuracy drop and reported on the held-out split:
  static     one exit for every batch (unified exit); adjacent exits are mixed at random so the budget is met exactly
             (if no intermediate exit is within the budget, the last one is mixed with the full model)
  adaptive   stop at the first exit where the batch's mean confidence reaches tau (one threshold)
Why it matters: with random batching the batch mean confidence has standard deviation ~ sigma / sqrt(B), so at
large B every batch looks alike and an adaptive stop cannot tell easy batches from hard ones. The script measures
that standard deviation and the depth the adaptive rule saves beyond the static one.

--metric accuracy (default): the budget is on next-token / label accuracy. --metric fidelity: the budget is on the
share of outputs that differ from the full model's (use larger budgets, e.g. 0.05 0.10).

Usage: python3 analysis/adaptivity_vs_batch.py [--tasks mnli qqp wikitext_llama wikitext_ls8b] [--budgets 0.01 0.03]
Output: results/adaptivity_vs_batch.csv (results/adaptivity_vs_batch_fidelity.csv with --metric fidelity)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.simulator import load_traces  # noqa: E402

EXITS = {"wikitext": [8, 12, 16, 20, 24, 28], "wikitext_llama": [12, 16, 20, 24, 28, 32],
         "wikitext_ls8b": [12, 16, 20, 24, 28, 32], "wikitext_ls1b": None}
BS = [1, 2, 4, 8, 16, 32, 64]
TAUS = np.linspace(0.30, 1.0, 141)


def batches(n: int, B: int, cap: int, seed: int) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(n)
    k = min(n // B, cap)
    return perm[: k * B].reshape(k, B)


def adaptive(tr, ok_all: np.ndarray, idx: np.ndarray, tau: float, depth: np.ndarray) -> tuple[float, float]:
    """(mean depth fraction, accuracy) of 'stop at the first exit whose batch-mean confidence >= tau'."""
    mc = tr.conf[idx].mean(1)                                   # [K, E]
    hit = mc[:, :-1] >= tau
    ex = np.where(hit.any(1), hit.argmax(1), mc.shape[1] - 1)   # [K]
    ok = ok_all[idx, ex[:, None]]
    return float(depth[ex].mean()), float(ok.mean())


def static_frontier(ok_all: np.ndarray, depth: np.ndarray, budget: float) -> tuple[int, float, float]:
    """Shallowest random mix of two adjacent exits whose accuracy drop is <= budget. Returns (e_lo, p_lo, depth).
    The last "exit" is the full model (drop 0), so the loop always returns."""
    acc = ok_all.mean(0)
    drop = acc[-1] - acc
    E = len(acc)
    for e in range(E):
        if drop[e] <= budget:
            if e == 0:
                return 0, 1.0, float(depth[0])
            # mix exit e-1 (prob p) and e (prob 1-p) so that the drop equals the budget
            p = (budget - drop[e]) / (drop[e - 1] - drop[e]) if drop[e - 1] > drop[e] else 0.0
            p = float(np.clip(p, 0, 1))
            return e - 1, p, float(p * depth[e - 1] + (1 - p) * depth[e])
    raise AssertionError("unreachable: the full model has drop 0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["mnli", "qqp", "wikitext_llama", "wikitext_ls8b"])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.03])
    ap.add_argument("--cap", type=int, default=20000, help="max batches per split and batch size")
    ap.add_argument("--metric", default="accuracy", choices=["accuracy", "fidelity"],
                    help="what the budget is on: accuracy against the label, or agreement with the full model")
    args = ap.parse_args()
    rows = []
    for task in args.tasks:
        path = ROOT / f"traces/{task}_val.parquet"
        if not path.exists():
            print("missing", path)
            continue
        meta = ROOT / f"traces/{task}_val.meta.json"            # written by decoder/train_exit_lora.py
        ex = __import__("json").loads(meta.read_text())["exit_layers"] if meta.exists() else EXITS.get(task)
        ev, ho = load_traces(path, "evolve", exit_layers=ex), load_traces(path, "heldout", exit_layers=ex)
        depth = np.array(ev.exit_layers) / ev.exit_layers[-1]
        if args.metric == "fidelity":
            ok_ev, ok_ho = ev.pred == ev.pred[:, [-1]], ho.pred == ho.pred[:, [-1]]
        else:
            ok_ev, ok_ho = ev.correct, ho.correct
        full_ev, full_ho = ok_ev[:, -1].mean(), ok_ho[:, -1].mean()
        agree = (ho.pred[:, :-1] == ho.pred[:, [-1]]).mean(0)
        print(f"\n{task}: full acc {ho.correct[:, -1].mean():.4f}; agreement with the full model per exit {np.round(agree, 3).tolist()}")
        for budget in args.budgets:
            e_lo, p, _ = static_frontier(ok_ev, depth, budget)
            acc_ho = ok_ho.mean(0)
            st_depth = p * depth[e_lo] + (1 - p) * depth[e_lo + 1]
            st_drop = full_ho - (p * acc_ho[e_lo] + (1 - p) * acc_ho[e_lo + 1])
            for B in BS:
                be, bh = batches(len(ev), B, args.cap, B), batches(len(ho), B, args.cap, 100 + B)
                best = (1.0, 1.01)
                for tau in TAUS:
                    dpt, acc = adaptive(ev, ok_ev, be, tau, depth)
                    if full_ev - acc <= budget and dpt < best[0]:
                        best = (dpt, tau)
                dpt, acc = adaptive(ho, ok_ho, bh, best[1], depth)
                sd = float(ho.conf[bh].mean(1).std(0)[:-1].mean())        # spread of the batch-mean confidence
                rows.append({"task": task, "metric": args.metric, "budget_pts": 100 * budget, "B": B, "static_depth": st_depth,
                             "static_drop_pts": 100 * st_drop, "adaptive_depth": dpt,
                             "adaptive_drop_pts": 100 * (full_ho - acc), "tau": best[1],
                             "static_speedup": 1 / st_depth, "adaptive_speedup": 1 / dpt,
                             "gain_over_static_pct": 100 * (st_depth / dpt - 1), "batch_mean_conf_sd": sd})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / f"results/adaptivity_vs_batch{'_fidelity' if args.metric == 'fidelity' else ''}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
