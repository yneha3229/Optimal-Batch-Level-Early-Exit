"""Shared scorer for evolved programs, random search and BO (equal budget = equal number of scored candidates).

score(regime, B) = speedup * exp(-100 * max(0, full_acc - 0.01 - acc))     (speedup only counts at matched accuracy)
combined_score   = geometric mean over regimes x batch sizes
Two regimes (small-rho encoder cost, large-rho decoder-like cost) so a program only scores well everywhere if it
adapts to rho = a / (b n), which is what the theory rule does.

Evolved programs only ever receive confidences through decide(); labels are used here, in the judge, never by them.
Search uses the evolve split; report.py re-scores the winner on the held-out split.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.simulator import Cost, draw_batches, load_traces, simulate  # noqa: E402

TRACES = Path(os.environ.get("IE624_TRACES", ROOT / "traces/mnli_val.parquet"))
REGIMES = ("enc_eager_meas", "dec_eager")
BATCH_SIZES = (16, 64)
N_BATCHES = int(os.environ.get("IE624_EVOLVE_BATCHES", 80))
TOL = 0.01


@lru_cache(maxsize=None)
def costs() -> dict[str, Cost]:
    from simulator.run_sim import load_costs
    return {c.name: c for c in load_costs()}


@lru_cache(maxsize=None)
def traces(split: str):
    return load_traces(TRACES, split)


@lru_cache(maxsize=None)
def batches(split: str, B: int):
    return draw_batches(len(traces(split)), B, N_BATCHES, seed=1234 + B + (0 if split == "evolve" else 1))


def score_decide(decide, split: str = "evolve", regimes=REGIMES, batch_sizes=BATCH_SIZES) -> dict:
    tr = traces(split)
    out, parts = {}, []
    for reg in regimes:
        cost = costs()[reg]
        for B in batch_sizes:
            bt = batches(split, B)
            full_acc = float(tr.correct[np.concatenate(bt), -1].mean())
            r = simulate(decide, tr, cost, B, len(bt), batches=bt)
            viol = max(0.0, full_acc - TOL - r["accuracy"])
            s = r["speedup"] * float(np.exp(-100 * viol))
            parts.append(s)
            out[f"{reg}_B{B}_speedup"] = r["speedup"]
            out[f"{reg}_B{B}_acc_drop"] = full_acc - r["accuracy"]
            out[f"{reg}_B{B}_score"] = s
    out["combined_score"] = float(np.exp(np.mean(np.log(np.maximum(parts, 1e-6)))))
    return out


if __name__ == "__main__":
    from simulator.policies import threshold
    print(json.dumps(score_decide(threshold(0.9)), indent=1))
