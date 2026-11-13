"""Equal-budget search baselines for FunSearch: random search and Bayesian optimisation (GP + expected
improvement) over the hand-parameterised collective rule  param_collective(tau, frac_cut, conf_floor).
Both use the same scorer and split as the evolution, so "programs tried" is directly comparable.

Usage: ~/venv/bin/python evolve/baselines.py --budget 300
Output: results/evolve_baselines_<traces>.csv  (method, i, params, score, best_so_far)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evolve.score import TRACES, score_decide  # noqa: E402
from simulator.policies import param_collective  # noqa: E402

LO, HI = np.array([0.5, 0.0, 0.3]), np.array([1.0, 0.5, 0.95])


def f(x: np.ndarray) -> float:
    return score_decide(param_collective(*map(float, x)))["combined_score"]


def random_search(budget: int, seed: int) -> list[tuple]:
    rng = np.random.default_rng(seed)
    return [(x, f(x)) for x in (LO + rng.random(3) * (HI - LO) for _ in range(budget))]


def bayes_opt(budget: int, seed: int, n_init: int = 10) -> list[tuple]:
    rng = np.random.default_rng(seed)
    X = [LO + rng.random(3) * (HI - LO) for _ in range(n_init)]
    out = [(x, f(x)) for x in X]
    gp = GaussianProcessRegressor(Matern(nu=2.5) + WhiteKernel(1e-4), normalize_y=True, random_state=seed)
    while len(out) < budget:
        Xa = np.array([o[0] for o in out]); ya = np.array([o[1] for o in out])
        gp.fit((Xa - LO) / (HI - LO), ya)
        cand = rng.random((3000, 3))
        mu, sd = gp.predict(cand, return_std=True)
        z = (mu - ya.max()) / np.maximum(sd, 1e-9)
        ei = (mu - ya.max()) * norm.cdf(z) + sd * norm.pdf(z)
        x = LO + cand[ei.argmax()] * (HI - LO)
        out.append((x, f(x)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    rows = []
    for seed in args.seeds:
        for name, fn in [("random", random_search), ("bo", bayes_opt)]:
            best = -1.0
            for i, (x, s) in enumerate(fn(args.budget, seed)):
                best = max(best, s)
                rows.append({"method": name, "seed": seed, "i": i + 1, "params": json.dumps(np.round(x, 4).tolist()),
                             "score": s, "best_so_far": best})
            print(f"{name} seed {seed}: best {best:.4f}", flush=True)
    out = ROOT / f"results/evolve_baselines_{TRACES.stem}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    main()
