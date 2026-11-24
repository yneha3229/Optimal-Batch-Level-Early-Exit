"""Model selection on validation graphs (never the test graphs): refit the Q-function with different horizons
(gamma) on the same training transitions and compare on fresh graphs from the training generator.

    python gap-agent/validate.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gap_agent.agent import LearnedAgent, RandomSwitch, run  # noqa: E402
from gap_agent.graph import FAMILIES, make_graph  # noqa: E402
from gap_agent.learn import fitted_q  # noqa: E402
from gap_agent.strategies import compile_all  # noqa: E402
from bench import FixedPair  # noqa: E402

GAMMAS = (0.0, 0.5, 0.8, 0.97)


def val_graph(i):
    rng = np.random.default_rng(500_000 + i)
    return make_graph(FAMILIES[i % 3], (500, 1000, 2000)[(i // 3) % 3], rng)


def _job(args):
    i, key = args
    g = val_graph(i)
    if key.startswith("gamma"):
        ctrl = LearnedAgent(pickle.loads((HERE / "models" / f"val_{key}.pkl").read_bytes()))
    else:
        ctrl = {"random": RandomSwitch(0.0, 1), "fixedpair": FixedPair()}[key]
    r = run(g, ctrl, T=20.0, seed=i)
    return {"i": i, "key": key, "integral": r["integral"], "share": r["share"]}


def main():
    trans = pickle.load(open(HERE / "data" / "transitions.pkl", "rb"))
    for gm in GAMMAS:
        p = HERE / "models" / f"val_gamma{gm}.pkl"
        if not p.exists():
            p.write_bytes(pickle.dumps(fitted_q(trans, iters=1 if gm == 0 else 12, gamma=gm, log=lambda *a: None)))
    keys = [f"gamma{gm}" for gm in GAMMAS] + ["random", "fixedpair"]
    jobs = [(i, k) for i in range(18) for k in keys]
    with mp.get_context("spawn").Pool(3, initializer=compile_all) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "validation.json").write_text(json.dumps(rows))
    for k in keys:
        rs = [r for r in rows if r["key"] == k]
        fam = {f: np.mean([r["integral"] for r in rs if r["i"] % 3 == j]) for j, f in enumerate(FAMILIES)}
        print(f"{k:10s} mean {np.mean([r['integral'] for r in rs]):.3f}  " + "  ".join(f"{f} {v:.3f}" for f, v in fam.items()))


if __name__ == "__main__":
    main()
