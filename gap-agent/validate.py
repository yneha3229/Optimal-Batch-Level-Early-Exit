"""Model selection on validation graphs (never the test graphs): fresh graphs from the training generator with
3000, 6000 and 10000 receivers, 30 s each.

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

from bench import FixedPair  # noqa: E402
from gap_agent.agent import Bandit, LearnedAgent, LearnedBandit, run  # noqa: E402
from gap_agent.graph import FAMILIES, make_graph  # noqa: E402
from gap_agent.learn import rate_model  # noqa: E402
from gap_agent.strategies import compile_all  # noqa: E402

SIZES = (3000, 6000, 10000)


def val_graph(i):
    rng = np.random.default_rng(500_000 + i)
    return make_graph(FAMILIES[i % 3], SIZES[(i // 3) % 3], rng)


def make(key):
    if key == "fqi":
        return LearnedAgent(pickle.loads((HERE / "models" / "agent.pkl").read_bytes()))
    if key.startswith("prior"):
        k0 = float(key[5:])
        return LearnedBandit(pickle.loads((HERE / "models" / "rate.pkl").read_bytes()), k0=k0)
    return {"bandit": Bandit(), "fixedpair": FixedPair()}[key]


def _job(args):
    i, key = args
    r = run(val_graph(i), make(key), T=30.0, seed=i)
    return {"i": i, "key": key, "integral": r["integral"], "share": r["share"], "lo": r["lo"], "hi": r["hi"]}


def main(keys=("bandit", "fixedpair", "fqi", "prior1", "prior4", "prior16"), graphs=9):
    p = HERE / "models" / "rate.pkl"
    if not p.exists():
        p.write_bytes(pickle.dumps(rate_model(pickle.load(open(HERE / "data" / "transitions.pkl", "rb")))))
    jobs = [(i, k) for i in range(graphs) for k in keys]
    with mp.get_context("spawn").Pool(3, initializer=compile_all) as pool:
        rows = pool.map(_job, jobs, chunksize=1)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "validation.json").write_text(json.dumps(rows))
    for k in keys:
        rs = [r for r in rows if r["key"] == k]
        fam = {f: np.mean([r["integral"] for r in rs if r["i"] % 3 == j]) for j, f in enumerate(FAMILIES)}
        print(f"{k:10s} mean {np.mean([r['integral'] for r in rs]):.3f}  "
              + "  ".join(f"{f} {v:.3f}" for f, v in fam.items()), flush=True)


if __name__ == "__main__":
    main()
