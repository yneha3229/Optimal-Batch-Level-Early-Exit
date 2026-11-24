"""Train the learned agent (batch RL on real runs).

    python gap-agent/train.py            three rounds of data collection + fitted Q-iteration -> models/agent.pkl

Training graphs: 2000-8000 receivers, three families (random digraphs, planted cliques, layered digraphs), 30 s runs.
The test graphs in bench.py are different graphs, with 3000, 6000 and 10000 receivers.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gap_agent.agent import LearnedAgent, RandomSwitch, run  # noqa: E402
from gap_agent.graph import FAMILIES, make_graph  # noqa: E402
from gap_agent.learn import fitted_q  # noqa: E402
from gap_agent.strategies import compile_all  # noqa: E402

DATA = HERE / "data"
MODELS = HERE / "models"
T_TRAIN = 30.0
N_RANGE = (2000, 8001)
WORKERS = 3


def train_graph(i: int):
    rng = np.random.default_rng(10_000 + i)
    return make_graph(FAMILIES[i % 3], int(rng.integers(*N_RANGE)), rng)


def _init():
    compile_all()


def _episode(args):
    i, model_path, eps = args
    g = train_graph(i)
    if model_path is None:
        ctrl = RandomSwitch(stay=0.5, seed=i)
    else:
        ctrl = LearnedAgent(pickle.loads(Path(model_path).read_bytes()), epsilon=eps, seed=i)
    rec = []
    res = run(g, ctrl, T=T_TRAIN, seed=i, record=rec)
    return rec, {k: res[k] for k in ("graph", "n", "integral", "gap0", "gap")}


def collect(ids, model_path, eps):
    with mp.get_context("spawn").Pool(WORKERS, initializer=_init) as pool:
        out = pool.map(_episode, [(i, model_path, eps) for i in ids], chunksize=1)
    trans = [t for rec, _ in out for t in rec]
    integ = np.mean([s["integral"] for _, s in out])
    return trans, integ


def main():
    DATA.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    rounds = [(range(0, 150), None, 0.0), (range(150, 240), "round0", 0.15), (range(240, 330), "round1", 0.15)]
    allt = []
    for r, (ids, prev, eps) in enumerate(rounds):
        t0 = time.time()
        path = str(MODELS / f"{prev}.pkl") if prev else None
        trans, integ = collect(list(ids), path, eps)
        allt += trans
        print(f"round {r}: {len(ids)} runs, mean gap integral {integ:.3f} "
              f"({'random switching' if prev is None else 'learned agent, eps ' + str(eps)}), "
              f"{len(allt)} transitions, {time.time() - t0:.0f} s", flush=True)
        model = fitted_q(allt)
        (MODELS / f"round{r}.pkl").write_bytes(pickle.dumps(model))
    (MODELS / "agent.pkl").write_bytes(pickle.dumps(model))
    pickle.dump(allt, open(DATA / "transitions.pkl", "wb"))
    print("saved", MODELS / "agent.pkl")


if __name__ == "__main__":
    main()
