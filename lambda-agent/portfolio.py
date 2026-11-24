"""The live portfolio: an offline fine-tuned LLM controls many methods running in parallel to pin lambda_2.

    python lambda-agent/portfolio.py suites                 train.json / test.json graph suites
    python lambda-agent/portfolio.py log train|test         log every move alone (replay data), resumable
    python lambda-agent/portfolio.py data                   supervised examples for the LLM (from train logs)
    python lambda-agent/portfolio.py train                  LoRA fine-tune of the small LLM (CPU)
    python lambda-agent/portfolio.py replay                 score all controllers on the test logs (replay)
    python lambda-agent/portfolio.py live                   live runs on test graphs (real processes)
    python lambda-agent/portfolio.py headroom train hard    how much could ANY controller gain? (oracle vs rules)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lambda_agent.graph import Digraph  # noqa: E402

DATA = HERE / "portfolio_data"


def _random(n, p, two, rng, undirected=False):
    arcs = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() >= p:
                continue
            if undirected or rng.random() < two:
                arcs += [(i, j), (j, i)]
            else:
                arcs.append((i, j) if rng.random() < 0.5 else (j, i))
    return arcs


def _circulant(n, jumps, directed):
    arcs = {(i, (i + d) % n) for i in range(n) for d in jumps}
    if not directed:
        arcs |= {(j, i) for i, j in arcs}
    return sorted(arcs)


def _complement(n, arcs):
    s = set(arcs)
    return [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in s]


def _union(parts, rng, p_link):
    """Disjoint union of (n, arcs) parts with a few one-way arcs from earlier to later parts (several SCCs)."""
    arcs, off, offs = [], 0, []
    for n, a in parts:
        arcs += [(i + off, j + off) for i, j in a]
        offs.append((off, n))
        off += n
    for x in range(len(offs)):
        for y in range(x + 1, len(offs)):
            for i in range(offs[x][0], offs[x][0] + offs[x][1]):
                for j in range(offs[y][0], offs[y][0] + offs[y][1]):
                    if rng.random() < p_link:
                        arcs.append((i, j))
    return off, arcs


def make_suite(seed0: int, count: dict, big: bool) -> list[dict]:
    rng = random.Random(seed0)
    out = []

    def add(name, n, arcs):
        out.append(Digraph.from_arcs(n, arcs, f"{name}_{len(out)}").to_json())

    def structured():
        kind = rng.choice(["odd", "odd_c", "circ", "dcirc"])
        if kind in ("odd", "odd_c"):
            n = rng.choice([5, 7, 9, 11])
            a = _circulant(n, [1], False)
            return kind, n, (a if kind == "odd" else _complement(n, a))
        n = rng.randint(6, 13)
        jumps = rng.sample(range(1, n // 2 + 1), rng.randint(1, 2))
        return kind, n, _circulant(n, jumps, kind == "dcirc")

    for _ in range(count["digraph"]):
        n = rng.randint(8, 50 if big else 26)
        add("digraph", n, _random(n, rng.uniform(0.15, 0.6), rng.uniform(0.2, 0.8), rng))
    for _ in range(count["undirected"]):
        n = rng.randint(7, 30 if big else 18)
        add("undirected", n, _random(n, rng.uniform(0.2, 0.6), 1.0, rng, True))
    for _ in range(count["structured"]):
        kind, n, a = structured()
        add(kind, n, a)
    for _ in range(count["union"]):
        parts = [structured()[1:] for _ in range(rng.randint(2, 3))]
        n, a = _union(parts, rng, 0.05)
        add("union", n, a)
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    DATA.mkdir(exist_ok=True)
    if cmd == "suites":
        tr = make_suite(1, {"digraph": 90, "undirected": 40, "structured": 50, "union": 30}, big=False)
        te = make_suite(1000, {"digraph": 30, "undirected": 15, "structured": 20, "union": 15}, big=True)
        (DATA / "train.json").write_text(json.dumps(tr))
        (DATA / "test.json").write_text(json.dumps(te))
        print(len(tr), "train graphs,", len(te), "test graphs")
    elif cmd == "log":
        from lambda_agent.replay import log_suite

        which = sys.argv[2]
        log_suite(DATA / f"{which}.json", DATA / f"logs_{which}", cap=8.0, workers=3)
    elif cmd == "headroom":
        import glob
        import statistics as st

        from lambda_agent.controllers import OracleController, RuleController, StaticController
        from lambda_agent.replay import Sim

        for which in sys.argv[2:]:
            logs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(DATA / f"logs_{which}" / "*.json")))]
            print(f"== {which}: {len(logs)} graphs")
            for slots in (1, 3):
                for C in (StaticController, RuleController, OracleController):
                    rows = [Sim(L, C(), slots=slots, budget=60).run() for L in logs]
                    pinned = [r for r in rows if r["pinned"]]
                    print(f"  slots {slots} {C.name:30s} pinned {len(pinned)}/{len(rows)}  mean time on pinned "
                          f"{st.mean(r['time'] for r in pinned) if pinned else 0:.3f} s  total gap {sum(r['gap'] for r in rows):.2f}")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
