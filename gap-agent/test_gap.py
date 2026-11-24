import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap_agent.agent import A, Bandit, RandomSwitch, gap_integral, run  # noqa: E402
from gap_agent.graph import Graph, is_acyclic_set, is_clique_cover, make_graph  # noqa: E402
from gap_agent.strategies import STRATEGIES, Search, warm_start  # noqa: E402


def cycle(n):
    a = np.zeros((n, n), bool)
    for i in range(n):
        a[i, (i + 1) % n] = True
    return Graph(a, f"C{n}")


def test_checks():
    g = cycle(5)
    assert is_acyclic_set(g, [0, 1, 2, 3]) and not is_acyclic_set(g, range(5))
    two = Graph(np.array([[0, 1], [1, 0]], bool))
    assert not is_acyclic_set(two, [0, 1])          # a 2-cycle is a cycle
    assert is_clique_cover(two, np.array([0, 0]))   # mutual pair = one clique
    assert not is_clique_cover(g, np.zeros(5, int))


def test_every_strategy_keeps_valid_bounds():
    for fam in ("random", "planted", "layered"):
        g = make_graph(fam, 150, np.random.default_rng(1))
        s = Search(g, 0)
        warm_start(s)
        for name in STRATEGIES * 2:
            lo, hi = s.lo, s.hi
            s.run(name, 0.05)
            assert s.lo >= lo and s.hi <= hi
            assert is_acyclic_set(g, s.acyclic_set()) and len(s.acyclic_set()) == s.lo
            assert is_clique_cover(g, s.colour) and len(set(s.colour.tolist())) <= s.hi
            assert s.lo <= s.hi


def test_known_values():
    r = run(cycle(7), RandomSwitch(), T=1.0)       # directed 7-cycle: acyclic set 6, no mutual pairs -> cover 7
    assert (r["lo"], r["hi"]) == (6, 7)
    k = np.ones((6, 6), bool)
    r = run(Graph(k), RandomSwitch(), T=1.0)       # everyone knows everything else: one transmission
    assert (r["lo"], r["hi"]) == (1, 1)


def test_gap_integral():
    ev = [(5.0, "hi", 8, "x"), (5.0, "lo", 4, "y")]
    # gap 10 for 5 s then gap 4 for 5 s, normalised by 10, over 10 s
    assert abs(gap_integral(ev, 10.0, 2, 12, 10) - (10 * 5 + 4 * 5) / 10 / 10) < 1e-12


def test_controllers_and_record():
    g = make_graph("layered", 120, np.random.default_rng(2))
    rec = []
    r = run(g, Bandit(), T=1.0, record=rec)
    assert rec and all(0 <= a < A for _, a, _, _, _ in rec)
    assert r["lo"] <= r["hi"]
