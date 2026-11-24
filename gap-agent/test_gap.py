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


def test_learned_bandit_runs():
    from gap_agent.agent import LearnedBandit

    class Flat:  # stand-in model: predicts the same rate for every strategy
        def predict(self, X):
            return np.ones(len(X))

    g = make_graph("planted", 120, np.random.default_rng(3))
    r = run(g, LearnedBandit(Flat()), T=1.0)
    assert r["lo"] <= r["hi"] and r["decisions"] > 0


def _brute_mais(adj):
    import itertools

    g = Graph(adj)
    for size in range(adj.shape[0], -1, -1):
        for S in itertools.combinations(range(adj.shape[0]), size):
            if is_acyclic_set(g, S):
                return size
    return 0


def test_kernel_rules_keep_the_optimum():
    from gap_agent.structure import kernel, lift_acyclic

    rng = np.random.default_rng(7)
    for _ in range(60):
        n = int(rng.integers(3, 9))
        a = rng.random((n, n)) < rng.uniform(0.15, 0.5)
        np.fill_diagonal(a, False)
        k = kernel(a)
        kadj = k.graph().adj if k.verts else np.zeros((0, 0), bool)
        assert _brute_mais(a) == len(k.keep) + (_brute_mais(kadj) if k.verts else 0)
        assert is_acyclic_set(Graph(a), lift_acyclic(k, []))


def test_cycle_of_receivers_saves_one_transmission():
    from gap_agent.lam import BlockSearch

    a = np.zeros((4, 4), bool)
    for i in range(4):
        a[i, (i + 1) % 4] = True                    # directed 4-cycle: clique cover 4, cycle code 3
    bs = BlockSearch(a, np.zeros(4, dtype=np.int64), np.arange(4))
    for _ in range(5):
        bs.cycle_step()
    assert int(bs.scc_cost[0]) == 3


def test_parallel_agent_small():
    import pickle

    from gap_agent.graph import make_structured
    from gap_agent.parallel import solve

    model = pickle.loads((Path(__file__).resolve().parent / "models" / "rate.pkl").read_bytes())
    g = make_structured("sparse", 400, np.random.default_rng(1))
    r = solve(g, model, T=4.0)
    assert r["verified"] and r["lo"] <= r["hi"]
