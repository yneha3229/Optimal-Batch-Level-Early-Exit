"""Tests for lambda_agent.  Run:  python -m pytest lambda-agent -q"""
import itertools
import random
import sys
from fractions import Fraction
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lambda_agent.agent import LambdaAgent  # noqa: E402
from lambda_agent.graph import Digraph, bits, blowup, scc  # noqa: E402
from lambda_agent.lower import mais, shannon  # noqa: E402
from lambda_agent.upper import clique_cover, frac_cover, lift_fix, minrank_sat  # noqa: E402
from lambda_agent.verify import fitting_rank, is_acyclic, rank_gf2  # noqa: E402


def ring(n):
    return Digraph.from_arcs(n, [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)], f"C{n}")


def rand(n, p, seed):
    rng = random.Random(seed)
    return Digraph.from_arcs(n, [(i, j) for i in range(n) for j in range(n) if i != j and rng.random() < p], f"r{seed}")


def brute_minrank(h: Digraph) -> int:
    """Smallest GF(2) rank over every matrix that fits h (enumerate the free entries)."""
    free = [(p, q) for p in range(h.n) for q in bits(h.out[p])]
    assert len(free) <= 16
    best = h.n
    for choice in itertools.product((0, 1), repeat=len(free)):
        rows = [1 << p for p in range(h.n)]
        for (p, q), b in zip(free, choice):
            if b:
                rows[p] |= 1 << q
        best = min(best, rank_gf2(rows))
    return best


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("t", [1, 2])
def test_sat_matches_brute_force(seed, t):
    g = rand(3 if t == 2 else 4, 0.45, seed)
    h = blowup(g, t)
    if sum(1 for p in range(h.n) for _ in bits(h.out[p])) > 16:
        pytest.skip("too many free entries")
    opt = brute_minrank(h)
    I = lift_fix(mais(g)[0], t)
    assert minrank_sat(h, opt, 20, I) not in (None, False)
    if opt > 1:
        assert minrank_sat(h, opt - 1, 20, I) is False
        assert minrank_sat(h, opt - 1, 20, None) is False


def test_blowup_and_lifted_fix():
    g = rand(6, 0.4, 3)
    h = blowup(g, 3)
    assert h.n == 18
    for i, j in g.arcs():
        assert h.knows(3 * i + 1, 3 * j + 2)
    assert not h.knows(0, 1)  # copies of the same receiver never know each other
    a, _ = mais(g)
    assert is_acyclic(h, lift_fix(a, 3))


def test_bounds_on_odd_cycles():
    for n in (5, 7):
        g = ring(n)
        assert shannon(g) == Fraction(n, 2)
        assert frac_cover(g)[0] == Fraction(n, 2)
        assert len(clique_cover(g)[0]) == (n + 1) // 2
        rows = minrank_sat(blowup(g, 2), n, 30, lift_fix(mais(g)[0], 2))
        assert fitting_rank(blowup(g, 2), rows) == n   # a vector-linear code of rate n/2


def test_fitting_rank_rejects_bad_matrix():
    g = ring(5)
    h = blowup(g, 2)
    rows = [1 << p for p in range(h.n)]
    rows[0] |= 1 << 1          # copy (0,1) is not known to (0,0)
    with pytest.raises(ValueError):
        fitting_rank(h, rows)


def test_agent_pins_and_adds_over_components():
    g = Digraph.from_arcs(8, ring(5).arcs() + [(5, 6), (6, 7), (7, 5), (0, 5)], "C5+dc3")
    assert [len(c) for c in scc(g)] == [5, 3]
    a = LambdaAgent(g, budget=30).run()
    assert a.lo == a.hi == Fraction(5, 2) + 2
