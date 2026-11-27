import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from minrank_agent import bounds as B  # noqa: E402
from minrank_agent import structgraph as SG  # noqa: E402
from minrank_agent.agent import SpectrumAgent, solve  # noqa: E402


def test_spectral_fits_and_rank():
    g = SG.kneser(7, 2)
    for p in (2, 3, 5):
        cost, c, lam = B.spectral_field(g.A, p)
        assert B.verify_spectral(g.A, p, c, cost)          # M = I + cA fits and has the claimed rank


def test_kneser_proven_cases_and_field():
    # the proof's examples: the gap closes only over the right field
    for n, k, beta, field in [(7, 2, 6, 3), (9, 2, 8, 5), (10, 3, 36, 2)]:
        g = SG.kneser(n, k)
        r = solve(g, SpectrumAgent())
        assert r["proven_beta"] == beta, (g.name, r["proven_beta"], beta)
        assert r["best_field"] == field, (g.name, r["best_field"], field)
        assert r["alpha"] == beta


def test_alpha_is_lower_bound_everywhere():
    for g in (SG.johnson(6, 2), SG.paley(13), SG.random_gnp(20, 0.5, 3)):
        r = solve(g, SpectrumAgent())
        assert r["lower"] <= r["upper"]
        assert r["alpha"] <= r["clique_cover"]


def test_clique_cover_is_a_clique_cover():
    g = SG.kneser(8, 3)
    k, colour, _ = B.clique_cover(g.A, 10)
    for c in np.unique(colour):
        vs = np.nonzero(colour == c)[0]
        assert g.A[np.ix_(vs, vs)].sum() == len(vs) * (len(vs) - 1)


def test_random_graph_gap_stays_open():
    g = SG.random_gnp(30, 0.5, 1)                            # no structure: spectral cannot close
    r = solve(g, SpectrumAgent())
    assert r["proven_beta"] is None


def test_kneser_agent_explicit_code():
    from minrank_agent.kneser_agent import simulate_code, solve as ksolve

    for n, k, beta, p in [(7, 2, 6, 3), (10, 3, 36, 2), (11, 3, 45, 5)]:
        g = SG.kneser(n, k)
        r = ksolve(g.A)
        assert r["exact"] and r["upper"] == beta and r["field"] == p
        c = r["code"]
        assert c["transmissions"] == beta
        eps = 1 if k % 2 else -1
        assert simulate_code(g.A, p, eps, c["encoder"], c["decoder"], trials=5, seed=1)
