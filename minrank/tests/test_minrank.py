"""Tests for minrank/: F_2 algebra, verifier, generators, baselines, lower bounds, SAT.
Run: python -m pytest minrank/tests -q
"""
from __future__ import annotations

import numpy as np
import pytest

from minrank.baselines.clique_cover import (clique_cover_rows, colors_to_rows, conflict_graph, dsatur, is_proper,
                                            num_colors, tabucol)
from minrank.baselines.factor_search import FactorState, constraint_mask, factor_search
from minrank.bench import generators as gen
from minrank.core.f2 import product_f2, rank_f2, rows_from_factors
from minrank.core.graph import Graph, bool_to_masks, masks_to_bool
from minrank.core.verify import fit_error, load_certificate, save_certificate, verify
from minrank.exact.lower import is_acyclic_induced, mais
from minrank.exact.sat import exact_minrank, minrank_bruteforce, rank_at_most


def rank_numpy_f2(a: np.ndarray) -> int:
    """Independent reference: dense Gaussian elimination over F_2."""
    a = a.copy().astype(np.uint8) % 2
    r = 0
    rows, cols = a.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if a[i, c]), None)
        if piv is None:
            continue
        a[[r, piv]] = a[[piv, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        r += 1
    return r


# ---- F_2 algebra -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_rank_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n, r = int(rng.integers(3, 40)), int(rng.integers(1, 12))
    U = rng.integers(0, 2, (n, r))
    V = rng.integers(0, 2, (n, r))
    P = product_f2(U, V)
    assert P.shape == (n, n) and P.dtype == np.uint8
    rows = rows_from_factors(U, V)
    assert rank_f2(rows) == rank_numpy_f2(P) <= min(r, n)


def test_rank_edge_cases():
    assert rank_f2([]) == 0
    assert rank_f2([0, 0]) == 0
    assert rank_f2([1, 1, 1]) == 1
    assert rank_f2([1 << k for k in range(70)]) == 70  # beyond 64 bits
    with pytest.raises(ValueError):
        rank_f2([-1])


def test_mask_roundtrip():
    rng = np.random.default_rng(0)
    a = rng.random((7, 9)) < 0.5
    assert (masks_to_bool(bool_to_masks(a), 9) == a).all()


# ---- graphs and verifier -------------------------------------------------------------------------------------------
def test_graph_validation_and_io(tmp_path):
    g = gen.gnp(30, 0.4, 1)
    p = tmp_path / "g.json"
    g.save(p)
    assert Graph.load(p) == g
    with pytest.raises(ValueError):
        Graph(2, (0b10, 0b00), directed=False)  # asymmetric
    with pytest.raises(ValueError):
        Graph(2, (0b01, 0b00), directed=True)  # self loop


def test_generators_basic():
    assert gen.cycle(5).num_edges() == 10
    assert gen.mycielski(3).n == 5 and gen.mycielski(4).n == 11 and gen.mycielski(4).num_edges() == 40
    k = gen.kneser(5, 2)  # Petersen graph
    assert k.n == 10 and all(m.bit_count() == 3 for m in k.out)
    p = gen.paley(13)
    assert all(m.bit_count() == 6 for m in p.out)
    d = gen.gnp(20, 0.5, 3, directed=True)
    assert d.directed and d.mutual().n == 20


def test_identity_and_clique_matrices_fit():
    g = gen.gnp(25, 0.5, 2)
    assert verify(g, [1 << i for i in range(g.n)]) == g.n
    rows = colors_to_rows(dsatur(conflict_graph(g)))
    assert fit_error(g, rows) is None


def test_verifier_rejects_bad_matrices(tmp_path):
    g = gen.cycle(5)
    assert fit_error(g, [0] + [1 << i for i in range(1, 5)]) is not None  # zero diagonal
    assert fit_error(g, [(1 << 0) | (1 << 2)] + [1 << i for i in range(1, 5)]) is not None  # 2 not in S_0
    rows = [1 << i for i in range(5)]
    path = tmp_path / "c.json"
    assert save_certificate(path, g, rows, "identity") == 5
    c = load_certificate(path)
    assert c["rows"] == rows and c["digest"] == g.digest()
    with pytest.raises(ValueError):
        verify(g, rows, claimed_rank=4)


# ---- known values --------------------------------------------------------------------------------------------------
def test_c5_minrank_is_3():
    g = gen.cycle(5)
    assert minrank_bruteforce(g) == 3
    assert rank_at_most(g, 2) is False
    rows = rank_at_most(g, 3)
    assert rows and verify(g, rows) <= 3
    assert mais(g).value == 2


def test_complete_and_empty_graphs():
    n = 6
    complete = Graph.from_edges(n, [(i, j) for i in range(n) for j in range(i + 1, n)], False)
    empty = Graph.from_edges(n, [], False)
    rows = rank_at_most(complete, 1)
    assert rows and verify(complete, rows) == 1
    assert rank_at_most(empty, n - 1) is False


@pytest.mark.parametrize("seed", range(12))
def test_sat_matches_bruteforce_small(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 7))
    directed = bool(seed % 2)
    g = gen.gnp(n, float(rng.uniform(0.2, 0.8)), seed, directed=directed)
    if g.num_edges() > 16:
        pytest.skip("too many free entries for brute force")
    bf = minrank_bruteforce(g)
    best, proven, rows = exact_minrank(g, [1 << i for i in range(n)])
    assert proven and best == bf
    lb = mais(g)
    assert lb.value <= bf
    assert is_acyclic_induced(g, lb.vertices)


# ---- baselines -----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["dsatur", "random_greedy", "tabucol"])
def test_clique_cover_methods(method):
    g = gen.gnp(40, 0.5, 4)
    rows = clique_cover_rows(g, method, 0.3, np.random.default_rng(0))
    r = verify(g, rows)
    assert mais(g, 5).value <= r


def test_tabucol_not_worse_than_dsatur():
    g = gen.gnp(50, 0.5, 5)
    conf = conflict_graph(g)
    t = tabucol(conf, 1.0, np.random.default_rng(1))
    assert is_proper(conf, t) and num_colors(t) <= num_colors(dsatur(conf))


def test_factor_state_deltas_are_exact():
    """Predicted delta of every move equals the actual change in violations."""
    g = gen.gnp(12, 0.5, 6)
    rng = np.random.default_rng(0)
    M = constraint_mask(g)
    U = rng.integers(0, 2, (12, 4))
    V = rng.integers(0, 2, (12, 4))
    st = FactorState(M, U, V)
    du, dv = st.deltas()
    for i in range(12):
        for k in range(4):
            s2 = FactorState(M, U, V)
            f0 = s2.f
            s2.flip_u(i, k)
            assert s2.f - f0 == int(du[i, k])
            s3 = FactorState(M, U, V)
            s3.flip_v(i, k)
            assert s3.f - f0 == int(dv[i, k])
            # incremental cost equals a fresh recount
            assert s2.f == FactorState(M, *s2.factors()).f


@pytest.mark.parametrize("mode", ["tabu", "anneal"])
def test_factor_search_valid_and_bounded(mode):
    g = gen.gnp(30, 0.5, 7)
    res = factor_search(g, 1.5, np.random.default_rng(0), mode=mode)
    assert verify(g, res.rows) == res.rank
    assert res.rank <= res.U.shape[1] + 1
    lb = mais(g, 5).value
    assert lb <= res.rank <= num_colors(dsatur(conflict_graph(g)))


def test_factor_search_reaches_c5_optimum():
    res = factor_search(gen.cycle(5), 1.0, np.random.default_rng(0))
    assert res.rank == 3


def test_directed_pipeline():
    g = gen.gnp(20, 0.6, 8, directed=True)
    res = factor_search(g, 1.0, np.random.default_rng(0))
    assert verify(g, res.rows) == res.rank
    assert mais(g, 5).value <= res.rank
