"""Phase-2 tests: certificate checks against brute force (n <= 12), Freivalds catches corrupted factorizations, the
decoding simulation passes on a proven code and fails on a corrupted one, and the Solver's rank engine agrees with the
Verifier's separate elimination."""
from __future__ import annotations

import itertools
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kagent.graphs import kneser, relabel  # noqa: E402
from kagent.rank import DenseMatrix, SpectralMatrix, rank_factor  # noqa: E402
from kagent.verify import checks as V  # noqa: E402

RNG = np.random.default_rng(2026)


def random_graph(n, p_edge, rng):
    A = np.triu(rng.random((n, n)) < p_edge, 1)
    return A | A.T


def small_graphs():
    out = [kneser(5, 2), kneser(6, 2)[:12, :12]]
    for n in (5, 7, 9, 10, 12):
        for pe in (0.3, 0.6):
            out.append(random_graph(n, pe, RNG))
    return out


def brute_rank(M, p):
    """rank over F_p by counting the row space: |span| = p^rank."""
    rows, cols = M.shape
    span = set()
    for coef in itertools.product(range(p), repeat=rows):
        span.add(tuple((np.array(coef) @ M) % p))
    r = 0
    while p ** r < len(span):
        r += 1
    return r


def is_indep_bf(A, S):
    return all(not A[u, v] for u, v in itertools.combinations(S, 2))


def is_clique_bf(A, S):
    return all(A[u, v] for u, v in itertools.combinations(S, 2))


# ------------------------------------------------------------------------------------------------ brute force
@pytest.mark.parametrize("gi", range(12))
def test_independent_set_check_matches_brute_force(gi):
    A = small_graphs()[gi]
    n = A.shape[0]
    for _ in range(40):
        S = [v for v in range(n) if RNG.random() < 0.35]
        ok, _, _ = V.check_independent_set(A, S)
        assert ok == is_indep_bf(A, S)


@pytest.mark.parametrize("gi", range(12))
def test_clique_cover_check_matches_brute_force(gi):
    A = small_graphs()[gi]
    n = A.shape[0]
    for _ in range(40):
        lab = RNG.integers(0, max(1, n // 2), n)
        parts = [np.nonzero(lab == x)[0].tolist() for x in np.unique(lab)]
        if RNG.random() < 0.3:
            parts = parts[:-1]                                     # sometimes drop a block (uncovered vertices)
        ok, _, _ = V.check_clique_cover(A, parts)
        expect = all(is_clique_bf(A, P) for P in parts) and sorted(v for P in parts for v in P) == list(range(n))
        assert ok == expect


@pytest.mark.parametrize("gi", range(12))
def test_fractional_cover_check_matches_brute_force(gi):
    A = small_graphs()[gi]
    n = A.shape[0]
    cliques = [list(c) for k in (1, 2, 3) for c in itertools.combinations(range(n), k) if is_clique_bf(A, c)]
    for _ in range(30):
        pick = [cliques[i] for i in RNG.choice(len(cliques), size=min(len(cliques), n), replace=False)]
        w = [Fraction(int(RNG.integers(0, 4)), 2) for _ in pick]
        if RNG.random() < 0.3:
            pick = pick + [list(range(n))]                          # usually not a clique
            w = w + [Fraction(1, 3)]
        ok, _, val = V.check_fractional_cover(A, pick, w)
        cov = [sum((wi for C, wi in zip(pick, w) if v in C), Fraction(0)) for v in range(n)]
        expect = all(is_clique_bf(A, C) for C in pick) and min(cov) >= 1
        assert ok == expect
        if ok:
            assert val == sum(w, Fraction(0))


@pytest.mark.parametrize("p", [2, 3, 5])
def test_own_rank_matches_brute_force(p):
    for _ in range(25):
        r, c = int(RNG.integers(1, 6)), int(RNG.integers(1, 7))
        if p ** r > 4000:
            r = 4
        M = RNG.integers(0, p, (r, c))
        assert V.own_rank_mod_p(M, p) == brute_rank(M, p)


@pytest.mark.parametrize("p", [2, 3, 5, 7, 11])
def test_rank_engine_matches_verifier_elimination(p):
    for _ in range(20):
        n = int(RNG.integers(4, 40))
        k = int(RNG.integers(0, n + 1))
        M = (RNG.integers(0, p, (n, k)) @ RNG.integers(0, p, (k, n))) % p
        res = rank_factor(DenseMatrix(M, p))
        assert res.rank == V.own_rank_mod_p(M, p)
        assert np.array_equal((res.C @ M[res.S]) % p, M)


def test_rank_engine_cap():
    A = kneser(7, 2)
    full = rank_factor(SpectralMatrix(A, 3, 2))
    assert full.rank is not None
    assert rank_factor(SpectralMatrix(A, 3, 2), cap=full.rank - 1).rank is None
    assert rank_factor(SpectralMatrix(A, 3, 2), cap=full.rank).rank == full.rank


def test_spectral_rank_agrees_with_verifier_on_kneser():
    for (n, k) in ((7, 2), (9, 2), (8, 3)):
        A, _ = relabel(kneser(n, k), 1)
        for p in (2, 3, 5, 7):
            for c in range(1, p):
                res = rank_factor(SpectralMatrix(A, p, c))
                assert res.rank == V.own_rank_mod_p(V.spectral_matrix(A, p, c), p)


# ------------------------------------------------------------------------------------------------ Freivalds / decoding
def proven_code():
    """K(7,2) over F_3, c = -1/1: rank 6 = alpha (the verifier re-derives the rank; nothing is assumed here)."""
    A, _ = relabel(kneser(7, 2), 0)
    p, c = 3, 2
    res = rank_factor(SpectralMatrix(A, p, c))
    return A, p, c, res


def test_freivalds_accepts_true_factorization():
    A, p, c, res = proven_code()
    r = V.check_spectral_code(A, p, c, res.S, res.C)
    assert r["ok"] and r["freivalds_passed"] == 30 and r["exact_rank_own"] == res.rank


@pytest.mark.parametrize("trial", range(10))
def test_freivalds_catches_corrupted_factorization(trial):
    A, p, c, res = proven_code()
    rng = np.random.default_rng(trial)
    C = res.C.copy()
    i, j = rng.integers(0, C.shape[0]), rng.integers(0, C.shape[1])
    C[i, j] = (C[i, j] + 1 + rng.integers(0, p - 1)) % p          # one wrong coefficient
    r = V.check_spectral_code(A, p, c, res.S, C, exact_max=0)
    assert not r["ok"]


def test_freivalds_catches_dropped_basis_row():
    A, p, c, res = proven_code()
    S, C = res.S[:-1], res.C[:, :-1]                                # claims a shorter code
    r = V.check_spectral_code(A, p, c, S, C)
    assert not r["ok"]


def test_decoding_passes_on_proven_code():
    A, p, c, res = proven_code()
    sim = V.simulate_decoding(A, p, c, res.S, res.C)
    assert sim["ok"] and sim["failed_decodings"] == 0


def test_decoding_fails_on_corrupted_code():
    A, p, c, res = proven_code()
    C = res.C.copy()
    C[0] = (C[0] + 1) % p
    assert not V.simulate_decoding(A, p, c, res.S, C)["ok"]
    sim = V.simulate_decoding(A, p, (c + 1) % p or 1, res.S, res.C)  # wrong field constant
    assert not sim["ok"]


def test_clique_decoding():
    A, _ = relabel(kneser(6, 2), 0)
    from kagent.tools.covers import clique_cover

    cl = clique_cover(A, 1.0)
    assert V.simulate_clique_decoding(A, cl)["ok"]
    bad = [cl[0] + cl[1]] + cl[2:]                                   # merged blocks: not a clique
    assert not V.simulate_clique_decoding(A, bad)["ok"]


# ------------------------------------------------------------------------------------------------ tools
def test_tools_on_small_kneser():
    from kagent.tools.covers import clique_cover, frac_cover
    from kagent.tools.indep import indep_cpsat, indep_ils

    A, _ = relabel(kneser(8, 3), 0)
    S = indep_ils(A, 1.0)
    assert V.check_independent_set(A, S)[0]
    S2, opt, ub = indep_cpsat(A, 10)
    assert opt and len(S2) == ub >= len(S)
    cl = clique_cover(A, 2.0)
    assert V.check_clique_cover(A, cl)[0]
    fc, w, val, _ = frac_cover(A, 5.0, init_cliques=cl)
    ok, _, v2 = V.check_fractional_cover(A, fc, w)
    assert ok and v2 == val and val <= len(cl)


def test_verifier_does_not_import_solver_code():
    import ast

    for f in ("checks.py", "stage.py"):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "kagent" / "verify" / f).read_text())
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods.append(("." * node.level) + (node.module or ""))
        for m in mods:
            assert "rank" not in m and "tools" not in m and "flint" not in m, (f, m)
