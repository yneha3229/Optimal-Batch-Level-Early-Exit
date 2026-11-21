"""Tests for Option B and baselines B3/B4/B10: LDG, cycle-of-cliques cover, SDP rounding, symmetry breaking,
F_q minrank, exact driver, separations.
Run: python -m pytest minrank/tests/test_option_b.py -q
"""
from __future__ import annotations

import csv

import numpy as np
import pytest

from minrank.baselines.ldg import ldg_groups, ldg_rows
from minrank.baselines.partial_cover import cycle_cover_rows, partial_cover_rows
from minrank.bench import generators as gen
from minrank.core.graph import Graph
from minrank.core.verify import verify
from minrank.exact.lower import mais
from minrank.exact.sat import minrank_bruteforce
from minrank.exact.sat_fq import (exact_minrank_fq, fit_error_fq, minrank_bruteforce_fq, rank_at_most_fq, rank_fq,
                                  verify_fq)
from minrank.exact.symmetry import VarPool, automorphisms, is_automorphism, lex_leq, rank_at_most_sym


def directed_cycle(m: int) -> Graph:
    return Graph.from_edges(m, [(i, (i + 1) % m) for i in range(m)], True, f"dcycle_{m}")


def exact_sym(g: Graph, mode: str) -> int:
    """Smallest r with rank_at_most_sym SAT (no time limit)."""
    for r in range(1, g.n + 1):
        if rank_at_most_sym(g, r, None, mode=mode):
            return r
    raise AssertionError("identity always fits")


# ---- B3 LDG --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("seed,directed", [(0, False), (1, False), (2, True), (3, True)])
def test_ldg_valid_and_bounded(seed, directed):
    g = gen.gnp(40, 0.5, seed, directed=directed)
    rows = ldg_rows(g)
    r = verify(g, rows)
    assert r == len(ldg_groups(g))
    assert mais(g, 5).value <= r <= g.n


def test_ldg_groups_are_mutual_cliques_and_partition():
    g = gen.gnp(30, 0.6, 5)
    groups = ldg_groups(g)
    assert sorted(v for grp in groups for v in grp) == list(range(g.n))
    for grp in groups:
        for a in grp:
            for b in grp:
                assert a == b or ((g.out[a] >> b) & 1 and (g.out[b] >> a) & 1)


def test_ldg_complete_graph_one_group():
    n = 7
    g = Graph.from_edges(n, [(i, j) for i in range(n) for j in range(i + 1, n)], False)
    assert verify(g, ldg_rows(g)) == 1


# ---- B4 cycle-of-cliques cover ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("m", [3, 4, 5, 8])
def test_directed_cycle_gets_m_minus_1(m):
    g = directed_cycle(m)
    rows = partial_cover_rows(g, 0.2, np.random.default_rng(0))
    assert verify(g, rows) == m - 1
    assert minrank_bruteforce(g) == m - 1  # and that is optimal


def test_cycle_of_cliques():
    # cliques {0,1}, {2,3}, {4} with arcs C0 -> C1 -> C2 -> C0 (all members to all members)
    edges = [(0, 1), (1, 0), (2, 3), (3, 2)]
    edges += [(a, b) for a in (0, 1) for b in (2, 3)] + [(a, 4) for a in (2, 3)] + [(4, b) for b in (0, 1)]
    g = Graph.from_edges(5, edges, True)
    rows, ncyc = cycle_cover_rows(g, [[0, 1], [2, 3], [4]])
    assert ncyc == 1 and verify(g, rows) == 2


@pytest.mark.parametrize("seed", range(4))
def test_partial_cover_valid_directed(seed):
    g = gen.gnp(30, 0.5, seed, directed=True)
    rows = partial_cover_rows(g, 0.3, np.random.default_rng(seed))
    assert mais(g, 5).value <= verify(g, rows)


# ---- B10 SDP -------------------------------------------------------------------------------------------------------
def test_sdp_rounding_valid():
    pytest.importorskip("cvxpy")
    from minrank.baselines.sdp import sdp_rows
    for g in (gen.gnp(20, 0.5, 0), gen.cycle(5), gen.kneser(5, 2)):
        r = verify(g, sdp_rows(g, 10.0, np.random.default_rng(0)))
        assert mais(g, 5).value <= r <= g.n


# ---- symmetry breaking ---------------------------------------------------------------------------------------------
def test_lex_leq_semantics():
    """Brute-force check of the lex encoding on all assignments of 2 x 3 bits."""
    from itertools import product as iproduct

    from pysat.solvers import Solver
    for L in (1, 2, 3):
        x = list(range(1, L + 1))
        y = list(range(L + 1, 2 * L + 1))
        pool = VarPool(2 * L)
        cls = lex_leq(x, y, pool)
        for bits in iproduct((0, 1), repeat=2 * L):
            assum = [v if b else -v for v, b in zip(x + y, bits)]
            with Solver(name="cadical153", bootstrap_with=cls) as s:
                sat = s.solve(assumptions=assum)
            assert sat == (list(bits[:L]) <= list(bits[L:]))


def test_automorphisms_valid():
    for g in (gen.cycle(7), gen.kneser(5, 2), gen.paley(13), directed_cycle(6)):
        gens = automorphisms(g)
        assert gens, g.name
        assert all(is_automorphism(g, s) for s in gens)
    asym = Graph.from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 4)], True)  # check only validity
    assert all(is_automorphism(asym, s) for s in automorphisms(asym))


@pytest.mark.parametrize("mode", ["none", "u0", "cols", "aut"])
@pytest.mark.parametrize("seed", range(10))
def test_symmetry_modes_agree_with_bruteforce(mode, seed):
    rng = np.random.default_rng(100 + seed)
    n = int(rng.integers(3, 7))
    g = gen.gnp(n, float(rng.uniform(0.2, 0.8)), 100 + seed, directed=bool(seed % 2))
    if g.num_edges() > 16:
        pytest.skip("too many free entries for brute force")
    assert exact_sym(g, mode) == minrank_bruteforce(g)


@pytest.mark.parametrize("mode", ["cols", "aut"])
def test_symmetry_modes_on_symmetric_graphs(mode):
    for g, want in ((gen.cycle(5), 3), (gen.kneser(5, 2), 5), (directed_cycle(5), 4)):
        assert exact_sym(g, mode) == want


def test_rank_at_most_sym_timeout():
    import time
    g = gen.gnp(100, 0.3, 0)
    t = time.perf_counter()
    res = rank_at_most_sym(g, 40, 0.5, mode="cols")
    assert time.perf_counter() - t < 0.5 + 0.3  # limit covers encoding (~1 s here) and solving
    assert res is None or res is False or verify(g, res) <= 40


# ---- F_q -----------------------------------------------------------------------------------------------------------
def test_rank_fq_basic():
    assert rank_fq([[1, 1], [1, 1]], 3) == 1
    assert rank_fq([[1, 2], [2, 1]], 3) == 1  # second row = 2 * first mod 3
    assert rank_fq([[1, 2], [2, 1]], 5) == 2
    assert rank_fq([[1, 1], [1, 1]], 2) == 1
    with pytest.raises(ValueError):
        rank_fq([[1]], 4)


def test_rank_fq_matches_f2_rank_on_random():
    from minrank.core.f2 import rank_f2
    from minrank.core.graph import bool_to_masks
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 2, (8, 8))
        assert rank_fq(a.tolist(), 2) == rank_f2(bool_to_masks(a))


def test_fit_error_fq():
    g = gen.cycle(5)
    ident = [[1 if i == j else 0 for j in range(5)] for i in range(5)]
    assert fit_error_fq(g, ident, 3) is None
    bad = [row[:] for row in ident]
    bad[0][2] = 1  # 2 not in S_0
    assert fit_error_fq(g, bad, 3) is not None
    bad2 = [row[:] for row in ident]
    bad2[1][1] = 3  # 0 mod 3
    assert fit_error_fq(g, bad2, 3) is not None


@pytest.mark.parametrize("seed", range(8))
def test_f3_sat_matches_bruteforce(seed):
    rng = np.random.default_rng(200 + seed)
    n = int(rng.integers(2, 5))
    g = gen.gnp(n, float(rng.uniform(0.3, 0.8)), 200 + seed, directed=bool(seed % 2))
    if g.num_edges() > 9:
        pytest.skip("too many free entries for F_3 brute force")
    bf = minrank_bruteforce_fq(g, 3)
    best, proven, A = exact_minrank_fq(g, 3, upper=g.n)
    assert proven and best == bf
    if A is not None:
        assert verify_fq(g, A, 3) == best


def test_f3_c5_and_q2_delegation():
    g = gen.cycle(5)
    assert rank_at_most_fq(g, 2, 3) is False
    A = rank_at_most_fq(g, 3, 3)
    assert A and verify_fq(g, A, 3) <= 3
    A2 = rank_at_most_fq(g, 3, 2)
    assert A2 and verify_fq(g, A2, 2) <= 3


# ---- driver and separations ----------------------------------------------------------------------------------------
def _tiny_bench(tmp_path):
    from minrank.bench.instances import Instance, materialize
    gs = [gen.cycle(5), gen.kneser(5, 2)]
    materialize(tmp_path / "graphs", [Instance(g.name, "test", lambda g=g: g) for g in gs])
    return gs


def test_driver_runs_and_is_resumable(tmp_path, capsys):
    from minrank.exact.driver import main
    _tiny_bench(tmp_path)
    args = ["--graphs", str(tmp_path / "graphs"), "--out", str(tmp_path / "exact.csv"), "--certs",
            str(tmp_path / "certs"), "--time-per-call", "30", "--lb-time", "5", "--cover-time", "0.2",
            "--workers", "1"]
    main(args)
    rows = list(csv.DictReader((tmp_path / "exact.csv").open()))
    got = {r["graph"]: (int(r["best_rank"]), r["proven_optimal"]) for r in rows}
    assert got == {"cycle_5": (3, "True"), "kneser_5_2": (5, "True")}
    assert (tmp_path / "certs" / "cycle_5.json").exists()
    main(args)
    out = capsys.readouterr().out
    assert "2 already proven, 0 to run" in out
    assert len(list(csv.DictReader((tmp_path / "exact.csv").open()))) == 2


def test_separations_smoke(tmp_path):
    from minrank.exact.separations import main
    main(["--random-n", "5", "--random-count", "1", "--no-structured", "--time-per-call", "20", "--lb-time", "2",
          "--cover-time", "0.1", "--workers", "1", "--out", str(tmp_path / "sep.csv"), "--out-dir",
          str(tmp_path / "sep")])
    rows = list(csv.DictReader((tmp_path / "sep.csv").open()))
    assert len(rows) == 3 + 4  # 3 random + 2 orthogonality + 2 planted
    orth = {r["graph"]: r for r in rows}["orth_5"]
    assert int(orth["minrk2"]) == 5 and orth["proven2"] == "True"
    for r in rows:
        assert int(r["lb"]) <= int(r["minrk2"]) <= int(r["cover"])


def test_run_baselines_registers_new_methods():
    from minrank import run_baselines as rb
    assert {"ldg", "partial_cover"} <= set(rb.METHODS) and "sdp" in rb.OPTIONAL_METHODS
    g = gen.gnp(15, 0.5, 1)
    for m in ("ldg", "partial_cover"):
        assert verify(g, rb.run_method(g, m, 0.2, 0)) >= mais(g, 5).value
