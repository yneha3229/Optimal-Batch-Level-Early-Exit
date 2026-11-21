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


# ---- planted families and B5 ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("directed", [False, True])
def test_planted_rank_is_an_upper_bound(directed):
    g = gen.planted(30, 5, 0.1, 3, directed=directed)
    rows = rank_at_most(g, 5, time_limit=60)
    assert rows and verify(g, rows) <= 5
    assert mais(g, 5).value <= 5


def test_orthogonality_graph():
    g = gen.orthogonality(5)
    assert g.n == 16
    assert mais(g, 5).value <= 5
    rows = rank_at_most(g, 5, time_limit=60)
    assert rows and verify(g, rows) <= 5


def test_solve_subset_satisfies_pivots():
    from minrank.baselines.alt_f2 import solve_subset
    rng = np.random.default_rng(0)
    for _ in range(50):
        m, r = int(rng.integers(1, 20)), int(rng.integers(1, 8))
        x_true = rng.integers(0, 2, r)
        A = rng.integers(0, 2, (m, r))
        b = (A @ x_true) & 1  # consistent system
        x = solve_subset(A, b, rng.permutation(m), rng)
        assert (((A @ x) & 1) == b).all()


def test_decode_never_drops_the_diagonal():
    from minrank.baselines.alt_f2 import best_decode
    rng = np.random.default_rng(1)
    A = rng.integers(0, 2, (12, 4))
    A[0] = [1, 0, 0, 0]
    b = np.zeros(12, dtype=np.int64)
    b[0] = 1
    x, _ = best_decode(A, b, np.zeros(4, dtype=np.int64), 16, rng, hard=0)
    assert ((A[0] @ x) & 1) == 1


def test_alt_search_valid():
    from minrank.baselines.alt_f2 import alt_search
    g = gen.gnp(25, 0.5, 9)
    res = alt_search(g, 1.5, np.random.default_rng(0))
    assert verify(g, res.rows) == res.rank
    assert mais(g, 5).value <= res.rank <= num_colors(dsatur(conflict_graph(g)))


# ---- review fixes: budgets, child errors, certificates, CSV dedup, descent -----------------------------------------
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SLACK = 0.75  # seconds allowed past a budget for process kill / final verification


def test_rank_at_most_time_limit_includes_encoding():
    g = gen.gnp(100, 0.3, 0)  # ~7,000 constrained pairs; encoding alone takes ~1 s at r = 40
    t = time.perf_counter()
    res = rank_at_most(g, 40, time_limit=0.5)
    assert time.perf_counter() - t < 0.5 + 0.3  # before the fix: 0.5 s + full encoding time
    assert res is None or res is False or verify(g, res) <= 40


def test_sat_descend_respects_budget():
    from minrank.exact.sat import sat_descend
    g = gen.gnp(100, 0.3, 0)
    t = time.perf_counter()
    rows = sat_descend(g, 1.5, np.random.default_rng(0))
    assert time.perf_counter() - t < 1.5 + SLACK
    verify(g, rows)


def test_alt_search_respects_budget():
    from minrank.baselines.alt_f2 import alt_search
    g = gen.planted(100, 10, 0.0, 0, directed=True)
    t = time.perf_counter()
    res = alt_search(g, 1.0, np.random.default_rng(0), trials=32)  # one full sweep takes ~0.5-0.9 s here
    assert time.perf_counter() - t < 1.0 + 0.25  # before the fix the deadline was only checked between sweeps
    assert verify(g, res.rows) == res.rank


def test_sat_child_error_is_reported():
    with pytest.raises(RuntimeError, match="SAT child failed"):
        rank_at_most(gen.cycle(5), 3, time_limit=30, solver="no-such-solver")


def test_sat_after_threaded_cpsat_in_same_process():
    g = gen.planted(30, 5, 0.0, 1, directed=False)
    assert mais(g, 2, workers=4).value <= 5  # leaves CP-SAT worker threads behind in this process
    rows = rank_at_most(g, 5, time_limit=60)
    assert rows and verify(g, rows) <= 5


def test_sat_descend_reaches_orthogonality_optimum():
    from minrank.exact.sat import sat_descend
    g = gen.orthogonality(6)  # minrk = 6 (lower bound 6, planted 6); clique cover 8
    assert verify(g, sat_descend(g, 20, np.random.default_rng(0))) == 6


def test_factor_search_descends_below_its_start():
    g = gen.gnp(20, 0.5, 4)  # DSatur gives 7; 6 is reachable
    assert num_colors(dsatur(conflict_graph(g))) == 7
    res = factor_search(g, 3, np.random.default_rng(0), cover_fraction=0.0)  # start = DSatur, no TabuCol
    assert res.history[0][1] == 7 and res.rank <= 6 and len(res.history) >= 2


@pytest.mark.xfail(strict=False, reason="target for Option A: local search does not yet find planted gaps (day-1)")
@pytest.mark.parametrize("method", ["tabu", "anneal", "alt"])
def test_local_search_finds_orthogonality_gap(method):
    from minrank.baselines.alt_f2 import alt_search
    g = gen.orthogonality(6)
    rng = np.random.default_rng(0)
    res = alt_search(g, 5, rng) if method == "alt" else factor_search(g, 5, rng, mode=method)
    assert res.rank <= 7  # clique cover is 8


def test_keep_best_rejects_foreign_and_stale_certificates(tmp_path):
    from minrank.run_baselines import keep_best
    a = gen.gnp(20, 0.5, 1)
    b = Graph(a.n, gen.gnp(20, 0.5, 2).out, False, a.name, a.family)  # same name, different graph
    assert a.digest() != b.digest()
    rows_a = clique_cover_rows(a, "tabucol", 0.3, np.random.default_rng(0))
    rows_b = [1 << i for i in range(b.n)]  # identity: rank 20, worse than any clique cover
    assert keep_best(tmp_path, a, rows_a, "a")
    assert keep_best(tmp_path, b, rows_b, "b")  # foreign certificate must not block b
    assert load_certificate(tmp_path / f"{a.name}.json")["digest"] == b.digest()
    assert not keep_best(tmp_path, b, rows_b, "b-again")  # equal rank: keep existing
    rows_b2 = clique_cover_rows(b, "tabucol", 0.3, np.random.default_rng(0))
    assert keep_best(tmp_path, b, rows_b2, "better")  # strictly better replaces
    # corrupt the stored rows (same digest): must be treated as absent
    import json
    p = tmp_path / f"{b.name}.json"
    d = json.loads(p.read_text())
    d["rows"] = ["0"] * b.n
    p.write_text(json.dumps(d))
    assert keep_best(tmp_path, b, rows_b, "after-corruption")


def test_sweep_rerun_adds_no_duplicate_rows(tmp_path):
    from minrank.bench.instances import Instance, materialize
    g = gen.gnp(20, 0.5, 3)
    materialize(tmp_path / "graphs", [Instance(g.name, "val", lambda: g)])
    cmd = [sys.executable, "-m", "minrank.run_baselines", "--graphs", str(tmp_path / "graphs"), "--split", "val",
           "--out", str(tmp_path / "res.csv"), "--certs", str(tmp_path / "certs"), "--methods", "dsatur",
           "random_greedy", "--budgets", "0.2", "0.2", "--seeds", "0", "--workers", "1"]
    for _ in range(2):
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    lines = (tmp_path / "res.csv").read_text().strip().splitlines()
    assert len(lines) == 1 + 2  # header + dsatur + one random_greedy row (repeated budget deduplicated too)


def test_existing_keys_rejects_foreign_header(tmp_path):
    from minrank.run_baselines import existing_keys
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit):
        existing_keys(p)
