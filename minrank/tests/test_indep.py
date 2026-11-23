"""Tests for the independent-set symmetry breaking (modes "indep" / "indep_cols" of minrank/exact/symmetry.py).

The fix is checked exhaustively on small graphs: for every acyclic vertex set I (not only maximum ones) and every r,
"minrk_2 <= r" with I fixed must be SAT exactly when brute force says so.
Run: python -m pytest minrank/tests/test_indep.py -q
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from minrank.bench import generators as gen
from minrank.core.graph import Graph
from minrank.core.verify import verify
from minrank.exact.lower import is_acyclic_induced, mais
from minrank.exact.sat import minrank_bruteforce
from minrank.exact.symmetry import INDEP_MODES, check_indep, encode_sym, rank_at_most_sym

MODES = list(INDEP_MODES)


def acyclic_sets(g: Graph) -> list[list[int]]:
    return [list(c) for k in range(1, g.n + 1) for c in combinations(range(g.n), k) if is_acyclic_induced(g, list(c))]


def exact_indep(g: Graph, mode: str, indep: list[int] | None = None) -> int:
    for r in range(1, g.n + 1):
        if rank_at_most_sym(g, r, None, mode=mode, indep=indep):
            return r
    raise AssertionError("identity always fits")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("seed", range(60))
def test_every_acyclic_set_agrees_with_bruteforce(mode, seed):
    """All acyclic sets I and all r with |I| <= r: SAT iff r >= minrk (brute force). Odd seeds are directed, so sets
    with arcs (partial fix) are covered too."""
    rng = np.random.default_rng(300 + seed)
    n = int(rng.integers(3, 7))
    g = gen.gnp(n, float(rng.uniform(0.2, 0.8)), 300 + seed, directed=bool(seed % 2))
    if g.num_edges() > 16:
        pytest.skip("too many free entries for brute force")
    want = minrank_bruteforce(g)
    for I in acyclic_sets(g):
        for r in range(len(I), g.n + 1):
            res = rank_at_most_sym(g, r, None, mode=mode, indep=I)
            assert bool(res) == (r >= want), (g.name, I, r, want)
            if res:
                assert verify(g, res) <= r


@pytest.mark.parametrize("mode", MODES)
def test_known_values(mode):
    cases = [(gen.cycle(5), 3), (gen.kneser(5, 2), 5), (gen.mycielski(4), 6), (gen.paley(13), 5),
             (Graph.from_edges(5, [(i, (i + 1) % 5) for i in range(5)], True, "dcycle_5"), 4),
             (gen.kneser(7, 2), 7), (gen.orthogonality(5), 5)]
    for g, want in cases:
        assert exact_indep(g, mode) == want, g.name


def test_partial_fix_used_for_acyclic_sets_with_arcs():
    path = Graph.from_edges(3, [(0, 1), (1, 2)], True, "dpath_3")  # acyclic, has arcs
    assert check_indep(path, [0, 1, 2]) is False
    assert check_indep(path, [0, 2]) is True
    cyc = Graph.from_edges(3, [(0, 1), (1, 2), (2, 0)], True, "dcycle_3")
    with pytest.raises(ValueError):
        check_indep(cyc, [0, 1, 2])
    with pytest.raises(ValueError):
        check_indep(gen.cycle(5), [0, 1])  # adjacent vertices
    with pytest.raises(ValueError):
        check_indep(gen.cycle(5), [0, 0])


def test_more_vertices_than_r_is_unsat_without_cnf():
    g = gen.kneser(7, 2)
    I = mais(g, 10).vertices
    assert len(I) == 6
    assert rank_at_most_sym(g, 5, None, mode="indep", indep=I) is False
    with pytest.raises(ValueError):
        encode_sym(g, 5, "indep", indep=I)


@pytest.mark.parametrize("seed", range(6))
def test_agrees_with_cols_mode_on_medium_graphs(seed):
    """n = 9..11, beyond brute force: the indep modes give the same minrank as the established "cols" mode."""
    rng = np.random.default_rng(500 + seed)
    n = int(rng.integers(9, 12))
    g = gen.gnp(n, float(rng.uniform(0.3, 0.7)), 500 + seed, directed=bool(seed % 2))
    want = None
    for r in range(1, n + 1):
        if rank_at_most_sym(g, r, None, mode="cols"):
            want = r
            break
    for mode in MODES:
        assert exact_indep(g, mode) == want, (g.name, mode)


def test_families_runner(tmp_path, capsys):
    """End to end: proven values, resumability, and the DIMACS of every UNSAT call really is UNSAT."""
    import csv
    import gzip

    from pysat.formula import CNF
    from pysat.solvers import Solver

    from minrank.exact import families

    out, certs, dim = tmp_path / "fam.csv", tmp_path / "certs", tmp_path / "cnf"
    specs = ["cycle:5", "kneser:5,2", "mycielski:4", "paley:13", "kneser:7,2"]
    args = ["--spec", *specs, "--out", str(out), "--certs", str(certs), "--dimacs-dir", str(dim),
            "--time-per-call", "60", "--lb-time", "5", "--cover-time", "1", "--workers", "2"]
    families.main(args)
    rows = {r["spec"]: r for r in csv.DictReader(out.open())}
    want = {"cycle:5": 3, "kneser:5,2": 5, "mycielski:4": 6, "paley:13": 5, "kneser:7,2": 7}
    for s, v in want.items():
        assert rows[s]["proven"] == "True" and int(rows[s]["lower"]) == int(rows[s]["upper"]) == v, rows[s]
    files = sorted(dim.glob("*.cnf.gz"))
    assert files  # e.g. mycielski_4 r=5, paley_13 r=3 and r=4
    for p in files:
        with gzip.open(p, "rt") as f:
            cnf = CNF(from_string=f.read())
        with Solver(name="glucose4", bootstrap_with=cnf.clauses) as s:
            assert s.solve() is False, p.name
    families.main(args)  # resumable: everything proven, nothing rerun
    assert "0 to run" in capsys.readouterr().out
    assert len(list(csv.DictReader(out.open()))) == len(specs)


def test_driver_indep_mode_writes_unsat_cnf(tmp_path):
    """The benchmark driver with --mode indep proves a small graph and writes the CNF of its UNSAT call."""
    from minrank.exact import driver

    graphs = tmp_path / "graphs"
    graphs.mkdir()
    g = gen.mycielski(4)
    g.save(graphs / "m4.json")
    (graphs / "index.csv").write_text("name,split,family,n,density,digest,path\n"
                                      f"{g.name},test,mycielski,{g.n},0.36,{g.digest()},m4.json\n")
    out, cnf = tmp_path / "exact.csv", tmp_path / "cnf"
    driver.main(["--graphs", str(graphs), "--split", "test", "--mode", "indep", "--time-per-call", "60",
                 "--lb-time", "5", "--cover-time", "1", "--workers", "1", "--out", str(out),
                 "--certs", str(tmp_path / "certs"), "--dimacs-dir", str(cnf)])
    import csv
    row = next(csv.DictReader(out.open()))
    assert (row["proven_optimal"], row["best_rank"], row["status"]) == ("True", "6", "unsat")  # alpha 5, cover 6
    assert (cnf / f"{g.name}_r5_indep.cnf.gz").exists()
