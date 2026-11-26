"""Exact scalar minrank over F_2 by SAT (n <= 150): Pincer's encoding with the independent-set symmetry fix
(minrank.exact.symmetry.rank_at_most_sym, mode "indep_cols"), reused unchanged.

is minrk_2(G) <= r ?   SAT    -> fitting rows over F_2 of rank <= r (upper bound on minrk_2 and beta)
                       UNSAT  -> minrk_2(G) >= r + 1 (a lower bound on minrk_2 only, not on beta); the CNF is saved
                                 gzipped so the Verifier can re-solve it with a different SAT solver.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAT_MAX_N = 150


def graph_of(A: np.ndarray):
    from minrank.core.graph import Graph

    return Graph.from_adjacency(A, directed=False)


def sat_minrank_at(A: np.ndarray, r: int, time_limit: float, indep: list[int], cnf_path: Path | None = None,
                   max_n: int = SAT_MAX_N):
    """rows (list of int bitmasks) | False (UNSAT) | None (timeout or too large)."""
    if A.shape[0] > max_n:
        return None
    from minrank.exact.symmetry import encode_sym, rank_at_most_sym

    g = graph_of(A)
    I = sorted(indep)[:r]
    res = rank_at_most_sym(g, r, time_limit=time_limit, mode="indep_cols", indep=I)
    if res is False and cnf_path is not None:
        clauses = encode_sym(g, r, "indep_cols", None, I)
        nv = max((abs(l) for c in clauses for l in c), default=0)
        cnf_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(cnf_path, "wt") as f:
            f.write(f"p cnf {nv} {len(clauses)}\n")
            for c in clauses:
                f.write(" ".join(map(str, c)) + " 0\n")
    return res


def rows_to_dense(rows: list[int], n: int) -> np.ndarray:
    M = np.zeros((n, n), dtype=np.int64)
    for i, m in enumerate(rows):
        for j in range(n):
            M[i, j] = (m >> j) & 1
    return M
