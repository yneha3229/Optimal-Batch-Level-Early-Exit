"""Alternating minimisation over F_2 with information-set decoding (baseline B5).

Contract
--------
Same factorisation as factor_search: A = U V^T, U, V in F_2^{n x r}, constraints <u_i, v_j> = [i == j] on the
constrained mask M. With U fixed the constraints on column v_j are LINEAR:

    U[C_j] v_j = t_j   over F_2,   C_j = {i : M[i, j]},  t_j = indicator of (i == j) restricted to C_j,

so updating v_j is a decoding problem: minimise the number of violated equations (Hamming distance between
U[C_j] v_j and t_j). Exact minimum-distance decoding is NP-hard; we use Prange-style information-set decoding: pick
a random order of the equations, take the first r linearly independent ones, solve them exactly (free variables
random), count violations of all equations, keep the best of `trials` attempts and of the current v_j. The update is
therefore monotone: the total cost never increases. Rows u_i are updated symmetrically with V fixed.

Rank reduction is as in factor_search: start from a clique cover, drop the cheapest column, sweep until cost 0 or a
stall, then restart from a perturbation; on success go one rank lower.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.f2 import rank_f2, rows_from_factors
from ..core.graph import Graph
from ..core.verify import fit_error
from .clique_cover import conflict_graph, num_colors, tabucol
from .factor_search import SearchResult, colors_to_factors, constraint_mask, drop_best_column


def solve_subset(Aeq: np.ndarray, b: np.ndarray, order: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Solve the first independent equations of Aeq x = b taken in `order` (Gauss-Jordan over F_2).

    Aeq: (m, r) 0/1, b: (m,) 0/1. Returns x (r,) 0/1 satisfying every equation that was accepted as a pivot row;
    free variables are random. Exact integer arithmetic on bit-packed Python ints.
    """
    m, r = Aeq.shape
    weights = 1 << np.arange(r, dtype=object)
    piv_rows: dict[int, tuple[int, int]] = {}  # pivot column -> (row bits, rhs), fully reduced
    for e in order:
        row = int((Aeq[e].astype(object) * weights).sum())
        rhs = int(b[e])
        for c, (prow, prhs) in piv_rows.items():  # reduce against existing pivots
            if (row >> c) & 1:
                row ^= prow
                rhs ^= prhs
        if row == 0:
            continue
        c = row.bit_length() - 1
        for c2 in list(piv_rows):  # keep the basis fully reduced (Gauss-Jordan)
            prow, prhs = piv_rows[c2]
            if (prow >> c) & 1:
                piv_rows[c2] = (prow ^ row, prhs ^ rhs)
        piv_rows[c] = (row, rhs)
        if len(piv_rows) == r:
            break
    x = rng.integers(0, 2, r).astype(np.int64)
    for c in piv_rows:
        x[c] = 0
    # each pivot row: x_c + sum_{free f in row} x_f = rhs  (other pivots were eliminated)
    for c, (row, rhs) in piv_rows.items():
        s = rhs
        rest = row & ~(1 << c)
        while rest:
            low = rest & -rest
            s ^= int(x[low.bit_length() - 1])
            rest ^= low
        x[c] = s
    return x


def best_decode(Aeq: np.ndarray, b: np.ndarray, current: np.ndarray, trials: int, rng: np.random.Generator,
                hard: int | None = None) -> tuple[np.ndarray, int]:
    """Best of `current` and `trials` information-set solutions. Returns (x, weighted violations).

    `hard` is the index of the diagonal equation <u_j, v_j> = 1. It is always the first pivot and carries weight m+1,
    so no solution trades it away: otherwise x = 0 satisfies every "= 0" equation and the search collapses to the
    trivial state with exactly one (diagonal) violation per column.
    """
    m = Aeq.shape[0]
    w = np.ones(m, dtype=np.int64)
    if hard is not None:
        w[hard] = m + 1

    def cost(x: np.ndarray) -> int:
        return int(((((Aeq @ x) & 1) != b) * w).sum())  # (m,) -> scalar

    best_x, best_c = current, cost(current)
    for _ in range(trials):
        if best_c == 0:
            break
        order = rng.permutation(m)
        if hard is not None:
            order = np.concatenate(([hard], order[order != hard]))
        x = solve_subset(Aeq, b, order, rng)
        c = cost(x)
        if c < best_c:
            best_x, best_c = x, c
    return best_x, best_c


def total_cost(M: np.ndarray, U: np.ndarray, V: np.ndarray) -> int:
    P = (U.astype(np.int64) @ V.astype(np.int64).T) & 1  # (n, n)
    return int((M & (P != np.eye(M.shape[0], dtype=np.int64))).sum())


def sweep(M: np.ndarray, U: np.ndarray, V: np.ndarray, trials: int, rng: np.random.Generator) -> int:
    """One pass over all columns of V then all rows of U, in random order. In place. Returns the new cost."""
    n = M.shape[0]
    for j in rng.permutation(n):  # v_j with U fixed: equations over C_j = {i : M[i, j]}
        C = np.flatnonzero(M[:, j])
        b = (C == j).astype(np.int64)
        V[j], _ = best_decode(U[C].astype(np.int64), b, V[j].astype(np.int64), trials, rng,
                              hard=int(np.flatnonzero(C == j)[0]))
    for i in rng.permutation(n):  # u_i with V fixed: equations over {j : M[i, j]}
        C = np.flatnonzero(M[i])
        b = (C == i).astype(np.int64)
        U[i], _ = best_decode(V[C].astype(np.int64), b, U[i].astype(np.int64), trials, rng,
                              hard=int(np.flatnonzero(C == i)[0]))
    return total_cost(M, U, V)


def search_fixed_rank_alt(M: np.ndarray, U0: np.ndarray, V0: np.ndarray, deadline: float, rng: np.random.Generator,
                          trials: int = 8, patience: int = 3, kick: float = 0.05
                          ) -> tuple[np.ndarray, np.ndarray] | None:
    """Alternate decoding sweeps until cost 0. After `patience` sweeps without improvement, flip a `kick` fraction of
    the bits of U and V (restart from a perturbation of the best state)."""
    U, V = U0.astype(np.int64).copy(), V0.astype(np.int64).copy()
    best_c = total_cost(M, U, V)
    best = (U.copy(), V.copy())
    stall = 0
    while best_c > 0:
        if time.perf_counter() > deadline:
            return None
        c = sweep(M, U, V, trials, rng)
        if c < best_c:
            best_c, best, stall = c, (U.copy(), V.copy()), 0
        else:
            stall += 1
            if stall >= patience:
                U, V = best[0].copy(), best[1].copy()
                U ^= (rng.random(U.shape) < kick).astype(np.int64)
                V ^= (rng.random(V.shape) < kick).astype(np.int64)
                stall = 0
    return best[0].astype(np.uint8), best[1].astype(np.uint8)


def alt_search(g: Graph, time_limit: float, rng: np.random.Generator, cover_fraction: float = 0.1,
               trials: int = 8) -> SearchResult:
    """B5: clique cover start, then rank reduction with alternating information-set decoding. Verified output."""
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    colors = tabucol(conflict_graph(g), time_limit * cover_fraction, rng)
    U, V = colors_to_factors(colors)
    M = constraint_mask(g)
    best = SearchResult(num_colors(colors), rows_from_factors(U, V), U, V, [(time.perf_counter() - t0, num_colors(colors))])
    while best.rank > 1 and time.perf_counter() < deadline:
        U1, V1 = drop_best_column(M, best.U, best.V)
        res = search_fixed_rank_alt(M, U1, V1, deadline, rng, trials)
        if res is None:
            break
        U, V = res
        rows = rows_from_factors(U, V)
        err = fit_error(g, rows)
        if err:
            raise RuntimeError(f"alt search returned a non-fitting matrix on {g.name}: {err}")
        best = SearchResult(rank_f2(rows), rows, U, V, best.history + [(time.perf_counter() - t0, rank_f2(rows))])
    return best
