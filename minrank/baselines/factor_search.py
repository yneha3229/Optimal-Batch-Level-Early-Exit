"""Local search on the factorisation A = U V^T over F_2 (baselines B6 tabu and B7 simulated annealing).

Contract
--------
rank_F2(A) <= r  iff  A = U V^T with U, V in F_2^{n x r}. Write u_i, v_j for the rows. A fits G iff
    <u_i, v_i> = 1                for every i              (diagonal)
    <u_i, v_j> = 0                for every forbidden (i, j) (j not in S_i, j != i)
and entries on edges are free. So "minrk_2(G) <= r" is a constraint-satisfaction problem on 2 n r bits.

- Constrained mask M (n, n): M[i, j] = (i == j) or (j forbidden for i). Target T = identity on M.
- State: U (n, r), V (n, r) in {0, 1}; P = U V^T mod 2 (n, n); violations viol = M & (P != T); cost f = sum(viol).
- Flipping U[i, k] toggles P[i, j] for every j with V[j, k] = 1. With the signed matrix
      S = M * (1 - 2 * viol)   (+1 on satisfied constrained entries, -1 on violated, 0 off M)
  the cost change of every U-flip is deltaU = S @ V  (n, r), and of every V-flip deltaV = S^T @ U  (n, r).
- Only rows (columns) that contain a violation are candidate U (V) moves: a flip in a violation-free row only touches
  satisfied entries, so its delta is >= 0.
- Rank reduction: start at r = number of cliques of a clique cover (cost 0: u_i = v_i = e_colour(i)); after reaching
  cost 0 at rank r, delete the column of U, V whose removal breaks the fewest constraints and search at r - 1.

The n x n matrices P, S, M are materialised on purpose: the constraint set itself has Theta(n^2) entries (every
forbidden pair), so no method can avoid touching them. Arithmetic is exact (integers; float32 matmul of 0/+-1
entries with sums <= n < 2^24 is exact).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from ..core.f2 import product_f2, rank_f2, rows_from_factors
from ..core.graph import Graph
from ..core.verify import fit_error
from .clique_cover import conflict_graph, dsatur, num_colors, tabucol


@dataclass
class SearchResult:
    rank: int
    rows: list[int]
    U: np.ndarray
    V: np.ndarray
    history: list[tuple[float, int]] = field(default_factory=list)  # (seconds since start, rank reached)
    iters: int = 0


def constraint_mask(g: Graph) -> np.ndarray:
    """M (n, n) bool: True where the entry of a fitting matrix is fixed (diagonal = 1, forbidden = 0)."""
    m = ~g.adjacency()
    np.fill_diagonal(m, True)
    return m


class FactorState:
    """Incremental state for fixed rank r. All arrays live on the CPU; sizes are O(n^2 + n r)."""

    def __init__(self, M: np.ndarray, U: np.ndarray, V: np.ndarray):
        n, r = U.shape
        if M.shape != (n, n) or V.shape != (n, r):
            raise ValueError(f"shapes: M {M.shape}, U {U.shape}, V {V.shape}")
        if r < 1:
            raise ValueError("rank must be >= 1")
        self.M = M
        self.T = np.eye(n, dtype=np.uint8)  # targets on M (off-M entries are ignored)
        self.U = U.astype(np.float32)  # (n, r) 0/1
        self.V = V.astype(np.float32)  # (n, r) 0/1
        self.P = product_f2(U, V)  # (n, n) uint8
        self.viol = M & (self.P != self.T)  # (n, n) bool
        self.S = np.where(M, 1.0 - 2.0 * self.viol, 0.0).astype(np.float32)  # (n, n)
        self.f = int(self.viol.sum())

    def deltas(self) -> tuple[np.ndarray, np.ndarray]:
        return self.S @ self.V, self.S.T @ self.U  # (n, r), (n, r)

    def flip_u(self, i: int, k: int) -> None:
        self.U[i, k] = 1.0 - self.U[i, k]
        self.P[i] ^= self.V[:, k].astype(np.uint8)  # row i toggles where V[:, k] = 1
        self._refresh_row(i)

    def flip_v(self, j: int, k: int) -> None:
        self.V[j, k] = 1.0 - self.V[j, k]
        self.P[:, j] ^= self.U[:, k].astype(np.uint8)  # column j toggles where U[:, k] = 1
        self._refresh_col(j)

    def _refresh_row(self, i: int) -> None:
        old = int(self.viol[i].sum())
        self.viol[i] = self.M[i] & (self.P[i] != self.T[i])
        self.S[i] = np.where(self.M[i], 1.0 - 2.0 * self.viol[i], 0.0)
        self.f += int(self.viol[i].sum()) - old

    def _refresh_col(self, j: int) -> None:
        old = int(self.viol[:, j].sum())
        self.viol[:, j] = self.M[:, j] & (self.P[:, j] != self.T[:, j])
        self.S[:, j] = np.where(self.M[:, j], 1.0 - 2.0 * self.viol[:, j], 0.0)
        self.f += int(self.viol[:, j].sum()) - old

    def factors(self) -> tuple[np.ndarray, np.ndarray]:
        return self.U.astype(np.uint8), self.V.astype(np.uint8)


def search_fixed_rank(M: np.ndarray, U0: np.ndarray, V0: np.ndarray, deadline: float, rng: np.random.Generator,
                      mode: str = "tabu", max_iters: int = 10 ** 9, temp0: float = 2.0, cooling: float = 0.9995,
                      ) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Drive the cost to 0 at the rank of U0/V0. Returns (U, V, iterations) or None at the deadline.

    mode="tabu": best non-tabu move among conflict rows/columns (aspiration on a new best), tenure 0.6 f + U{0..9}.
    mode="anneal": random candidate move, Metropolis acceptance at temperature temp0 * cooling^t (reheated on stalls).
    """
    if mode not in ("tabu", "anneal"):
        raise ValueError(f"unknown mode {mode}")
    st = FactorState(M, U0, V0)
    n, r = st.U.shape
    tabu_u = np.zeros((n, r), dtype=np.int64)
    tabu_v = np.zeros((n, r), dtype=np.int64)
    best_f, it, temp = st.f, 0, temp0
    while st.f > 0:
        if it >= max_iters or ((it & 31) == 0 and time.perf_counter() > deadline):
            return None
        it += 1
        rows = np.flatnonzero(st.viol.any(1))  # rows with a violation
        cols = np.flatnonzero(st.viol.any(0))  # columns with a violation
        du, dv = st.deltas()
        du, dv = du[rows], dv[cols]  # (|rows|, r), (|cols|, r)
        if mode == "tabu":
            au = np.where((tabu_u[rows] <= it) | (st.f + du < best_f), du, np.inf)
            av = np.where((tabu_v[cols] <= it) | (st.f + dv < best_f), dv, np.inf)
            au = au + rng.random(au.shape) * 1e-3
            av = av + rng.random(av.shape) * 1e-3
            iu, iv = int(np.argmin(au)), int(np.argmin(av))
            if not (np.isfinite(au.flat[iu]) or np.isfinite(av.flat[iv])):
                continue
            if au.flat[iu] <= av.flat[iv]:
                i, k = divmod(iu, r)
                st.flip_u(int(rows[i]), k)
                tabu_u[rows[i], k] = it + int(0.6 * st.f) + int(rng.integers(0, 10))
            else:
                j, k = divmod(iv, r)
                st.flip_v(int(cols[j]), k)
                tabu_v[cols[j], k] = it + int(0.6 * st.f) + int(rng.integers(0, 10))
        else:
            nu, nv = du.size, dv.size
            pick = int(rng.integers(0, nu + nv))
            d = float(du.flat[pick]) if pick < nu else float(dv.flat[pick - nu])
            if d <= 0 or rng.random() < math.exp(-d / max(temp, 1e-9)):
                if pick < nu:
                    i, k = divmod(pick, r)
                    st.flip_u(int(rows[i]), k)
                else:
                    j, k = divmod(pick - nu, r)
                    st.flip_v(int(cols[j]), k)
            temp *= cooling
            if temp < 0.05:
                temp = temp0  # reheat
        best_f = min(best_f, st.f)
    U, V = st.factors()
    return U, V, it


def drop_best_column(M: np.ndarray, U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove the column k minimising the violations of U[:, -k] V[:, -k]^T. Returns (n, r-1) factors."""
    r = U.shape[1]
    if r < 2:
        raise ValueError("cannot drop below rank 1")
    P = product_f2(U, V).astype(np.int32)  # (n, n)
    T = np.eye(M.shape[0], dtype=np.int32)
    best_k, best_v = 0, None
    for k in range(r):
        Pk = P ^ np.outer(U[:, k], V[:, k]).astype(np.int32)  # remove the rank-1 term of column k
        v = int((M & (Pk != T)).sum())
        if best_v is None or v < best_v:
            best_k, best_v = k, v
    keep = [c for c in range(r) if c != best_k]
    return U[:, keep].copy(), V[:, keep].copy()


def colors_to_factors(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clique cover with k cliques -> U = V = one-hot colour matrix (n, k); U V^T fits with rank k."""
    colors = np.unique(colors, return_inverse=True)[1]
    k = int(colors.max()) + 1
    U = np.zeros((len(colors), k), dtype=np.uint8)
    U[np.arange(len(colors)), colors] = 1
    return U, U.copy()


def factor_search(g: Graph, time_limit: float, rng: np.random.Generator, mode: str = "tabu",
                  cover_fraction: float = 0.2) -> SearchResult:
    """B6 (mode="tabu") / B7 (mode="anneal").

    Spend `cover_fraction` of the budget on TabuCol (clique cover = rank-k feasible start), the rest on reducing r.
    The returned matrix is verified (fits, rank <= r) before returning.
    """
    if not 0.0 <= cover_fraction < 1.0:
        raise ValueError("cover_fraction must be in [0, 1)")
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    conf = conflict_graph(g)
    colors = tabucol(conf, time_limit * cover_fraction, rng) if cover_fraction > 0 else dsatur(conf)
    U, V = colors_to_factors(colors)
    M = constraint_mask(g)
    best = SearchResult(num_colors(colors), rows_from_factors(U, V), U, V,
                        [(time.perf_counter() - t0, num_colors(colors))])
    while best.rank > 1 and time.perf_counter() < deadline:
        U1, V1 = drop_best_column(M, best.U, best.V)
        res = search_fixed_rank(M, U1, V1, deadline, rng, mode)
        if res is None:
            break
        U, V, iters = res
        rows = rows_from_factors(U, V)
        err = fit_error(g, rows)
        if err:  # would be a bug in the incremental bookkeeping: fail loudly
            raise RuntimeError(f"factor search returned a non-fitting matrix on {g.name}: {err}")
        r = rank_f2(rows)
        if r > U.shape[1]:
            raise RuntimeError(f"rank {r} exceeds factor width {U.shape[1]}")
        # If r < width the next drop still starts from these factors; it simply reaches cost 0 immediately.
        best = SearchResult(r, rows, U, V, best.history + [(time.perf_counter() - t0, r)], best.iters + iters)
    return best
