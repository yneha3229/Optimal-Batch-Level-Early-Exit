"""Exact rank over F_p with a rank factorization, for fitting matrices with up to ~10^4 rows.

    rank_factor(mat, cap) -> RankResult(rank, S, Q, C)   with   M = C . M[S, :]   exactly over F_p

Method (exact, no probability):
  1. Sample rows S0 and columns J. Numba elimination on the small block M[S0, J] gives rows S and columns Q with
     M[S, Q] invertible over F_p (early stop once more than `cap` independent rows are found).
  2. T = M[S, Q]^{-1} by python-flint (nmod_mat.inv), and C = M[:, Q] . T.
  3. Residual R = M - C . M[S, :], computed for every row in float64 BLAS blocks. All products are exact because
     every entry is < p and r . (p-1)^2 < 2^53 (asserted). A row with R_i != 0 is outside span(M[S, :]); its row and
     a nonzero residual column are added (the Schur complement is R_ij != 0, so the bordered block stays
     invertible) and steps 1-3 repeat. When every residual is zero, rank_p(M) = |S| exactly, M[S, :] is a basis.
Cost: O(n^2 r) in BLAS plus O(s^2 r) in numba for the sample, instead of flint's O(n^3) plus ~0.35 us per entry
to move a 10^4 x 10^4 matrix into flint.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import flint
import numpy as np
from numba import njit

EXACT = 2 ** 53


# ------------------------------------------------------------------------------------------------ matrices
class SpectralMatrix:
    """M = I + c A over F_p, generated on the fly from the boolean adjacency matrix (never stored densely)."""

    symmetric = True

    def __init__(self, A: np.ndarray, p: int, c: int) -> None:
        self.A, self.p, self.c, self.n = A, int(p), int(c) % int(p), A.shape[0]

    def block(self, rows, cols=None) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if cols is None:
            X = self.A[rows].astype(np.int64) * self.c
            X[np.arange(len(rows)), rows] += 1
        else:
            cols = np.asarray(cols, dtype=np.int64)
            X = self.A[np.ix_(rows, cols)].astype(np.int64) * self.c
            X[rows[:, None] == cols[None, :]] += 1
        return X % self.p


class DenseMatrix:
    symmetric = False

    def __init__(self, M: np.ndarray, p: int) -> None:
        self.M, self.p, self.n = np.asarray(M, dtype=np.int64) % p, int(p), M.shape[0]
        self.ncols = M.shape[1]

    def block(self, rows, cols=None) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        return self.M[rows] if cols is None else self.M[np.ix_(rows, np.asarray(cols, dtype=np.int64))]


# ------------------------------------------------------------------------------------------------ kernels
@njit(cache=True)
def _echelon(Y, p, cap):
    """Row-by-row elimination of Y (s x m, entries in [0, p)) over F_p. Returns (pivot rows, pivot cols) of a
    maximal set of independent rows, stopping after cap + 1 of them. Y is overwritten."""
    s, m = Y.shape
    prow = np.empty(min(s, cap + 1), dtype=np.int64)
    pcol = np.empty(min(s, cap + 1), dtype=np.int64)
    basis = np.empty((min(s, cap + 1), m), dtype=np.int64)
    r = 0
    for i in range(s):
        v = Y[i].copy()                                   # kept unreduced: |v| < (r + 1) p^2 < 2^63
        for b in range(r):
            f = v[pcol[b]] % p
            if f:
                g = p - f
                for j in range(m):
                    v[j] += g * basis[b, j]
        piv = -1
        for j in range(m):
            v[j] %= p
            if piv < 0 and v[j] != 0:
                piv = j
        if piv < 0:
            continue
        inv = 1
        e, base = p - 2, v[piv]                           # Fermat inverse (p prime)
        while e:
            if e & 1:
                inv = inv * base % p
            base = base * base % p
            e >>= 1
        for j in range(m):
            v[j] = v[j] * inv % p
        basis[r] = v
        prow[r], pcol[r] = i, piv
        r += 1
        if r > cap:
            break
    return prow[:r], pcol[:r]


def _inv_mod(B: np.ndarray, p: int) -> np.ndarray:
    r = B.shape[0]
    F = flint.nmod_mat(r, r, [int(x) for x in B.ravel()], p)
    Finv = F.inv()
    return np.array([int(x) for x in Finv.entries()], dtype=np.int64).reshape(r, r)


@dataclass
class RankResult:
    rank: int | None            # None: rank > cap (not computed exactly)
    S: np.ndarray | None        # basis rows
    Q: np.ndarray | None        # columns with M[S, Q] invertible
    C: np.ndarray | None        # n x r, M = C . M[S, :] over F_p
    p: int
    seconds: float
    rounds: int

    def ok(self) -> bool:
        return self.rank is not None


def rank_factor(mat, cap: int | None = None, seed: int = 0, block: int = 1024, extra: int = 32,
                stop=None) -> RankResult:
    """Exact rank_p(M) and factorization M = C . M[S, :], or rank=None if rank_p(M) > cap.
    `stop` is an optional callable checked between blocks (returns True to abandon: rank=None)."""
    t0 = time.perf_counter()
    n, p = mat.n, mat.p
    ncols = getattr(mat, "ncols", n)
    cap = min(n, ncols) if cap is None else int(cap)
    if cap * (p - 1) ** 2 >= EXACT:
        raise ValueError(f"p = {p}, cap = {cap}: float64 products would not be exact")
    rng = np.random.default_rng(seed)
    S0 = rng.permutation(n)[: min(n, cap + extra)]
    J = rng.permutation(ncols)[: min(ncols, 2 * cap + 2 * extra)]
    rounds = 0
    while True:
        rounds += 1
        Y = mat.block(S0, J)
        pr, pc = _echelon(Y.copy(), p, cap)
        if len(pr) > cap:
            return RankResult(None, None, None, None, p, time.perf_counter() - t0, rounds)
        S, Q = S0[pr], J[pc]
        r = len(S)
        if r == 0:                                            # M = 0 (cannot happen for a fitting matrix)
            return RankResult(0, S, Q, np.zeros((n, 0), np.int64), p, time.perf_counter() - t0, rounds)
        T = _inv_mod(mat.block(S, Q), p).astype(np.float64)
        MS = mat.block(S).astype(np.float64)                  # r x ncols
        C = np.empty((n, r), dtype=np.int64)
        bad_rows, bad_cols = [], []
        for a in range(0, n, block):
            if stop is not None and stop():
                return RankResult(None, None, None, None, p, time.perf_counter() - t0, rounds)
            idx = np.arange(a, min(n, a + block))
            MQ = mat.block(idx, Q).astype(np.float64)
            Cb = np.mod(MQ @ T, p)
            C[idx] = Cb.astype(np.int64)
            R = np.mod(mat.block(idx).astype(np.float64) - Cb @ MS, p)
            nz = np.nonzero(R.any(axis=1))[0]
            for i in nz[: max(0, 64 - len(bad_rows))]:
                bad_rows.append(int(idx[i]))
                bad_cols.append(int(np.nonzero(R[i])[0][0]))
            if len(bad_rows) >= 64:
                break
        if not bad_rows:
            return RankResult(r, S, Q, C, p, time.perf_counter() - t0, rounds)
        if r >= cap:                                          # a row outside span(M[S]) => rank > r >= cap
            return RankResult(None, None, None, None, p, time.perf_counter() - t0, rounds)
        S0 = np.concatenate([S, np.setdiff1d(np.array(bad_rows), S)])
        J = np.concatenate([Q, np.setdiff1d(np.array(bad_cols), Q),
                            np.setdiff1d(J, Q)[: extra]])
        J = np.array(list(dict.fromkeys(J.tolist())), dtype=np.int64)


def rank_flint_full(M: np.ndarray, p: int) -> int:
    """Plain flint rank of a dense matrix (benchmark / cross-check only: building the flint matrix is slow)."""
    F = flint.nmod_mat(M.shape[0], M.shape[1], (np.asarray(M) % p).ravel().tolist(), p)
    return F.rank()
