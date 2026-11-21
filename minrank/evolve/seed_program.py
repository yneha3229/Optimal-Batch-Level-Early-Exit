"""Seed program for OpenEvolve: a minrank heuristic over F_2.

Problem
-------
A side-information graph on n vertices is given as bitmasks: bit j of out[i] is set iff receiver i knows message j.
A matrix A over F_2 FITS the graph iff A[i][i] = 1 for every i and A[i][j] = 0 whenever j is not in out[i] (j != i);
entries where bit j of out[i] is set are free. Goal: return a fitting matrix of rank as low as possible.
Return format: list of n ints, bit j of rows[i] = A[i][j].

Method (the fixed skeleton, outside the EVOLVE block)
-----------------------------------------------------
- rank(A) <= r  iff  A = U V^T with U, V in F_2^{n x r}. Constraints: <u_i, v_i> = 1 (diagonal) and <u_i, v_j> = 0 on
  every forbidden pair. M[i, j] = True marks a constrained entry, target T = identity on M.
- Start: a clique cover of the mutual graph (DSatur colouring of its complement) gives u_i = v_i = e_colour(i), a
  fitting matrix of rank = number of colours. This is also the fallback answer: solve() always returns a fitting
  matrix.
- Rank reduction: drop one column (column_to_drop), optionally re-initialise (init_factors), then drive the number of
  violated constraints f to 0 at that rank with a tabu bit-flip search (choose_move, tenure) and, when it stalls,
  alternating information-set-decoding sweeps (each column of V / row of U is a linear system over F_2 with U / V
  fixed; the diagonal equation is a hard pivot). On f = 0 the rank went down by one; repeat until the budget ends.
- Exact incremental deltas: with S = M * (1 - 2*viol), flipping U[i, k] changes f by (S @ V)[i, k] and flipping
  V[j, k] changes f by (S^T @ U)[j, k].

Only the code between the EVOLVE-BLOCK markers is rewritten by the LLM. Allowed imports: numpy, math, random,
itertools, collections, heapq, time. No file or network access.
"""
import math  # noqa: F401  (available to evolved code)
import random
import time

import numpy as np


# EVOLVE-BLOCK-START
# Tunable parameters of the search. All are read by the fixed skeleton below.
PARAMS = {
    "isd_after_stall": 300,   # tabu iterations without a new best f before an ISD sweep is tried
    "isd_trials": 8,          # information sets tried per column/row in an ISD sweep
    "kick_after_stall": 3000, # tabu iterations without a new best f before a random kick of the best state
    "kick_fraction": 0.03,    # fraction of bits of U and V flipped by a kick
}


def tenure(f, it, rng):
    """Tabu tenure (iterations) for the bit just flipped. f = current violations, it = iteration counter.

    Classic TabuCol rule: proportional to the number of conflicts plus a small random term.
    """
    return int(0.6 * f) + rng.randint(0, 9)


def choose_move(du, dv, f, best_f, it, allow_u, allow_v, rng):
    """Pick the next bit flip.

    du: float array (a, r): change of f if U[rows[x], k] is flipped, for the a rows that contain a violation.
    dv: float array (b, r): change of f if V[cols[y], k] is flipped, for the b columns that contain a violation.
    allow_u / allow_v: bool arrays of the same shapes; False = tabu (already includes aspiration: a tabu move that
    reaches a new best f is allowed).
    Return ("u", x, k) or ("v", y, k) with indices into du / dv, or None to skip this iteration.

    Default: best allowed move, ties broken at random.
    """
    big = 1e18
    au = np.where(allow_u, du, big) + np.array([[rng.random() * 1e-3 for _ in range(du.shape[1])] for _ in range(du.shape[0])]) if du.size else du
    av = np.where(allow_v, dv, big) + np.array([[rng.random() * 1e-3 for _ in range(dv.shape[1])] for _ in range(dv.shape[0])]) if dv.size else dv
    best = None
    if du.size:
        x = int(np.argmin(au))
        if au.flat[x] < big:
            best = ("u", x // du.shape[1], x % du.shape[1], au.flat[x])
    if dv.size:
        y = int(np.argmin(av))
        if av.flat[y] < big and (best is None or av.flat[y] < best[3]):
            best = ("v", y // dv.shape[1], y % dv.shape[1], av.flat[y])
    return None if best is None else best[:3]


def column_to_drop(U, V, M):
    """Index of the column of U, V to remove when trying rank r - 1.

    Default: the column whose removal violates the fewest constraints.
    """
    n, r = U.shape
    P = (U.astype(np.int64) @ V.astype(np.int64).T) & 1
    T = np.eye(n, dtype=np.int64)
    best_k, best_v = 0, None
    for k in range(r):
        Pk = P ^ np.outer(U[:, k], V[:, k]).astype(np.int64)
        v = int((M & (Pk != T)).sum())
        if best_v is None or v < best_v:
            best_k, best_v = k, v
    return best_k


def init_factors(n, r, M, rng, prev):
    """Starting factors (U, V), each an (n, r) 0/1 int array, for the search at rank r.

    prev = (U, V) after column_to_drop removed one column from the last solution. Default: start from prev.
    """
    return prev
# EVOLVE-BLOCK-END


# ---------------------------------------------------------------------------------------------------------------------
# Fixed skeleton. Do not change below this line (OpenEvolve restores it).
# ---------------------------------------------------------------------------------------------------------------------
def _adjacency(n, out):
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        m = out[i]
        while m:
            low = m & -m
            adj[i, low.bit_length() - 1] = True
            m ^= low
    np.fill_diagonal(adj, False)
    return adj


def _dsatur(conf):
    n = conf.shape[0]
    colors = [-1] * n
    sat = [set() for _ in range(n)]
    deg = conf.sum(1)
    for _ in range(n):
        v = max((u for u in range(n) if colors[u] < 0), key=lambda u: (len(sat[u]), int(deg[u]), -u))
        c = 0
        while c in sat[v]:
            c += 1
        colors[v] = c
        for u in np.flatnonzero(conf[v]):
            sat[int(u)].add(c)
    return colors


def _rows(U, V):
    P = (U.astype(np.int64) @ V.astype(np.int64).T) & 1
    return [sum(1 << int(j) for j in np.flatnonzero(P[i])) for i in range(P.shape[0])]


def _fits(rows, n, out):
    full = (1 << n) - 1
    for i in range(n):
        if not (rows[i] >> i) & 1:
            return False
        if rows[i] & full & ~(out[i] | (1 << i)):
            return False
    return True


def _decode(Aeq, b, cur, trials, hard, rng):
    """Information-set decoding of Aeq x = b over F_2; equation `hard` is always the first pivot (weight m + 1)."""
    m, r = Aeq.shape
    w = np.ones(m, dtype=np.int64)
    w[hard] = m + 1

    def cost(x):
        return int(((((Aeq @ x) & 1) != b) * w).sum())

    best_x, best_c = cur, cost(cur)
    rows_bits = [sum(1 << k for k in range(r) if Aeq[e, k]) for e in range(m)]
    for _ in range(trials):
        if best_c == 0:
            break
        order = [hard] + [e for e in rng.sample(range(m), m) if e != hard]
        piv = {}
        for e in order:
            row, rhs = rows_bits[e], int(b[e])
            for c, (prow, prhs) in piv.items():
                if (row >> c) & 1:
                    row ^= prow
                    rhs ^= prhs
            if row == 0:
                continue
            c = row.bit_length() - 1
            for c2 in list(piv):
                prow, prhs = piv[c2]
                if (prow >> c) & 1:
                    piv[c2] = (prow ^ row, prhs ^ rhs)
            piv[c] = (row, rhs)
            if len(piv) == r:
                break
        x = np.array([rng.randint(0, 1) for _ in range(r)], dtype=np.int64)
        for c in piv:
            x[c] = 0
        for c, (row, rhs) in piv.items():
            s = rhs
            rest = row & ~(1 << c)
            while rest:
                low = rest & -rest
                s ^= int(x[low.bit_length() - 1])
                rest ^= low
            x[c] = s
        cst = cost(x)
        if cst < best_c:
            best_x, best_c = x, cst
    return best_x


class _State:
    def __init__(self, M, U, V):
        self.M = M
        self.n, self.r = U.shape
        self.U = U.astype(np.float32)
        self.V = V.astype(np.float32)
        self.T = np.eye(self.n, dtype=np.uint8)
        self.P = ((U.astype(np.int64) @ V.astype(np.int64).T) & 1).astype(np.uint8)
        self.viol = M & (self.P != self.T)
        self.S = np.where(M, 1.0 - 2.0 * self.viol, 0.0).astype(np.float32)
        self.f = int(self.viol.sum())

    def flip_u(self, i, k):
        self.U[i, k] = 1.0 - self.U[i, k]
        self.P[i] ^= self.V[:, k].astype(np.uint8)
        old = int(self.viol[i].sum())
        self.viol[i] = self.M[i] & (self.P[i] != self.T[i])
        self.S[i] = np.where(self.M[i], 1.0 - 2.0 * self.viol[i], 0.0)
        self.f += int(self.viol[i].sum()) - old

    def flip_v(self, j, k):
        self.V[j, k] = 1.0 - self.V[j, k]
        self.P[:, j] ^= self.U[:, k].astype(np.uint8)
        old = int(self.viol[:, j].sum())
        self.viol[:, j] = self.M[:, j] & (self.P[:, j] != self.T[:, j])
        self.S[:, j] = np.where(self.M[:, j], 1.0 - 2.0 * self.viol[:, j], 0.0)
        self.f += int(self.viol[:, j].sum()) - old

    def factors(self):
        return self.U.astype(np.int64), self.V.astype(np.int64)


def _isd_sweep(st, trials, rng, deadline):
    U, V = st.factors()
    M, n = st.M, st.n
    for j in rng.sample(range(n), n):
        if time.perf_counter() > deadline:
            break
        C = np.flatnonzero(M[:, j])
        V[j] = _decode(U[C], (C == j).astype(np.int64), V[j], trials, int(np.flatnonzero(C == j)[0]), rng)
    for i in rng.sample(range(n), n):
        if time.perf_counter() > deadline:
            break
        C = np.flatnonzero(M[i])
        U[i] = _decode(V[C], (C == i).astype(np.int64), U[i], trials, int(np.flatnonzero(C == i)[0]), rng)
    return _State(M, U, V)


def _search_rank(M, U0, V0, deadline, rng):
    """Drive violations to 0 at the rank of U0. Returns (U, V) with f = 0, or None at the deadline."""
    st = _State(M, U0, V0)
    n, r = st.n, st.r
    tabu_u = np.zeros((n, r), dtype=np.int64)
    tabu_v = np.zeros((n, r), dtype=np.int64)
    best_f, best_state, it, last_improve = st.f, st.factors(), 0, 0
    while st.f > 0:
        it += 1
        if (it & 15) == 0 and time.perf_counter() > deadline:
            return None
        rows = np.flatnonzero(st.viol.any(1))
        cols = np.flatnonzero(st.viol.any(0))
        du = (st.S @ st.V)[rows]
        dv = (st.S.T @ st.U)[cols]
        allow_u = (tabu_u[rows] <= it) | (st.f + du < best_f)
        allow_v = (tabu_v[cols] <= it) | (st.f + dv < best_f)
        mv = choose_move(du, dv, st.f, best_f, it, allow_u, allow_v, rng)
        if mv is not None:
            kind, x, k = mv
            if kind == "u":
                i = int(rows[int(x)])
                st.flip_u(i, int(k))
                tabu_u[i, int(k)] = it + max(0, int(tenure(st.f, it, rng)))
            else:
                j = int(cols[int(x)])
                st.flip_v(j, int(k))
                tabu_v[j, int(k)] = it + max(0, int(tenure(st.f, it, rng)))
        if st.f < best_f:
            best_f, best_state, last_improve = st.f, st.factors(), it
        stall = it - last_improve
        if stall and stall % int(PARAMS["isd_after_stall"]) == 0:
            cand = _isd_sweep(st, int(PARAMS["isd_trials"]), rng, deadline)
            if cand.f < st.f:
                st = cand
                if st.f < best_f:
                    best_f, best_state, last_improve = st.f, st.factors(), it
        if stall and stall % int(PARAMS["kick_after_stall"]) == 0:
            U, V = best_state[0].copy(), best_state[1].copy()
            fr = float(PARAMS["kick_fraction"])
            U ^= (np.array([[rng.random() < fr for _ in range(r)] for _ in range(n)])).astype(np.int64)
            V ^= (np.array([[rng.random() < fr for _ in range(r)] for _ in range(n)])).astype(np.int64)
            st = _State(M, U, V)
    return st.factors()


def solve(n, out, budget_s, seed):
    """Return rows (list of n ints) of a matrix over F_2 that fits the graph, of rank as low as found in budget_s s."""
    t0 = time.perf_counter()
    deadline = t0 + max(0.0, float(budget_s)) * 0.95
    rng = random.Random(seed)
    adj = _adjacency(n, out)
    mutual = adj & adj.T
    conf = ~mutual
    np.fill_diagonal(conf, False)
    colors = _dsatur(conf)
    k = max(colors) + 1
    U = np.zeros((n, k), dtype=np.int64)
    U[np.arange(n), colors] = 1
    V = U.copy()
    best_rows = _rows(U, V)  # clique cover: always fits
    M = ~adj
    np.fill_diagonal(M, True)
    try:
        while U.shape[1] > 1 and time.perf_counter() < deadline:
            kd = int(column_to_drop(U, V, M))
            keep = [c for c in range(U.shape[1]) if c != kd]
            prev = (U[:, keep].copy(), V[:, keep].copy())
            U1, V1 = init_factors(n, U.shape[1] - 1, M, rng, prev)
            res = _search_rank(M, np.asarray(U1, dtype=np.int64) & 1, np.asarray(V1, dtype=np.int64) & 1, deadline, rng)
            if res is None:
                break
            rows = _rows(res[0], res[1])
            if not _fits(rows, n, out):
                break
            U, V = res
            best_rows = rows
    except Exception:
        pass  # keep the best fitting answer found so far
    return best_rows
