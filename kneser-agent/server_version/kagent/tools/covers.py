"""Clique covers (upper bound on beta: one XOR per clique) and fractional clique covers (upper bound on beta).

clique_cover        DSatur + Culberson iterated greedy + TabuCol on the complement (gap-agent numba kernels).
clique_cover_cpsat  CP-SAT colouring of the complement (n <= 300 in the agent; the baseline tries larger n).
frac_cover          column generation: LP over cliques (HiGHS, sparse) + pricing by randomised greedy maximum-
                    weight cliques (numba), and exact CP-SAT pricing when n <= 300 (then the optimum is proven).
                    Any feasible LP point is an upper bound; it is rationalised and scaled so every vertex is
                    covered >= 1 in exact arithmetic.
"""
from __future__ import annotations

import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "gap-agent"))
from gap_agent import kernels as K  # noqa: E402  (DSatur, iterated greedy, TabuCol; reused unchanged)


def conflict_csr(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CSR of the complement graph (two receivers conflict unless adjacent)."""
    n = A.shape[0]
    C = ~A
    np.fill_diagonal(C, False)
    deg = C.sum(1)
    cp = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=cp[1:])
    c = np.empty(int(cp[-1]), dtype=np.int32)
    for a in range(0, n, 1024):
        r, j = np.nonzero(C[a:a + 1024])
        c[cp[a]:cp[min(n, a + 1024)]] = j
    return cp, c


def colour_to_cliques(col: np.ndarray) -> list[list[int]]:
    out: dict[int, list[int]] = {}
    for v, x in enumerate(col.tolist()):
        out.setdefault(x, []).append(v)
    return list(out.values())


def clique_cover(A: np.ndarray, time_limit: float, seed: int = 0, on_improve=None, stop=None, floor: int = 0,
                 tabu_iters: int = 20000, csr=None) -> list[list[int]]:
    """Anytime clique cover. on_improve(cliques) on every improvement. Stops at `floor` (a known lower limit)."""
    t0 = time.perf_counter()
    cp, c = csr if csr is not None else conflict_csr(A)
    K.seed(seed)
    rng = np.random.default_rng(seed)
    best_col, best_k = None, A.shape[0] + 1

    def improve(col, k):
        nonlocal best_col, best_k
        col = np.unique(col, return_inverse=True)[1].astype(np.int32)
        k = int(col.max()) + 1
        if k < best_k:
            best_col, best_k = col.copy(), k
            if on_improve is not None:
                on_improve(colour_to_cliques(best_col))

    for noise in (0.0, 0.5, 0.5):
        col, k = K.dsatur(cp, c, noise)
        improve(col, k)
    left = lambda: time_limit - (time.perf_counter() - t0)  # noqa: E731
    col = best_col.copy()
    for _ in range(3):
        k = K.iterated_greedy(cp, c, col, best_k, 50)
        improve(col, k)
        if left() <= 0 or (stop and stop()) or best_k <= floor:
            return colour_to_cliques(best_col)
    tabu = None
    while left() > 0 and best_k > max(1, floor) and not (stop and stop()):
        if tabu is None or tabu[0] >= best_k:
            k = best_k - 1
            col = best_col.copy()
            while int(col.max()) + 1 > k:
                kk = int(col.max()) + 1
                drop = int(np.argmin(np.bincount(col, minlength=kk)))
                idx = np.nonzero(col == drop)[0]
                others = np.array([x for x in range(kk) if x != drop])
                col[idx] = others[rng.integers(0, len(others), len(idx))]
                col = np.unique(col, return_inverse=True)[1].astype(np.int32)
            gamma, f = K.tabu_init(cp, c, col, k)
            tabu = [k, col, gamma, np.zeros((A.shape[0], k), dtype=np.int64), int(f), 0]
        k, col, gamma, tb, f, it = tabu
        f, it = K.tabu_search(cp, c, col, gamma, tb, k, f, it, tabu_iters)
        tabu[4], tabu[5] = f, it
        if f == 0:
            improve(col, k)
            tabu = None
    return colour_to_cliques(best_col)


def clique_cover_cpsat(A: np.ndarray, time_limit: float, workers: int = 4, hint: list[list[int]] | None = None,
                       on_improve=None) -> tuple[list[list[int]], bool]:
    """Exact clique cover = colouring of the complement, CP-SAT with symmetry breaking (vertex 0 gets colour 0)."""
    from ortools.sat.python import cp_model

    n = A.shape[0]
    K_ = len(hint) if hint else n
    m = cp_model.CpModel()
    col = [m.NewIntVar(0, K_ - 1, f"c{i}") for i in range(n)]
    mx = m.NewIntVar(0, K_ - 1, "mx")
    m.Add(col[0] == 0)
    iu, ju = np.nonzero(np.triu(~A, 1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        m.Add(col[i] != col[j])
    for i in range(n):
        m.Add(col[i] <= mx)
    m.Minimize(mx)
    if hint:
        for ci, C in enumerate(hint):
            for v in C:
                m.AddHint(col[v], ci)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.best = n + 1

        def on_solution_callback(self):
            k = len({self.Value(x) for x in col})
            if k < self.best:
                self.best = k
                if on_improve is not None:
                    on_improve(colour_to_cliques(np.array([self.Value(x) for x in col])))

    st = s.Solve(m, CB())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], False
    colour = np.array([s.Value(x) for x in col])
    return colour_to_cliques(colour), st == cp_model.OPTIMAL


# ------------------------------------------------------------------------------------------------ fractional cover
@njit(cache=True)
def _greedy_weight_clique(A, w, order_noise, tries):
    """Randomised greedy maximum-weight clique: start from a heavy vertex, add the heaviest common neighbour."""
    n = A.shape[0]
    best_w = 0.0
    best = np.zeros(n, dtype=np.bool_)
    for t in range(tries):
        key = w * (1.0 + order_noise * np.random.random(n))
        order = np.argsort(-key)
        cand = np.ones(n, dtype=np.bool_)
        cur = np.zeros(n, dtype=np.bool_)
        tw = 0.0
        start = order[np.random.randint(min(n, 8))] if t else order[0]
        for v in np.concatenate((np.array([start]), order)):
            if cand[v] and not cur[v]:
                cur[v] = True
                tw += w[v]
                row = A[v]
                for u in range(n):
                    if not row[u]:
                        cand[u] = False
        if tw > best_w:
            best_w = tw
            best[:] = cur
    return best, best_w


def _rationalise(cols: list[list[int]], x: np.ndarray, n: int) -> tuple[list[list[int]], list[Fraction], Fraction]:
    """Round the LP point to exact fractions on a 1/D grid (rounding up), then scale so coverage >= 1 exactly."""
    D = 720720
    keep = [k for k in range(len(cols)) if x[k] > 1e-12]
    ws = [Fraction(int(np.ceil(x[k] * D - 1e-9)), D) for k in keep]
    cover = [Fraction(0)] * n
    for k, wk in zip(keep, ws):
        for v in cols[k]:
            cover[v] += wk
    low = min(cover)
    if low < 1:
        ws = [w / low for w in ws]
    cl = [cols[k] for k in keep]
    return cl, ws, sum(ws, Fraction(0))


def frac_cover(A: np.ndarray, time_limit: float, init_cliques: list[list[int]] | None = None, seed: int = 0,
               on_improve=None, stop=None, exact_pricing_max_n: int = 300, floor: float = 0.0):
    """Column generation. Returns (cliques, weights, value: Fraction, proven_optimal)."""
    from scipy.optimize import linprog
    from scipy.sparse import csc_matrix

    t0 = time.perf_counter()
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    cols: list[list[int]] = [[v] for v in range(n)]
    seen = {(v,) for v in range(n)}
    for C in init_cliques or []:
        t = tuple(sorted(C))
        if t not in seen:
            seen.add(t)
            cols.append(list(t))
    best_val = None
    best = None
    proven = False
    Af = A
    while True:
        if stop is not None and stop():
            break
        indptr = np.zeros(len(cols) + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(C) for C in cols])
        indices = np.concatenate([np.array(C, dtype=np.int64) for C in cols])
        M = csc_matrix((np.ones(len(indices)), indices, indptr), shape=(n, len(cols)))
        res = linprog(np.ones(len(cols)), A_ub=-M, b_ub=-np.ones(n), bounds=(0, None), method="highs")
        if res.status != 0:
            break
        x = res.x
        cl, ws, val = _rationalise(cols, x, n)
        if best_val is None or val < best_val:
            best_val, best = val, (cl, ws)
            if on_improve is not None:
                on_improve(cl, ws, val)
        if best_val <= floor:
            break
        if time.perf_counter() - t0 > time_limit:
            break
        pi = -np.asarray(res.ineqlin.marginals)
        new = []
        if n <= exact_pricing_max_n:
            C, opt = _price_cpsat(A, pi, max(0.1, time_limit - (time.perf_counter() - t0)))
            if C is not None and pi[C].sum() > 1 + 1e-7:
                new.append(C)
            elif C is not None and opt:
                proven = True
                break
        else:
            for _ in range(4):
                mask, wv = _greedy_weight_clique(Af, pi, 0.3, 16)
                if wv > 1 + 1e-7:
                    new.append(np.nonzero(mask)[0].tolist())
        added = 0
        for C in new:
            t = tuple(sorted(C))
            if t not in seen:
                seen.add(t)
                cols.append(list(t))
                added += 1
        if not added:
            break
    cl, ws = best
    return cl, ws, best_val, proven


def _price_cpsat(A: np.ndarray, pi: np.ndarray, time_limit: float):
    from ortools.sat.python import cp_model

    n = A.shape[0]
    m = cp_model.CpModel()
    x = [m.NewBoolVar("") for _ in range(n)]
    iu, ju = np.nonzero(np.triu(~A, 1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        m.AddBoolOr([x[i].Not(), x[j].Not()])
    scale = 10 ** 6
    m.Maximize(sum(int(round(pi[i] * scale)) * x[i] for i in range(n)))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 4
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False
    C = [i for i in range(n) if s.Value(x[i])]
    # proven only if even the rounded objective bound cannot exceed 1 (+ rounding slack)
    opt = st == cp_model.OPTIMAL and s.BestObjectiveBound() <= scale * (1 + 1e-7) + n
    return C, opt
