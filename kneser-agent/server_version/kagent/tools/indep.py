"""Independent sets (lower bound on beta: alpha(G) = MAIS(G) <= beta(G) for undirected G) and upper bounds on alpha.

indep_ils   iterated local search with (1,2)-swaps and forced insertions (Andrade, Resende & Werneck 2012), numba,
            on the dense adjacency matrix. Anytime; restarts from random greedy solutions.
indep_cpsat exact maximum independent set by CP-SAT (n <= 300): also proves alpha.
alpha_hoffman  ratio bound n(-l_min)/(l_max - l_min) for regular graphs (numerical, information only).
alpha_lovasz   Lovasz theta by SDP (n <= 300, numerical, information only).
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


@njit(cache=True)
def _add(A, v, insol, tight):
    insol[v] = True
    row = A[v]
    for u in range(A.shape[0]):
        if row[u]:
            tight[u] += 1


@njit(cache=True)
def _remove(A, v, insol, tight):
    insol[v] = False
    row = A[v]
    for u in range(A.shape[0]):
        if row[u]:
            tight[u] -= 1


@njit(cache=True)
def _fill_free(A, insol, tight):
    n = A.shape[0]
    order = np.random.permutation(n)
    added = 0
    for v in order:
        if not insol[v] and tight[v] == 0:
            _add(A, v, insol, tight)
            added += 1
    return added


@njit(cache=True)
def _two_improve(A, insol, tight):
    """Find x in S with two non-adjacent 1-tight neighbours u, w; swap x out, u and w in. Returns True if done."""
    n = A.shape[0]
    sol = np.nonzero(insol)[0]
    perm = np.random.permutation(len(sol))
    cand = np.empty(n, dtype=np.int64)
    for a in range(len(sol)):
        x = sol[perm[a]]
        k = 0
        row = A[x]
        for u in range(n):
            if row[u] and tight[u] == 1 and not insol[u]:
                cand[k] = u
                k += 1
        if k < 2:
            continue
        for i in range(k):
            u = cand[i]
            for j in range(i + 1, k):
                w = cand[j]
                if not A[u, w]:
                    _remove(A, x, insol, tight)
                    _add(A, u, insol, tight)
                    _add(A, w, insol, tight)
                    _fill_free(A, insol, tight)
                    return True
    return False


@njit(cache=True)
def _local_search(A, insol, tight):
    _fill_free(A, insol, tight)
    while _two_improve(A, insol, tight):
        pass
    return insol.sum()


@njit(cache=True)
def _perturb(A, insol, tight, strength):
    n = A.shape[0]
    for _ in range(strength):
        while True:
            v = np.random.randint(n)
            if not insol[v]:
                break
        row = A[v]
        for u in range(n):
            if row[u] and insol[u]:
                _remove(A, u, insol, tight)
        _add(A, v, insol, tight)


@njit(cache=True)
def ils_round(A, insol, tight, best, iters):
    """`iters` ILS iterations from the current solution. Keeps `best` (bool vector) updated. Returns best size."""
    bsize = best.sum()
    cur = insol.sum()
    for it in range(iters):
        save = insol.copy()
        save_t = tight.copy()
        strength = 1 if np.random.random() < 0.5 else 1 + np.random.randint(3)
        _perturb(A, insol, tight, strength)
        size = _local_search(A, insol, tight)
        if size > bsize:
            bsize = size
            best[:] = insol
        if size >= cur or np.random.random() < 1.0 / (1 + 4 * (cur - size)):
            cur = size
        else:
            insol[:] = save
            tight[:] = save_t
    return bsize


def indep_ils(A: np.ndarray, time_limit: float, seed: int = 0, restarts: int = 4, on_improve=None, stop=None,
              target: int | None = None) -> list[int]:
    """Anytime maximum independent set. on_improve(vertices) is called on every improvement."""
    n = A.shape[0]
    _seed(seed)
    t0 = time.perf_counter()
    best_all = np.zeros(n, dtype=np.bool_)
    per = max(0.5, time_limit / max(1, restarts))
    r = 0
    while time.perf_counter() - t0 < time_limit:
        insol = np.zeros(n, dtype=np.bool_)
        tight = np.zeros(n, dtype=np.int32)
        _local_search(A, insol, tight)
        best = insol.copy()
        t1 = time.perf_counter()
        it = 8
        while time.perf_counter() - t1 < per and time.perf_counter() - t0 < time_limit:
            ils_round(A, insol, tight, best, it)
            if best.sum() > best_all.sum():
                best_all[:] = best
                if on_improve is not None:
                    on_improve(np.nonzero(best_all)[0].tolist())
            if target is not None and best_all.sum() >= target:
                return np.nonzero(best_all)[0].tolist()
            if stop is not None and stop():
                return np.nonzero(best_all)[0].tolist()
            it = min(4096, it * 2)
        r += 1
        _seed(seed + 7919 * r)
    return np.nonzero(best_all)[0].tolist()


def indep_cpsat(A: np.ndarray, time_limit: float, workers: int = 4, hint: list[int] | None = None,
                on_improve=None) -> tuple[list[int], bool, int]:
    """CP-SAT maximum independent set. Returns (set, proven optimal, proven upper bound on alpha)."""
    from ortools.sat.python import cp_model

    n = A.shape[0]
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(n)]
    iu, ju = np.nonzero(np.triu(A, 1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        m.AddBoolOr([x[i].Not(), x[j].Not()])
    m.Maximize(sum(x))
    if hint:
        hs = set(hint)
        for i in range(n):
            m.AddHint(x[i], i in hs)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = workers

    class CB(cp_model.CpSolverSolutionCallback):
        def __init__(self):
            super().__init__()
            self.best = 0

        def on_solution_callback(self):
            v = int(round(self.ObjectiveValue()))
            if v > self.best and on_improve is not None:
                self.best = v
                on_improve([i for i in range(n) if self.Value(x[i])])

    st = s.Solve(m, CB())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], False, n
    S = [i for i in range(n) if s.Value(x[i])]
    return S, st == cp_model.OPTIMAL, int(np.floor(s.BestObjectiveBound() + 1e-6))


def alpha_hoffman(n: int, regular: bool, lmin: float, lmax: float) -> float | None:
    if not regular or lmax - lmin <= 0:
        return None
    return n * (-lmin) / (lmax - lmin)


def alpha_lovasz(A: np.ndarray, time_limit: float = 30.0) -> float | None:
    """Lovasz theta (an upper bound on alpha) by SDP: max <J, X>, tr X = 1, X_ij = 0 on edges, X psd."""
    import cvxpy as cp

    n = A.shape[0]
    if n > 300:
        return None
    X = cp.Variable((n, n), symmetric=True)
    iu, ju = np.nonzero(np.triu(A, 1))
    cons = [X >> 0, cp.trace(X) == 1]
    if len(iu):
        cons.append(X[iu, ju] == 0)
    prob = cp.Problem(cp.Maximize(cp.sum(X)), cons)
    try:
        prob.solve(solver="SCS", max_iters=20000, eps=1e-6, time_limit_secs=time_limit)
    except Exception:
        return None
    return float(prob.value) if prob.value is not None else None
