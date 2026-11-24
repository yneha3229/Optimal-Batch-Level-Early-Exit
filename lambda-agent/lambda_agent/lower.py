"""Lower bounds on beta, hence on lambda_F (every linear code is a code): lambda_F >= beta >= lower bound.

mais(g)     largest acyclic induced subgraph by CP-SAT (ordering model). The set is re-checked.
shannon(g)  polymatroid (Shannon) LP bound (Blasiak, Kleinberg, Lubetzky 2011): with g(S) = H(Y, X_S) / t,
              g(V) = n;  g(N(i)+i) = g(N(i));  g(K+i) + g(K+j) >= g(K+i+j) + g(K);  g(K+x) <= g(K) + 1;
              |K| <= g(K) <= n;  beta >= g(empty).
            The reported bound comes from rounded Lagrange multipliers through exact weak duality, so float error
            cannot make it wrong:  bound = -y.b_ub - mu.b_eq + sum_k min(d_k lb_k, d_k ub_k),
            d = c + A_ub^T y + A_eq^T mu,  y >= 0.  Limited to n <= 10 per component (2^n variables).
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from .graph import Digraph, bits
from .verify import is_acyclic

SHANNON_MAX_N = 10


def mais(g: Digraph, time_limit: float = 10.0) -> tuple[list[int], bool]:
    """(acyclic set, proven maximum)."""
    n = g.n
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{v}") for v in range(n)]
    pos = [m.NewIntVar(0, max(0, n - 1), f"p{v}") for v in range(n)]
    for u in range(n):
        for v in bits(g.out[u]):
            if g.knows(v, u):
                if u < v:
                    m.AddBoolOr([y[u].Not(), y[v].Not()])
            else:
                m.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
    m.Maximize(sum(y))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [0] if n else [], False
    verts = [v for v in range(n) if s.Value(y[v])]
    assert is_acyclic(g, verts)
    return verts, st == cp_model.OPTIMAL


def _shannon_lp(g: Digraph):
    n = g.n
    N, full = 1 << n, (1 << n) - 1
    A_ub, b_ub = [], []
    for K in range(N):
        free = full & ~K
        for i in bits(free):
            A_ub.append({K | 1 << i: 1, K: -1})
            b_ub.append(1)
            for j in bits(free >> (i + 1) << (i + 1)):
                A_ub.append({K | 1 << i | 1 << j: 1, K: 1, K | 1 << i: -1, K | 1 << j: -1})
                b_ub.append(0)
    for i in range(n):
        A_ub.append({full & ~(1 << i): 1, full: -1})
        b_ub.append(0)
    A_eq, b_eq = [{full: 1}], [n]
    for i in range(n):
        A_eq.append({g.out[i] | 1 << i: 1, g.out[i]: -1})
        b_eq.append(0)
    lb = [K.bit_count() for K in range(N)]
    return A_ub, b_ub, A_eq, b_eq, lb, [n] * N


def _sparse(rows, ncols):
    r, c, v = [], [], []
    for k, row in enumerate(rows):
        for j, a in row.items():
            r.append(k)
            c.append(j)
            v.append(a)
    return coo_matrix((v, (r, c)), shape=(len(rows), ncols)).tocsr()


def _certified(A_ub, b_ub, A_eq, b_eq, lb, ub, y, mu) -> Fraction:
    d = [Fraction(0)] * len(lb)
    d[0] = Fraction(1)
    const = Fraction(0)
    for rows, bs, mult in ((A_ub, b_ub, y), (A_eq, b_eq, mu)):
        for row, b, m in zip(rows, bs, mult):
            if m:
                const -= m * b
                for j, a in row.items():
                    d[j] += m * a
    for k, dk in enumerate(d):
        if dk:
            const += min(dk * lb[k], dk * ub[k])
    return const


def shannon(g: Digraph) -> Fraction:
    """Certified Shannon lower bound on beta(g) (exact rational, never above the true LP optimum)."""
    if g.n > SHANNON_MAX_N:
        raise ValueError(f"Shannon LP limited to n <= {SHANNON_MAX_N}")
    if g.n == 0:
        return Fraction(0)
    A_ub, b_ub, A_eq, b_eq, lb, ub = _shannon_lp(g)
    N = 1 << g.n
    c = np.zeros(N)
    c[0] = 1.0
    res = linprog(c, A_ub=_sparse(A_ub, N), b_ub=b_ub, A_eq=_sparse(A_eq, N), b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs-ipm")
    if res.status != 0:
        raise RuntimeError(res.message)
    target = Fraction(res.fun).limit_denominator(1000)
    best = None
    for den in (12, 60, 1000, 10**5):
        y = [max(Fraction(0), Fraction(float(-x)).limit_denominator(den)) for x in res.ineqlin.marginals]
        mu = [Fraction(float(-x)).limit_denominator(den) for x in res.eqlin.marginals]
        b = _certified(A_ub, b_ub, A_eq, b_eq, lb, ub, y, mu)
        if best is None or b > best:
            best = b
        if b >= target:
            break
    if best < target:  # rounding missed the optimum: solve the optimal basis exactly (rational arithmetic)
        ex = _exact_basis_duals(A_ub, b_ub, A_eq, b_eq, lb, ub, N)
        if ex is not None:
            best = max(best, _certified(A_ub, b_ub, A_eq, b_eq, lb, ub, *ex))
    if best.denominator > 10**9:  # report a short number below the certified one (still valid)
        best = Fraction(math.floor(best * 10**6), 10**6)
    return best


def _exact_basis_duals(A_ub, b_ub, A_eq, b_eq, lb, ub, N):
    """Optimal basis from HiGHS (interior point + crossover); its square system B^T y = c_B solved over the
    rationals with python-flint. Only used through _certified, so a bad basis can cost tightness, never validity."""
    try:
        import flint
        import highspy
    except ImportError:
        return None
    rows = A_ub + A_eq
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "on")
    c = np.zeros(N)
    c[0] = 1.0
    h.addVars(N, np.array(lb, float), np.array(ub, float))
    h.changeColsCost(N, np.arange(N, dtype=np.int32), c)
    M = _sparse(rows, N)
    h.addRows(len(rows), np.array([-inf] * len(A_ub) + list(map(float, b_eq))),
              np.array(list(map(float, b_ub)) + list(map(float, b_eq))), M.nnz,
              M.indptr[:-1].astype(np.int32), M.indices.astype(np.int32), M.data.astype(float))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    basis = h.getBasis()
    B = highspy.HighsBasisStatus.kBasic
    bcols = [j for j, st in enumerate(basis.col_status) if st == B]
    nrows = [i for i, st in enumerate(basis.row_status) if st != B]
    if len(bcols) != len(nrows) or not bcols:
        return None
    cpos = {j: t for t, j in enumerate(bcols)}
    mat = flint.fmpz_mat(len(bcols), len(bcols))
    for t, i in enumerate(nrows):
        for j, a in rows[i].items():
            if j in cpos:
                mat[cpos[j], t] = a
    try:
        sol = mat.solve(flint.fmpz_mat(len(bcols), 1, [int(c[j]) for j in bcols]))
    except Exception:
        return None
    yhat = [Fraction(0)] * len(rows)
    for t, i in enumerate(nrows):
        q = sol[t, 0]
        yhat[i] = Fraction(int(q.p), int(q.q))
    m = len(A_ub)
    return [max(Fraction(0), -v) for v in yhat[:m]], [-v for v in yhat[m:]]
