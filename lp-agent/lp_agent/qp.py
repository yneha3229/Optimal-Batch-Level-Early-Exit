"""Convex QPs: loader, independent certificate check, and the existing QP methods.

Form:  min 1/2 x'Qx + c'x + c0   s.t.  rl <= Ax <= ru,  cl <= x <= cu      (Q symmetric positive semidefinite)

Certificate check (`kkt`), from x and row duals y only:
  z = Qx + c - A'y   (multipliers of the variable bounds)
  primal residual  ||bound violations||_2 / (1 + ||finite row bounds||_2)
  dual residual    ||sign violations of y and z||_2 / (1 + ||c||_2 + ||Qx||_2)
  gap              |p - d| / (1 + |p| + |d|),  p = 1/2 x'Qx + c'x + c0,
                   d = -1/2 x'Qx + (bound terms of y and z) + c0                       (Wolfe dual)
Certified at accuracy eps when all three are <= eps. No solver's own claim is trusted.

Methods (each one thread, each run in its own forked process with a hard time limit):
  highs     HiGHS active-set QP           clarabel  Clarabel interior point
  piqp      PIQP proximal interior point  proxqp    ProxQP proximal augmented Lagrangian
  osqp      OSQP (ADMM) + polishing       scs       SCS (ADMM, conic form)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

INF = 1e30


@dataclass
class QP:
    name: str
    Q: sp.csc_matrix
    c: np.ndarray
    c0: float
    A: sp.csr_matrix
    rl: np.ndarray
    ru: np.ndarray
    cl: np.ndarray
    cu: np.ndarray

    @property
    def shape(self):
        return self.A.shape


def load_qps(path: str, name: str | None = None) -> QP:
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    path = str(path)
    if not path.lower().endswith((".mps", ".qps")):
        tmp = os.path.join(tempfile.mkdtemp(), "model.mps")
        shutil.copy(path, tmp)
        path = tmp
    h.readModel(path)
    model = h.getModel()
    lp, hes = model.lp_, model.hessian_
    inf = h.getInfinity()
    m, n = lp.num_row_, lp.num_col_
    a = lp.a_matrix_
    A = sp.csc_matrix((np.asarray(a.value_), np.asarray(a.index_), np.asarray(a.start_)), shape=(m, n)).tocsr()
    if hes.dim_ > 0:
        L = sp.csc_matrix((np.asarray(hes.value_), np.asarray(hes.index_), np.asarray(hes.start_)), shape=(n, n))
        if str(hes.format_).endswith("kTriangular"):
            Q = (L + L.T - sp.diags(L.diagonal())).tocsc()
        else:
            Q = L.tocsc()
    else:
        Q = sp.csc_matrix((n, n))

    def fix(v):
        v = np.asarray(v, dtype=np.float64).copy()
        v[v >= inf] = np.inf
        v[v <= -inf] = -np.inf
        return v

    c, c0 = np.asarray(lp.col_cost_, dtype=np.float64), float(lp.offset_)
    if str(lp.sense_).endswith("kMaximize"):
        c, c0, Q = -c, -c0, -Q
    return QP(name or path, Q, c, c0, A, fix(lp.row_lower_), fix(lp.row_upper_), fix(lp.col_lower_),
              fix(lp.col_upper_))


def _dual_part(dual, lo, hi):
    pos, neg = np.maximum(dual, 0), np.maximum(-dual, 0)
    lo_f, hi_f = np.isfinite(lo), np.isfinite(hi)
    return (np.sum(pos[lo_f] * lo[lo_f]) - np.sum(neg[hi_f] * hi[hi_f]),
            np.concatenate([pos[~lo_f], neg[~hi_f]]))


def kkt(qp: QP, x, y) -> dict:
    bad = {"primal": np.inf, "dual": np.inf, "gap": np.inf, "kkt": np.inf, "pobj": np.nan, "dobj": np.nan}
    if x is None or y is None:
        return bad
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != (qp.shape[1],) or y.shape != (qp.shape[0],) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return bad
    Ax, Qx = qp.A @ x, qp.Q @ x
    pviol = np.concatenate([np.maximum(qp.rl - Ax, 0), np.maximum(Ax - qp.ru, 0),
                            np.maximum(qp.cl - x, 0), np.maximum(x - qp.cu, 0)])
    pviol = pviol[np.isfinite(pviol)]
    bnorm = np.concatenate([qp.rl[np.isfinite(qp.rl)], qp.ru[np.isfinite(qp.ru)]])
    z = Qx + qp.c - qp.A.T @ y
    o1, i1 = _dual_part(y, qp.rl, qp.ru)
    o2, i2 = _dual_part(z, qp.cl, qp.cu)
    xQx = float(x @ Qx)
    pobj = 0.5 * xQx + float(qp.c @ x) + qp.c0
    dobj = -0.5 * xQx + float(o1 + o2) + qp.c0
    r = {"primal": float(np.linalg.norm(pviol) / (1 + np.linalg.norm(bnorm))),
         "dual": float(np.linalg.norm(np.concatenate([i1, i2])) / (1 + np.linalg.norm(qp.c) + np.linalg.norm(Qx))),
         "gap": float(abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj))), "pobj": pobj, "dobj": dobj}
    r["kkt"] = max(r["primal"], r["dual"], r["gap"])
    return r


# ----------------------------------------------------------------------------------------------------- methods
def _rows(qp):
    """Row index sets: equalities, finite upper sides, finite lower sides (of non-equalities)."""
    eq = np.isfinite(qp.rl) & np.isfinite(qp.ru) & (qp.rl == qp.ru)
    up = np.isfinite(qp.ru) & ~eq
    lo = np.isfinite(qp.rl) & ~eq
    return eq, up, lo


def _highs(qp, tol, tl):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("threads", 1)
    h.setOptionValue("time_limit", float(max(0.01, tl)))
    h.setOptionValue("primal_feasibility_tolerance", max(tol, 1e-10))
    h.setOptionValue("dual_feasibility_tolerance", max(tol, 1e-10))
    m, n = qp.shape
    model = highspy.HighsModel()
    lp = model.lp_
    lp.num_col_, lp.num_row_ = n, m
    lp.col_cost_, lp.offset_ = qp.c, qp.c0
    lp.col_lower_ = np.where(np.isfinite(qp.cl), qp.cl, -INF)
    lp.col_upper_ = np.where(np.isfinite(qp.cu), qp.cu, INF)
    lp.row_lower_ = np.where(np.isfinite(qp.rl), qp.rl, -INF)
    lp.row_upper_ = np.where(np.isfinite(qp.ru), qp.ru, INF)
    A = qp.A.tocsc()
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = A.indptr, A.indices, A.data
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = n, m
    L = sp.tril(qp.Q).tocsc()
    if L.nnz:
        hes = model.hessian_
        hes.dim_ = n
        hes.format_ = highspy.HessianFormat.kTriangular
        hes.start_, hes.index_, hes.value_ = L.indptr, L.indices, L.data
    h.passModel(model)
    h.run()
    sol = h.getSolution()
    return (np.asarray(sol.col_value) if sol.value_valid else None,
            np.asarray(sol.row_dual) if sol.dual_valid else None, h.modelStatusToString(h.getModelStatus()))


def _conic(qp):
    """Conic form A_c x + s = b, s in Zero x Nonneg, for Clarabel and SCS."""
    eq, up, lo = _rows(qp)
    n = qp.shape[1]
    I = sp.identity(n, format="csr")
    cu_f, cl_f = np.isfinite(qp.cu), np.isfinite(qp.cl)
    Ac = sp.vstack([qp.A[eq], qp.A[up], -qp.A[lo], I[cu_f], -I[cl_f]]).tocsc()
    b = np.concatenate([qp.rl[eq], qp.ru[up], -qp.rl[lo], qp.cu[cu_f], -qp.cl[cl_f]])
    return Ac, b, int(eq.sum()), (eq, up, lo)


def _row_duals_from_conic(qp, zc, n_eq, sets):
    eq, up, lo = sets
    y = np.zeros(qp.shape[0])
    k = 0
    y[eq] = -zc[k:k + n_eq]
    k += n_eq
    nu = int(up.sum())
    y[up] -= zc[k:k + nu]
    k += nu
    nl = int(lo.sum())
    y[lo] += zc[k:k + nl]
    return y


def _clarabel(qp, tol, tl):
    import clarabel

    Ac, b, n_eq, sets = _conic(qp)
    s = clarabel.DefaultSettings()
    s.verbose = False
    s.max_threads = 1
    s.time_limit = float(max(0.01, tl))
    s.tol_gap_abs = s.tol_gap_rel = s.tol_feas = s.tol_ktratio = tol
    cones = ([clarabel.ZeroConeT(n_eq)] if n_eq else []) + [clarabel.NonnegativeConeT(Ac.shape[0] - n_eq)]
    sol = clarabel.DefaultSolver(sp.triu(qp.Q).tocsc(), qp.c, Ac, b, cones, s).solve()
    return np.asarray(sol.x), _row_duals_from_conic(qp, np.asarray(sol.z), n_eq, sets), str(sol.status)


def _scs(qp, tol, tl):
    import scs

    Ac, b, n_eq, sets = _conic(qp)
    data = {"P": sp.triu(qp.Q).tocsc(), "A": Ac, "b": b, "c": qp.c}
    solver = scs.SCS(data, {"z": n_eq, "l": Ac.shape[0] - n_eq}, verbose=False, eps_abs=tol, eps_rel=tol,
                     time_limit_secs=float(max(0.01, tl)), max_iters=10**8)
    sol = solver.solve()
    return sol["x"], _row_duals_from_conic(qp, sol["y"], n_eq, sets), sol["info"]["status"]


def _osqp(qp, tol, tl):
    import osqp

    m, n = qp.shape
    A = sp.vstack([qp.A, sp.identity(n)]).tocsc()
    l = np.concatenate([qp.rl, qp.cl])
    u = np.concatenate([qp.ru, qp.cu])
    s = osqp.OSQP()
    s.setup(P=sp.triu(qp.Q).tocsc(), q=qp.c, A=A, l=l, u=u, verbose=False, eps_abs=tol, eps_rel=tol,
            time_limit=float(max(0.01, tl)), max_iter=10**8, polishing=True)
    r = s.solve()
    return np.asarray(r.x), -np.asarray(r.y)[:m], str(r.info.status)


def _piqp(qp, tol, tl):
    import piqp

    eq, up, lo = _rows(qp)
    ineq = up | lo
    s = piqp.SparseSolver()
    s.settings.verbose = False
    s.settings.eps_abs = tol
    s.settings.eps_rel = tol
    s.settings.max_iter = 10_000
    G = qp.A[ineq].tocsc()
    s.setup(sp.triu(qp.Q).tocsc(), qp.c, qp.A[eq].tocsc(), qp.rl[eq], G, qp.rl[ineq], qp.ru[ineq], qp.cl, qp.cu)
    s.solve()
    r = s.result
    y = np.zeros(qp.shape[0])
    y[eq] = -np.asarray(r.y)
    y[ineq] = np.asarray(r.z_l) - np.asarray(r.z_u)
    return np.asarray(r.x), y, str(r.info.status)


def _proxqp(qp, tol, tl):
    from proxsuite import proxqp

    eq, up, lo = _rows(qp)
    m, n = qp.shape
    ineq = up | lo
    box = np.isfinite(qp.cl) | np.isfinite(qp.cu)
    C = sp.vstack([qp.A[ineq], sp.identity(n, format="csr")[box]]).tocsc()
    l = np.concatenate([qp.rl[ineq], qp.cl[box]])
    u = np.concatenate([qp.ru[ineq], qp.cu[box]])
    l = np.where(np.isfinite(l), l, -INF)
    u = np.where(np.isfinite(u), u, INF)
    s = proxqp.sparse.QP(n, int(eq.sum()), C.shape[0])
    s.settings.eps_abs = tol
    s.settings.eps_rel = 0.0
    s.settings.verbose = False
    s.settings.max_iter = 10**6
    s.init(qp.Q.tocsc(), qp.c, qp.A[eq].tocsc(), qp.rl[eq], C, l, u)
    s.solve()
    r = s.results
    y = np.zeros(m)
    y[eq] = -np.asarray(r.y)
    y[ineq] = -np.asarray(r.z)[: int(ineq.sum())]
    return np.asarray(r.x), y, str(r.info.status)


METHODS = {"highs": _highs, "clarabel": _clarabel, "piqp": _piqp, "proxqp": _proxqp, "osqp": _osqp, "scs": _scs}


def _child(fn, qp, tol, tl, conn):
    try:
        t = time.perf_counter()
        x, y, st = fn(qp, tol, tl)
        conn.send((x, y, st, time.perf_counter() - t))
    except Exception as e:  # noqa: BLE001
        conn.send((None, None, f"error: {e!r}"[:200], 0.0))


def preload():
    """Import every solver once in the parent, so forked runs do not pay import time."""
    import clarabel  # noqa: F401
    import highspy  # noqa: F401
    import osqp  # noqa: F401
    import piqp  # noqa: F401
    import scs  # noqa: F401
    from proxsuite import proxqp  # noqa: F401


def run(name: str, qp: QP, tol: float, time_limit: float):
    """One run in a forked process, killed at the time limit. Returns (x, y, status, seconds)."""
    preload()
    ctx = mp.get_context("fork")
    a, b = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(METHODS[name], qp, tol, time_limit, b), daemon=True)
    t = time.perf_counter()
    p.start()
    if a.poll(time_limit + 1.0):
        out = a.recv()
        p.join(1)
    else:
        out = (None, None, "killed at time limit", time.perf_counter() - t)
    if p.is_alive():
        p.kill()
        p.join()
    return out
