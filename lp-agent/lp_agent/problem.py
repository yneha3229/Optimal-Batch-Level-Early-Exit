"""LP instances and the independent certificate check.

Every LP is held in one form:   min c'x + c0   s.t.   rl <= A x <= ru,   cl <= x <= cu   (bounds may be infinite).
Integer variables of MIPLIB models are relaxed (the LP relaxation is the LP we solve). Maximisation is negated.

Certificate check (`kkt`): from a primal x and row duals y alone, recompute everything:
  reduced costs  z = c - A'y
  primal residual  ||violation of row and column bounds||_2 / (1 + ||finite row bounds||_2)
  dual residual    ||dual sign violations||_2 / (1 + ||c||_2)  (a dual pushing on an infinite bound is infeasible)
  gap              |primal objective - dual objective| / (1 + |primal objective| + |dual objective|)
the relative criterion used by PDLP. A solution is certified at accuracy eps when all three are <= eps. No solver's
own claim is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass
class LP:
    name: str
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

    @property
    def nnz(self):
        return self.A.nnz


def load_mps(path: str, name: str | None = None) -> LP:
    import highspy

    import os
    import shutil
    import tempfile

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    path = str(path)
    if not path.lower().endswith((".mps", ".mps.gz", ".lp")):   # e.g. Netlib .SIF files are free MPS
        tmp = os.path.join(tempfile.mkdtemp(), "model.mps")
        shutil.copy(path, tmp)
        path = tmp
    h.readModel(path)
    lp = h.getLp()
    inf = h.getInfinity()
    m, n = lp.num_row_, lp.num_col_
    a = lp.a_matrix_
    A = sp.csc_matrix((np.asarray(a.value_), np.asarray(a.index_), np.asarray(a.start_)), shape=(m, n)).tocsr()

    def fix(v):
        v = np.asarray(v, dtype=np.float64).copy()
        v[v >= inf] = np.inf
        v[v <= -inf] = -np.inf
        return v

    c = np.asarray(lp.col_cost_, dtype=np.float64)
    c0 = float(lp.offset_)
    if int(lp.sense_) == -1 or str(lp.sense_).endswith("kMaximize"):
        c, c0 = -c, -c0
    return LP(name or str(path), c, c0, A, fix(lp.row_lower_), fix(lp.row_upper_), fix(lp.col_lower_),
              fix(lp.col_upper_))


def _dual_part(dual, lo, hi):
    """Dual objective contribution and infeasibility of multipliers on bounds lo <= . <= hi."""
    pos, neg = np.maximum(dual, 0), np.maximum(-dual, 0)
    lo_f, hi_f = np.isfinite(lo), np.isfinite(hi)
    obj = np.sum(pos[lo_f] * lo[lo_f]) - np.sum(neg[hi_f] * hi[hi_f])
    infeas = np.concatenate([pos[~lo_f], neg[~hi_f]])
    return obj, infeas


def kkt(lp: LP, x, y) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != (lp.shape[1],) or y.shape != (lp.shape[0],) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return {"primal": np.inf, "dual": np.inf, "gap": np.inf, "kkt": np.inf, "pobj": np.nan, "dobj": np.nan}
    Ax = lp.A @ x
    pviol = np.concatenate([np.maximum(lp.rl - Ax, 0), np.maximum(Ax - lp.ru, 0),
                            np.maximum(lp.cl - x, 0), np.maximum(x - lp.cu, 0)])
    pviol = pviol[np.isfinite(pviol)]
    bnorm = np.concatenate([lp.rl[np.isfinite(lp.rl)], lp.ru[np.isfinite(lp.ru)]])
    z = lp.c - lp.A.T @ y
    o1, i1 = _dual_part(y, lp.rl, lp.ru)
    o2, i2 = _dual_part(z, lp.cl, lp.cu)
    pobj = float(lp.c @ x + lp.c0)
    dobj = float(o1 + o2 + lp.c0)
    r = {"primal": float(np.linalg.norm(pviol) / (1 + np.linalg.norm(bnorm))),
         "dual": float(np.linalg.norm(np.concatenate([i1, i2])) / (1 + np.linalg.norm(lp.c))),
         "gap": float(abs(pobj - dobj) / (1 + abs(pobj) + abs(dobj))), "pobj": pobj, "dobj": dobj}
    r["kkt"] = max(r["primal"], r["dual"], r["gap"])
    return r
