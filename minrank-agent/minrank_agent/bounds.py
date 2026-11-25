"""Certified bound tools.  Each returns a value and, where it makes sense, an explicit certificate re-checked here.

Lower bounds on beta(G):
  independence(A)      alpha = max independent set (CP-SAT); the independent set is the certificate
Upper bounds on beta(G) (every explicit code is a fitting matrix, so its rank bounds minrk_F >= beta):
  clique_cover(A)      smallest clique cover (CP-SAT colouring of the complement)
  spectral_field(A,p)  the construction  M = I + c A  over F_p, c chosen to kill the highest-multiplicity nonzero
                       eigenvalue; cost = rank_p(M).  New: run over several p and keep the best field.
  f2_minrank(A,r)      is there a rank-<= r fitting matrix over F_2?  (SAT; exact minrk_2 by searching r)

If a spectral upper bound meets alpha (with alpha proven optimal) then beta = minrk_{F_p} = alpha is PROVEN, and the
matrix M is an explicit optimal scalar linear index code over F_p.
"""
from __future__ import annotations

import sys
from pathlib import Path

import flint
import numpy as np
from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gap-agent"))
from gap_agent.lam import fit_rank, rank_gf2  # noqa: E402  (reuse the F_2 minrank SAT encoder)


def _rank_mod(ent_flat, n, p):
    return flint.nmod_mat(n, n, [int(x) % p for x in ent_flat], p).rank()


def independence(A, limit=10.0):
    n = A.shape[0]
    m = cp_model.CpModel()
    x = [m.NewBoolVar(str(i)) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                m.Add(x[i] + x[j] <= 1)
    m.Maximize(sum(x))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 4
    st = s.Solve(m)
    S = [i for i in range(n) if s.Value(x[i])]
    assert all(not A[i, j] for a, i in enumerate(S) for j in S[a + 1:])    # re-check independence
    return len(S), S, st == cp_model.OPTIMAL


def clique_cover(A, limit=10.0):
    n = A.shape[0]
    comp = 1 - A
    np.fill_diagonal(comp, 0)
    m = cp_model.CpModel()
    K = n
    col = [m.NewIntVar(0, K - 1, str(i)) for i in range(n)]
    mx = m.NewIntVar(0, K - 1, "mx")
    for i in range(n):
        for j in range(i + 1, n):
            if comp[i, j]:
                m.Add(col[i] != col[j])
        m.Add(col[i] <= mx)
    m.Minimize(mx)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 4
    st = s.Solve(m)
    colour = np.array([s.Value(col[i]) for i in range(n)])
    for c in np.unique(colour):                                            # re-check each class is a clique
        vs = np.nonzero(colour == c)[0]
        assert A[np.ix_(vs, vs)].sum() == len(vs) * (len(vs) - 1)
    return int(colour.max()) + 1, colour, st == cp_model.OPTIMAL


def spectral_field(A, p):
    """Best single-construction bound over F_p: min over lambda != 0 of rank_p(A - lambda I) = rank_p(I + cA).
    Returns (cost, c, lam) with the explicit c; the fitting matrix is (I + cA) mod p."""
    n = A.shape[0]
    flat = A.reshape(-1).tolist()
    best = (n, 0, 0)
    for lam in range(1, p):
        ent = [x % p for x in flat]
        for i in range(n):
            ent[i * n + i] = (-lam) % p
        r = _rank_mod(ent, n, p)
        if r < best[0]:
            c = (-pow(lam, -1, p)) % p                                     # M = I + cA, c = -1/lambda
            best = (r, c, lam)
    return best


def spectral_matrix(A, p, c):
    n = A.shape[0]
    M = (np.eye(n, dtype=np.int64) + c * A) % p
    return M


def verify_spectral(A, p, c, cost):
    """Re-check: M = I + cA fits G (nonzero diagonal, zero off a non-edge) and has rank_p = cost."""
    n = A.shape[0]
    M = spectral_matrix(A, p, c)
    if any(M[i, i] == 0 for i in range(n)):
        return False
    nz = (M != 0)
    off = nz & (np.eye(n, dtype=bool) == 0)
    if (off & (A == 0)).any():                                            # nonzero only allowed on edges
        return False
    return _rank_mod(M.reshape(-1).tolist(), n, p) == cost


def spectral_multifield(A, fields=(2, 3, 5, 7, 11, 13)):
    out = {}
    for p in fields:
        cost, c, lam = spectral_field(A, p)
        assert verify_spectral(A, p, c, cost)
        out[p] = {"cost": cost, "c": c, "lam": lam}
    best_p = min(out, key=lambda p: out[p]["cost"])
    return out, best_p, out[best_p]["cost"]


def f2_minrank_at(A, r, budget=40000):
    """A rank-<= r fitting matrix over F_2, or None, or 'unknown' (SAT gave up). Verified if found."""
    n = A.shape[0]
    rows = fit_rank(A.astype(bool), np.arange(n), r, budget=budget)
    if rows is None:
        return None
    if rows is False:
        return False
    assert rank_gf2(rows) <= r
    for i in range(n):                                                     # re-check fit
        assert (rows[i] >> i) & 1
        for j in range(n):
            if j != i and (rows[i] >> j) & 1:
                assert A[i, j]
    return rows
