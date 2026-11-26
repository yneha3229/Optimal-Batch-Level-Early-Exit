"""A theory-guided agent for Kneser-type index coding instances.

Input: only an adjacency matrix (vertices are NOT given as sets).  Steps:

  1. recognise    find (n, k) with  #vertices = C(n,k)  and every degree = C(n-k,k)
  2. choose field the structural rule for this family names the prime p directly (no search over fields);
                  if the rule names none, the instance is one of the finitely many exceptions -> step 5
  3. upper bound  the scalar linear code  M = I + eps*A  over F_p  (eps = (-1)^(k-1));  its length is rank_p(M),
                  certified by ONE rank computation mod p
  4. lower bound  a greedy independent set (on these graphs it lands on a maximum one); if its size equals the
                  code length, beta is PROVEN exact: no index code of any kind can be shorter
  5. fallback     anything that does not certify (wrong guess, exception, non-Kneser input) goes to the general
                  minrank agent, so a wrong hint can never produce a wrong answer

Every reported number comes with its certificate re-checked: the code fits the graph and has the stated rank over
F_p; the independent set has no edge.
"""
from __future__ import annotations

import time
from math import comb

import flint
import numpy as np

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]


def recognise(A):
    """(n, k) such that A could be the Kneser graph K(n,k): vertex count and (regular) degree match."""
    N = A.shape[0]
    deg = A.sum(1)
    if deg.min() != deg.max():
        return None
    d = int(deg[0])
    for k in range(2, 40):
        n = 2 * k + 1
        while comb(n, k) <= N:
            if comb(n, k) == N and comb(n - k, k) == d:
                return n, k
            n += 1
    return None


def _q(p, k):
    q = 1
    while q < k:
        q *= p
    return q


def rule_prime(n, k):
    """The field named by the structural rule for K(n,k), or None (an exception instance)."""
    for p in PRIMES:
        if (n - 2 * k) % _q(p, k) == 0:
            return p
    return None


def code_length(A, p, eps):
    """rank over F_p of the fitting matrix I + eps*A (the code's number of transmissions)."""
    N = A.shape[0]
    M = (np.eye(N, dtype=np.int64) + eps * A) % p
    return flint.nmod_mat(N, N, M.reshape(-1).tolist(), p).rank()


def greedy_independent_set(A):
    N = A.shape[0]
    nonadj = A == 0
    np.fill_diagonal(nonadj, False)
    cand = np.ones(N, dtype=bool)
    score = nonadj.sum(1).astype(np.int64)        # candidates each vertex keeps (kept up to date incrementally)
    S = []
    while cand.any():
        idx = np.nonzero(cand)[0]
        v = int(idx[np.argmax(score[idx])])
        S.append(v)
        removed = cand & ~nonadj[v]                # candidates that leave because they are adjacent to v (and v)
        cand &= nonadj[v]
        removed[v] = True
        if removed.any():
            score -= nonadj[:, removed].sum(1)
    return S


def is_independent(A, S):
    S = np.asarray(S)
    return not A[np.ix_(S, S)].any()


def solve(A, fallback=None, fallback_limit=120.0):
    t0 = time.perf_counter()
    out = {"recognised": None, "field": None, "upper": None, "lower": None, "exact": False, "path": ""}
    rec = recognise(A)
    S = []
    if rec is not None:
        n, k = rec
        out["recognised"] = f"K({n},{k})"
        p = rule_prime(n, k)
        out["field"] = p
        if p is not None:
            eps = 1 if k % 2 == 1 else -1
            r = code_length(A, p, eps)
            S = greedy_independent_set(A)
            assert is_independent(A, S)
            out.update(upper=r, lower=len(S), certificate_fits=True)
            if len(S) >= r:
                out.update(exact=True, path="rule: one field, one rank, greedy star",
                           alpha_formula=comb(n - 1, k - 1), s=time.perf_counter() - t0)
                return out
            out["path"] = "rule did not certify"
        else:
            out["path"] = "exception instance"
            S = greedy_independent_set(A)
            assert is_independent(A, S)
    if fallback is not None:                       # general agent (searches fields, then the classic bounds)
        g = fallback(A, fallback_limit)
        lower = max(g["lower"] or 0, len(S))       # keep the greedy star found above
        out.update(upper=g["upper"], lower=lower, exact=lower >= g["upper"] - 1e-9, field=g.get("field"),
                   path=(out["path"] + " -> general agent").strip(" ->"))
    out["s"] = time.perf_counter() - t0
    return out
