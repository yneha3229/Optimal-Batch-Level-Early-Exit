"""Numerical checks for the general-k theorem (docs/MINRANK_THEORY.md section 11).

Theorem 4: for k >= 2, n > 2k and a prime p > k with p | n - 2k,
    rank_p(I + (-1)^(k-1) D_{K(n,k)}) = C(n-1, k-1) = alpha,   hence   minrk_p(K(n,k)) = alpha.

Checks (ranks from FLINT, python-flint):
  [lemma]   rank_F(P_m - diag((-1)^(m-1-v))) = floor(m/2) over F_p, p odd, m = 1..40 (P_m = upper Pascal C(v,u)).
  [reduce]  the Ducey et al. block formula sum_s (mu_s - mu_{s-1}) rank_p(T_s + diag(eps theta_j)) equals the
            directly computed rank_p(I + eps D) (all primes <= 13, small n, k = 2..5).
  [direct]  Theorem 4 on full matrices: k = 2..5, every (n, p) with p > k, p | n - 2k, C(n,k) <= 6200.
  [blocks]  for k <= 12 and 2k < n <= 80, every p > k with p | n - 2k: the block formula gives exactly alpha.

Usage:  python -m minrank.analysis.verify_general_k
"""
from __future__ import annotations

import itertools
from math import comb

import numpy as np
from flint import nmod_mat


def frank(a, p: int) -> int:
    a = np.asarray(a, dtype=object)
    r, c = a.shape
    return nmod_mat(r, c, [int(v) % p for v in a.ravel()], p).rank()


def primes_upto(m: int) -> list[int]:
    return [q for q in range(2, m + 1) if all(q % d for d in range(2, int(q ** 0.5) + 1))]


def kneser(n: int, k: int) -> np.ndarray:
    x = np.zeros((comb(n, k), n), dtype=np.int64)
    for i, s in enumerate(itertools.combinations(range(n), k)):
        x[i, list(s)] = 1
    return (x @ x.T == 0).astype(np.int64)


def block(n: int, k: int, s: int) -> list[list[int]]:
    """T_s + diag(eps * theta_j), j = s..k: unimodularly equivalent (T M_s) to Ducey et al.'s M_s for I + eps D."""
    eps = (-1) ** (k - 1)
    idx = range(s, k + 1)
    return [[comb(j - s, i - s) if j >= i else 0 for j in idx] for i in idx] if False else \
        [[(comb(j - s, i - s) if j >= i else 0) + (eps * (-1) ** j * comb(n - k - j, k - j) if i == j else 0)
          for j in idx] for i in idx]


def ducey_block(n: int, k: int, s: int) -> list[list[int]]:
    """Ducey et al. Def. 3.1 / Lemma 4.10 for L = D - lam I with lam = -eps, times eps (so it represents I + eps D)."""
    eps = (-1) ** (k - 1)
    idx = range(s, k + 1)
    return [[(1 if i == j else 0) + eps * (comb(j - s, i - s) if j >= i else 0) * (-1) ** i * comb(n - k - j, k - j)
             for j in idx] for i in idx]


def block_rank(n: int, k: int, p: int, which=block) -> int:
    tot = 0
    for s in range(k + 1):
        mult = comb(n, s) - 2 * (comb(n, s - 1) if s >= 1 else 0) + (comb(n, s - 2) if s >= 2 else 0)
        tot += mult * frank(np.array(which(n, k, s), dtype=object), p)
    return tot


def main() -> None:
    # [lemma]
    for p in [q for q in primes_upto(31) if q > 2]:
        for m in range(1, 41):
            P = np.array([[comb(v, u) for v in range(m)] for u in range(m)], dtype=object)
            N = P - np.diag([(-1) ** (m - 1 - v) for v in range(m)])
            assert frank(N, p) == m // 2, (p, m)
    print("[lemma] rank(P_m - diag((-1)^(m-1-v))) = floor(m/2) for m <= 40, odd p <= 31")

    # [reduce] block formula (both forms) vs direct rank, including n below 3k - 1 where it may fail
    small = {2: range(5, 16), 3: range(7, 17), 4: range(9, 16), 5: range(11, 16)}
    fails_below = []
    for k, ns in small.items():
        eps = (-1) ** (k - 1)
        for n in ns:
            A = np.eye(comb(n, k), dtype=np.int64) + eps * kneser(n, k)
            for p in primes_upto(13):
                d = frank(A, p)
                b1, b2 = block_rank(n, k, p, block), block_rank(n, k, p, ducey_block)
                assert b1 == b2, ("T M_s equivalence", n, k, p)
                if n >= 3 * k - 1:
                    assert d == b1, ("reduction", n, k, p, d, b1)
                elif d != b1:
                    fails_below.append((n, k, p))
    print(f"[reduce] block formula = direct rank for all n >= 3k - 1 (k = 2..5, p <= 13);"
          f" n < 3k - 1 disagreements (outside Ducey's range): {fails_below}")

    # [direct] Theorem 4 on full matrices
    cases = []
    for k in range(2, 6):
        for n in range(2 * k + 1, 40):
            if comb(n, k) > 6200:
                break
            for p in primes_upto(n):
                if p > k and (n - 2 * k) % p == 0:
                    cases.append((k, n, p))
    for k, n, p in cases:
        A = np.eye(comb(n, k), dtype=np.int64) + (-1) ** (k - 1) * kneser(n, k)
        r = frank(A, p)
        assert r == comb(n - 1, k - 1), (k, n, p, r)
    print(f"[direct] Theorem 4 holds on {len(cases)} full-matrix cases: {cases}")

    # [blocks] large k, n via the block formula
    cnt = 0
    for k in range(2, 13):
        for n in range(2 * k + 1, 81):
            for p in primes_upto(n):
                if p > k and (n - 2 * k) % p == 0:
                    assert block_rank(n, k, p) == comb(n - 1, k - 1), (k, n, p)
                    cnt += 1
    print(f"[blocks] block formula gives alpha in all {cnt} cases k <= 12, n <= 80")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
