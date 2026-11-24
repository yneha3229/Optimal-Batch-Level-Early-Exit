"""Checks for Theorem 6 (docs/MINRANK_THEORY.md section 12): every prime p, including p = 2 and p <= k.

Theorem 6: let p be prime, k >= 2, n > 2k, and q = the smallest power of p with q >= k. If q | n - 2k then
rank_p(I + (-1)^(k-1) D_{K(n,k)}) = C(n-1, k-1) = alpha, so minrk_p(K(n,k)) = alpha.

  [lemma2]  char 2: rank_2(P_m + I) = floor(m/2) for m <= 64 (and, for m >= 2, with the (0,0) entry flipped: same rank).
  [lemmap]  odd p:  rank_p(P_m - diag((-1)^(m-1-v))) = floor(m/2), and the same with entry (0,0) changed arbitrarily
            when m - 1 is odd (the boundary case k = q = p^c).
  [period]  C(w, r) = C(-1, r) (mod p) whenever w = -1 (mod q) and r < q.
  [iff]     block formula (Ducey et al.): rank_p = alpha  <=>  q | n - 2k, for k <= 12, 3k - 1 <= n <= 120, p <= 13.
  [direct]  full-matrix ranks (FLINT) for new cases the theorem predicts: k = 4, p = 3 (q = 9), k = 4/5, p = 2, ...

Usage:  python -m minrank.analysis.verify_all_primes
"""
from __future__ import annotations

from math import comb

import numpy as np

from .verify_general_k import block_rank, frank, kneser, primes_upto


def qpow(p: int, k: int) -> int:
    q = 1
    while q < k:
        q *= p
    return q


def binom_poly_mod(w: int, r: int, p: int) -> int:
    """C(w, r) mod p for any integer w (generalised binomial)."""
    num = 1
    for i in range(r):
        num *= (w - i)
    den = 1
    for i in range(1, r + 1):
        den *= i
    return (num // den) % p  # exact division over Z


def main() -> None:
    # [lemma2]
    for m in range(1, 65):
        P = np.array([[comb(v, u) for v in range(m)] for u in range(m)], dtype=object)
        N = P + np.eye(m, dtype=object)
        assert frank(N, 2) == m // 2, m
        if m >= 2:  # the flipped case only occurs in the s = 0 block, of size k + 1 >= 3
            N2 = N.copy(); N2[0, 0] += 1
            assert frank(N2, 2) == m // 2, ("flip", m)
    print("[lemma2] rank_2(P_m + I) = floor(m/2), also with (0,0) flipped, m <= 64")
    # [lemmap]
    for p in [q for q in primes_upto(31) if q > 2]:
        for m in range(1, 41):
            P = np.array([[comb(v, u) for v in range(m)] for u in range(m)], dtype=object)
            N = P - np.diag([(-1) ** (m - 1 - v) for v in range(m)])
            assert frank(N, p) == m // 2
            if (m - 1) % 2 == 1:
                for c in range(p):
                    N2 = N.copy(); N2[0, 0] = c
                    assert frank(N2, p) == m // 2, (p, m, c)
    print("[lemmap] odd p <= 31, m <= 40: rank floor(m/2); boundary entry arbitrary when m - 1 is odd")
    # [period]
    for p in primes_upto(13):
        for q in [p ** e for e in range(1, 5) if p ** e <= 81]:
            for w in range(-1 - 5 * q, 5 * q, q):
                for r in range(q):
                    assert binom_poly_mod(w, r, p) == ((-1) ** r) % p, (p, q, w, r)
    print("[period] C(w, r) = (-1)^r mod p for w = -1 mod q, r < q")
    # [iff]
    tot = hits = 0
    for k in range(2, 13):
        for n in range(max(2 * k + 1, 3 * k - 1), 121):
            a = comb(n - 1, k - 1)
            for p in primes_upto(13):
                hit = block_rank(n, k, p) == a
                pred = (n - 2 * k) % qpow(p, k) == 0
                assert hit == pred, ("iff", k, n, p, hit, pred)
                tot += 1; hits += hit
    print(f"[iff] block formula: rank_p = alpha iff q | n - 2k in all {tot} cases ({hits} hits), k <= 12, n <= 120")
    # [direct]
    cases = [(4, 17, 3), (4, 12, 2), (4, 16, 2), (4, 20, 2), (2, 6, 2), (2, 8, 2), (3, 10, 2), (3, 15, 3),
             (5, 18, 2), (4, 14, 3), (4, 11, 3), (4, 18, 2), (5, 15, 5)]
    for k, n, p in cases:
        A = np.eye(comb(n, k), dtype=np.int64) + (-1) ** (k - 1) * kneser(n, k)
        r = frank(A, p)
        pred = (n - 2 * k) % qpow(p, k) == 0
        a = comb(n - 1, k - 1)
        assert (r == a) == pred, ("direct", k, n, p, r, a)
        print(f"[direct] k={k} n={n} p={p} q={qpow(p, k)}: rank={r} alpha={a} predicted={'alpha' if pred else '> alpha'} ok",
              flush=True)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
