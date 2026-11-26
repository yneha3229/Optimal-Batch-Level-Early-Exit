"""Step-by-step numerical verification of the k = 3 proofs in docs/MINRANK_THEORY.md section 6.

Independent of minrank/analysis/kneser_prank.py: all ranks come from FLINT (python-flint, nmod_mat), not from our own
elimination code. Every intermediate claim of the proofs is checked separately:

  [identity]  I + D = W_0^T W_0 - W_1^T W_1 + W_2^T W_2 over Z (so I + D = M^T Q M, and mod 2 Q = I).
  [fits]      I + D (mod p) has nonzero diagonal and zeros on intersecting pairs (it fits K(n,3)).
  [wilson]    rank_p M = C(n,2) for p = 2 and p >= 5 (dim C^perp = n + 1).
  [lemma1]    the explicit parametrisation of C^perp (F_2: f(a,b,x); F_p: g(c,t,x)) lies in C^perp and spans it.
  [form]      the Gram matrix of that basis under Q equals the closed-form B (resp. B_Q) of the proof.
  [lemma2]    rank B = 2 / 3 / n + 1 for n = 2, 0 mod 4 / odd (p = 2).
  [rankfml]   rank_p(I + D) = rank_p M - (N - rank_p M) + rank B_Q (the reduction to the small form).
  [thm]       rank_p(I + D) = alpha  <=>  p | n - 6 (p >= 5), 4 | n - 6 (p = 2).
  [cor]       Corollary 3 arithmetic for 9 <= n <= 10^6, and alpha < cover bound iff 3 does not divide n.

Usage:  python -m minrank.analysis.verify_kneser_proof [--nmax 22] [--primes 2 3 5 7 11 13 17 19 23]
Needs:  pip install python-flint
"""
from __future__ import annotations

import argparse
import itertools
from math import comb

import numpy as np

try:
    from flint import nmod_mat
except ImportError as e:  # pragma: no cover
    raise SystemExit("pip install python-flint") from e


def frank(a: np.ndarray, p: int) -> int:
    a = np.asarray(a, dtype=np.int64) % p
    return nmod_mat(a.shape[0], a.shape[1], [int(v) for v in a.ravel()], p).rank()


def setup(n: int):
    S = [frozenset(s) for s in itertools.combinations(range(n), 3)]
    levels = [[frozenset(a) for a in itertools.combinations(range(n), t)] for t in range(3)]
    W = [np.array([[int(a <= s) for s in S] for a in lev], dtype=np.int64) for lev in levels]
    X = np.zeros((len(S), n), dtype=np.int64)
    for i, s in enumerate(S):
        X[i, list(s)] = 1
    D = (X @ X.T == 0).astype(np.int64)
    inter = X @ X.T
    return S, levels, W, D, inter


def check_n(n: int, primes: list[int], log: list[str]) -> None:
    S, levels, W, D, inter = setup(n)
    nS = len(S)
    I = np.eye(nS, dtype=np.int64)
    alpha = comb(n - 1, 2)
    M = np.vstack(W)
    N = M.shape[0]
    q = np.concatenate([np.full(len(levels[0]), 1), np.full(len(levels[1]), -1), np.full(len(levels[2]), 1)])

    # [identity] over Z
    G = W[0].T @ W[0] - W[1].T @ W[1] + W[2].T @ W[2]
    assert np.array_equal(G, I + D), f"identity fails n={n}"

    pairs = levels[2]
    idx_single = {next(iter(a)): len(levels[0]) + i for i, a in enumerate(levels[1])}
    idx_pair = {a: len(levels[0]) + len(levels[1]) + i for i, a in enumerate(pairs)}

    for p in primes:
        A = (I + D) % p
        # [fits]
        assert np.all(np.diag(A) != 0) and np.all(A[(inter > 0) & (I == 0)] == 0), "fit"
        rM = frank(M, p)
        rA = frank(A, p)
        if p == 2 or p >= 5:
            assert rM == comb(n, 2), f"[wilson] n={n} p={p}: rank M={rM}"
        # explicit basis of C^perp (gauge fixed by x_{n-1} = 0)
        basis = []
        if p == 2:
            def vec(a, b, x):
                v = np.zeros(N, dtype=np.int64)
                v[0] = a
                for i in range(n):
                    v[idx_single[i]] = b
                for e, j in idx_pair.items():
                    i1, i2 = tuple(e)
                    v[j] = (a + b + x[i1] + x[i2]) % 2
                return v
            params = [(1, 0, np.zeros(n, int)), (0, 1, np.zeros(n, int))] + \
                     [(0, 0, np.eye(n, dtype=int)[i]) for i in range(n - 1)]
            basis = [vec(*pr) for pr in params]
        else:
            def vec(c, t, x):
                v = np.zeros(N, dtype=np.int64)
                v[0] = -3 * (t + c)
                for i in range(n):
                    v[idx_single[i]] = t - 2 * x[i]
                for e, j in idx_pair.items():
                    i1, i2 = tuple(e)
                    v[j] = c + x[i1] + x[i2]
                return v % p
            params = [(1, 0, np.zeros(n, int)), (0, 1, np.zeros(n, int))] + \
                     [(0, 0, np.eye(n, dtype=int)[i]) for i in range(n - 1)]
            basis = [vec(*pr) for pr in params]
        B = np.array(basis)
        in_perp = np.all((B @ M) % p == 0)
        indep = frank(B, p) == n + 1
        spans = (N - rM) == n + 1
        if p == 2 or p >= 5:
            assert in_perp and indep and spans, f"[lemma1] n={n} p={p}: in={in_perp} indep={indep} span={spans}"
        # [form] numeric Gram vs closed form
        Qd = q % p if p != 2 else np.ones(N, dtype=np.int64)
        gram = (B * Qd) @ B.T % p
        closed = np.zeros_like(gram)
        for u, (a1, b1, x1) in enumerate(params):
            for w, (a2, b2, x2) in enumerate(params):
                X1, X2, P = int(x1.sum()), int(x2.sum()), int(x1 @ x2)
                if p == 2:
                    c1, c2 = a1 + b1, a2 + b2
                    val = a1 * a2 + n * b1 * b2 + comb(n, 2) * c1 * c2 + (n - 1) * (c1 * X2 + c2 * X1) \
                        + (n - 2) * P + X1 * X2
                else:
                    c1, t1, c2, t2 = a1, b1, a2, b2
                    val = 9 * (t1 + c1) * (t2 + c2) - (n * t1 * t2 - 2 * (t1 * X2 + t2 * X1) + 4 * P) \
                        + comb(n, 2) * c1 * c2 + (n - 1) * (c1 * X2 + c2 * X1) + (n - 2) * P + X1 * X2
                closed[u, w] = val % p
        if p == 2 or p >= 5:
            assert np.array_equal(gram, closed), f"[form] n={n} p={p}"
        rB = frank(gram, p)
        if p == 2:
            exp = {2: 2, 0: 3, 1: n + 1, 3: n + 1}[n % 4]
            assert rB == exp, f"[lemma2] n={n}: rank B={rB}, expected {exp}"
        if p == 2 or p >= 5:
            assert rA == rM - (N - rM) + rB, f"[rankfml] n={n} p={p}: {rA} vs {rM}-{N - rM}+{rB}"
        pred = (n - 6) % 4 == 0 if p == 2 else (n - 6) % p == 0
        if p == 2 or p >= 5:
            assert (rA == alpha) == pred, f"[thm] n={n} p={p}: rank={rA} alpha={alpha} pred={pred}"
        log.append(f"n={n:2d} p={p:2d}  rank(I+D)={rA:4d}  alpha={alpha:4d}  rank M={rM:4d}  rank B={rB:3d}  "
                   f"hit={rA == alpha!s:5}  {'(p=3: not covered by the proof)' if p == 3 else 'all checks ok'}")


def check_corollary(nmax: int) -> None:
    for n in range(9, nmax + 1):
        m = n - 6
        mm = m
        while mm % 2 == 0:
            mm //= 2
        while mm % 3 == 0:
            mm //= 3
        assert mm > 1 or m % 4 == 0 or n % 3 == 0, f"[cor] n={n}"
        if n % 3 == 0:
            assert comb(n, 3) // (n // 3) == comb(n - 1, 2) and comb(n, 3) % (n // 3) == 0
        cover_lb = -(-comb(n, 3) // (n // 3))
        assert (cover_lb > comb(n - 1, 2)) == (n % 3 != 0), f"[cor] cover n={n}"


def check_ducey_route(nmax: int, primes: list[int]) -> int:
    """Third route: Ducey et al. (arXiv:2310.09227, Thm 4.1) reduce the Smith group of B = D + I in the Johnson
    algebra to small blocks M_s(i, j) = C(j-s, i-s) (-1)^i C(n-3-j, 3-j) + delta_ij (s <= i, j <= 3) with
    multiplicity C(n,s) - 2 C(n,s-1) + C(n,s-2), valid for n >= 3k - 1 = 8. Returns the number of (n, p) that
    disagree with Theorems 1-2."""
    bad = 0
    for n in range(8, nmax + 1):
        for p in primes:
            tot = 0
            for s0 in range(4):
                idx = range(s0, 4)
                blk = [[(comb(j - s0, i - s0) if j >= i else 0) * ((-1) ** i * comb(n - 3 - j, 3 - j))
                        + (1 if i == j else 0) for j in idx] for i in idx]
                mult = comb(n, s0) - 2 * (comb(n, s0 - 1) if s0 else 0) + (comb(n, s0 - 2) if s0 >= 2 else 0)
                tot += mult * frank(np.array(blk), p)
            pred = (n - 6) % 4 == 0 if p == 2 else (n - 6) % p == 0
            if p != 3 and (tot == comb(n - 1, 2)) != pred:
                bad += 1
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nmin", type=int, default=7)
    ap.add_argument("--nmax", type=int, default=22)
    ap.add_argument("--primes", type=int, nargs="+", default=[2, 3, 5, 7, 11, 13, 17, 19, 23])
    ap.add_argument("--cor-nmax", type=int, default=10 ** 6)
    a = ap.parse_args()
    log: list[str] = []
    for n in range(a.nmin, a.nmax + 1):
        check_n(n, a.primes, log)
        print("\n".join(log[-len(a.primes):]), flush=True)
    bad = check_ducey_route(a.nmax, a.primes)
    assert bad == 0, f"Ducey et al. route disagrees on {bad} cases"
    print(f"Ducey et al. Thm 4.1 route agrees for n = 8..{a.nmax}")
    check_corollary(a.cor_nmax)
    print(f"corollary arithmetic ok for 9 <= n <= {a.cor_nmax}")
    print(f"ALL CHECKS PASSED: n = {a.nmin}..{a.nmax}, primes {a.primes}")


if __name__ == "__main__":
    main()
