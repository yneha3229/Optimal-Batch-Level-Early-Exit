"""Usage: python minrank/analysis/verify_independent.py  (see proof.md section 8)"""
"""Independent check of: q | n-2k  =>  rank_GF(p)(I + (-1)^(k-1) D) = C(n-1,k-1) = alpha(K(n,k)).
Own Gaussian elimination mod p (numpy int64), cross-checked with FLINT; also checks the matrix fits K(n,k)
and that a star (all k-sets containing element 0) is an independent set of size C(n-1,k-1)."""
import itertools, time
from math import comb
import numpy as np
from flint import nmod_mat

def qpow(p, k):
    q = 1
    while q < k: q *= p
    return q

def primes(m): return [x for x in range(2, m+1) if all(x % d for d in range(2, int(x**.5)+1))]

def rank_mod_p(A, p):
    A = A.copy() % p; r = 0; rows, cols = A.shape
    for c in range(cols):
        piv = np.nonzero(A[r:, c])[0]
        if len(piv) == 0: continue
        i = r + piv[0]
        if i != r: A[[r, i]] = A[[i, r]]
        inv = pow(int(A[r, c]), -1, p)
        A[r] = (A[r] * inv) % p
        nz = np.nonzero(A[:, c])[0]; nz = nz[nz != r]
        if len(nz): A[nz] = (A[nz] - np.outer(A[nz, c], A[r])) % p
        r += 1
        if r == rows: break
    return r

def build(n, k):
    S = list(itertools.combinations(range(n), k))
    X = np.zeros((len(S), n), dtype=np.int64)
    for i, s in enumerate(S): X[i, list(s)] = 1
    inter = X @ X.T
    D = (inter == 0).astype(np.int64)
    return S, inter, D

rows = []; t0 = time.time()
for k in range(2, 7):
    for n in range(2*k+1, 40):
        N = comb(n, k)
        if N > 3500: break
        S, inter, D = build(n, k)
        eps = (-1) ** (k-1)
        H = np.eye(N, dtype=np.int64) + eps * D
        # fits K(n,k): diagonal 1, zero on distinct intersecting pairs
        off = ~np.eye(N, dtype=bool)
        assert np.all(H[off & (inter > 0)] == 0) and np.all(np.diag(H) == 1)
        # star = independent set of size alpha
        star = [i for i, s in enumerate(S) if 0 in s]
        assert len(star) == comb(n-1, k-1) and np.all(D[np.ix_(star, star)] == 0)
        for p in primes(n):
            q = qpow(p, k)
            if (n - 2*k) % q: continue
            r1 = rank_mod_p(H, p)
            r2 = nmod_mat(N, N, [int(v) % p for v in H.ravel()], p).rank()
            a = comb(n-1, k-1)
            rows.append((k, n, p, q, N, r1, r2, a, n % k == 0))
            assert r1 == r2 == a, (k, n, p, r1, r2, a)
print(f"{len(rows)} cases, all rank = alpha, both methods agree ({time.time()-t0:.0f}s)")
new = [r for r in rows if not r[8]]
print(f"{len(rows)-len(new)} cases have k | n (already = alpha by Baranyai's clique cover); {len(new)} have k not dividing n:")
for k, n, p, q, N, r1, r2, a, b in new:
    print(f"  K({n},{k}) over GF({p}) [q={q}]: {N}x{N}, rank {r1} = alpha {a}")
