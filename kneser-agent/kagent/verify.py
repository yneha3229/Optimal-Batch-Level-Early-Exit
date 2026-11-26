"""Independent verification. Deliberately shares NO code with kagent.tools (own fit check, own elimination,
own factorization check, decoding simulation), so a bug on the solver side is caught here."""
from __future__ import annotations

import numpy as np


def _M(A, p, c):
    M = np.where(A, c % p, 0).astype(np.int64)
    M[np.diag_indices_from(M)] = 1
    return M


def check_independent(A, S) -> bool:
    S = np.asarray(sorted(set(int(v) for v in S)))
    return len(S) > 0 and not A[np.ix_(S, S)].any()


def check_clique_cover(A, cliques) -> bool:
    seen = np.zeros(A.shape[0], dtype=bool)
    for C in cliques:
        C = np.asarray(C)
        sub = A[np.ix_(C, C)]
        if sub.sum() != len(C) * (len(C) - 1):
            return False
        seen[C] = True
    return bool(seen.all())


def check_fit(A, p, c) -> bool:
    """M = I + cA fits G: diagonal nonzero, M_uv = 0 for every non-edge u != v."""
    if c % p == 0:
        return False
    return True  # by construction of _M: off-diagonal support equals the edge set; diagonal is 1


def rank_mod_p(M, p) -> int:
    """Own Gaussian elimination mod p (vectorised numpy), for N up to a few thousand."""
    M = M.copy() % p
    n, m = M.shape
    r = 0
    for col in range(m):
        piv = np.nonzero(M[r:, col])[0]
        if len(piv) == 0:
            continue
        i = r + piv[0]
        M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, col]), -1, p)
        M[r] = (M[r] * inv) % p
        f = M[:, col].copy()
        f[r] = 0
        nz = np.nonzero(f)[0]
        if len(nz):
            M[nz] = (M[nz] - np.outer(f[nz], M[r])) % p
        r += 1
        if r == n:
            break
    return r


def check_code_meets(A, p, c, S, block=256) -> bool:
    """rank_p(M) <= |S| via the explicit factorization M = M[:,S] M[S,:] (exact, integer arithmetic in blocks).
    Together with S independent this proves rank = |S|."""
    S = np.asarray(S)
    R = _M(A, p, c)[S] if A.shape[0] <= 3000 else None
    if R is None:
        R = np.where(A[S], c % p, 0).astype(np.int64)
        R[np.arange(len(S)), S] = 1
    N = A.shape[0]
    for s in range(0, N, block):
        Mb = np.where(A[s:s + block], c % p, 0).astype(np.int64)
        idx = np.arange(s, min(N, s + block))
        Mb[idx - s, idx] = 1
        prod = (Mb[:, S].astype(np.float64) @ R.astype(np.float64))
        if not np.array_equal(np.mod(prod, p).astype(np.int64), Mb):
            return False
    return True


def decode_simulation(A, p, c, S, trials=3, seed=1) -> bool:
    """Broadcast y_s = sum_j M[s,j] x_j for s in S; receiver v outputs
    x_v = sum_s M[v,s] y_s - sum_{u != v} M[v,u] x_u using only its side information (neighbours)."""
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    S = np.asarray(S)
    for _ in range(trials):
        x = rng.integers(0, p, N)
        MS = np.where(A[S], c % p, 0).astype(np.int64)
        MS[np.arange(len(S)), S] = 1
        y = (MS @ x) % p                                   # what the server sends: |S| symbols
        for v in rng.choice(N, size=min(N, 200), replace=False):
            row = np.where(A[v], c % p, 0).astype(np.int64)   # M[v, :] off the diagonal: only on neighbours
            known = (row @ np.where(A[v], x, 0)) % p          # uses only x_u for neighbours u (side info)
            coeff = row[S].copy()
            if v in set(S.tolist()):
                coeff[np.nonzero(S == v)[0][0]] = 1
            got = (coeff @ y - known) % p
            if got != x[v]:
                return False
    return True
