"""Independent certificate checks. This package must not import kagent.rank, kagent.tools or python-flint, so a bug
in the Solver's linear algebra cannot hide itself here. Plain numpy (and pysat's Glucose for UNSAT records).

check_independent_set      no edge inside I (O(|I|^2) bit tests)
check_clique_cover         every block a clique, union = V
check_fractional_cover     every block a clique, every vertex covered with weight >= 1 (exact fractions)
check_spectral_code        (1) M = I + cA fits G; (2) M = C . M[S, :] by Freivalds with 30 random vectors
                           (error <= p^-30); (3) n <= EXACT_MAX: exact rank by our own Gaussian elimination mod p
check_dense_code           fit on G (or G^(t)) + exact rank by our own elimination
check_unsat                re-solve the stored CNF with Glucose 4 (a different solver from the one that produced it)
simulate_decoding          3 random message vectors over F_p: every receiver decodes x_v from y and its side info
"""
from __future__ import annotations

import gzip
from fractions import Fraction

import numpy as np

EXACT_MAX = 3000


def check_independent_set(A: np.ndarray, I) -> tuple[bool, str, int]:
    I = sorted(set(int(v) for v in I))
    n = A.shape[0]
    if any(v < 0 or v >= n for v in I):
        return False, "vertex out of range", 0
    for a in range(len(I)):
        u = I[a]
        for w in I[a + 1:]:
            if A[u, w]:
                return False, f"edge {u}-{w} inside the set", 0
    return True, f"{len(I)} vertices, no edge inside ({len(I) * (len(I) - 1) // 2} pairs tested)", len(I)


def _is_clique(A, C) -> bool:
    C = list(C)
    for a in range(len(C)):
        row = A[C[a]]
        for b in C[a + 1:]:
            if not row[b]:
                return False
    return True


def check_clique_cover(A: np.ndarray, cliques) -> tuple[bool, str, int]:
    n = A.shape[0]
    covered = np.zeros(n, dtype=bool)
    for C in cliques:
        if not _is_clique(A, C):
            return False, "a block is not a clique", 0
        covered[list(C)] = True
    if not covered.all():
        return False, f"{int((~covered).sum())} vertices uncovered", 0
    return True, f"{len(cliques)} cliques cover all {n} vertices", len(cliques)


def check_fractional_cover(A: np.ndarray, cliques, weights) -> tuple[bool, str, Fraction]:
    n = A.shape[0]
    cov = [Fraction(0)] * n
    ws = [Fraction(w) for w in weights]
    for C, w in zip(cliques, ws):
        if w < 0:
            return False, "negative weight", Fraction(0)
        if not _is_clique(A, C):
            return False, "a block is not a clique", Fraction(0)
        for v in C:
            cov[v] += w
    lo = min(cov)
    if lo < 1:
        return False, f"a vertex is covered with weight {lo} < 1", Fraction(0)
    val = sum(ws, Fraction(0))
    return True, f"{len(cliques)} weighted cliques, min coverage {lo}, value {val}", val


def matvec_mod(A: np.ndarray, x: np.ndarray, p: int) -> np.ndarray:
    """(A x) mod p for boolean A, exact (int64, blocked)."""
    n = A.shape[0]
    out = np.empty(n, dtype=np.int64)
    xi = x.astype(np.int64)
    for a in range(0, n, 512):
        out[a:a + 512] = (A[a:a + 512].astype(np.int64) @ xi) % p
    return out


def own_rank_mod_p(M: np.ndarray, p: int) -> int:
    """Gaussian elimination over F_p, numpy int64, written separately from the Solver's rank engine."""
    M = np.array(M, dtype=np.int64) % p
    rows, cols = M.shape
    r = 0
    for j in range(cols):
        if r == rows:
            break
        piv = np.nonzero(M[r:, j])[0]
        if len(piv) == 0:
            continue
        i = r + int(piv[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        inv = pow(int(M[r, j]), p - 2, p)
        M[r] = (M[r] * inv) % p
        below = np.nonzero(M[r + 1:, j])[0] + r + 1
        if len(below):
            M[below] = (M[below] - np.outer(M[below, j], M[r])) % p
        r += 1
    return r


def spectral_matrix(A: np.ndarray, p: int, c: int) -> np.ndarray:
    M = (c % p) * A.astype(np.int64)
    np.fill_diagonal(M, 1)
    return M % p


def check_spectral_code(A: np.ndarray, p: int, c: int, S, C: np.ndarray, rounds: int = 30, seed: int = 12345,
                        exact_max: int = EXACT_MAX) -> dict:
    n = A.shape[0]
    S = np.asarray(S, dtype=np.int64)
    C = np.asarray(C, dtype=np.int64) % p
    out = {"p": p, "c": c, "claimed_rank": int(len(S))}
    # (1) fit: diagonal of I + cA is 1 (nonzero mod p), off-diagonal support is c*A, inside the edges by construction;
    # we still check it explicitly on the entries we generate.
    diag_ok = (1 % p) != 0
    out["fit"] = bool(diag_ok and (c % p) != 0)
    if C.shape != (n, len(S)):
        out.update(ok=False, reason=f"C has shape {C.shape}, expected {(n, len(S))}")
        return out
    # (2) Freivalds: M x == C (M[S,:] x) for random x
    rng = np.random.default_rng(seed)
    passed = 0
    for _ in range(rounds):
        x = rng.integers(0, p, n)
        Mx = (x + (c % p) * matvec_mod(A, x, p)) % p
        lhs = Mx
        rhs = (C @ Mx[S]) % p
        if not np.array_equal(lhs, rhs):
            break
        passed += 1
    out["freivalds_passed"] = passed
    out["freivalds_rounds"] = rounds
    out["freivalds_error_bound"] = f"p^-{rounds}"
    ok = out["fit"] and passed == rounds
    # (3) exact rank by our own elimination for small n
    if n <= exact_max:
        r = own_rank_mod_p(spectral_matrix(A, p, c), p)
        out["exact_rank_own"] = r
        ok = ok and r <= len(S)
    out["ok"] = bool(ok)
    return out


def check_dense_code(A: np.ndarray, M: np.ndarray, p: int, claimed: int) -> dict:
    """Fit of a dense fitting matrix on (possibly blown-up) adjacency A, and its exact rank."""
    n = A.shape[0]
    M = np.asarray(M, dtype=np.int64) % p
    diag = bool((np.diag(M) != 0).all())
    off = (M != 0) & ~np.eye(n, dtype=bool)
    support = bool(not (off & ~A).any())
    r = own_rank_mod_p(M, p)
    return {"fit_diag": diag, "fit_support": support, "exact_rank_own": r, "claimed_rank": claimed,
            "ok": bool(diag and support and r <= claimed)}


def check_unsat(cnf_path: str) -> dict:
    from pysat.solvers import Glucose4

    clauses = []
    with gzip.open(cnf_path, "rt") as f:
        for line in f:
            if line.startswith(("p", "c")):
                continue
            lits = [int(x) for x in line.split()]
            if lits and lits[-1] == 0:
                lits = lits[:-1]
            clauses.append(lits)
    with Glucose4(bootstrap_with=clauses) as s:
        sat = s.solve()
    return {"solver": "glucose4", "clauses": len(clauses), "ok": not sat}


def simulate_decoding(A: np.ndarray, p: int, c: int, S, C: np.ndarray, trials: int = 3, seed: int = 99) -> dict:
    """Transmit y = M[S,:] x; receiver v computes C[v] . y = (M x)_v = x_v + c * sum_{u ~ v} x_u, removes the side
    information it holds (its neighbours' messages) and recovers x_v."""
    n = A.shape[0]
    S = np.asarray(S, dtype=np.int64)
    C = np.asarray(C, dtype=np.int64) % p
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(trials):
        x = rng.integers(0, p, n)
        y = (x[S] + (c % p) * matvec_mod(A[S], x, p)) % p                  # the |S| transmissions
        side = matvec_mod(A, x, p)                                         # sum of the neighbours' messages
        decoded = ((C @ y) % p - (c % p) * side) % p                       # diagonal of M is 1
        fails += int((decoded != x).sum())
    return {"trials": trials, "receivers": n, "transmissions": int(len(S)), "failed_decodings": fails,
            "ok": fails == 0}


def simulate_clique_decoding(A: np.ndarray, cliques, trials: int = 3, seed: int = 99) -> dict:
    """Clique-cover code over F_2: one XOR per clique; receiver v XORs its clique-mates' messages back out."""
    n = A.shape[0]
    rng = np.random.default_rng(seed)
    fails = 0
    for _ in range(trials):
        x = rng.integers(0, 2, n)
        for C in cliques:
            y = int(np.bitwise_xor.reduce(x[list(C)])) if len(C) else 0
            for v in C:
                if any(not A[v, u] for u in C if u != v):
                    fails += 1
                    continue
                d = y
                for u in C:
                    if u != v:
                        d ^= int(x[u])
                fails += int(d != x[v])
    return {"trials": trials, "transmissions": len(cliques), "failed_decodings": fails, "ok": fails == 0}
