"""Minrank over a prime field F_q (q = 2, 3, 5, ...): verifier, brute force, and SAT encoding.

A matrix over F_q is a list of rows, each a list of ints in [0, q). A fits G iff A[i][i] != 0 and A[i][j] = 0 for
every j not in S_i u {i}. Scaling a row by a nonzero constant keeps fit and rank, so the brute force fixes A[i][i]=1.

SAT encoding of "minrk_q(G) <= r" (q > 2; q = 2 delegates to the XOR encoding in sat.py / symmetry.py):
- one-hot variables x[e][a] for every entry e of U, V (n x r each) and value a in F_q, exactly one true;
- for each constrained entry (i, j) and coordinate k: one-hot product p_k with (u_ik = a and v_jk = b) -> p_k = ab;
- one-hot running sums s_1 = p_1, s_k = s_{k-1} + p_k (mod q): (s_{k-1} = a and p_k = b) -> s_k = a + b;
- target: s_r != 0 on the diagonal, s_r = 0 on forbidden entries;
- symmetry breaking: u_0 = e_1 (sound: u_0 != 0 since A[0][0] != 0, and GL(r, F_q) maps any nonzero vector to e_1;
  row scaling of A is absorbed by GL as well).
Implications from all (a, b) pairs plus at-most-one on p and s make every p, s determined by u, v.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

from ..core.graph import Graph, iter_bits
from .symmetry import VarPool, solve_timed

Matrix = list[list[int]]


def _check_q(q: int) -> None:
    if q < 2 or any(q % d == 0 for d in range(2, int(q ** 0.5) + 1)):
        raise ValueError(f"q must be prime, got {q}")


def rank_fq(rows: Sequence[Sequence[int]], q: int) -> int:
    """Rank over F_q by Gaussian elimination with exact Python ints."""
    _check_q(q)
    m = [list(int(x) % q for x in r) for r in rows]
    if not m:
        return 0
    ncol = len(m[0])
    rank = 0
    for c in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][c]), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][c], q - 2, q)  # Fermat inverse
        m[rank] = [(x * inv) % q for x in m[rank]]
        for i in range(len(m)):
            if i != rank and m[i][c]:
                f = m[i][c]
                m[i] = [(a - f * b) % q for a, b in zip(m[i], m[rank])]
        rank += 1
        if rank == len(m):
            break
    return rank


def fit_error_fq(g: Graph, A: Sequence[Sequence[int]], q: int) -> str | None:
    if len(A) != g.n or any(len(r) != g.n for r in A):
        return f"matrix must be {g.n} x {g.n}"
    for i in range(g.n):
        if A[i][i] % q == 0:
            return f"A[{i}][{i}] = 0"
        for j in iter_bits(g.forbidden(i)):
            if A[i][j] % q:
                return f"A[{i}][{j}] != 0 but {j} is not in S_{i}"
    return None


def verify_fq(g: Graph, A: Sequence[Sequence[int]], q: int) -> int:
    err = fit_error_fq(g, A, q)
    if err:
        raise ValueError(f"{g.name}: matrix does not fit over F_{q}: {err}")
    return rank_fq(A, q)


def minrank_bruteforce_fq(g: Graph, q: int, max_free: int = 10) -> int:
    """Enumerate fitting matrices with unit diagonal (q^#edges). Tests only."""
    _check_q(q)
    free = [(i, j) for i in range(g.n) for j in iter_bits(g.out[i])]
    if len(free) > max_free:
        raise ValueError(f"{len(free)} free entries is too many for brute force")
    best = g.n
    for vals in product(range(q), repeat=len(free)):
        A = [[1 if i == j else 0 for j in range(g.n)] for i in range(g.n)]
        for (i, j), x in zip(free, vals):
            A[i][j] = x
        best = min(best, rank_fq(A, q))
    return best


def _exactly_one(lits: list[int]) -> list[list[int]]:
    cls = [list(lits)]
    cls += [[-a, -b] for x, a in enumerate(lits) for b in lits[x + 1:]]
    return cls


def encode_fq(g: Graph, r: int, q: int) -> list[list[int]]:
    """CNF for minrk_q(G) <= r (one-hot encoding; variables of entry (M, i, k), value a: see `_entry_var`)."""
    _check_q(q)
    n = g.n
    if not 1 <= r <= n:
        raise ValueError(f"need 1 <= r <= n, got r={r}, n={n}")
    pool = VarPool(2 * n * r * q)
    cnf: list[list[int]] = []
    for M in (0, 1):
        for i in range(n):
            for k in range(r):
                cnf += _exactly_one([_entry_var(n, r, q, M, i, k, a) for a in range(q)])
    for i in range(n):
        for j in [i] + list(iter_bits(g.forbidden(i))):
            s_prev: list[int] | None = None
            for k in range(r):
                p = [pool.new() for _ in range(q)]
                cnf += [[-a, -b] for x, a in enumerate(p) for b in p[x + 1:]]  # at most one
                for a in range(q):
                    for b in range(q):
                        cnf.append([-_entry_var(n, r, q, 0, i, k, a), -_entry_var(n, r, q, 1, j, k, b), p[(a * b) % q]])
                if s_prev is None:
                    s_prev = p
                    continue
                s = [pool.new() for _ in range(q)]
                cnf += [[-a, -b] for x, a in enumerate(s) for b in s[x + 1:]]
                for a in range(q):
                    for b in range(q):
                        cnf.append([-s_prev[a], -p[b], s[(a + b) % q]])
                s_prev = s
            cnf.append([-s_prev[0]] if i == j else [s_prev[0]])
    cnf.append([_entry_var(n, r, q, 0, 0, 0, 1)])  # u_0 = e_1
    cnf += [[_entry_var(n, r, q, 0, 0, k, 0)] for k in range(1, r)]
    return cnf


def _entry_var(n: int, r: int, q: int, M: int, i: int, k: int, a: int) -> int:
    """Variable of "entry (i, k) of U (M=0) or V (M=1) equals a"."""
    return 1 + ((M * n + i) * r + k) * q + a


def rank_at_most_fq(g: Graph, r: int, q: int, time_limit: float | None = None, solver: str = "cadical153"
                    ) -> Matrix | bool | None:
    """Verified fitting matrix over F_q of rank <= r if SAT; False if UNSAT; None on timeout."""
    _check_q(q)
    if q == 2:
        from ..core.graph import masks_to_bool
        from .symmetry import rank_at_most_sym

        res = rank_at_most_sym(g, r, time_limit, mode="cols", solver=solver)
        if res is None or res is False:
            return res
        return masks_to_bool(res, g.n).astype(int).tolist()
    res = solve_timed(encode_fq, (g, r, q), time_limit, solver)
    if res is None or res is False:
        return res
    model = set(x for x in res if x > 0)
    n = g.n

    def val(M: int, i: int, k: int) -> int:
        vals = [a for a in range(q) if _entry_var(n, r, q, M, i, k, a) in model]
        if len(vals) != 1:
            raise RuntimeError("one-hot violated in model")
        return vals[0]

    U = [[val(0, i, k) for k in range(r)] for i in range(n)]
    V = [[val(1, j, k) for k in range(r)] for j in range(n)]
    A = [[sum(U[i][k] * V[j][k] for k in range(r)) % q for j in range(n)] for i in range(n)]
    if verify_fq(g, A, q) > r:
        raise RuntimeError("SAT model exceeds the rank bound")
    return A


def exact_minrank_fq(g: Graph, q: int, upper: int, lower: int = 1, time_limit_per_call: float | None = None
                     ) -> tuple[int, bool, Matrix | None]:
    """Descend from a known upper bound `upper` (e.g. a clique cover, valid over every field).
    Returns (best rank, proven optimal, matrix for best or None if no SAT call succeeded)."""
    best, best_A = upper, None
    while best > lower:
        res = rank_at_most_fq(g, best - 1, q, time_limit_per_call)
        if res is None:
            return best, False, best_A
        if res is False:
            return best, True, best_A
        best_A, best = res, rank_fq(res, q)
    return best, True, best_A
