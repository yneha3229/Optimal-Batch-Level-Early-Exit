"""Exact minrank over F_2 by SAT (B9 and Option B), plus a brute-force reference for tests.

Encoding of "minrk_2(G) <= r":
- Variables u[i][k], v[j][k] for i, j in [n], k in [r].
- For every constrained entry (i, j) (diagonal or forbidden): z_k = u[i][k] AND v[j][k] (3 clauses each), and an XOR
  chain x_1 = z_1, x_k = x_{k-1} XOR z_k (4 clauses each), with the unit clause x_r = [i == j].
- Symmetry breaking (sound): A = U V^T = (U W)(V W^{-T})^T for any invertible W, and u_0 != 0 because A[0, 0] = 1,
  so some W maps u_0 to e_1: fix u[0] = e_1.
Entries on edges are unconstrained and get no clauses.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from itertools import product

import numpy as np
from pysat.solvers import Solver

from ..core.f2 import rank_f2, rows_from_factors
from ..core.graph import Graph, iter_bits
from ..core.verify import verify


class _Pool:
    def __init__(self) -> None:
        self.top = 0

    def new(self) -> int:
        self.top += 1
        return self.top


def encode(g: Graph, r: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """CNF clauses, u vars (n x r), v vars (n x r)."""
    if not 1 <= r <= g.n:
        raise ValueError(f"need 1 <= r <= n, got r={r}, n={g.n}")
    pool = _Pool()
    u = [[pool.new() for _ in range(r)] for _ in range(g.n)]
    v = [[pool.new() for _ in range(r)] for _ in range(g.n)]
    assert u[-1][-1] == _var_u(g.n, r, g.n - 1, r - 1) and v[-1][-1] == _var_v(g.n, r, g.n - 1, r - 1)
    cnf: list[list[int]] = []
    for i in range(g.n):
        for j in [i] + list(iter_bits(g.forbidden(i))):
            prev = None
            for k in range(r):
                z = pool.new()
                a, b = u[i][k], v[j][k]
                cnf += [[-z, a], [-z, b], [z, -a, -b]]
                if prev is None:
                    prev = z
                    continue
                x = pool.new()  # x = prev XOR z
                cnf += [[-x, prev, z], [-x, -prev, -z], [x, -prev, z], [x, prev, -z]]
                prev = x
            cnf.append([prev] if i == j else [-prev])
    cnf.append([u[0][0]])
    cnf += [[-u[0][k]] for k in range(1, r)]
    return cnf, u, v


def _var_u(n: int, r: int, i: int, k: int) -> int:
    """Index of u[i][k] in `encode` (u vars are allocated first, row-major)."""
    return 1 + i * r + k


def _var_v(n: int, r: int, j: int, k: int) -> int:
    """Index of v[j][k] in `encode` (v vars follow the n*r u vars, row-major)."""
    return 1 + n * r + j * r + k


def _encode_and_solve(g: Graph, r: int, solver: str) -> list[int] | None:
    """Model (list of literals) if SAT, None if UNSAT."""
    cnf, _, _ = encode(g, r)
    with Solver(name=solver, bootstrap_with=cnf) as s:
        return s.get_model() if s.solve() else None


def _solve_child(conn, g: Graph, r: int, solver: str) -> None:
    """Child entry point: encode + solve, send ("ok", model) or ("err", traceback text)."""
    import traceback

    try:
        msg = ("ok", _encode_and_solve(g, r, solver))
    except BaseException:  # report everything, including MemoryError, to the parent
        msg = ("err", traceback.format_exc())
    try:
        conn.send(msg)
    finally:
        conn.close()


_CTX = None


def _context():
    """forkserver context: children fork from a clean single-threaded server, never from a caller that may hold
    native threads (CP-SAT, BLAS). The server preloads this module so each child starts fast."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context("forkserver")
        _CTX.set_forkserver_preload(["minrank.exact.sat"])
    return _CTX


def rank_at_most(g: Graph, r: int, time_limit: float | None = None, solver: str = "cadical153"
                 ) -> list[int] | bool | None:
    """Fitting rows of rank <= r if SAT; False if UNSAT (proof minrk > r); None if the time limit hit.

    With a time limit, encoding and solving both run in a child process that is killed when `time_limit` seconds
    (measured from entry to this function, so process start-up and encoding count) have passed. pysat's
    solve_limited does not release the GIL, so an in-process timer cannot interrupt the solver. Errors in the child
    are re-raised here with the child's traceback.
    """
    if not 1 <= r <= g.n:
        raise ValueError(f"need 1 <= r <= n, got r={r}, n={g.n}")
    t0 = time.perf_counter()
    if time_limit is None:
        model_list = _encode_and_solve(g, r, solver)
    else:
        ctx = _context()
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_solve_child, args=(child, g, r, solver), daemon=True)
        started = False
        try:
            p.start()
            started = True
            child.close()
            left = max(0.0, time_limit - (time.perf_counter() - t0))
            if not parent.poll(left):
                p.kill()
                return None
            try:
                status, payload = parent.recv()
            except EOFError:
                p.join()
                raise RuntimeError(f"SAT child exited without a result (exit code {p.exitcode})") from None
            if status == "err":
                raise RuntimeError(f"SAT child failed:\n{payload}")
            model_list = payload
        finally:
            parent.close()
            if started:  # joining an unstarted process would mask the real start-up error
                p.join()
    if model_list is None:
        return False
    model = set(x for x in model_list if x > 0)
    n = g.n
    U = np.array([[1 if _var_u(n, r, i, k) in model else 0 for k in range(r)] for i in range(n)], dtype=np.uint8)
    V = np.array([[1 if _var_v(n, r, i, k) in model else 0 for k in range(r)] for i in range(n)], dtype=np.uint8)
    rows = rows_from_factors(U, V)
    rk = verify(g, rows)
    if rk > r:
        raise RuntimeError(f"SAT model has rank {rk} > {r}")
    return rows


def exact_minrank(g: Graph, upper_rows: list[int], lower: int = 1, time_limit_per_call: float | None = None
                  ) -> tuple[int, bool, list[int]]:
    """Walk down from a known solution. Returns (best rank, proven optimal, rows)."""
    best_rows = upper_rows
    best = verify(g, upper_rows)
    while best > lower:
        res = rank_at_most(g, best - 1, time_limit_per_call)
        if res is None:
            return best, False, best_rows
        if res is False:
            return best, True, best_rows
        best_rows = res
        best = rank_f2(res)
    return best, True, best_rows


def minrank_bruteforce(g: Graph) -> int:
    """Enumerate every fitting matrix (2^#edges of them). Tests only; keep #edges <= ~16."""
    free = [(i, j) for i in range(g.n) for j in iter_bits(g.out[i])]
    if len(free) > 20:
        raise ValueError(f"{len(free)} free entries is too many for brute force")
    base = [1 << i for i in range(g.n)]
    best = g.n
    for bits in product((0, 1), repeat=len(free)):
        rows = base.copy()
        for (i, j), b in zip(free, bits):
            if b:
                rows[i] |= 1 << j
        best = min(best, rank_f2(rows))
    return best


def sat_descend(g: Graph, time_limit: float, rng: np.random.Generator, cover_fraction: float = 0.1,
                info: dict | None = None) -> list[int]:
    """B9 anytime: clique cover (TabuCol) for `cover_fraction` of the budget, then SAT for rank r-1, r-2, ... until the
    budget is spent or UNSAT (then the result is proven optimal). Returns the best verified rows.
    Total wall-clock stays within `time_limit` plus process-kill latency: every SAT call gets only what is left.
    If `info` is a dict, info["proven"] is set to True iff the descent ended with UNSAT (the result is optimal)."""
    from ..baselines.clique_cover import clique_cover_rows

    deadline = time.perf_counter() + time_limit
    rows = clique_cover_rows(g, "tabucol", time_limit * cover_fraction, rng)
    best = verify(g, rows)
    while best > 1:
        left = deadline - time.perf_counter()
        if left <= 0.05:
            break
        res = rank_at_most(g, best - 1, time_limit=left)
        if res is None or res is False:
            if info is not None:
                info["proven"] = res is False
            break
        rows, best = res, verify(g, res)
    return rows
