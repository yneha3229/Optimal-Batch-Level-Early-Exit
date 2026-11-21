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


def _solve(cnf: list[list[int]], solver: str) -> list[int] | None:
    """Model (list of literals) if SAT, None if UNSAT."""
    with Solver(name=solver, bootstrap_with=cnf) as s:
        return s.get_model() if s.solve() else None


def _solve_child(conn, cnf: list[list[int]], solver: str) -> None:
    conn.send(_solve(cnf, solver))
    conn.close()


def rank_at_most(g: Graph, r: int, time_limit: float | None = None, solver: str = "cadical153"
                 ) -> list[int] | bool | None:
    """Fitting rows of rank <= r if SAT; False if UNSAT (proof minrk > r); None if the time limit hit.

    With a time limit the solver runs in a forked child that is killed at the deadline: pysat's solve_limited does
    not release the GIL, so an in-process timer thread can never fire.
    """
    cnf, u, v = encode(g, r)
    if time_limit is None:
        model_list = _solve(cnf, solver)
    else:
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        p = ctx.Process(target=_solve_child, args=(child, cnf, solver), daemon=True)
        p.start()
        child.close()
        if not parent.poll(time_limit):
            p.kill()
            p.join()
            return None
        model_list = parent.recv()
        p.join()
    if model_list is None:
        return False
    model = set(x for x in model_list if x > 0)
    U = np.array([[1 if u[i][k] in model else 0 for k in range(r)] for i in range(g.n)], dtype=np.uint8)
    V = np.array([[1 if v[i][k] in model else 0 for k in range(r)] for i in range(g.n)], dtype=np.uint8)
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


def sat_descend(g: Graph, time_limit: float, rng: np.random.Generator, cover_fraction: float = 0.1) -> list[int]:
    """B9 anytime: clique cover (TabuCol) for `cover_fraction` of the budget, then SAT for rank r-1, r-2, ... until the
    budget is spent or UNSAT (then the result is proven optimal). Returns the best verified rows."""
    import time

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
            break
        rows, best = res, verify(g, res)
    return rows
