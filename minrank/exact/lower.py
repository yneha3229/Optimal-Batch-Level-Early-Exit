"""Certified lower bounds (L1): MAIS(G) <= minrk_q(G) for every field.

If the induced subgraph on a vertex set I is acyclic, order I topologically; the submatrix A[I, I] of any fitting
matrix is then triangular with unit diagonal, so rank(A) >= |I|. For an undirected graph (every edge a 2-cycle) an
acyclic induced subgraph is an independent set, so the bound is alpha(G).

The bound we report is the size of a *found* acyclic set, re-checked independently, so it is valid even when CP-SAT
stops at its time limit. `proven_max` says whether CP-SAT also proved no larger set exists.
"""
from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from ..core.graph import Graph, iter_bits


@dataclass
class LowerBound:
    value: int
    vertices: list[int]
    proven_max: bool
    upper: int  # CP-SAT bound on MAIS itself (not a bound on minrk)


def is_acyclic_induced(g: Graph, verts: list[int]) -> bool:
    """Kahn's algorithm on the subgraph induced by `verts`."""
    vs = set(verts)
    if len(vs) != len(verts):
        return False
    mask = sum(1 << v for v in vs)
    indeg = {v: 0 for v in vs}
    for u in vs:
        for v in iter_bits(g.out[u] & mask):
            indeg[v] += 1
    stack = [v for v, d in indeg.items() if d == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for v in iter_bits(g.out[u] & mask):
            indeg[v] -= 1
            if indeg[v] == 0:
                stack.append(v)
    return seen == len(vs)


def mais(g: Graph, time_limit: float = 30.0, workers: int = 1, seed: int = 0) -> LowerBound:
    """Maximum acyclic induced subgraph by CP-SAT.

    Variables: y_v (kept), pos_v in [0, n-1]. For a 2-cycle {u, v}: not (y_u and y_v). For a one-way edge u -> v
    with both kept: pos_u < pos_v (any strict order of kept vertices consistent with all one-way edges exists iff the
    kept set is acyclic, because 2-cycles are already excluded).
    """
    n = g.n
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{v}") for v in range(n)]
    one_way = [(u, v) for u in range(n) for v in iter_bits(g.out[u]) if not (g.out[v] >> u) & 1]
    pos = [m.NewIntVar(0, n - 1, f"p{v}") for v in range(n)] if one_way else []
    for u in range(n):
        for v in iter_bits(g.out[u]):
            if u < v and (g.out[v] >> u) & 1:
                m.AddBoolOr([y[u].Not(), y[v].Not()])
    for u, v in one_way:
        m.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
    m.Maximize(sum(y))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(time_limit)
    s.parameters.num_workers = int(workers)
    s.parameters.random_seed = int(seed)
    status = s.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return LowerBound(1, [0], False, n)  # a single vertex is always acyclic
    verts = [v for v in range(n) if s.Value(y[v])]
    if not is_acyclic_induced(g, verts):
        raise RuntimeError(f"CP-SAT returned a cyclic set on {g.name}")
    return LowerBound(len(verts), verts, status == cp_model.OPTIMAL, int(s.BestObjectiveBound()))
