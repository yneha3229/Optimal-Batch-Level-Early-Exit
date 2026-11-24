"""Split a side-information digraph into strongly connected components (SCCs) and lift answers back.

Why this is free (all four quantities Pincer bounds are additive over SCCs):
- MAIS: a directed cycle lies inside one SCC, so a union of acyclic sets, one per SCC, is acyclic.
- Clique cover: a bidirected pair is a 2-cycle, so every clique of the mutual graph lies inside one SCC.
- minrk_q: order the SCCs topologically. Every arc between two SCCs then points the same way, so a fitting matrix
  is block triangular with a fitting matrix of each SCC on the diagonal, and rank of a block-triangular matrix is at
  least the sum of the diagonal ranks. Conversely the block-diagonal matrix of per-SCC solutions fits G.
- beta (broadcast rate): Tahmasbi, Shahrasbi, Gohari, "Critical graphs in index coding" (ISIT 2014, JSAC 2015,
  arXiv:1312.0132) show the capacity region is obtained by time-sharing over the SCCs, so beta(G) = sum of beta(SCC).
  Pincer only uses the direction beta(G) >= sum of lower bounds; tests check it against the LP on small graphs.
"""
from __future__ import annotations

from dataclasses import dataclass

from minrank.core.graph import Graph, iter_bits


@dataclass(frozen=True)
class Component:
    """An induced subgraph on `verts` (sorted, global labels). Local vertex k is global vertex verts[k]."""

    index: int
    verts: tuple[int, ...]
    graph: Graph

    def to_global(self, local: list[int]) -> list[int]:
        return [self.verts[k] for k in local]


def induced(g: Graph, verts: list[int] | tuple[int, ...], name: str = "") -> Graph:
    """Subgraph induced on `verts`, relabelled 0..len-1 in the given order."""
    pos = {v: k for k, v in enumerate(verts)}
    out = []
    for v in verts:
        m = 0
        for u in iter_bits(g.out[v]):
            k = pos.get(u)
            if k is not None:
                m |= 1 << k
        out.append(m)
    directed = g.directed and any((out[j] >> i) & 1 == 0 for i in range(len(out)) for j in iter_bits(out[i]))
    return Graph(len(verts), tuple(out), directed, name or f"{g.name}[{len(verts)}]", g.family, {})


def scc(g: Graph) -> list[list[int]]:
    """Strongly connected components (iterative Tarjan), each sorted, listed in reverse topological order."""
    n = g.n
    index = [-1] * n
    low = [0] * n
    on = [False] * n
    stack: list[int] = []
    comps: list[list[int]] = []
    counter = 0
    succ = [list(iter_bits(g.out[v])) for v in range(n)]
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            if pi < len(succ[v]):
                work[-1] = (v, pi + 1)
                w = succ[v][pi]
                if index[w] == -1:
                    work.append((w, 0))
                elif on[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                u = work[-1][0]
                low[u] = min(low[u], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
    return comps


def decompose(g: Graph) -> list[Component]:
    """SCCs as Components, largest first (singletons last)."""
    comps = sorted(scc(g), key=lambda c: (-len(c), c[0]))
    return [Component(k, tuple(c), induced(g, c, f"{g.name}#scc{k}")) for k, c in enumerate(comps)]


def lift_rows(g: Graph, comps: list[Component], local_rows: list[list[int]]) -> list[int]:
    """Block-diagonal global fitting matrix from per-component fitting matrices (local bit k = global verts[k])."""
    rows = [0] * g.n
    for comp, lr in zip(comps, local_rows):
        if len(lr) != len(comp.verts):
            raise ValueError(f"component {comp.index}: {len(lr)} rows for {len(comp.verts)} vertices")
        for k, row in enumerate(lr):
            gm = 0
            for b in iter_bits(row):
                gm |= 1 << comp.verts[b]
            rows[comp.verts[k]] = gm
    return rows
