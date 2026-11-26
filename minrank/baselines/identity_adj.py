"""I + A(G) over F_2: a one-line upper bound on minrk_2 of an undirected graph.

The matrix with ones on the diagonal and on every edge fits G (zeros exactly on the non-edges), so
minrk_2(G) <= rank_2(I + A(G)). It is the only fitting matrix invariant under all of Aut(G) apart from I when Aut(G)
is transitive on ordered edges and on ordered non-edges, e.g. Kneser graphs under S_n; there it is optimal on
K(8,3) (22), K(10,3) (36 = alpha) and K(n,2), see docs/MINRANK_THEORY.md. Directed graphs work the same way with A
the arc matrix.
"""
from __future__ import annotations

from ..core.graph import Graph


def identity_plus_adjacency_rows(g: Graph) -> list[int]:
    return [(1 << i) | g.out[i] for i in range(g.n)]
