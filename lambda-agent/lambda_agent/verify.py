"""Checkers. Every bound the agent reports passes one of these; solvers only propose.

- acyclic set           -> lambda >= beta >= |set|            (Kahn's algorithm)
- clique cover          -> beta <= lambda <= #groups           (every group a clique of two-way pairs)
- fractional cover      -> lambda <= sum of weights            (exact rational coverage)
- fitting matrix of G^(t) with rank r over GF(2)  -> lambda_2 <= r / t
      (diagonal blocks = identity, block (i, j) zero unless i knows j; rank by exact elimination)
"""
from __future__ import annotations

from fractions import Fraction

from .graph import Digraph, bits


def is_acyclic(g: Digraph, verts: list[int]) -> bool:
    vs = set(verts)
    if len(vs) != len(verts) or any(not 0 <= v < g.n for v in vs):
        return False
    mask = sum(1 << v for v in vs)
    indeg = {v: 0 for v in vs}
    for u in vs:
        for w in bits(g.out[u] & mask):
            indeg[w] += 1
    stack = [v for v, d in indeg.items() if d == 0]
    seen = 0
    while stack:
        u = stack.pop()
        seen += 1
        for w in bits(g.out[u] & mask):
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    return seen == len(vs)


def is_clique(g: Digraph, grp) -> bool:
    return all(g.mutual(a, b) for k, a in enumerate(grp) for b in grp[k + 1:])


def is_clique_cover(g: Digraph, groups) -> bool:
    seen = sorted(v for grp in groups for v in grp)
    return seen == list(range(g.n)) and all(is_clique(g, grp) for grp in groups)


def frac_cover_value(g: Digraph, cliques, weights) -> Fraction:
    """Sum of weights if every clique is valid and every receiver is covered with total weight >= 1; else raises."""
    cover = [Fraction(0)] * g.n
    for C, w in zip(cliques, weights):
        w = Fraction(w)
        if w < 0 or not is_clique(g, C):
            raise ValueError("invalid fractional cover")
        for v in C:
            cover[v] += w
    if any(c < 1 for c in cover):
        raise ValueError("a receiver is covered with weight < 1")
    return sum((Fraction(w) for w in weights), Fraction(0))


def rank_gf2(rows) -> int:
    basis: dict[int, int] = {}
    for row in rows:
        x = row
        while x:
            p = x.bit_length() - 1
            if p not in basis:
                basis[p] = x
                break
            x ^= basis[p]
    return len(basis)


def fitting_rank(h: Digraph, rows) -> int:
    """Rank over GF(2) of a matrix that fits h (A[p][p] = 1, A[p][q] = 0 unless p knows q); raises if it does not fit.
    For h = G^(t) this is exactly a vector-linear code for G: diagonal blocks identity, block (i, j) zero unless i
    knows j."""
    if len(rows) != h.n:
        raise ValueError("wrong number of rows")
    for p, row in enumerate(rows):
        if not (row >> p) & 1 or row >> h.n:
            raise ValueError(f"row {p}: diagonal entry not 1 or out of range")
        if row & ~(h.out[p] | (1 << p)):
            raise ValueError(f"row {p}: nonzero where the receiver lacks the message")
    return rank_gf2(rows)
