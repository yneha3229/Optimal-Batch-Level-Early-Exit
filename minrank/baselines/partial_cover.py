"""B4: clique-and-cycle cover for directed side-information graphs (an F_2-valid partial clique cover).

Background
----------
Birk & Kol's partial clique cover serves a set K of k receivers, each knowing at least k - 1 - d others in K, with
d + 1 transmissions of an MDS code. The local chromatic number bound (Shanmugam, Dimakis, Langberg, ISIT 2013) also
relies on MDS codes. MDS codes of the needed length do not exist over F_2 in general, and those constructions are
vector/large-field codes, not F_2 fitting matrices, so they are NOT valid minrk_2 upper bounds and are not reported
here. What is implemented is a construction that always yields a scalar-linear F_2 fitting matrix:

Cycle of cliques
----------------
Let C_1, ..., C_m be disjoint cliques of the mutual graph (every member knows every other member) such that every
member of C_t knows every member of C_{t+1} (indices mod m). Give every member of C_t the row
    row(C_t) = 1[C_t] + 1[C_{t+1}].
It fits: the 1-positions other than i itself are members of C_t (known: clique) or of C_{t+1} (known: arc). The m
distinct rows sum to 0 and any m - 1 of them are independent (supports are disjoint cliques in a cycle), so the block
has rank m - 1: one transmission is saved per cycle. A singleton clique is a vertex, so a plain directed cycle
v_1 -> v_2 -> ... -> v_m -> v_1 (v_t knows v_{t+1}) gets rank m - 1, the classic cycle-cover saving.

Algorithm
---------
1. Clique cover of the mutual graph (TabuCol, `cover_fraction` of the budget) -> cliques.
2. Super-digraph on cliques: C -> D iff every member of C knows every member of D.
3. Greedy disjoint cycle packing: repeatedly take a shortest cycle (BFS from every unused node) among unused cliques;
   each saves 1. Shortest first leaves the most nodes for further cycles.
Rank = #cliques - #cycles. Output is verified.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core.f2 import rank_f2
from ..core.graph import Graph
from ..core.verify import fit_error
from .clique_cover import conflict_graph, tabucol


def _cliques_from_colors(colors: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for v, c in enumerate(colors.tolist()):
        groups.setdefault(c, []).append(v)
    return list(groups.values())


def super_digraph(g: Graph, cliques: list[list[int]]) -> list[list[int]]:
    """succ[a] = cliques b != a such that every member of a knows every member of b."""
    masks = [sum(1 << v for v in c) for c in cliques]
    succ: list[list[int]] = []
    for a, ca in enumerate(cliques):
        know_all = (1 << g.n) - 1
        for v in ca:
            know_all &= g.out[v]  # messages every member of a knows
        succ.append([b for b in range(len(cliques)) if b != a and masks[b] & ~know_all == 0])
    return succ


def shortest_cycle(succ: list[list[int]], used: list[bool]) -> list[int] | None:
    """Shortest directed cycle among unused nodes (BFS from each node); None if acyclic."""
    best: list[int] | None = None
    for s in range(len(succ)):
        if used[s]:
            continue
        parent = {s: -1}
        q = deque([s])
        found = None
        while q and found is None:
            u = q.popleft()
            for w in succ[u]:
                if used[w]:
                    continue
                if w == s:
                    found = u
                    break
                if w not in parent:
                    parent[w] = u
                    q.append(w)
        if found is not None:
            cyc = [found]
            while cyc[-1] != s:
                cyc.append(parent[cyc[-1]])
            cyc.reverse()  # s -> ... -> found -> s
            if best is None or len(cyc) < len(best):
                best = cyc
                if len(best) == 2:
                    return best
    return best


def cycle_cover_rows(g: Graph, cliques: list[list[int]]) -> tuple[list[int], int]:
    """Rows from a clique partition plus greedy cycle packing on the super-digraph. Returns (rows, #cycles)."""
    succ = super_digraph(g, cliques)
    used = [False] * len(cliques)
    cycles: list[list[int]] = []
    while True:
        cyc = shortest_cycle(succ, used)
        if cyc is None:
            break
        for c in cyc:
            used[c] = True
        cycles.append(cyc)
    masks = [sum(1 << v for v in c) for c in cliques]
    rows = [0] * g.n
    for a, c in enumerate(cliques):  # cliques not on a cycle: one transmission each
        if not used[a]:
            for v in c:
                rows[v] = masks[a]
    for cyc in cycles:
        for t, a in enumerate(cyc):
            row = masks[a] | masks[cyc[(t + 1) % len(cyc)]]  # disjoint cliques: OR == XOR
            for v in cliques[a]:
                rows[v] = row
    err = fit_error(g, rows)
    if err:
        raise RuntimeError(f"cycle cover produced a non-fitting matrix on {g.name}: {err}")
    return rows, len(cycles)


def partial_cover_rows(g: Graph, time_limit: float, rng: np.random.Generator) -> list[int]:
    """B4: TabuCol clique cover of the mutual graph, then cycle-of-cliques savings. Verified rows."""
    colors = tabucol(conflict_graph(g), time_limit, rng)
    cliques = _cliques_from_colors(colors)
    rows, ncyc = cycle_cover_rows(g, cliques)
    r = rank_f2(rows)
    if r > len(cliques) - ncyc:
        raise RuntimeError(f"rank {r} exceeds #cliques - #cycles = {len(cliques) - ncyc}")
    return rows
