"""Block-minrank large-neighbourhood search: an anytime minrk_2 upper bound for graphs too big for global SAT.

Idea. Partition V into blocks B_1..B_m. If A_t fits the induced subgraph G[B_t], the block-diagonal matrix
diag(A_1..A_m) fits G (a receiver only uses side information inside its own block), with rank sum_t rank(A_t).
A clique cover is the special case where every block is a clique of rank 1; the "cycle of cliques" partial cover
is a block of m cliques with rank m - 1. Here blocks are solved *exactly* by SAT when small, so any sub-structure
that saves transmissions is found, not only cliques and cycles.

Search. Start from a clique cover. Repeatedly pick k = 2..k_max blocks with combined size <= cap, ask SAT whether
G[union] has minrank <= (sum of their ranks) - 1, and if so replace them by the merged block. Which groups of
blocks to try is decided by `group_score` (evolvable by FunSearch; default: arc density between the blocks, since
a merge can only save if the blocks know each other's messages). Pairs that failed are remembered (tabu) until a
neighbouring block changes.

Every accepted block is verified (fit + rank) and the final global matrix is verified again by the caller.
"""
from __future__ import annotations

import itertools
import random
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from minrank.baselines.clique_cover import clique_cover_rows
from minrank.core.f2 import rank_f2
from minrank.core.graph import Graph, iter_bits
from minrank.core.verify import verify
from minrank.exact.symmetry import rank_at_most_sym

from .decompose import induced
from .mais import greedy_fvs


@dataclass
class Block:
    verts: tuple[int, ...]
    rows: list[int]      # local fitting matrix of G[verts]
    rank: int


@dataclass
class LNSStats:
    tried: int = 0
    accepted: int = 0
    sat_seconds: float = 0.0
    history: list[tuple[float, int]] = field(default_factory=list)  # (elapsed s, total rank)


def blocks_from_rows(g: Graph, rows: list[int]) -> list[Block]:
    """Clique cover rows (every row = indicator of its clique) -> one block per distinct row."""
    groups: dict[int, list[int]] = {}
    for v, r in enumerate(rows):
        groups.setdefault(r, []).append(v)
    out = []
    for vs in groups.values():
        vs = sorted(vs)
        m = (1 << len(vs)) - 1
        out.append(Block(tuple(vs), [m] * len(vs), 1))
    return out


def global_rows(g: Graph, blocks: list[Block]) -> list[int]:
    rows = [0] * g.n
    for b in blocks:
        for k, lr in enumerate(b.rows):
            gm = 0
            for t in iter_bits(lr):
                gm |= 1 << b.verts[t]
            rows[b.verts[k]] = gm
    return rows


def default_group_score(sizes: list[int], ranks: list[int], arcs_between: int, possible_between: int) -> float:
    """Higher = try first. Fraction of possible cross arcs present, favouring small unions (cheap SAT)."""
    return arcs_between / max(1, possible_between) - 0.002 * sum(sizes)


def _cross_arcs(g: Graph, vs_a: tuple[int, ...], mask_b: int) -> int:
    return sum((g.out[v] & mask_b).bit_count() for v in vs_a)


def solve_block(g: Graph, verts: tuple[int, ...], target: int, time_limit: float) -> list[int] | None:
    """Local rows of rank <= target for G[verts], or None (UNSAT or timeout)."""
    sub = induced(g, list(verts))
    if target < 1:
        return None
    acyc = greedy_fvs(sub)
    if len(acyc) > target:
        return None  # MAIS lower bound already exceeds the target
    res = rank_at_most_sym(sub, target, time_limit=time_limit, mode="indep_cols", indep=acyc)
    if res is None or res is False:
        return None
    if verify(sub, res) > target:
        raise RuntimeError("block solution above target")
    return list(res)


def block_lns(g: Graph, time_limit: float, rng: random.Random, init_rows: list[int] | None = None,
              cap: int = 24, k_max: int = 3, call_limit: float = 2.0, candidates: int = 400,
              group_score: Callable[..., float] = default_group_score) -> tuple[list[int], LNSStats]:
    """Anytime block LNS. `init_rows` must be a clique-cover-style matrix (rows = clique indicators) or any
    fitting matrix (then it is used only if it beats the clique cover found here)."""
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    stats = LNSStats()
    cover = clique_cover_rows(g, "tabucol", min(2.0, 0.1 * time_limit), np.random.default_rng(rng.randrange(2**31)))
    blocks = blocks_from_rows(g, cover)
    total = sum(b.rank for b in blocks)
    stats.history.append((time.perf_counter() - t0, total))
    tabu: set[frozenset[tuple[int, ...]]] = set()
    while time.perf_counter() < deadline - 0.05:
        masks = [sum(1 << v for v in b.verts) for b in blocks]
        # candidate groups: pairs with any cross arc, extended greedily to triples
        nb = len(blocks)
        pairs = []
        for a in range(nb):
            for b in range(a + 1, nb):
                if len(blocks[a].verts) + len(blocks[b].verts) > cap:
                    continue
                ab = _cross_arcs(g, blocks[a].verts, masks[b])
                ba = _cross_arcs(g, blocks[b].verts, masks[a])
                if ab == 0 or ba == 0:
                    continue  # one-way arcs only: the union is not strongly connected, no saving possible
                pairs.append((a, b, ab + ba))
        groups = []
        for a, b, arcs in pairs:
            sz = [len(blocks[a].verts), len(blocks[b].verts)]
            poss = 2 * sz[0] * sz[1]
            groups.append(((a, b), group_score(sz, [blocks[a].rank, blocks[b].rank], arcs, poss)))
        if k_max >= 3 and pairs:
            # triples a-b-c where (a,b) and (b,c) are pairs: catch 3-cycles of cliques
            adj: dict[int, list[int]] = {}
            for a, b, _ in pairs:
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
            for b, nbrs in adj.items():
                for a, c in itertools.combinations(nbrs, 2):
                    grp = tuple(sorted((a, b, c)))
                    sz = [len(blocks[x].verts) for x in grp]
                    if sum(sz) > cap:
                        continue
                    arcs = sum(_cross_arcs(g, blocks[x].verts, masks[y]) for x in grp for y in grp if x != y)
                    poss = sum(sz[i] * sz[j] for i in range(3) for j in range(3) if i != j)
                    groups.append((grp, group_score(sz, [blocks[x].rank for x in grp], arcs, poss)))
        groups = [(grp, s) for grp, s in groups if frozenset(blocks[x].verts for x in grp) not in tabu]
        if not groups:
            break
        groups.sort(key=lambda t: -t[1])
        groups = groups[:candidates]
        progressed = False
        for grp, _ in groups:
            if time.perf_counter() >= deadline - 0.05:
                break
            key = frozenset(blocks[x].verts for x in grp)
            verts = tuple(sorted(v for x in grp for v in blocks[x].verts))
            target = sum(blocks[x].rank for x in grp) - 1
            stats.tried += 1
            ts = time.perf_counter()
            rows = solve_block(g, verts, target, min(call_limit, deadline - time.perf_counter()))
            stats.sat_seconds += time.perf_counter() - ts
            if rows is None:
                tabu.add(key)
                continue
            new = Block(verts, rows, rank_f2(rows))
            blocks = [b for i, b in enumerate(blocks) if i not in grp] + [new]
            total = sum(b.rank for b in blocks)
            stats.accepted += 1
            stats.history.append((time.perf_counter() - t0, total))
            progressed = True
            break  # recompute candidates around the new block
        if not progressed:
            break
    rows = global_rows(g, blocks)
    r = verify(g, rows)
    if init_rows is not None and verify(g, init_rows) < r:
        return list(init_rows), stats
    return rows, stats
