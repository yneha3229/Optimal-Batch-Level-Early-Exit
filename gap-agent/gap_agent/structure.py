"""Structure analysis: split the graph into independent pieces and shrink each piece without changing the bounds.

1. Strongly connected components (SCCs).  Relabelling the receivers SCC by SCC puts the adjacency matrix in block
   triangular form.  Arcs between different SCCs lie on no cycle, so
     * the largest acyclic set is the union of the largest acyclic sets of the SCCs,
     * every clique of mutual receivers lies inside one SCC, so the clique cover adds up too,
     * beta and lambda_2 add up as well (arcs on no directed cycle can be removed without changing the broadcast
       rate: Arbabjolfaei & Kim, "Structural properties of index coding capacity", ISIT 2015).
2. Mutual components: inside an SCC, cliques live in the connected components of the mutual graph, so the clique
   cover is the sum over those.
3. Kernel for the acyclic set (equivalently the minimum directed feedback vertex set F, |acyclic| = n - |F|), with
   the classic reduction rules (Levy & Low 1988; Lin & Jou 2000), applied until nothing changes:
     LOOP   a receiver with a self-loop (created by contractions) is in F
     IN0/OUT0  no in-arc or no out-arc: on no cycle, keep it in the acyclic set
     IN1    exactly one in-arc u -> v: every cycle through v passes u, so v can be kept; contract v into u
            (u gets v's out-arcs)
     OUT1   exactly one out-arc v -> w: symmetric, contract v into w
     CORE   all of v's arcs are two-way and its neighbours are pairwise two-way: keep v, its neighbours go to F
     PIE    a one-way arc between two different SCCs of the one-way subgraph can be deleted: any cycle through it
            also uses a two-way pair, and one end of every two-way pair is in F anyway
   The acyclic set of the original piece = (receivers kept by the rules) + (an acyclic set of the kernel).
Every final certificate is re-checked on the original graph, so a reduction can never produce a wrong bound.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .graph import Graph

DENSE_ARCS = 300_000   # pieces with more arcs than this skip the set-based kernel rules (they rarely fire there)


def scc_labels(adj: np.ndarray) -> tuple[int, np.ndarray]:
    return connected_components(csr_matrix(adj), directed=True, connection="strong")


def mutual_labels(adj: np.ndarray) -> tuple[int, np.ndarray]:
    return connected_components(csr_matrix(adj & adj.T), directed=False)


class Kernel:
    """Acyclic-set kernel of one piece: `keep` (piece-local ids kept by the rules), `drop` (put in F), and the
    remaining kernel graph on `verts` (piece-local ids) with arcs `arcs` (kernel-local ids, may include arcs created
    by contractions)."""

    def __init__(self, n, keep, drop, verts, arcs, rules):
        self.n, self.keep, self.drop, self.verts, self.arcs, self.rules = n, keep, drop, verts, arcs, rules

    def graph(self) -> Graph:
        m = len(self.verts)
        a = np.zeros((m, m), dtype=bool)
        if self.arcs is None:
            raise ValueError("dense kernel: use the piece itself")
        if self.arcs:
            ij = np.asarray(self.arcs)
            a[ij[:, 0], ij[:, 1]] = True
        return Graph(a, "kernel")


def kernel(adj: np.ndarray, max_arcs: int = DENSE_ARCS) -> Kernel:
    n = adj.shape[0]
    rules = {k: 0 for k in ("LOOP", "IN0/OUT0", "IN1", "OUT1", "CORE", "PIE")}
    if int(adj.sum()) > max_arcs:  # dense piece: no rules, the kernel is the piece itself (arcs not listed)
        return Kernel(n, [], [], list(range(n)), None, rules)
    succ = [set(np.nonzero(adj[v])[0].tolist()) for v in range(n)]
    pred = [set(np.nonzero(adj[:, v])[0].tolist()) for v in range(n)]
    alive = set(range(n))
    keep, drop = [], []

    def remove(v):
        alive.discard(v)
        for w in succ[v]:
            pred[w].discard(v)
        for u in pred[v]:
            succ[u].discard(v)
        nb = (succ[v] | pred[v]) - {v}
        succ[v], pred[v] = set(), set()
        return nb

    def add_arc(u, w):
        succ[u].add(w)
        pred[w].add(u)

    def local_rules(queue):
        while queue:
            v = queue.pop()
            if v not in alive:
                continue
            if v in succ[v]:
                rules["LOOP"] += 1
                drop.append(v)
                queue |= remove(v)
            elif not pred[v] or not succ[v]:
                rules["IN0/OUT0"] += 1
                keep.append(v)
                queue |= remove(v)
            elif len(pred[v]) == 1:
                rules["IN1"] += 1
                (u,) = pred[v]
                out = list(succ[v])
                keep.append(v)
                queue |= remove(v)
                for w in out:
                    add_arc(u, w)
                queue.add(u)
            elif len(succ[v]) == 1:
                rules["OUT1"] += 1
                (w,) = succ[v]
                inn = list(pred[v])
                keep.append(v)
                queue |= remove(v)
                for u in inn:
                    add_arc(u, w)
                queue.add(w)
            elif succ[v] == pred[v] and len(succ[v]) <= 25:
                nb = list(succ[v])
                if all(b in succ[a] and a in succ[b] for i, a in enumerate(nb) for b in nb[i + 1:]):
                    rules["CORE"] += 1
                    keep.append(v)
                    queue |= remove(v)
                    for a in nb:
                        if a in alive:
                            drop.append(a)
                            queue |= remove(a)

    def pie():
        vs = sorted(alive)
        if not vs:
            return set()
        idx = {v: k for k, v in enumerate(vs)}
        rows, cols = [], []
        for u in vs:
            for w in succ[u]:
                if u not in succ[w]:
                    rows.append(idx[u])
                    cols.append(idx[w])
        if not rows:
            return set()
        m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(vs), len(vs)))
        _, lab = connected_components(m, directed=True, connection="strong")
        touched = set()
        for r, c in zip(rows, cols):
            if lab[r] != lab[c]:
                u, w = vs[r], vs[c]
                succ[u].discard(w)
                pred[w].discard(u)
                rules["PIE"] += 1
                touched |= {u, w}
        return touched

    queue = set(range(n))
    while True:
        local_rules(queue)
        queue = pie()
        if not queue:
            break
    verts = sorted(alive)
    idx = {v: k for k, v in enumerate(verts)}
    arcs = [(idx[u], idx[w]) for u in verts for w in succ[u]]
    return Kernel(n, keep, drop, verts, arcs, rules)


def lift_acyclic(k: Kernel, kernel_set) -> list[int]:
    """Acyclic set of the piece (piece-local ids) from an acyclic set of the kernel (kernel-local ids)."""
    return sorted(set(k.keep) | {k.verts[i] for i in kernel_set})


class Pieces:
    """The decomposition of a graph: SCC label per receiver, and per SCC its receivers (sorted original ids)."""

    def __init__(self, g: Graph):
        self.count, self.label = scc_labels(g.adj)
        order = np.argsort(self.label, kind="stable")
        bounds = np.searchsorted(self.label[order], np.arange(self.count + 1))
        self.verts = [order[bounds[s]:bounds[s + 1]] for s in range(self.count)]
        self.sizes = np.array([len(v) for v in self.verts])
