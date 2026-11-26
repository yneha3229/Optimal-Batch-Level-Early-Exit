"""Side-information digraphs for index coding, stored for speed on large graphs.

Arc i -> j means receiver i already has message x_j.  Two receivers are *mutual* when each has the other's message.
  lower bound on beta: an acyclic set (receivers whose induced subgraph has no directed cycle, 2-cycles included)
  upper bound on beta: a clique cover of the mutual graph (= a proper colouring of the "conflict" graph, in which two
                       receivers conflict unless they are mutual)
"""
from __future__ import annotations

import numpy as np


class Graph:
    def __init__(self, adj: np.ndarray, name: str = "g"):
        adj = np.asarray(adj, dtype=bool).copy()
        np.fill_diagonal(adj, False)
        self.adj, self.n, self.name = adj, adj.shape[0], name
        mutual = adj & adj.T
        conflict = ~mutual
        np.fill_diagonal(conflict, False)
        self.succ_ptr, self.succ = _csr(adj)
        self.pred_ptr, self.pred = _csr(adj.T)
        self.conf_ptr, self.conf = _csr(conflict)
        n = self.n
        pairs = max(1, n * (n - 1))
        self.features = {
            "n": n,
            "arc_density": float(adj.sum()) / pairs,
            "mutual_density": float(mutual.sum()) / pairs,
        }

    def to_json(self) -> dict:
        i, j = np.nonzero(self.adj)
        return {"name": self.name, "n": self.n, "arcs": np.stack([i, j], 1).tolist()}

    @staticmethod
    def from_json(d: dict) -> "Graph":
        a = np.zeros((d["n"], d["n"]), dtype=bool)
        if d["arcs"]:
            ij = np.asarray(d["arcs"])
            a[ij[:, 0], ij[:, 1]] = True
        return Graph(a, d.get("name", "g"))


def _csr(m: np.ndarray):
    deg = m.sum(1)
    ptr = np.zeros(m.shape[0] + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    idx = np.nonzero(m)[1].astype(np.int32)
    return ptr, idx


# ----------------------------------------------------------------------------- checks (used on every reported bound)
def is_acyclic_set(g: Graph, s) -> bool:
    s = np.asarray(sorted(set(int(v) for v in s)), dtype=np.int64)
    if len(s) == 0:
        return True
    sub = g.adj[np.ix_(s, s)].copy()
    indeg = sub.sum(0)
    stack = list(np.nonzero(indeg == 0)[0])
    seen = 0
    while stack:
        v = stack.pop()
        seen += 1
        for w in np.nonzero(sub[v])[0]:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack.append(w)
    return seen == len(s)


def is_clique_cover(g: Graph, colour) -> bool:
    colour = np.asarray(colour)
    if colour.shape != (g.n,) or colour.min() < 0:
        return False
    mutual = g.adj & g.adj.T
    for c in np.unique(colour):
        vs = np.nonzero(colour == c)[0]
        sub = mutual[np.ix_(vs, vs)]
        if sub.sum() != len(vs) * (len(vs) - 1):
            return False
    return True


# --------------------------------------------------------------------------------------------- graph families
def random_digraph(n, p, recip, rng, name=None):
    """Each pair is linked with prob. p; a linked pair is two-way with prob. recip, else one random direction."""
    up = np.triu(rng.random((n, n)) < p, 1)
    two = np.triu(rng.random((n, n)) < recip, 1)
    dirn = np.triu(rng.random((n, n)) < 0.5, 1)
    a = (up & (two | dirn)) | (up & (two | ~dirn)).T
    return Graph(a, name or f"rand_n{n}_p{p:.2f}_r{recip:.2f}")


def planted_cliques(n, k, p_in, p_out, recip_out, rng, name=None):
    """k hidden groups that are (nearly) mutual inside, random one-/two-way noise between groups."""
    lab = rng.integers(0, k, n)
    same = lab[:, None] == lab[None, :]
    up = np.triu(rng.random((n, n)), 1)
    two = np.triu(rng.random((n, n)) < recip_out, 1)
    dirn = np.triu(rng.random((n, n)) < 0.5, 1)
    link_in = same & (up < p_in)
    link_out = ~same & (up < p_out)
    a = link_in | link_in.T | (link_out & (two | dirn)) | (link_out & (two | ~dirn)).T
    return Graph(a, name or f"planted_n{n}_k{k}")


def layered(n, layers, p_fwd, p_back, recip, rng, name=None):
    """Receivers in layers; most arcs point forward, a few back arcs create long cycles (hard acyclic sets)."""
    lay = np.sort(rng.integers(0, layers, n))
    fwd = lay[:, None] < lay[None, :]
    r = rng.random((n, n))
    a = (fwd & (r < p_fwd)) | (fwd.T & (r < p_back))
    m = np.triu(rng.random((n, n)) < recip, 1) & np.triu(a | a.T, 1)
    a = a | m | m.T
    return Graph(a, name or f"layered_n{n}_L{layers}")


def make_graph(family: str, n: int, rng) -> Graph:
    if family == "random":
        return random_digraph(n, rng.uniform(0.3, 0.9), rng.uniform(0.2, 0.8), rng)
    if family == "planted":
        return planted_cliques(n, int(rng.integers(max(2, n // 40), max(3, n // 8))), rng.uniform(0.85, 1.0),
                               rng.uniform(0.2, 0.7), rng.uniform(0.1, 0.5), rng)
    if family == "layered":
        return layered(n, int(rng.integers(5, 30)), rng.uniform(0.3, 0.8), rng.uniform(0.02, 0.2),
                       rng.uniform(0.1, 0.6), rng)
    raise ValueError(family)


FAMILIES = ("random", "planted", "layered")


def communities(n, rng, name=None):
    """Groups of 4-200 receivers (dense inside: random arcs, many two-way), ordered; earlier groups know some messages
    of later groups (one-way), and a few back links tie some groups together into larger strongly connected parts.
    Models side information that comes in clusters (shared caches, neighbourhoods, interest groups)."""
    sizes = []
    while sum(sizes) < n:
        sizes.append(int(np.exp(rng.uniform(np.log(4), np.log(200)))))
    sizes[-1] -= sum(sizes) - n
    sizes = [s for s in sizes if s > 0]
    a = np.zeros((n, n), dtype=bool)
    off = np.cumsum([0] + sizes)
    for k, s in enumerate(sizes):
        sub = random_digraph(s, rng.uniform(0.4, 0.9), rng.uniform(0.2, 0.9), rng).adj
        a[off[k]:off[k + 1], off[k]:off[k + 1]] = sub
    q, px, back = rng.uniform(0.02, 0.1), rng.uniform(0.05, 0.3), rng.uniform(0.0, 0.003)
    for i in range(len(sizes)):
        for j in range(i + 1, len(sizes)):
            r = rng.random()
            if r < q:
                blk = rng.random((sizes[i], sizes[j])) < px
                a[off[i]:off[i + 1], off[j]:off[j + 1]] |= blk
            elif r < q + back:
                u, v = rng.integers(off[i], off[i + 1]), rng.integers(off[j], off[j + 1])
                a[u, v] = a[v, u] = True
    return Graph(a, name or f"communities_n{n}")


def sparse_digraph(n, rng, name=None):
    """Sparse side information: each receiver knows a few messages (average 1.5-6), some pairs two-way."""
    d, recip = rng.uniform(1.5, 6.0), rng.uniform(0.2, 0.6)
    m = int(d * n / (1 + recip))
    i, j = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = i != j
    i, j = i[keep], j[keep]
    a = np.zeros((n, n), dtype=bool)
    a[i, j] = True
    two = rng.random(len(i)) < recip
    a[j[two], i[two]] = True
    return Graph(a, name or f"sparse_n{n}")


STRUCTURED = ("communities", "sparse")


def make_structured(family: str, n: int, rng) -> Graph:
    return communities(n, rng) if family == "communities" else sparse_digraph(n, rng)
