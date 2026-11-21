"""Graph families for the benchmark. All generators are deterministic given their arguments (and seed)."""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..core.graph import Graph


def gnp(n: int, p: float, seed: int, directed: bool = False) -> Graph:
    """Erdos-Renyi G(n, p) (undirected) or D(n, p) (each ordered pair independently)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    rng = np.random.default_rng(seed)
    if directed:
        adj = rng.random((n, n)) < p  # (n, n)
        np.fill_diagonal(adj, False)
    else:
        upper = np.triu(rng.random((n, n)) < p, k=1)  # (n, n), strictly upper
        adj = upper | upper.T
    kind = "dnp" if directed else "gnp"
    return Graph.from_adjacency(adj, directed, f"{kind}_n{n}_p{p}_s{seed}", kind, {"p": p, "seed": seed})


def kneser(m: int, k: int) -> Graph:
    """Kneser graph K(m, k): k-subsets of [m], adjacent iff disjoint."""
    if not 1 <= k <= m:
        raise ValueError("need 1 <= k <= m")
    verts = [frozenset(c) for c in combinations(range(m), k)]
    edges = [(a, b) for a, b in combinations(range(len(verts)), 2) if not (verts[a] & verts[b])]
    return Graph.from_edges(len(verts), edges, False, f"kneser_{m}_{k}", "kneser", {"m": m, "k": k})


def _is_prime(q: int) -> bool:
    return q >= 2 and all(q % d for d in range(2, int(q ** 0.5) + 1))


def paley(q: int) -> Graph:
    """Paley graph on Z_q, q prime with q = 1 mod 4: i ~ j iff i - j is a nonzero square."""
    if not _is_prime(q) or q % 4 != 1:
        raise ValueError(f"paley needs a prime q = 1 mod 4, got {q}")
    squares = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i, j in combinations(range(q), 2) if (j - i) % q in squares]
    return Graph.from_edges(q, edges, False, f"paley_{q}", "paley", {"q": q})


def circulant(n: int, jumps: tuple[int, ...]) -> Graph:
    """Circulant graph: i ~ j iff (j - i) mod n is in +-jumps."""
    s = {d % n for d in jumps} | {(-d) % n for d in jumps}
    s.discard(0)
    edges = [(i, j) for i, j in combinations(range(n), 2) if (j - i) % n in s]
    name = f"circ_{n}_" + "-".join(map(str, sorted(set(jumps))))
    return Graph.from_edges(n, edges, False, name, "circulant", {"jumps": list(jumps)})


def cycle_power(n: int, k: int) -> Graph:
    """k-th power of the n-cycle (i ~ j iff cyclic distance <= k)."""
    g = circulant(n, tuple(range(1, k + 1)))
    return Graph(g.n, g.out, False, f"cyclepow_{n}_{k}", "cycle_power", {"k": k})


def mycielski(k: int) -> Graph:
    """Mycielski graph M_k (M_2 = K_2, M_3 = C_5, M_4 = Groetzsch on 11 vertices, ...)."""
    if k < 2:
        raise ValueError("k >= 2")
    n, edges = 2, {(0, 1)}
    for _ in range(k - 2):
        # vertices 0..n-1 originals, n..2n-1 shadows, 2n apex
        new = set(edges)
        for u, v in edges:
            new.add((u, n + v))
            new.add((v, n + u))
        for i in range(n):
            new.add((n + i, 2 * n))
        edges, n = new, 2 * n + 1
    return Graph.from_edges(n, sorted(edges), False, f"mycielski_{k}", "mycielski", {"k": k})


def cycle(n: int) -> Graph:
    g = circulant(n, (1,))
    return Graph(g.n, g.out, False, f"cycle_{n}", "cycle", {})
