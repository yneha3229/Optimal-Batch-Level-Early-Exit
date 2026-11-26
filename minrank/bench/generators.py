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


def orthogonality(k: int) -> Graph:
    """Odd-weight vectors of F_2^k; i knows j iff <v_i, v_j> = 1 (mod 2).

    A[i, j] = <v_i, v_j> fits by construction (diagonal 1, zero exactly on non-edges), so minrk_2 <= k with
    n = 2^(k-1) vertices; clique covers are typically much larger.
    """
    if not 2 <= k <= 12:
        raise ValueError("k in [2, 12]")
    vecs = np.array([[(x >> b) & 1 for b in range(k)] for x in range(1, 1 << k) if bin(x).count("1") % 2],
                    dtype=np.int64)  # (n, k)
    gram = (vecs @ vecs.T) & 1  # (n, n)
    adj = gram.astype(bool)
    np.fill_diagonal(adj, False)
    return Graph.from_adjacency(adj, False, f"orth_{k}", "orthogonality", {"k": k, "planted_rank": k})


def planted(n: int, r: int, extra_p: float, seed: int, directed: bool = True) -> Graph:
    """Graph with a planted fitting matrix of rank <= r.

    Draw u_i, v_i in F_2^r with <u_i, v_i> = 1 (resample until true). Side information S_i = {j : <u_i, v_j> = 1}
    plus each other pair independently with probability extra_p (extra edges only add free entries, so the planted
    matrix still fits). Undirected: u_i = v_i and the support is symmetrised (still fits since <u_i,u_j> is symmetric).
    """
    if not 1 <= r <= n:
        raise ValueError("need 1 <= r <= n")
    rng = np.random.default_rng(seed)
    U = np.zeros((n, r), dtype=np.int64)
    V = np.zeros((n, r), dtype=np.int64)
    for i in range(n):
        while True:
            u = rng.integers(0, 2, r)
            v = u.copy() if not directed else rng.integers(0, 2, r)
            if int(u @ v) % 2 == 1:
                U[i], V[i] = u, v
                break
    adj = ((U @ V.T) & 1).astype(bool)  # (n, n): A[i, j] = <u_i, v_j>
    extra = rng.random((n, n)) < extra_p
    if not directed:
        extra = np.triu(extra, 1)
        extra = extra | extra.T
    adj = adj | extra
    np.fill_diagonal(adj, False)
    kind = "planted_d" if directed else "planted"
    return Graph.from_adjacency(adj, directed, f"{kind}_n{n}_r{r}_e{extra_p}_s{seed}", kind,
                                {"planted_rank": r, "extra_p": extra_p, "seed": seed})


# ------------------------------------------------------------------ set systems and codes (WORK.md T1, 24 Nov)
def _subset_graph(m: int, k: int, adjacent, name: str, family: str, params: dict) -> Graph:
    verts = [frozenset(c) for c in combinations(range(m), k)]
    edges = [(a, b) for a, b in combinations(range(len(verts)), 2) if adjacent(len(verts[a] & verts[b]))]
    return Graph.from_edges(len(verts), edges, False, name, family, params)


def johnson(m: int, k: int) -> Graph:
    """Johnson graph J(m, k): k-subsets of [m], adjacent iff |A & B| = k - 1."""
    if not 1 <= k < m:
        raise ValueError("need 1 <= k < m")
    return _subset_graph(m, k, lambda s: s == k - 1, f"johnson_{m}_{k}", "johnson", {"m": m, "k": k})


def generalized_kneser(m: int, k: int, s: int) -> Graph:
    """Generalized Kneser graph: k-subsets of [m], adjacent iff |A & B| <= s (s = 0 is K(m, k))."""
    if not (1 <= k <= m and 0 <= s < k):
        raise ValueError("need 1 <= k <= m and 0 <= s < k")
    return _subset_graph(m, k, lambda t: t <= s, f"gkneser_{m}_{k}_{s}", "generalized_kneser",
                         {"m": m, "k": k, "s": s})


def gkneser(m: int, k: int, s: int) -> Graph:
    """Short alias of generalized_kneser for family specs (gkneser:9,3,1)."""
    return generalized_kneser(m, k, s)


def odd_graph(k: int) -> Graph:
    """Odd graph O_k = K(2k + 1, k)."""
    if k < 1:
        raise ValueError("k >= 1")
    return _subset_graph(2 * k + 1, k, lambda t: t == 0, f"odd_{k}", "odd", {"k": k, "m": 2 * k + 1})


def co_kneser(m: int, k: int) -> Graph:
    """Complement of K(m, k): k-subsets of [m], adjacent iff they intersect (and differ)."""
    if not 1 <= k <= m:
        raise ValueError("need 1 <= k <= m")
    return _subset_graph(m, k, lambda t: t > 0, f"cokneser_{m}_{k}", "co_kneser", {"m": m, "k": k})


def hamming(d: int, q: int, dist: int) -> Graph:
    """Words of Z_q^d, adjacent iff their Hamming distance equals dist (dist = 1: the Hamming graph H(d, q);
    q = 2, dist = 1: the hypercube Q_d; q = 2, dist = d: a perfect matching)."""
    if not (d >= 1 and q >= 2 and 1 <= dist <= d and q ** d <= 4096):
        raise ValueError("need d >= 1, q >= 2, 1 <= dist <= d, q^d <= 4096")
    words = np.array(np.unravel_index(np.arange(q ** d), (q,) * d)).T  # (q^d, d)
    hd = (words[:, None, :] != words[None, :, :]).sum(-1)  # (q^d, q^d)
    adj = hd == dist
    return Graph.from_adjacency(adj, False, f"hamming_{d}_{q}_{dist}", "hamming", {"d": d, "q": q, "dist": dist})
