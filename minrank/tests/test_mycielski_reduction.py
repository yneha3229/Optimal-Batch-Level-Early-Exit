"""The shadow-basis reduction agrees with the direct SAT encoding: minrk_2(mu(G)) == |V(G)| iff the reduced
instance is SAT (small graphs with |V| >= 2 alpha(G), so that the shadows are a maximum independent set)."""
import numpy as np
from pysat.solvers import Solver

from minrank.core.graph import Graph
from minrank.core.verify import verify
from minrank.exact.lower import mais
from minrank.exact.mycielski_reduction import build_matrix, encode
from minrank.exact.symmetry import rank_at_most_sym


def mu(g: Graph) -> Graph:
    n = g.n
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (g.out[u] >> v) & 1:
                edges |= {(u, v), (u, n + v), (v, n + u)}
    edges |= {(n + i, 2 * n) for i in range(n)}
    return Graph.from_edges(2 * n + 1, sorted(edges), False, f"mu_{g.name}", "mu", {})


def test_reduction_matches_direct():
    rng = np.random.default_rng(1)
    seen = {True: 0, False: 0}
    for t in range(60):
        n = int(rng.integers(3, 6))
        up = np.triu(rng.random((n, n)) < rng.uniform(0.4, 0.95), 1)
        g = Graph.from_adjacency(up | up.T, False, f"r{t}", "rand", {})
        if 2 * mais(g, 5).value > n:
            continue
        h = mu(g)
        c, A, B, x, z = encode(g)
        with Solver(name="cadical153", bootstrap_with=c.cl) as S:
            red = S.solve()
            model = set(l for l in S.get_model() if l > 0) if red else None
        direct = rank_at_most_sym(h, n, None, mode="cols")        # in-process, no limit (tiny instances)
        assert bool(red) == (direct is not False), (g.name, red, direct)
        if red:
            assert verify(h, build_matrix(g, model, A, B, x, z)) == n
        seen[bool(red)] += 1
    assert seen[True] > 0 and seen[False] > 0, seen
