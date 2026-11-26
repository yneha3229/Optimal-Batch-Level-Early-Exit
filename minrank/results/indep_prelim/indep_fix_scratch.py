"""SAT for "minrk_2(G) <= r" with independent-set symmetry breaking.

Lemma (undirected G). If I is independent, A[I,I] = Id for every fitting A, so U_I (|I| x r) has full row rank.
Some W in GL(r) gives U_I W = [Id | 0]; then V_I W^{-T} = [Id | X], and the stabiliser of U_I = [Id|0]
(W = [[Id,0],[C,D]]) maps X to (X - C^T) D^{-T}, so C^T = X gives X = 0. Hence WLOG U_I = V_I = [Id | 0].
Directed G with an acyclic induced I: only U_I = [Id | 0] and V_I[:, |I|:] = 0 (A[I,I] is unit-triangular).
"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/home/neha/ie624")
from pysat.solvers import Solver
from minrank.core.graph import Graph
from minrank.exact.sat import encode
from minrank.exact.lower import mais
from minrank.core.verify import verify


def encode_indep(g: Graph, r: int, I: list[int]):
    cnf, u, v = encode(g, r)
    cnf = cnf[:-r]                      # drop the u_0 = e_1 fix (a different GL(r) normalisation)
    s = len(I)
    assert s <= r
    for k, i in enumerate(sorted(I)):
        for c in range(r):
            cnf.append([u[i][c]] if c == k else [-u[i][c]])
            if not g.directed:
                cnf.append([v[i][c]] if c == k else [-v[i][c]])
            elif c >= s:
                cnf.append([-v[i][c]])
    return cnf, u, v


def solve(g, r, I, limit=None, solver="cadical153"):
    cnf, u, v = encode_indep(g, r, I)
    t = time.perf_counter()
    with Solver(name=solver, bootstrap_with=cnf) as S:
        if limit:
            from threading import Timer
            tm = Timer(limit, S.interrupt); tm.start()
            res = S.solve_limited(expect_interrupt=True); tm.cancel()
        else:
            res = S.solve()
        model = S.get_model() if res else None
    dt = time.perf_counter() - t
    if res is None:
        return "timeout", dt, None
    if not res:
        return "unsat", dt, None
    m = set(x for x in model if x > 0)
    U = [[1 if u[i][k] in m else 0 for k in range(r)] for i in range(g.n)]
    V = [[1 if v[i][k] in m else 0 for k in range(r)] for i in range(g.n)]
    import numpy as np
    from minrank.core.f2 import rows_from_factors
    rows = rows_from_factors(np.array(U, dtype=np.uint8), np.array(V, dtype=np.uint8))
    return "sat", dt, rows


if __name__ == "__main__":
    path, r, limit = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    g = Graph.load(path)
    lb = mais(g, time_limit=60)
    I = lb.vertices[: min(len(lb.vertices), r)]
    st, dt, rows = solve(g, r, I, limit)
    extra = f" verified_rank={verify(g, rows)}" if rows else ""
    print(f"{g.name} n={g.n} r={r} |I|={len(I)} alpha_found={lb.value}: {st} in {dt:.1f}s{extra}", flush=True)
