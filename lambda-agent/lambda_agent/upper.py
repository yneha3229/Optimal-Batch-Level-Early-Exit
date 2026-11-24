"""Upper bounds on lambda_2 (all achieved by explicit linear codes over GF(2)).

clique_cover(g)      one XOR per clique of two-way pairs (CP-SAT, strengthened colouring model). Rate = #groups.
frac_cover(g)        time sharing over cliques (an LP); a vector-linear code over any field. Rate = sum of weights.
minrank_sat(g,t,r)   is there a vector-linear code with block length t and r transmissions, i.e. a fitting matrix of
                     G^(t) with GF(2) rank <= r?  CNF for A = U V^T (U, V: (n t) x r), one AND/XOR chain per
                     constrained entry. Optional symmetry fix (`fix`): for an acyclic set I of G^(t) with |I| <= r,
                     some basis change gives U_I = [Id | 0] (and V_I = [Id | 0] if I has no arcs inside), so these
                     variables are fixed. The returned matrix is re-checked with verify.fitting_rank.
"""
from __future__ import annotations

import multiprocessing as mp
import time
from fractions import Fraction

import networkx as nx
import numpy as np
from ortools.sat.python import cp_model
from scipy.optimize import linprog

from .graph import Digraph, bits
from .verify import fitting_rank, frac_cover_value, is_acyclic, is_clique_cover


# ------------------------------------------------------------------------------------------- clique cover
def _dsatur(n, conf):
    color = [-1] * n
    for _ in range(n):
        best, key = -1, None
        for v in range(n):
            if color[v] < 0:
                k = (len({color[u] for u in range(n) if conf[v][u] and color[u] >= 0}), sum(conf[v]))
                if key is None or k > key:
                    best, key = v, k
        used = {color[u] for u in range(n) if conf[best][u] and color[u] >= 0}
        c = 0
        while c in used:
            c += 1
        color[best] = c
    return color


def clique_cover(g: Digraph, time_limit: float = 10.0) -> tuple[list[list[int]], bool]:
    n = g.n
    if n == 0:
        return [], True
    conf = [[u != v and not g.mutual(u, v) for v in range(n)] for u in range(n)]
    start = _dsatur(n, conf)
    k = max(start) + 1
    m = cp_model.CpModel()
    x = [[m.NewBoolVar("") for _ in range(k)] for _ in range(n)]
    used = [m.NewBoolVar("") for _ in range(k)]
    for v in range(n):
        m.AddExactlyOne(x[v])
        for c in range(k):
            m.AddImplication(x[v][c], used[c])
            m.AddHint(x[v][c], start[v] == c)
    for u in range(n):
        for v in range(u + 1, n):
            if conf[u][v]:
                for c in range(k):
                    m.AddBoolOr([x[u][c].Not(), x[v][c].Not()])
    for c in range(k - 1):
        m.AddImplication(used[c + 1], used[c])
    clique = []
    for v in sorted(range(n), key=lambda v: -sum(conf[v])):
        if all(conf[v][u] for u in clique):
            clique.append(v)
    for c, v in enumerate(clique[:k]):
        m.Add(x[v][c] == 1)
    m.Minimize(sum(used))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        groups = {}
        for v, c in enumerate(start):
            groups.setdefault(c, []).append(v)
        return list(groups.values()), False
    groups = {}
    for v in range(n):
        for c in range(k):
            if s.Value(x[v][c]):
                groups.setdefault(c, []).append(v)
    out = list(groups.values())
    assert is_clique_cover(g, out)
    return out, st == cp_model.OPTIMAL


# --------------------------------------------------------------------------------------- fractional cover
def frac_cover(g: Digraph) -> tuple[Fraction, list[list[int]], list[Fraction]]:
    """Exact fractional clique cover over all maximal cliques (fine for the small components it is used on)."""
    if g.n == 0:
        return Fraction(0), [], []
    M = nx.Graph()
    M.add_nodes_from(range(g.n))
    M.add_edges_from((i, j) for i in range(g.n) for j in range(i + 1, g.n) if g.mutual(i, j))
    cliques = [sorted(c) for c in nx.find_cliques(M)]
    A = np.zeros((g.n, len(cliques)))
    for k, C in enumerate(cliques):
        A[C, k] = 1
    res = linprog(np.ones(len(cliques)), A_ub=-A, b_ub=-np.ones(g.n), bounds=(0, None), method="highs")
    best = None
    for den in (12, 60, 1000, 720720):
        w = [Fraction(float(x)).limit_denominator(den) for x in res.x]
        cov = [sum((w[k] for k, C in enumerate(cliques) if v in C), Fraction(0)) for v in range(g.n)]
        worst = min(cov)
        if worst <= 0:
            continue
        if worst < 1:
            w = [x / worst for x in w]
        val = frac_cover_value(g, cliques, w)
        if best is None or val < best[0]:
            best = (val, w)
        if val <= Fraction(res.fun).limit_denominator(1000):
            break
    keep = [k for k, x in enumerate(best[1]) if x > 0]
    return best[0], [cliques[k] for k in keep], [best[1][k] for k in keep]


# ------------------------------------------------------------------------------------- minrank via SAT
def _encode(N: int, out: list[int], r: int, fix: list[int] | None):
    """CNF: exists U, V in GF(2)^{N x r} with (U V^T)[p][p] = 1 and (U V^T)[p][q] = 0 when p lacks q."""
    u = lambda p, k: 1 + p * r + k                       # noqa: E731
    v = lambda q, k: 1 + N * r + q * r + k               # noqa: E731
    nxt = [1 + 2 * N * r]
    cnf = []

    def new():
        nxt[0] += 1
        return nxt[0] - 1

    full = (1 << N) - 1
    for p in range(N):
        for q in [p] + list(bits(full & ~out[p] & ~(1 << p))):
            prev = None
            for k in range(r):
                z = new()
                a, b = u(p, k), v(q, k)
                cnf += [[-z, a], [-z, b], [z, -a, -b]]
                if prev is None:
                    prev = z
                    continue
                x = new()
                cnf += [[-x, prev, z], [-x, -prev, -z], [x, -prev, z], [x, prev, -z]]
                prev = x
            cnf.append([prev] if p == q else [-prev])
    if fix:
        s = len(fix)
        mask = sum(1 << w for w in fix)
        arc_free = all(not (out[w] & mask) for w in fix)
        for k, w in enumerate(sorted(fix)):
            for c in range(r):
                cnf.append([u(w, c)] if c == k else [-u(w, c)])
                if arc_free:
                    cnf.append([v(w, c)] if c == k else [-v(w, c)])
                elif c >= s:
                    cnf.append([-v(w, c)])
    return cnf, u, v


def _child(conn, N, out, r, fix):
    from pysat.solvers import Solver

    cnf, u, v = _encode(N, out, r, fix)
    with Solver(name="cadical153", bootstrap_with=cnf) as s:
        if not s.solve():
            conn.send(None)
            return
        model = set(x for x in s.get_model() if x > 0)
    U = [[1 if u(p, k) in model else 0 for k in range(r)] for p in range(N)]
    V = [[1 if v(q, k) in model else 0 for k in range(r)] for q in range(N)]
    rows = []
    for p in range(N):
        row = 0
        for q in range(N):
            if sum(U[p][k] & V[q][k] for k in range(r)) & 1:
                row |= 1 << q
        rows.append(row)
    conn.send(rows)


_CTX = None


def _context():
    """A forkserver started once (with this module preloaded): each SAT call then costs milliseconds to start,
    not the ~0.6 s of a fresh interpreter, and the child can still be killed at its time limit."""
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context("forkserver")
        _CTX.set_forkserver_preload(["lambda_agent.upper"])
    return _CTX


def minrank_sat(h: Digraph, r: int, time_limit: float, fix: list[int] | None = None):
    """For h = G^(t): fitting rows of rank <= r (a code of rate r/t), False if none exists, None on timeout."""
    if r < 1:
        return False
    if fix is not None:
        fix = sorted(fix)[:r]
        assert is_acyclic(h, fix)
    ctx = _context()
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_child, args=(child, h.n, list(h.out), r, fix), daemon=True)
    p.start()
    child.close()
    try:
        if not parent.poll(max(0.05, time_limit)):
            p.kill()
            return None
        rows = parent.recv()
    finally:
        parent.close()
        p.join()
    if rows is None:
        return False
    if fitting_rank(h, rows) > r:
        raise RuntimeError("SAT model above the target rank")
    return rows


def lift_fix(acyclic_g: list[int], t: int) -> list[int]:
    """An acyclic set I of G gives the acyclic set I x [t] of G^(t)."""
    return [v * t + a for v in acyclic_g for a in range(t)]


def timed(f, *a, **k):
    t0 = time.perf_counter()
    out = f(*a, **k)
    return out, time.perf_counter() - t0
