"""Exact solving of small pieces (used by the structure worker once the graph has been split).

  acyclic_exact(m, arcs, limit)   largest acyclic set of a kernel graph (CP-SAT ordering model); anytime
  cover_exact(adj, limit)         smallest clique cover of one mutual component (CP-SAT colouring model)
  lambda_exact(adj, budget)       the full lambda-agent pipeline (Shannon LP lower bound, vector-linear GF(2) codes
                                  by SAT on blow-ups) for SCCs of at most 10 receivers: lambda_2, often exactly
"""
from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from . import kernels as K
from .graph import Graph

LAMBDA_MAX_N = 10


def acyclic_exact(m: int, arcs, limit: float):
    """Returns (size, kernel-local set, proven optimal)."""
    if m == 0:
        return 0, [], True
    arcset = set(map(tuple, arcs))
    model = cp_model.CpModel()
    y = [model.NewBoolVar("") for _ in range(m)]
    pos = [model.NewIntVar(0, m - 1, "") for _ in range(m)]
    for u, v in arcset:
        if u == v:
            model.Add(y[u] == 0)
        elif (v, u) in arcset:
            if u < v:
                model.AddBoolOr([y[u].Not(), y[v].Not()])
        else:
            model.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
    model.Maximize(sum(y))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(0.05, limit)
    s.parameters.num_workers = 1
    st = s.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0, [], False
    chosen = [v for v in range(m) if s.Value(y[v])]
    return len(chosen), chosen, st == cp_model.OPTIMAL


def cover_exact(adj: np.ndarray, limit: float):
    """adj: adjacency of one mutual component. Returns (number of cliques, colour per receiver, proven optimal)."""
    n = adj.shape[0]
    mutual = adj & adj.T
    if n == 1 or mutual.sum() == n * (n - 1):
        return 1, np.zeros(n, dtype=np.int32), True
    g = Graph(adj)
    col, k = K.dsatur(g.conf_ptr, g.conf, 0.0)
    model = cp_model.CpModel()
    x = [[model.NewBoolVar("") for _ in range(k)] for _ in range(n)]
    used = [model.NewBoolVar("") for _ in range(k)]
    for v in range(n):
        model.AddExactlyOne(x[v])
        for c in range(k):
            model.AddImplication(x[v][c], used[c])
        model.AddHint(x[v][int(col[v])], 1)
    for u in range(n):
        for e in range(g.conf_ptr[u], g.conf_ptr[u + 1]):
            v = int(g.conf[e])
            if u < v:
                for c in range(k):
                    model.AddBoolOr([x[u][c].Not(), x[v][c].Not()])
    for c in range(k - 1):
        model.AddImplication(used[c + 1], used[c])
    model.Minimize(sum(used))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(0.05, limit)
    s.parameters.num_workers = 1
    st = s.Solve(model)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return k, col, False
    out = np.array([next(c for c in range(k) if s.Value(x[v][c])) for v in range(n)], dtype=np.int32)
    kk = len(set(out.tolist()))
    if kk > k:
        return k, col, False
    return kk, out, st == cp_model.OPTIMAL


def _lambda_agent():
    here = Path(__file__).resolve().parents[2] / "lambda-agent"
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    from lambda_agent.agent import LambdaAgent
    from lambda_agent.graph import Digraph

    return LambdaAgent, Digraph


def lambda_exact(adj: np.ndarray, budget: float):
    """lambda-agent on a small SCC. Returns (lower, upper, code) with lower <= beta <= lambda_2 <= upper (Fractions);
    code = (t, rows) for the best linear code found (rows of the fitting matrix of the blow-up), or None."""
    LambdaAgent, Digraph = _lambda_agent()
    n = adj.shape[0]
    i, j = np.nonzero(adj)
    d = Digraph.from_arcs(n, list(zip(i.tolist(), j.tolist())), "piece")
    a = LambdaAgent(d, budget=budget).run()
    code = None
    for c in a.comps:
        if getattr(c, "code", None) is not None:
            code = c.code
    return Fraction(a.lo), Fraction(a.hi), code
