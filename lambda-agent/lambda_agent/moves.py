"""Moves: every method the controller can run, as a killable worker process that streams progress events.

A move works on ONE strongly connected component and sends events (seconds since its start, kind, payload):
  ("lower", value, certificate)   a lower bound on lambda_2 (acyclic set, or Shannon LP value)
  ("upper", value, certificate)   an explicit linear code (clique cover, fractional cover, or a SAT code matrix)
  ("no", None, None)              a SAT question answered "no such code" (rules out that (t, r) only)
  ("done", proven, None)          the move finished; proven = its bound is optimal for its method
The runtime re-checks every certificate before using it (see live.py), except the Shannon LP value, which is
computed by lower.shannon's exact weak-duality certificate inside the worker.

Move names:
  acyclic          largest acyclic set, CP-SAT, reports each better set as found (anytime)
  lp               Shannon LP lower bound (components with <= 10 receivers)
  cover            smallest clique cover, CP-SAT, reports each better cover as found (anytime)
  frac             fractional clique cover (components with <= 25 receivers)
  code:t:r         "is there a linear code with block length t and r transmissions?" (SAT on G^(t), with the
                   blown-up acyclic set fixed for symmetry breaking)
"""
from __future__ import annotations

import multiprocessing as mp
import time
from fractions import Fraction

from ortools.sat.python import cp_model

from .graph import Digraph, bits, blowup
from .lower import SHANNON_MAX_N, shannon
from .upper import _child as _sat_child  # noqa: F401  (keeps the SAT encoder importable in the forkserver)
from .upper import _encode, frac_cover, lift_fix

FRAC_MAX_N = 25


def _now(t0: float) -> float:
    return round(time.perf_counter() - t0, 4)


def _acyclic(g: Digraph, emit, t0, limit):
    n = g.n
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{v}") for v in range(n)]
    pos = [m.NewIntVar(0, max(0, n - 1), "") for _ in range(n)]
    for u in range(n):
        for v in bits(g.out[u]):
            if g.knows(v, u):
                if u < v:
                    m.AddBoolOr([y[u].Not(), y[v].Not()])
            else:
                m.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
    m.Maximize(sum(y))

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            emit(("lower", len([v for v in range(n) if self.Value(y[v])]), [v for v in range(n) if self.Value(y[v])]))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m, CB())
    emit(("done", st == cp_model.OPTIMAL, None))


def _cover(g: Digraph, emit, t0, limit):
    from .upper import _dsatur

    n = g.n
    conf = [[u != v and not g.mutual(u, v) for v in range(n)] for u in range(n)]
    start = _dsatur(n, conf)
    groups = {}
    for v, c in enumerate(start):
        groups.setdefault(c, []).append(v)
    emit(("upper", len(groups), list(groups.values())))
    k = max(start) + 1
    m = cp_model.CpModel()
    x = [[m.NewBoolVar("") for _ in range(k)] for _ in range(n)]
    used = [m.NewBoolVar("") for _ in range(k)]
    for v in range(n):
        m.AddExactlyOne(x[v])
        for c in range(k):
            m.AddImplication(x[v][c], used[c])
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

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            gr = {}
            for v in range(n):
                for c in range(k):
                    if self.Value(x[v][c]):
                        gr.setdefault(c, []).append(v)
            emit(("upper", len(gr), list(gr.values())))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m, CB())
    emit(("done", st == cp_model.OPTIMAL, None))


def _code(g: Digraph, emit, t0, t: int, r: int, fix_g: list[int]):
    from pysat.solvers import Solver

    h = blowup(g, t)
    fix = sorted(lift_fix(fix_g, t))[:r] if fix_g else None
    cnf, u, v = _encode(h.n, list(h.out), r, fix)
    with Solver(name="cadical153", bootstrap_with=cnf) as s:
        if not s.solve():
            emit(("no", None, None))
            emit(("done", True, None))
            return
        model = set(x for x in s.get_model() if x > 0)
    N = h.n
    rows = []
    for p in range(N):
        row = 0
        for q in range(N):
            if sum(1 for k in range(r) if u(p, k) in model and v(q, k) in model) & 1:
                row |= 1 << q
        rows.append(row)
    emit(("upper", str(Fraction(r, t)), {"t": t, "rows": rows}))
    emit(("done", True, None))


def run_move(name: str, g_json: dict, fix_g: list[int], limit: float, q) -> None:
    """Worker entry point: runs one move and puts (seconds, kind, value, certificate) tuples on queue q."""
    t0 = time.perf_counter()
    g = Digraph.from_json(g_json)

    def emit(ev):
        q.put((_now(t0),) + tuple(ev))

    try:
        if name == "acyclic":
            _acyclic(g, emit, t0, limit)
        elif name == "lp":
            emit(("lower", str(shannon(g)), None))
            emit(("done", True, None))
        elif name == "cover":
            _cover(g, emit, t0, limit)
        elif name == "frac":
            val, cl, w = frac_cover(g)
            emit(("upper", str(val), {"cliques": cl, "weights": [str(x) for x in w]}))
            emit(("done", True, None))
        elif name.startswith("code:"):
            _, t, r = name.split(":")
            _code(g, emit, t0, int(t), int(r), fix_g)
        else:
            raise ValueError(name)
    except Exception as e:  # report, never hang the runtime
        q.put((_now(t0), "error", repr(e)[:300], None))


def max_t_for(n: int) -> int:
    """Largest block length offered on a component of n receivers (the CNF grows with (n t)^2 r)."""
    return 4 if n <= 12 else 3 if n <= 20 else 2 if n <= 30 else 1


def available(name: str, n: int) -> bool:
    if name == "lp":
        return 1 < n <= SHANNON_MAX_N
    if name == "frac":
        return 1 < n <= FRAC_MAX_N
    return n > 1


_CTX = None


def context():
    global _CTX
    if _CTX is None:
        _CTX = mp.get_context("forkserver")
        _CTX.set_forkserver_preload(["lambda_agent.moves"])
    return _CTX
