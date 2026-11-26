"""Maximum acyclic induced subgraph (MAIS), the lower bound L <= beta <= minrk.

Three solvers, all returning a vertex set that `is_acyclic_induced` re-checks:

- `greedy_fvs(g, priority)`: the classic feedback-vertex-set reduction loop. Repeatedly (1) move every vertex with
  no in-arc or no out-arc among the remaining vertices into the acyclic set (it cannot be on a remaining cycle),
  (2) otherwise delete the remaining vertex with the highest `priority`. Then re-insert deleted vertices greedily.
  `priority(v, ctx)` is the function FunSearch evolves (see pincer/funsearch/specs.py); the default is the classic
  in-degree x out-degree rule.
- `sa_order(g, ...)`: simulated annealing on a topological order of the kept set (the move set of Galinier, Lemamou
  and Bouzidi, "Applying local search to the feedback vertex set problem", J. Heuristics 2013): insert an outside
  vertex just after its last kept in-neighbour or just before its first kept out-neighbour and drop the vertices
  that then point backwards.
- `exact_lazy(g, ...)`: CP-SAT on y_v with cycle constraints added lazily (sum over a cycle of y <= len - 1), warm
  started. Returns the best acyclic set found and the solver's upper bound on MAIS (so the controller knows when
  MAIS cannot raise L any further).

Arc convention (minrank): bit j of out[i] means receiver i knows x_j, arc i -> j.
"""
from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from minrank.core.graph import Graph, iter_bits
from minrank.exact.lower import is_acyclic_induced


def in_masks(g: Graph) -> list[int]:
    inn = [0] * g.n
    for i in range(g.n):
        for j in iter_bits(g.out[i]):
            inn[j] |= 1 << i
    return inn


@dataclass
class MaisResult:
    verts: list[int]
    upper: int          # proven upper bound on MAIS (n if unknown)
    proven: bool        # verts is a maximum acyclic set
    seconds: float

    @property
    def value(self) -> int:
        return len(self.verts)


# ------------------------------------------------------------------------------------------- greedy (FVS)
class Ctx:
    """What a priority function may read. All fields are plain ints / lists (safe to hand to a sandboxed program).

    n, out, inn: the whole graph (bitmasks). alive: bitmask of undecided vertices. kept: bitmask of vertices already
    in the acyclic set. deleted: bitmask of deleted vertices. Helpers give degrees inside `alive`.
    """

    __slots__ = ("n", "out", "inn", "alive", "kept", "deleted", "step")

    def __init__(self, n: int, out: list[int], inn: list[int]) -> None:
        self.n, self.out, self.inn = n, out, inn
        self.alive = (1 << n) - 1
        self.kept = 0
        self.deleted = 0
        self.step = 0

    def out_deg(self, v: int) -> int:
        return (self.out[v] & self.alive).bit_count()

    def in_deg(self, v: int) -> int:
        return (self.inn[v] & self.alive).bit_count()

    def mutual_deg(self, v: int) -> int:
        return (self.out[v] & self.inn[v] & self.alive).bit_count()


def default_priority(v: int, ctx: Ctx) -> float:
    return ctx.in_deg(v) * ctx.out_deg(v)


def _reachable(start: int, allowed: int, out: list[int]) -> int:
    """Bitmask of vertices reachable from the bitmask `start` inside `allowed` (start included if allowed)."""
    seen = start & allowed
    frontier = seen
    while frontier:
        nxt = 0
        for u in iter_bits(frontier):
            nxt |= out[u]
        nxt &= allowed & ~seen
        seen |= nxt
        frontier = nxt
    return seen


def can_add(v: int, kept: int, out: list[int], inn: list[int]) -> bool:
    """True iff kept + v is still acyclic (no kept path from an out-neighbour of v back to an in-neighbour)."""
    if out[v] & inn[v] & kept:
        return False
    reach = _reachable(out[v] & kept, kept, out)
    return not (reach & inn[v])


def greedy_insert(kept: int, order: list[int], out: list[int], inn: list[int]) -> int:
    for v in order:
        if not (kept >> v) & 1 and can_add(v, kept, out, inn):
            kept |= 1 << v
    return kept


def greedy_fvs(g: Graph, priority: Callable[[int, Ctx], float] = default_priority,
               rng: random.Random | None = None) -> list[int]:
    n = g.n
    out = list(g.out)
    inn = in_masks(g)
    ctx = Ctx(n, out, inn)
    deleted_order: list[int] = []
    while ctx.alive:
        changed = True
        while changed and ctx.alive:
            changed = False
            for v in list(iter_bits(ctx.alive)):
                if not (out[v] & ctx.alive & ~(1 << v)) or not (inn[v] & ctx.alive & ~(1 << v)):
                    ctx.alive &= ~(1 << v)
                    ctx.kept |= 1 << v
                    changed = True
        if not ctx.alive:
            break
        best, best_p = -1, -math.inf
        for v in iter_bits(ctx.alive):
            p = float(priority(v, ctx))
            if rng is not None:
                p += rng.random() * 1e-9
            if p > best_p:
                best, best_p = v, p
        ctx.alive &= ~(1 << best)
        ctx.deleted |= 1 << best
        deleted_order.append(best)
        ctx.step += 1
    # The "kept" vertices above are sources/sinks at removal time, so kept is acyclic. Re-insert deleted vertices,
    # most recently deleted first (they were the least harmful).
    kept = greedy_insert(ctx.kept, deleted_order[::-1], out, inn)
    return [v for v in range(n) if (kept >> v) & 1]


# ------------------------------------------------------------------------------- simulated annealing
def sa_order(g: Graph, time_limit: float, rng: random.Random, init: list[int] | None = None,
             t0: float = 0.6, alpha: float = 0.99, moves_per_temp: int | None = None) -> list[int]:
    """Galinier-style SA on a topological order of the kept set. Returns the best acyclic set seen."""
    deadline = time.perf_counter() + time_limit
    n = g.n
    out = list(g.out)
    inn = in_masks(g)
    seq = _topo(init if init is not None else greedy_fvs(g), out)
    best = list(seq)
    T = t0
    mpt = moves_per_temp or max(50, 2 * n)
    fails = 0
    while time.perf_counter() < deadline:
        improved = False
        for _ in range(mpt):
            kept = set(seq)
            outside = [v for v in range(n) if v not in kept]
            if not outside:
                return list(range(n))
            v = rng.choice(outside)
            pos = {u: k for k, u in enumerate(seq)}
            # candidate positions: just after the last kept in-neighbour, just before the first kept out-neighbour
            ins = [pos[u] for u in iter_bits(inn[v]) if u in pos]
            outs = [pos[u] for u in iter_bits(out[v]) if u in pos]
            p_minus = (max(ins) + 1) if ins else 0      # all in-neighbours before v: they stay valid
            p_plus = min(outs) if outs else len(seq)   # all out-neighbours after v
            best_move = None
            for p in {p_minus, p_plus}:
                # u -> v must have u before v; v -> u must have v before u. Conflicts at insertion slot p:
                conf = [u for u in iter_bits(inn[v]) if u in pos and pos[u] >= p]
                conf += [u for u in iter_bits(out[v]) if u in pos and pos[u] < p]
                conf = list(set(conf))
                delta = len(conf) - 1  # vertices lost minus gained (lower is better)
                if best_move is None or delta < best_move[0]:
                    best_move = (delta, p, conf)
            delta, p, conf = best_move
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                cs = set(conf)
                new = [u for u in seq[:p] if u not in cs] + [v] + [u for u in seq[p:] if u not in cs]
                seq = new
                if len(seq) > len(best):
                    best = list(seq)
                    improved = True
            if time.perf_counter() >= deadline:
                break
        T *= alpha
        fails = 0 if improved else fails + 1
        if fails > 50:  # reheat
            T, fails = t0, 0
    return sorted(best)


def _topo(verts: list[int], out: list[int]) -> list[int]:
    """A topological order (arc u -> v means u before v) of an acyclic vertex set."""
    vs = set(verts)
    mask = sum(1 << v for v in vs)
    indeg = {v: (0) for v in vs}
    for u in vs:
        for w in iter_bits(out[u] & mask):
            indeg[w] += 1
    q = deque(sorted(v for v in vs if indeg[v] == 0))
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for w in iter_bits(out[u] & mask):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    if len(order) != len(vs):
        raise ValueError("vertex set is not acyclic")
    return order


# ------------------------------------------------------------------------------------- exact, lazy cycles
def shortest_cycle_through(v: int, allowed: int, out: list[int]) -> list[int] | None:
    """Shortest directed cycle through v inside `allowed` (BFS), as a vertex list, or None."""
    if not (allowed >> v) & 1:
        return None
    parent = {v: -1}
    q = deque([v])
    while q:
        u = q.popleft()
        for w in iter_bits(out[u] & allowed):
            if w == v:
                cyc = [u]
                while parent[cyc[-1]] != -1:
                    cyc.append(parent[cyc[-1]])
                return cyc[::-1]
            if w not in parent:
                parent[w] = u
                q.append(w)
    return None


def exact_lazy(g: Graph, time_limit: float, init: list[int] | None = None, workers: int = 1,
               max_cuts_per_round: int = 200) -> MaisResult:
    """MAIS by CP-SAT with lazily added cycle constraints. Anytime: returns the best acyclic set found and the
    best proven upper bound when time runs out."""
    from ortools.sat.python import cp_model

    t_start = time.perf_counter()
    deadline = t_start + time_limit
    n = g.n
    out = list(g.out)
    inn = in_masks(g)
    full = (1 << n) - 1
    best = sorted(init) if init is not None else greedy_fvs(g)
    upper = n
    cycles: list[list[int]] = []
    seen_cycles: set[tuple[int, ...]] = set()

    def add_cycle(c: list[int]) -> None:
        key = tuple(sorted(c))
        if key not in seen_cycles:
            seen_cycles.add(key)
            cycles.append(c)

    for v in range(n):  # seed: 2-cycles and one shortest cycle per vertex
        for u in iter_bits(out[v] & inn[v]):
            if u > v:
                add_cycle([v, u])
        c = shortest_cycle_through(v, full, out)
        if c:
            add_cycle(c)
    hint = set(best)
    while True:
        left = deadline - time.perf_counter()
        if left <= 0.02:
            break
        m = cp_model.CpModel()
        y = [m.NewBoolVar(f"y{v}") for v in range(n)]
        for c in cycles:
            m.Add(sum(y[v] for v in c) <= len(c) - 1)
        m.Add(sum(y) >= len(best))  # no worse than the incumbent
        m.Maximize(sum(y))
        for v in range(n):
            m.AddHint(y[v], 1 if v in hint else 0)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = left
        s.parameters.num_workers = workers
        st = s.Solve(m)
        if st == cp_model.INFEASIBLE:
            upper = len(best)  # nothing at least as large as the incumbent... except the incumbent itself
            break
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        relax_bound = int(s.BestObjectiveBound())
        upper = min(upper, relax_bound)
        sol = [v for v in range(n) if s.Value(y[v])]
        mask = sum(1 << v for v in sol)
        new_cuts = 0
        for v in sol:
            c = shortest_cycle_through(v, mask, out)
            if c:
                before = len(cycles)
                add_cycle(c)
                new_cuts += len(cycles) - before
                if new_cuts >= max_cuts_per_round:
                    break
        if new_cuts == 0:  # relaxation solution is acyclic
            if len(sol) > len(best):
                best = sol
            if st == cp_model.OPTIMAL:
                upper = len(best)
            break
        # repair the cyclic solution into an acyclic one (warm start + incumbent)
        rep = _repair(sol, out, inn)
        if len(rep) > len(best):
            best = rep
        hint = set(rep)
        if upper <= len(best):
            upper = len(best)
            break
    assert is_acyclic_induced(g, best)
    return MaisResult(sorted(best), upper, upper == len(best), time.perf_counter() - t_start)


def _repair(sol: list[int], out: list[int], inn: list[int]) -> list[int]:
    """Greedy: keep vertices of `sol` in order of fewest arcs inside sol, skipping any that would close a cycle."""
    mask = sum(1 << v for v in sol)
    order = sorted(sol, key=lambda v: (out[v] & mask).bit_count() * (inn[v] & mask).bit_count())
    kept = greedy_insert(0, order, out, inn)
    return [v for v in iter_bits(kept)]


def best_mais(g: Graph, time_limit: float, seed: int = 0) -> MaisResult:
    """Convenience: greedy, then SA for half the time, then exact for the rest (warm-started)."""
    t0 = time.perf_counter()
    rng = random.Random(seed)
    s = greedy_fvs(g)
    s = sa_order(g, time_limit * 0.3, rng, init=s)
    r = exact_lazy(g, max(0.05, time_limit - (time.perf_counter() - t0)), init=s)
    return MaisResult(r.verts, r.upper, r.proven, time.perf_counter() - t0)
