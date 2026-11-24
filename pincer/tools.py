"""The controller's tools. Each tool works on one SCC within a time slice and reports through the checked
`CompState.offer_*` methods, so a buggy or adversarial tool cannot corrupt a bound (it can only waste time).

| tool          | moves            | anytime | notes                                                            |
|---------------|------------------|---------|------------------------------------------------------------------|
| quick         | L, U             | no      | greedy FVS + DSatur + I + A + cycle-of-cliques cover; first   |
| mais_sa       | L                | yes     | simulated annealing on topological orders                        |
| mais_lazy     | L (+ MAIS bound) | yes     | CP-SAT with lazy cycle cuts, proves MAIS upper bounds            |
| mais_order    | L (+ MAIS bound) | yes     | CP-SAT ordering model (minrank/exact/lower.py)                   |
| cover_tabu    | U                | yes     | TabuCol clique cover                                             |
| partial_cover | U                | yes     | clique cover + disjoint cycles of cliques                        |
| sat_down      | U, closes mr     | no      | is minrk <= U-1 ? (indep. set symmetry breaking)                 |
| sat_up        | L_mr, closes mr  | no      | is minrk <= L ?  UNSAT raises L_mr                               |
| shannon       | L_beta           | no      | polymatroid LP, n <= 12, exact dual certificate                  |
| frac_cover    | U_beta           | yes     | fractional clique cover, column generation                       |
| block_lns     | U                | yes     | block-minrank LNS (optional)                                     |
| evolved:*     | L or U           | yes     | programs from pincer/funsearch (registered at run time)          |
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from minrank.baselines.clique_cover import clique_cover_rows
from minrank.baselines.identity_adj import identity_plus_adjacency_rows
from minrank.baselines.partial_cover import partial_cover_rows
from minrank.exact.lower import mais as mais_order_cpsat
from minrank.exact.symmetry import rank_at_most_sym

from .lp import SHANNON_MAX_N, SHANNON_SYM_MAX_N, frac_clique_cover, shannon_bound, shannon_bound_sym
from .mais import exact_lazy, greedy_fvs, sa_order
from .state import CompState


@dataclass
class Outcome:
    status: str          # "improved" | "no_change" | "timeout" | "unsat" | "done"
    seconds: float
    gain: float = 0.0    # reduction of CompState.gap()


@dataclass
class Tool:
    name: str
    kind: str                                   # "lower" | "upper" | "beta_lower" | "beta_upper" | "both"
    anytime: bool                               # result improves smoothly with the slice
    run: Callable[[CompState, float, random.Random, float], str]
    useful: Callable[[CompState], bool]
    once: bool = False                          # nothing to gain from a second call (deterministic)
    max_n: int | None = None
    description: str = ""


def _mais_useful(s: CompState) -> bool:
    k = len(s.acyclic)
    if k >= s.mais_ub:
        return False
    return (k + 1 > s.mr_lo and not s.mr_closed) or (k + 1 > s.b_lo and not s.b_closed)


def _upper_useful(s: CompState) -> bool:
    return not s.mr_closed or (not s.b_closed and s.b_hi == s.mr_hi)


# ---------------------------------------------------------------------------------------------- tool bodies
def _quick(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    g = s.comp.graph
    ch = s.offer_acyclic(greedy_fvs(g), "quick", t)
    nrng = np.random.default_rng(rng.randrange(2**31))
    ch |= s.offer_rows(clique_cover_rows(g, "dsatur", budget, nrng), "quick", t)
    ch |= s.offer_rows(identity_plus_adjacency_rows(g), "quick", t)  # I + A: one line, optimal on some graphs
    if g.directed and budget > 0.05:
        ch |= s.offer_rows(partial_cover_rows(g, min(0.5, budget), nrng), "quick", t)
    if g.n == 1:
        s.offer_mais_ub(1, "quick", t)
    return "improved" if ch else "no_change"


def _mais_sa(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    v = sa_order(s.comp.graph, budget, rng, init=s.acyclic if len(s.acyclic) > 1 else None)
    return "improved" if s.offer_acyclic(v, "mais_sa", t) else "no_change"


def _mais_lazy(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    r = exact_lazy(s.comp.graph, budget, init=s.acyclic if len(s.acyclic) > 1 else None)
    ch = s.offer_acyclic(r.verts, "mais_lazy", t)
    before = s.mais_ub
    s.offer_mais_ub(r.upper, "mais_lazy", t)
    if r.proven:
        return "done"
    return "improved" if ch or s.mais_ub < before else "timeout"


def _mais_order(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    r = mais_order_cpsat(s.comp.graph, budget, seed=rng.randrange(1000))
    ch = s.offer_acyclic(r.vertices, "mais_order", t)
    before = s.mais_ub
    s.offer_mais_ub(r.upper, "mais_order", t)
    if r.proven_max:
        s.offer_mais_ub(r.value, "mais_order", t)
        return "done"
    return "improved" if ch or s.mais_ub < before else "timeout"


def _cover_tabu(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    rows = clique_cover_rows(s.comp.graph, "tabucol", budget, np.random.default_rng(rng.randrange(2**31)))
    return "improved" if s.offer_rows(rows, "cover_tabu", t) else "no_change"


def _partial(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    rows = partial_cover_rows(s.comp.graph, budget, np.random.default_rng(rng.randrange(2**31)))
    return "improved" if s.offer_rows(rows, "partial_cover", t) else "no_change"


def _fix_set(s: CompState, r: int, question: tuple, rng: random.Random) -> list[int]:
    """The acyclic set the SAT call fixes (independent-set symmetry breaking). SAT time depends heavily on which
    maximum set is fixed (planted n = 100, r = 10, five maximum independent sets: 1.6 s to > 60 s), so every retry
    of the same question uses a different set of the same size when a short SA run with a new seed finds one: a
    restart portfolio against heavy-tailed run times. Any acyclic subset is sound for the fix."""
    k = s.sat_attempts.get(question, 0)
    s.sat_attempts[question] = k + 1
    best = sorted(s.acyclic)
    if k > 0:
        alt = sa_order(s.comp.graph, min(0.3, 0.02 * s.n), random.Random(rng.randrange(2**31)),
                       init=greedy_fvs(s.comp.graph))
        if len(alt) >= len(best) and sorted(alt) != best:
            best = sorted(alt)
    return best[:r]


def _sat_down(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    r = s.mr_hi - 1
    if r < s.mr_lo:
        return "done"
    I = _fix_set(s, r, ("down", r), rng)
    t0 = time.perf_counter()
    res = rank_at_most_sym(s.comp.graph, r, time_limit=budget, mode="indep_cols", indep=I)
    if res is None:
        return "timeout"
    if res is False:
        s.offer_unsat(r, {"mode": "indep_cols", "indep": I, "seconds": round(time.perf_counter() - t0, 3)},
                      "sat_down", t)
        return "unsat"
    s.offer_rows(list(res), "sat_down", t)
    return "improved"


def _sat_up(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    r = s.mr_lo
    if r >= s.mr_hi:
        return "done"
    I = _fix_set(s, r, ("up", r), rng)
    t0 = time.perf_counter()
    res = rank_at_most_sym(s.comp.graph, r, time_limit=budget, mode="indep_cols", indep=I)
    if res is None:
        return "timeout"
    if res is False:
        s.offer_unsat(r, {"mode": "indep_cols", "indep": I, "seconds": round(time.perf_counter() - t0, 3)},
                      "sat_up", t)
        return "unsat"
    s.offer_rows(list(res), "sat_up", t)
    return "improved"


def _shannon(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    """Full LP for small asymmetric SCCs; the orbit-reduced LP when the SCC has nontrivial automorphisms (always
    faster) or is too large for the full LP (12 < n <= 16)."""
    from minrank.exact.symmetry import automorphisms

    g = s.comp.graph
    gens = automorphisms(g, max_gens=16) if g.n > 3 else []
    if gens and g.n <= SHANNON_SYM_MAX_N:
        r = shannon_bound_sym(g, gens)
        s.offer_shannon(r.bound, r.y, r.mu, "shannon", t, recheck=False, gens=r.gens)
    elif g.n <= SHANNON_MAX_N:
        r = shannon_bound(g)
        s.offer_shannon(r.bound, r.y, r.mu, "shannon", t, recheck=False)  # certified_bound already exact
    return "done"


def _frac(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    groups: dict[int, list[int]] = {}
    for v, row in enumerate(s.rows):
        groups.setdefault(row, []).append(v)
    init = list(groups.values())
    r = frac_clique_cover(s.comp.graph, budget, init_cliques=init)
    ch = s.offer_frac(r.cliques, r.weights, "frac_cover", t)
    if r.optimal:
        return "done"
    return "improved" if ch else "timeout"


def _block_lns(s: CompState, budget: float, rng: random.Random, t: float) -> str:
    from .block_lns import block_lns

    rows, _ = block_lns(s.comp.graph, budget, rng)
    return "improved" if s.offer_rows(rows, "block_lns", t) else "no_change"


def evolved_mais_tool(path: str) -> Tool:
    """A FunSearch program (reviewed and committed) as a controller tool: evolved greedy, then SA from it."""
    from pathlib import Path

    from .funsearch.evaluate import load_priority

    prio = load_priority(Path(path).read_text())

    def run(s: CompState, budget: float, rng: random.Random, t: float) -> str:
        t0 = time.perf_counter()
        v = greedy_fvs(s.comp.graph, prio)
        ch = s.offer_acyclic(v, "mais_evolved", t)
        left = budget - (time.perf_counter() - t0)
        if left > 0.05:
            v = sa_order(s.comp.graph, left, rng, init=s.acyclic if len(s.acyclic) > len(v) else v)
            ch |= s.offer_acyclic(v, "mais_evolved", t)
        return "improved" if ch else "no_change"

    return Tool("mais_evolved", "lower", True, run, _mais_useful,
                description="FunSearch-evolved greedy acyclic set, then simulated annealing from it")


def default_tools(include_lns: bool = False, evolved: str | None = None) -> dict[str, Tool]:
    tools = [
        Tool("quick", "both", False, _quick, lambda s: True, once=True,
             description="greedy acyclic set + DSatur clique cover + cycle-of-cliques cover (milliseconds)"),
        Tool("mais_sa", "lower", True, _mais_sa, _mais_useful,
             description="simulated annealing for a larger acyclic set (raises L)"),
        Tool("mais_lazy", "lower", True, _mais_lazy, _mais_useful,
             description="exact MAIS, CP-SAT with lazy cycle cuts; also proves MAIS upper bounds (best on sparse)"),
        Tool("mais_order", "lower", True, _mais_order, _mais_useful,
             description="exact MAIS, CP-SAT ordering model (best on dense graphs)"),
        Tool("cover_tabu", "upper", True, _cover_tabu, _upper_useful,
             description="TabuCol clique cover (lowers U)"),
        Tool("partial_cover", "upper", True, _partial, lambda s: s.comp.graph.directed and _upper_useful(s),
             description="clique cover plus disjoint cycles of cliques, directed graphs (lowers U)"),
        Tool("sat_down", "upper", False, _sat_down, lambda s: not s.mr_closed,
             description="SAT: is minrank <= U-1? SAT lowers U, UNSAT closes the minrank interval"),
        Tool("sat_up", "lower", False, _sat_up, lambda s: not s.mr_closed,
             description="SAT: is minrank <= L? UNSAT raises the minrank lower bound, SAT closes it"),
        Tool("shannon", "beta_lower", False, _shannon, lambda s: not s.b_closed and s.shannon is None,
             once=True, max_n=SHANNON_SYM_MAX_N,
             description="polymatroid LP lower bound on beta (fractional; n <= 12; closes odd-cycle-type gaps)"),
        Tool("frac_cover", "beta_upper", True, _frac, lambda s: not s.b_closed,
             description="fractional clique cover upper bound on beta (vector-linear codes)"),
    ]
    if evolved:
        tools.append(evolved_mais_tool(evolved))
    if include_lns:
        tools.append(Tool("block_lns", "upper", True, _block_lns, _upper_useful,
                          description="merge blocks of the cover and re-solve them exactly by SAT"))
    return {t.name: t for t in tools}


def predicted_seconds(tool: Tool, s: CompState) -> float:
    """Rough cost model for one-shot tools (used for budget checks, not for correctness)."""
    if tool.name == "shannon":
        return 0.004 * 2 ** max(0, s.n - 1) if s.n <= SHANNON_MAX_N else math.inf
    if tool.name == "quick":
        return 0.001 * s.n + 0.05
    return 0.0
