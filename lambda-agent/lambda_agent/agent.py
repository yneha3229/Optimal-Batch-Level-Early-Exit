"""The lambda agent: bound lambda_2(G), the best rate of any linear code over GF(2) (scalar or vector), by
squeezing it between certified lower and upper bounds, one strongly connected component at a time.

    lower:  lambda_2 >= beta >= max(|acyclic set|, Shannon LP)
    upper:  lambda_2 <= min(clique cover, fractional clique cover, r / t for every code found by SAT on G^(t))

Plan per component (each step logged with its reason):
  1. acyclic set and clique cover (cheap, CP-SAT). If they meet, done.
  2. Shannon LP (fractional lower bound) and fractional clique cover. If they meet, done.
  3. LP-guided search for linear codes: the LP says beta >= p/q, so the first question is the most informative one,
     "is there a code with block length q and p transmissions?" (if yes, lambda = beta = p/q is proven). Then the
     remaining targets r/t (t = 1..T) between the lower and the upper bound, smallest rate first. Each SAT call
     fixes the blown-up acyclic set (symmetry breaking) and gets a short time slice; timeouts are retried later with
     a doubled slice. UNSAT only rules out that (t, r), never a rate, because a longer block can still do better.
  Stop when the bounds meet or the budget is spent.
Totals add up over components (cycles and linear-code constraints never cross a component boundary).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction

from .graph import Digraph, blowup, induced, scc
from .lower import SHANNON_MAX_N, mais, shannon
from .upper import clique_cover, frac_cover, lift_fix, minrank_sat
from .verify import fitting_rank

MAX_T = 4


@dataclass
class Comp:
    verts: list[int]
    g: Digraph
    lo: Fraction = Fraction(0)
    hi: Fraction = Fraction(0)
    lo_by: str = ""
    hi_by: str = ""
    acyclic: list[int] = field(default_factory=list)
    code: tuple | None = None   # (t, rows of G^(t)) of the best SAT code


@dataclass
class Step:
    t: float
    comp: int
    tool: str
    why: str
    result: str
    lo: str
    hi: str


class LambdaAgent:
    def __init__(self, g: Digraph, budget: float = 60.0, max_t: int = MAX_T, use_fix: bool = True):
        self.g, self.budget, self.max_t, self.use_fix = g, budget, max_t, use_fix
        self.t0 = 0.0
        self.steps: list[Step] = []
        self.comps: list[Comp] = []

    def left(self) -> float:
        return self.budget - (time.perf_counter() - self.t0)

    def log(self, k: int, tool: str, why: str, result: str) -> None:
        c = self.comps[k]
        self.steps.append(Step(round(time.perf_counter() - self.t0, 3), k, tool, why, result, str(c.lo), str(c.hi)))

    @property
    def lo(self) -> Fraction:
        return sum((c.lo for c in self.comps), Fraction(0))

    @property
    def hi(self) -> Fraction:
        return sum((c.hi for c in self.comps), Fraction(0))

    def run(self) -> "LambdaAgent":
        self.t0 = time.perf_counter()
        self.comps = [Comp(c, induced(self.g, c)) for c in scc(self.g)]
        for k, c in enumerate(self.comps):
            self._cheap(k, c)
        pending = []  # (rate, t, r, comp, slice) still to try
        for k, c in enumerate(self.comps):
            if c.lo < c.hi:
                pending += self._targets(k, c)
        pending.sort()
        slice_s = 1.0
        while pending and self.left() > 0.05:
            retry = []
            for rate, t, r, k, _ in pending:
                c = self.comps[k]
                if c.lo >= c.hi or rate >= c.hi or self.left() <= 0.05:
                    continue
                h = blowup(c.g, t)
                fix = lift_fix(c.acyclic, t) if self.use_fix else None
                why = (f"The LP says beta >= {c.lo}: a code with block length {t} and {r} transmissions would prove "
                       f"lambda = {rate}." if rate == c.lo else
                       f"Look for a linear code of rate {r}/{t} = {float(rate):.3f} (current best {c.hi}).")
                res = minrank_sat(h, r, min(slice_s, self.left()), fix)
                if res is None:
                    retry.append((rate, t, r, k, slice_s))
                    self.log(k, f"sat(t={t}, r={r})", why, f"no answer within {slice_s:.1f} s; retry later")
                elif res is False:
                    self.log(k, f"sat(t={t}, r={r})", why, f"no code with block length {t} and {r} transmissions")
                else:
                    if fitting_rank(h, res) > r:
                        raise RuntimeError("bad certificate")
                    c.hi, c.hi_by, c.code = rate, f"linear code t={t}", (t, res)
                    self.log(k, f"sat(t={t}, r={r})", why, f"found a vector-linear code of rate {rate}")
            pending = sorted(retry)
            slice_s *= 2
        return self

    def _cheap(self, k: int, c: Comp) -> None:
        g = c.g
        if g.n == 1:
            c.lo = c.hi = Fraction(1)
            c.acyclic, c.lo_by, c.hi_by = [0], "single receiver", "single receiver"
            return
        c.acyclic, _ = mais(g, min(10.0, max(0.5, self.left() / 4)))
        c.lo, c.lo_by = Fraction(len(c.acyclic)), "acyclic set"
        groups, _ = clique_cover(g, min(10.0, max(0.5, self.left() / 4)))
        c.hi, c.hi_by = Fraction(len(groups)), "clique cover"
        self.log(k, "acyclic set + clique cover", "cheap integer bounds first",
                 f"{len(c.acyclic)} <= beta <= {len(groups)} on a component of {g.n} receivers")
        if c.lo == c.hi:
            return
        if g.n <= SHANNON_MAX_N:
            s = shannon(g)
            if s > c.lo:
                c.lo, c.lo_by = s, "Shannon LP"
            self.log(k, "Shannon LP", "a fractional lower bound can sit between the integers", f"beta >= {s}")
        f, _, _ = frac_cover(g)
        if f < c.hi:
            c.hi, c.hi_by = f, "fractional clique cover"
        self.log(k, "fractional clique cover", "time sharing over cliques is a vector-linear code", f"rate {f}")

    def _targets(self, k: int, c: Comp) -> list[tuple]:
        out = []
        for t in range(1, self.max_t + 1):
            r0 = math.ceil(c.lo * t)
            for r in range(r0, math.ceil(c.hi * t)):
                rate = Fraction(r, t)
                if rate < c.hi and rate >= c.lo:
                    out.append((rate, t, r, k, 0))
        # dedupe equal rates, keep the smallest block length (cheapest CNF) for each rate
        best = {}
        for x in out:
            if x[0] not in best or x[1] < best[x[0]][1]:
                best[x[0]] = x
        return sorted(best.values())

    def result(self) -> dict:
        return {"n": self.g.n, "components": [len(c.verts) for c in self.comps], "lo": str(self.lo),
                "hi": str(self.hi), "exact": self.lo == self.hi, "seconds": round(time.perf_counter() - self.t0, 3),
                "steps": len(self.steps)}
