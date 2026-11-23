"""Per-component bounds and the certificates behind them. Nothing enters the state without being checked.

Two intervals per strongly connected component (SCC):
- minrk_2 (best scalar linear F_2 code, an integer): mr_lo <= minrk_2 <= mr_hi.
    mr_hi: a fitting matrix (checked: fit + exact F_2 rank).
    mr_lo: max of |acyclic set| (checked: acyclic), ceil(Shannon bound) (checked: exact dual), r + 1 for a
           SAT call that returned UNSAT at rank r (trusted to the SAT solver; the CNF can be exported for DRAT).
- beta (broadcast rate, a rational): b_lo <= beta <= b_hi.
    b_lo: max of |acyclic set|, Shannon bound. (UNSAT results are NOT lower bounds on beta: they only rule out
          scalar linear F_2 codes.)
    b_hi: min of mr_hi and the fractional clique cover (checked: exact coverage).
Totals over components are sums (see pincer/decompose.py for why every quantity is additive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction

from minrank.core.f2 import rank_f2
from minrank.core.verify import verify
from minrank.exact.lower import is_acyclic_induced

from .decompose import Component
from .lp import check_frac_cover


@dataclass
class Event:
    t: float          # seconds since the start of the run
    tool: str
    what: str         # e.g. "mr_hi 21->18"


@dataclass
class CompState:
    comp: Component
    mr_lo: int = 1
    mr_hi: int = 0
    b_lo: Fraction = Fraction(1)
    b_hi: Fraction = Fraction(0)
    rows: list[int] = field(default_factory=list)          # local fitting matrix achieving mr_hi
    acyclic: list[int] = field(default_factory=list)       # local acyclic set
    mais_ub: int = 0                                       # proven upper bound on MAIS (n if unknown)
    shannon: dict | None = None                            # {"bound": Fraction, "y": {...}, "mu": {...}}
    frac: dict | None = None                               # {"value": Fraction, "cliques": [...], "weights": [...]}
    unsat: list[dict] = field(default_factory=list)        # [{"r": r, "mode": ..., "indep": [...], "seconds": s}]
    sources: dict = field(default_factory=dict)            # which tool set each bound
    events: list[Event] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = self.comp.graph.n
        if not self.rows:
            self.rows = [1 << i for i in range(n)]  # identity: rank n, always fits
            self.mr_hi = n
            self.b_hi = Fraction(n)
        if not self.acyclic:
            self.acyclic = [0]
        self.mais_ub = n
        if n == 1:
            self.mais_ub = 1

    # ---------------------------------------------------------------- views
    @property
    def n(self) -> int:
        return self.comp.graph.n

    @property
    def mr_closed(self) -> bool:
        return self.mr_lo >= self.mr_hi

    @property
    def b_closed(self) -> bool:
        return self.b_lo >= self.b_hi

    @property
    def closed(self) -> bool:
        return self.mr_closed and self.b_closed

    def gap(self) -> float:
        """Objective the controller drives to 0: minrank gap + beta gap."""
        return float(self.mr_hi - self.mr_lo) + float(self.b_hi - self.b_lo)

    # ---------------------------------------------------------------- offers (all checked)
    def _log(self, t: float, tool: str, what: str) -> None:
        self.events.append(Event(round(t, 3), tool, what))

    def offer_rows(self, rows: list[int], tool: str, t: float) -> bool:
        r = verify(self.comp.graph, rows)
        if r < self.mr_hi:
            self._log(t, tool, f"mr_hi {self.mr_hi}->{r}")
            self.mr_hi, self.rows = r, list(rows)
            self.sources["mr_hi"] = tool
            self._set_b_hi(Fraction(r), tool, t)
            return True
        return False

    def offer_acyclic(self, verts: list[int], tool: str, t: float) -> bool:
        verts = sorted(set(int(v) for v in verts))
        if not is_acyclic_induced(self.comp.graph, verts):
            raise ValueError(f"{tool} offered a cyclic vertex set")
        if len(verts) <= len(self.acyclic):
            return False
        self._log(t, tool, f"mais {len(self.acyclic)}->{len(verts)}")
        self.acyclic = verts
        if len(verts) > self.mr_lo:
            self.mr_lo = len(verts)
            self.sources["mr_lo"] = tool
        self._set_b_lo(Fraction(len(verts)), tool, t)
        return True

    def offer_mais_ub(self, ub: int, tool: str, t: float) -> None:
        if ub < self.mais_ub:
            self.mais_ub = max(ub, len(self.acyclic))
            self._log(t, tool, f"mais_ub {self.mais_ub}")

    def offer_unsat(self, r: int, record: dict, tool: str, t: float) -> bool:
        """SAT solver proved minrk_2 > r."""
        self.unsat.append({"r": r, **record})
        if r + 1 > self.mr_lo:
            self._log(t, tool, f"mr_lo {self.mr_lo}->{r + 1}")
            self.mr_lo = r + 1
            self.sources["mr_lo"] = tool
            return True
        return False

    def offer_shannon(self, bound: Fraction, y: dict, mu: dict, tool: str, t: float, recheck: bool = True) -> bool:
        if recheck:
            from .lp import check_shannon_certificate

            if check_shannon_certificate(self.comp.graph, y, mu) < bound:
                raise ValueError("Shannon certificate does not reproduce the bound")
        self.shannon = {"bound": bound, "y": y, "mu": mu}
        ch = self._set_b_lo(bound, tool, t)
        c = math.ceil(bound)
        if c > self.mr_lo:
            self._log(t, tool, f"mr_lo {self.mr_lo}->{c}")
            self.mr_lo = c
            self.sources["mr_lo"] = tool
            ch = True
        return ch

    def offer_frac(self, cliques: list[list[int]], weights: list[Fraction], tool: str, t: float) -> bool:
        value = check_frac_cover(self.comp.graph, cliques, weights)
        if self.frac is None or value < self.frac["value"]:
            self.frac = {"value": value, "cliques": cliques, "weights": weights}
        return self._set_b_hi(value, tool, t)

    def _set_b_lo(self, v: Fraction, tool: str, t: float) -> bool:
        if v > self.b_lo:
            self._log(t, tool, f"b_lo {float(self.b_lo):.4g}->{float(v):.4g}")
            self.b_lo = v
            self.sources["b_lo"] = tool
            return True
        return False

    def _set_b_hi(self, v: Fraction, tool: str, t: float) -> bool:
        if v < self.b_hi:
            self._log(t, tool, f"b_hi {float(self.b_hi):.4g}->{float(v):.4g}")
            self.b_hi = v
            self.sources["b_hi"] = tool
            return True
        return False

    def summary(self) -> dict:
        return {"comp": self.comp.index, "n": self.n, "mr": [self.mr_lo, self.mr_hi],
                "beta": [str(self.b_lo), str(self.b_hi)], "mais": len(self.acyclic), "mais_ub": self.mais_ub,
                "closed_mr": self.mr_closed, "closed_beta": self.b_closed}


def check_rank(rows: list[int]) -> int:
    return rank_f2(rows)
