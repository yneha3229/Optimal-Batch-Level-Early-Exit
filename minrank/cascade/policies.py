"""Stopping policies for evaluating a batch of candidate programs graph by graph.

Mapping from batch-level early exit (theory/THEORY.md) to candidate evaluation
-------------------------------------------------------------------------------
| early exit (encoder batch)          | evaluating evolved candidates                                  |
|-------------------------------------|----------------------------------------------------------------|
| batch of B inputs                   | batch of B candidate programs                                  |
| layers 1..L                         | graphs 1..L in the fixed evaluation order                      |
| exit e                              | checkpoint e: after the first checkpoints[e] graphs            |
| confidence of input j               | P(win | partial gap d_j, e) from the calibration table         |
| one input leaves                    | drop one candidate                                             |
| collective stop                     | stop the batch: no remaining candidate is evaluated further    |
| c(n) = a + b n per layer            | graph g on n live candidates costs a + b_g n                   |

Interface: `decide(ctx, e, prefix, alive) -> keep`, called after the first ctx.checkpoints[e] graphs (never after
the last one). prefix: (B, checkpoints[e]) scores of every candidate on the graphs evaluated so far (rows of dead
candidates are present but must be ignored); alive: (B,) bool. Returns keep (B,) bool with keep <= alive.
A policy sees only `prefix`, so it is causal by construction: later graphs cannot influence a decision.

Cost rule (ours), objective per batch: improvement of the incumbent - lambda * T / F, where T is the evaluation
time spent and F the time of evaluating the whole batch. With gain_j = P(win | d_j, e) * E[improvement | win]:
- individual exit:  drop j iff gain_j < lambda * (sum_{g >= c_e} b_g) / F
  (dropping one candidate saves only its per-candidate cost b_g on the remaining graphs);
- collective stop:  stop iff sum_{j in R} gain_j < lambda * (sum_{g >= c_e} (a + b_g |R|)) / F
  (stopping the batch also saves the fixed cost a of every remaining graph).
Approximation (stated): the lookahead is to the end only, because the table gives P(win) for the full evaluation,
not for an intermediate checkpoint. With lambda = 0 nothing is ever dropped (gains are >= 0).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .calibration import Calibration


@dataclass
class Context:
    """Everything a policy may use at a decision point (all of it known before the batch's later graphs)."""

    checkpoints: list[int]  # increasing graph counts, last = L
    incumbent: float  # best full fitness found so far (-inf if none)
    calibration: Calibration
    a: float  # fixed cost per evaluated graph (any number of live candidates)
    b: np.ndarray  # (L,) estimated per-candidate cost of each graph
    batch_size: int

    def __post_init__(self) -> None:
        if not self.checkpoints or any(x >= y for x, y in zip(self.checkpoints, self.checkpoints[1:])):
            raise ValueError(f"checkpoints must be strictly increasing, got {self.checkpoints}")
        if self.checkpoints[0] < 1 or self.checkpoints[-1] != len(self.b):
            raise ValueError("checkpoints must start >= 1 and end at L = len(b)")
        if self.a < 0 or (self.b < 0).any():
            raise ValueError("costs must be non-negative")

    @property
    def L(self) -> int:
        return len(self.b)

    @property
    def full_cost(self) -> float:
        """F: cost of evaluating the whole batch on every graph."""
        return float(self.L * self.a + self.batch_size * self.b.sum())


class Policy(ABC):
    name: str = "policy"

    @abstractmethod
    def decide(self, ctx: Context, e: int, prefix: np.ndarray, alive: np.ndarray) -> np.ndarray:
        ...

    def label(self) -> str:
        return self.name

    # optional hook used by the certified rule (simulator calls it after each certification batch)
    needs_audit: bool = False


def _means(prefix: np.ndarray) -> np.ndarray:
    return prefix.mean(axis=1)  # (B,)


class Full(Policy):
    name = "full"

    def decide(self, ctx, e, prefix, alive):
        return alive.copy()


@dataclass
class OpenEvolveCascade(Policy):
    """OpenEvolve-style fixed cascade (cascade_thresholds [0.5, 0.75] on a [0, 1] score).

    Adaptation to our score s in [-1, 1]: map to [0, 1] by (s + 1) / 2. Stage 1 (checkpoint 0): keep iff
    (p + 1) / 2 >= t1, i.e. p >= 2 t1 - 1 (t1 = 0.5: no worse than the clique cover on average). Stage 2
    (checkpoint 1): keep iff (p + 1) / 2 >= t2 * (incumbent + 1) / 2, i.e. relative to the incumbent like OpenEvolve's
    comparison with the best program. No drops after stage 2.
    """

    t1: float = 0.5
    t2: float = 0.75
    name: str = "openevolve_cascade"

    def decide(self, ctx, e, prefix, alive):
        p = _means(prefix)
        if e == 0:
            thr = 2.0 * self.t1 - 1.0
        elif e == 1:
            thr = self.t2 * (ctx.incumbent + 1.0) - 1.0 if math.isfinite(ctx.incumbent) else -math.inf
        else:
            return alive.copy()
        return alive & (p >= thr)


@dataclass
class SuccessiveHalving(Policy):
    """Keep the top ceil(n_alive / eta) by partial mean at every checkpoint (Karnin et al. 2013); never below 1."""

    eta: float = 2.0
    start: int = 0  # first checkpoint at which halving happens (used by Hyperband brackets)
    name: str = "successive_halving"

    def decide(self, ctx, e, prefix, alive):
        if e < self.start or alive.sum() <= 1:
            return alive.copy()
        idx = np.flatnonzero(alive)
        k = max(1, math.ceil(len(idx) / self.eta))
        p = _means(prefix)[idx]
        top = idx[np.argsort(-p, kind="stable")[:k]]
        keep = np.zeros_like(alive)
        keep[top] = True
        return keep


@dataclass
class Hyperband(Policy):
    """Simplified Hyperband (Li et al. 2018) for a fixed batch: candidates are dealt round-robin (by position) into
    brackets s = 0..n_brackets-1; bracket s runs successive halving that starts at checkpoint s (a larger minimum
    resource before the first elimination). Brackets halve independently."""

    eta: float = 2.0
    n_brackets: int = 3
    name: str = "hyperband"

    def decide(self, ctx, e, prefix, alive):
        keep = np.zeros_like(alive)
        B = len(alive)
        for s in range(self.n_brackets):
            members = np.zeros(B, dtype=bool)
            members[s::self.n_brackets] = True
            sub_alive = alive & members
            if not sub_alive.any():
                continue
            keep |= SuccessiveHalving(self.eta, start=s).decide(ctx, e, prefix, sub_alive)
        return keep


@dataclass
class HoeffdingRace(Policy):
    """Drop j when its Hoeffding upper bound p_j + range * sqrt(log(1/delta) / (2 m)) is below the incumbent
    (Maron & Moore 1994). Caveat: graphs are a fixed ordered list, not i.i.d. samples, so the bound is heuristic."""

    delta: float = 0.05
    score_range: float = 2.0
    name: str = "hoeffding_race"

    def decide(self, ctx, e, prefix, alive):
        if not math.isfinite(ctx.incumbent):
            return alive.copy()
        m = prefix.shape[1]
        ucb = _means(prefix) + self.score_range * math.sqrt(math.log(1.0 / self.delta) / (2.0 * m))
        return alive & (ucb >= ctx.incumbent)


@dataclass
class CostRule(Policy):
    """The cost-derived rule (module docstring). `lam` is the single knob."""

    lam: float = 0.1
    name: str = "cost_rule"

    def label(self) -> str:
        return f"{self.name}@{self.lam:g}"

    def gains(self, ctx: Context, e: int, prefix: np.ndarray) -> np.ndarray:
        d = _means(prefix) - (ctx.incumbent if math.isfinite(ctx.incumbent) else -math.inf)
        if not math.isfinite(ctx.incumbent):
            d = np.full(prefix.shape[0], np.inf)  # nothing to beat yet: every candidate maps to the top bin
        return ctx.calibration.p_win(e, d) * ctx.calibration.exp_improvement()  # (B,)

    def decide(self, ctx, e, prefix, alive):
        if self.lam <= 0 or not alive.any():
            return alive.copy()
        c = ctx.checkpoints[e]
        F = ctx.full_cost
        if F <= 0:
            return alive.copy()
        rest_b = float(ctx.b[c:].sum())
        g = self.gains(ctx, e, prefix)
        keep = alive & (g >= self.lam * rest_b / F)
        n_r = int(alive.sum())
        collective = float(((ctx.L - c) * ctx.a + n_r * rest_b))
        if g[alive].sum() < self.lam * collective / F:
            return np.zeros_like(alive)
        return keep


# ---- certified selection of lambda (Learn then Test) ---------------------------------------------------------------
def binom_cdf(k: int, m: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(m, p)."""
    if k < 0:
        return 0.0
    if k >= m:
        return 1.0
    return float(sum(math.comb(m, i) * p ** i * (1.0 - p) ** (m - i) for i in range(k + 1)))


def clopper_pearson_upper(k: int, m: int, delta: float) -> float:
    """One-sided (1 - delta) upper confidence bound on a binomial rate after k events in m trials."""
    if m <= 0:
        return 1.0
    if k >= m:
        return 1.0
    lo, hi = k / m, 1.0
    for _ in range(60):  # bisection on the monotone map p -> P(X <= k)
        mid = 0.5 * (lo + hi)
        if binom_cdf(k, m, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


@dataclass
class CostRuleCertified(Policy):
    """Cost rule whose lambda is chosen by fixed-sequence testing (Learn then Test, Angelopoulos et al. 2021).

    Risk = P(lost winner) per certification batch: the batch's best candidate (by full fitness) beats the incumbent
    but is not fully evaluated. Candidate lambdas are tested in increasing order (decreasing safety); lambda is
    certified iff the Clopper-Pearson (1 - delta) upper bound of its loss rate on the certification batches is
    <= alpha; testing stops at the first failure and the largest certified lambda is used (0 = full evaluation if
    none passes). Certification batches must be fully evaluated and must not feed the calibration table used to
    replay them (the simulator alternates audit batches between the two roles).
    """

    lams: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0)
    alpha: float = 0.1
    delta: float = 0.1
    name: str = "cost_rule_certified"
    needs_audit: bool = True
    lam: float = field(default=0.0, init=False)
    history: list[tuple[int, float]] = field(default_factory=list, init=False)

    def label(self) -> str:
        return f"{self.name}@a{self.alpha:g}"

    def decide(self, ctx, e, prefix, alive):
        return CostRule(self.lam).decide(ctx, e, prefix, alive)

    def recertify(self, audits: list[tuple[np.ndarray, float]], replay: Callable[[CostRule, np.ndarray, float], bool],
                  step: int = -1) -> float:
        """audits: (scores (B, L), incumbent at batch start) per certification batch.
        replay(rule, scores, incumbent) -> True if the batch's winner would be lost under `rule`."""
        chosen = 0.0
        m = len(audits)
        for lam in sorted(self.lams):
            k = sum(replay(CostRule(lam), s, inc) for s, inc in audits)
            if clopper_pearson_upper(k, m, self.delta) <= self.alpha:
                chosen = lam
            else:
                break
        self.lam = chosen
        self.history.append((step, chosen))
        return chosen


def make_policies(names: list[str], lams: list[float], alpha: float = 0.1, delta: float = 0.1) -> list[Policy]:
    """Policies by name; 'all' expands to every policy; cost_rule is expanded over `lams`."""
    allp = ["full", "openevolve_cascade", "successive_halving", "hyperband", "hoeffding_race", "cost_rule",
            "cost_rule_certified"]
    if "all" in names:
        names = allp
    out: list[Policy] = []
    for nm in names:
        if nm == "full":
            out.append(Full())
        elif nm == "openevolve_cascade":
            out.append(OpenEvolveCascade())
        elif nm == "successive_halving":
            out.append(SuccessiveHalving())
        elif nm == "hyperband":
            out.append(Hyperband())
        elif nm == "hoeffding_race":
            out.append(HoeffdingRace())
        elif nm == "cost_rule":
            out += [CostRule(lam) for lam in lams]
        elif nm == "cost_rule_certified":
            out.append(CostRuleCertified(tuple(lams), alpha, delta))
        else:
            raise ValueError(f"unknown policy {nm}; choose from {allp}")
    return out
