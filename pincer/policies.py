"""Controller policies: which (component, tool) to run next, and for how long.

All policies share `SliceSchedule` (so they differ only in the decision rule, not in how long tools run):
- anytime tools (heuristics, CP-SAT, LP column generation) start at a slice that grows with n and double after a
  call that changed nothing, up to `max_slice`; an arm is retired after two fruitless calls at the cap.
- one-shot SAT calls start at `sat_base` seconds and double after every timeout (a timeout is a censored
  observation: "needs more than s seconds"); the slice resets to the base when the question changes (a new r).
- "done" (proven optimal, or a deterministic tool already ran) retires the arm.

Policies
- `RoundRobinPolicy`: every useful arm in a fixed order. This is the "run every tool for an equal share" baseline.
- `IndexPolicy`: cost-aware index, the same rule as the early-exit project's stopping rule (continue while the
  expected gain per unit cost exceeds lambda). index = P(success) x expected gap reduction / slice, with
  P(success) a Beta posterior per (tool, size bucket, directed), shared across components and graphs, an optimism
  bonus, and a 0.7^k discount after k fruitless calls of the same arm. Priors can be loaded from a calibration
  file (pincer/calibrate.py). With `min_rate` > 0 it stops early once no arm is worth its time.
- `LLMPolicy`: an LLM reads the state and the index policy's top candidates (with their estimates) and picks one,
  or stops. Invalid replies or API errors fall back to the index choice. LLM latency counts against the budget.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_PRIORS = ROOT / "priors.json"

# default P(success) priors as (alpha, beta) pseudo-counts, before any calibration
DEFAULT_PRIOR = {
    "mais_sa": (2, 3), "mais_lazy": (2, 3), "mais_order": (1, 3), "cover_tabu": (1, 4), "partial_cover": (1, 4),
    "sat_down": (2, 2), "sat_up": (2, 2), "shannon": (2, 2), "frac_cover": (1, 2), "block_lns": (1, 6),
    "quick": (5, 1), "mais_evolved": (2, 3),
}
SAT_TOOLS = {"sat_down", "sat_up"}


def bucket(n: int, directed: bool) -> str:
    b = next(x for x in (15, 30, 60, 120, 10**9) if n <= x)
    return f"{'d' if directed else 'u'}{b}"


class SliceSchedule:
    def __init__(self, sat_base: float = 1.0, max_slice: float = 30.0) -> None:
        self.sat_base, self.max_slice = sat_base, max_slice
        self.slice: dict[tuple[int, str], float] = {}
        self.fails_at_cap: dict[tuple[int, str], int] = {}
        self.fails: dict[tuple[int, str], int] = {}
        self.retired: set[tuple[int, str]] = set()
        self.question: dict[tuple[int, str], tuple] = {}

    def base(self, tool: str, n: int) -> float:
        if tool in SAT_TOOLS:
            return self.sat_base
        if tool == "quick":
            return 0.2
        return min(5.0, max(0.5, 0.02 * n))

    def get(self, key: tuple[int, str], n: int, question: tuple = ()) -> float:
        if key[1] in SAT_TOOLS and self.question.get(key) != question:
            self.question[key] = question
            self.slice[key] = self.base(key[1], n)
        return self.slice.setdefault(key, self.base(key[1], n))

    def update(self, key: tuple[int, str], status: str, n: int) -> None:
        cur = self.slice.get(key, self.base(key[1], n))
        if status == "done":
            self.retired.add(key)
            return
        if status in ("improved", "unsat"):
            self.fails[key] = 0
            return
        self.fails[key] = self.fails.get(key, 0) + 1
        if cur >= self.max_slice:
            self.fails_at_cap[key] = self.fails_at_cap.get(key, 0) + 1
            if self.fails_at_cap[key] >= 2:
                self.retired.add(key)
        self.slice[key] = min(self.max_slice, cur * 2)


class BasePolicy:
    name = "base"

    def __init__(self, sat_base: float = 1.0, max_slice: float = 30.0) -> None:
        self.sched = SliceSchedule(sat_base, max_slice)
        self.agent = None

    def attach(self, agent) -> None:
        self.agent = agent
        self.sched.max_slice = min(self.sched.max_slice, max(1.0, agent.budget / 2))

    def exhausted(self, comp: int, tool: str) -> bool:
        return (comp, tool) in self.sched.retired

    def _question(self, comp: int, tool: str) -> tuple:
        s = self.agent.states[comp]
        return (s.mr_lo, s.mr_hi) if tool in SAT_TOOLS else ()

    def slice_for(self, arm) -> float:
        s = self.agent.states[arm.comp]
        return self.sched.get((arm.comp, arm.tool), s.n, self._question(arm.comp, arm.tool))

    def update(self, arm, step) -> None:
        s = self.agent.states[arm.comp]
        self.sched.update((arm.comp, arm.tool), step.status, s.n)

    def choose(self, arms):
        raise NotImplementedError


class RoundRobinPolicy(BasePolicy):
    name = "roundrobin"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.k = 0

    def choose(self, arms):
        arms = sorted(arms, key=lambda a: (a.comp, a.tool))
        arm = arms[self.k % len(arms)]
        self.k += 1
        return arm, self.slice_for(arm)


# prior mean gap reduction of a *successful* call, per tool (pseudo-count GAIN_K); learned per size bucket
DEFAULT_GAIN = {"mais_sa": 1.0, "mais_lazy": 1.5, "mais_order": 1.5, "mais_evolved": 1.0, "cover_tabu": 1.5,
                "partial_cover": 1.0, "sat_down": 2.0, "sat_up": 1.5, "shannon": 0.5, "frac_cover": 0.2,
                "block_lns": 1.0}
GAIN_K = 2.0


class IndexPolicy(BasePolicy):
    name = "index"

    def __init__(self, priors: str | Path | None = DEFAULT_PRIORS, min_rate: float = 0.0, optimism: float = 0.5,
                 **kw) -> None:
        super().__init__(**kw)
        self.min_rate, self.optimism = min_rate, optimism
        # (tool, bucket) -> [alpha, beta, sum of gains of successful calls, number of successful calls]
        self.counts: dict[tuple[str, str], list[float]] = {}
        self.n_decisions = 0
        if priors and Path(priors).exists():
            for k, v in json.loads(Path(priors).read_text()).items():
                tool, b = k.split("|")
                v = [float(x) for x in v] + [0.0] * (4 - len(v))
                self.counts[(tool, b)] = v[:4]

    def _ab(self, tool: str, b: str) -> list[float]:
        if (tool, b) not in self.counts:
            self.counts[(tool, b)] = list(map(float, DEFAULT_PRIOR.get(tool, (1, 2)))) + [0.0, 0.0]
        return self.counts[(tool, b)]

    def gain_cap(self, arm) -> float:
        """Largest gap reduction the call could achieve in the current state (structural, not learned)."""
        s = self.agent.states[arm.comp]
        mr_gap = s.mr_hi - s.mr_lo
        b_gap = float(s.b_hi - s.b_lo)
        b_tied = b_gap > 0 and s.b_hi == s.mr_hi
        t = arm.tool
        if t in ("sat_down", "sat_up"):
            return mr_gap + (min(b_gap, mr_gap) if b_tied or t == "sat_up" else 0.0)
        if t in ("cover_tabu", "partial_cover", "block_lns"):
            return mr_gap + (b_gap if b_tied else 0.0)
        if t.startswith("mais"):
            k = len(s.acyclic)
            return (mr_gap if k + 1 > s.mr_lo else 0.0) + (b_gap if k + 1 > s.b_lo else 0.0)
        if t == "shannon":
            return b_gap + (1.0 if mr_gap > 0 else 0.0)
        if t == "frac_cover":
            return b_gap
        return mr_gap + b_gap

    def estimate(self, arm) -> dict:
        s = self.agent.states[arm.comp]
        a, b, gsum, gn = self._ab(arm.tool, bucket(s.n, s.comp.graph.directed))
        p = a / (a + b)
        bonus = self.optimism * math.sqrt(math.log(self.n_decisions + 2) / (a + b))
        fails = self.sched.fails.get((arm.comp, arm.tool), 0)
        p_eff = min(1.0, p + bonus) * (0.7 ** fails)
        g0 = DEFAULT_GAIN.get(arm.tool, 1.0)
        mean_gain = (g0 * GAIN_K + gsum) / (GAIN_K + gn)
        gain = min(self.gain_cap(arm), mean_gain)
        sl = self.slice_for(arm)
        cost = sl
        tool = self.agent.tools[arm.tool]
        if tool.once:  # a one-shot tool runs to completion: charge its predicted run time, not the slice
            from .tools import predicted_seconds

            cost = max(sl, predicted_seconds(tool, s))
        return {"p": p_eff, "gain": gain, "slice": sl, "cost": cost, "index": p_eff * gain / cost}

    def choose(self, arms):
        self.n_decisions += 1
        scored = [(self.estimate(a), a) for a in arms]
        # a one-shot tool that cannot finish in the time left is skipped while anything else is available
        fits = [(e, a) for e, a in scored if e["cost"] <= 1.2 * self.agent.left()]
        scored = fits or scored
        est, arm = max(scored, key=lambda t: (t[0]["index"], -t[1].comp))
        if est["index"] < self.min_rate:
            return None
        return arm, est["slice"]

    def update(self, arm, step) -> None:
        super().update(arm, step)
        if arm.tool == "quick":
            return
        s = self.agent.states[arm.comp]
        ab = self._ab(arm.tool, bucket(s.n, s.comp.graph.directed))
        if step.gain > 1e-9:
            ab[0] += 1
            ab[2] += step.gain
            ab[3] += 1
        elif step.status != "done":
            ab[1] += 1


SYSTEM = """You control Pincer, a solver portfolio that bounds index-coding problems. For each strongly connected
component (SCC) it keeps two certified intervals: minrank over F_2 [mr_lo, mr_hi] (integers) and the broadcast
rate beta [b_lo, b_hi] (rationals). Goal: shrink the total gap (sum of mr_hi - mr_lo + b_hi - b_lo) as much as
possible before the time budget runs out. Every result is verified, so a bad choice only wastes time.
Tools: {tools}
Reply with one JSON object and nothing else: {{"choice": <candidate number>, "slice": <seconds>}} or {{"stop": true}}
if no candidate is worth its time. Prefer cheap, likely moves first; give SAT calls that timed out more time only
if the budget allows; one-shot tools ignore the slice."""


class LLMPolicy(IndexPolicy):
    name = "llm"

    def __init__(self, backend, every: int = 1, top_k: int = 10, **kw) -> None:
        super().__init__(**kw)
        self.backend, self.every, self.top_k = backend, every, top_k
        self.fallbacks = 0
        self.decisions = 0
        self.log: list[dict] = []

    def llm_usage(self) -> dict:
        u = self.backend.usage.as_dict()
        u.update({"decisions": self.decisions, "fallbacks": self.fallbacks, "backend": self.backend.name})
        return u

    def _prompt(self, cands) -> str:
        ag = self.agent
        lines = [f"time left: {ag.left():.1f} s of {ag.budget:.0f} s; total gap {ag.gap():.3f}",
                 "open components:"]
        for s in ag.open_states()[:15]:
            lines.append(f"  comp {s.comp.index}: n={s.n} {'directed' if s.comp.graph.directed else 'undirected'} "
                         f"minrank [{s.mr_lo},{s.mr_hi}] beta [{float(s.b_lo):.3f},{float(s.b_hi):.3f}] "
                         f"acyclic {len(s.acyclic)} (MAIS <= {s.mais_ub})")
        recent = ag.steps[-8:]
        if recent:
            lines.append("recent calls:")
            for st in recent:
                lines.append(f"  comp {st.comp} {st.tool} {st.slice:.1f}s -> {st.status} gain {st.gain:+.2f}")
        lines.append("candidates (estimates from a cost model: success prob, gap reduction if it works, slice):")
        for k, (est, a) in enumerate(cands):
            lines.append(f"  {k}: comp {a.comp} {a.tool} p={est['p']:.2f} gain={est['gain']:.2f} "
                         f"slice={est['slice']:.1f}s")
        return "\n".join(lines)

    def choose(self, arms):
        self.decisions += 1
        scored = sorted(((self.estimate(a), a) for a in arms), key=lambda t: -t[0]["index"])
        cands = scored[: self.top_k]
        self.n_decisions += 1
        if (self.decisions - 1) % self.every:
            return cands[0][1], cands[0][0]["slice"]
        tools = "; ".join(f"{t.name}: {t.description}" for t in self.agent.tools.values())
        try:
            text = self.backend.complete(SYSTEM.format(tools=tools), self._prompt(cands))
            m = re.search(r"\{.*\}", text, re.S)
            d = json.loads(m.group(0)) if m else {}
            if d.get("stop") is True:
                self.log.append({"stop": True})
                return None
            k = int(d["choice"])
            est, arm = cands[k]
            sl = float(d.get("slice", est["slice"]))
            sl = max(0.1, min(sl, self.sched.max_slice, self.agent.left()))
            self.log.append({"choice": k, "slice": sl})
            return arm, sl
        except Exception as e:  # any bad reply -> the index policy's choice
            self.fallbacks += 1
            self.log.append({"fallback": str(e)[:200]})
            return cands[0][1], cands[0][0]["slice"]


def make_policy(name: str, backend=None, **kw):
    if name == "roundrobin":
        return RoundRobinPolicy(**kw)
    if name == "index":
        return IndexPolicy(**kw)
    if name == "llm":
        if backend is None:
            raise ValueError("policy llm needs --llm")
        return LLMPolicy(backend, **kw)
    raise ValueError(f"unknown policy {name!r}")
