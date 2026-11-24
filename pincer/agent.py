"""Pincer: the controller agent. Decompose -> quick bounds -> loop {pick (component, tool, slice); run; verify}.

    python -m pincer.agent GRAPH.json --budget 60 --policy index [--llm anthropic] [--out bundle.json]

Policies (pincer/policies.py) only choose; tools only propose; `CompState` checks every proposal; the final bundle
is re-verified from scratch by `pincer.certify.verify_bundle` before anything is reported.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from minrank.core.graph import Graph

from .certify import make_bundle, totals, verify_bundle
from .decompose import decompose
from .state import CompState
from .tools import Outcome, Tool, default_tools


@dataclass
class Arm:
    comp: int
    tool: str


@dataclass
class Step:
    t: float
    comp: int
    tool: str
    slice: float
    status: str
    seconds: float
    gain: float
    gap_after: float


@dataclass
class RunResult:
    graph: str
    policy: str
    budget: float
    seconds: float
    totals: dict
    verified: dict
    steps: list[Step] = field(default_factory=list)
    trajectory: list[tuple[float, float]] = field(default_factory=list)   # (t, total gap)
    llm_usage: dict | None = None
    stopped: str = ""
    bundle: dict | None = None

    def summary(self) -> dict:
        return {"graph": self.graph, "policy": self.policy, "budget": self.budget, "seconds": round(self.seconds, 2),
                "mr": self.totals["mr"], "beta": self.totals["beta"], "components": self.totals["components"],
                "closed_mr": self.totals["closed_mr"], "closed_beta": self.totals["closed_beta"],
                "gap": round(total_gap_from_totals(self.totals), 6), "steps": len(self.steps), "stopped": self.stopped,
                "llm": self.llm_usage}


def total_gap_from_totals(t: dict) -> float:
    from fractions import Fraction

    return (t["mr"][1] - t["mr"][0]) + float(Fraction(t["beta"][1]) - Fraction(t["beta"][0]))


class Pincer:
    def __init__(self, g: Graph, budget: float, policy, seed: int = 0, tools: dict[str, Tool] | None = None,
                 quick_slice: float = 0.2) -> None:
        self.g, self.budget, self.policy = g, budget, policy
        self.rng = random.Random(seed)
        self.tools = tools or default_tools()
        self.quick_slice = quick_slice
        self.t0 = 0.0
        self.states: list[CompState] = []
        self.steps: list[Step] = []
        self.trajectory: list[tuple[float, float]] = []

    # ------------------------------------------------------------------ helpers used by policies
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def left(self) -> float:
        return self.budget - self.elapsed()

    def gap(self) -> float:
        return sum(s.gap() for s in self.states)

    def open_states(self) -> list[CompState]:
        return [s for s in self.states if not s.closed]

    def arms(self) -> list[Arm]:
        out = []
        for s in self.open_states():
            for name, tool in self.tools.items():
                if tool.once and any(st.comp == s.comp.index and st.tool == name for st in self.steps):
                    continue
                if tool.max_n is not None and s.n > tool.max_n:
                    continue
                if self.policy.exhausted(s.comp.index, name):
                    continue
                if tool.useful(s):
                    out.append(Arm(s.comp.index, name))
        return out

    # ------------------------------------------------------------------ main loop
    def run_tool(self, arm: Arm, slice_s: float) -> Outcome:
        s = self.states[arm.comp]
        tool = self.tools[arm.tool]
        before = s.gap()
        ts = time.perf_counter()
        status = tool.run(s, slice_s, self.rng, self.elapsed())
        sec = time.perf_counter() - ts
        out = Outcome(status, sec, before - s.gap())
        self.steps.append(Step(round(self.elapsed(), 3), arm.comp, arm.tool, round(slice_s, 3), status,
                               round(sec, 3), round(out.gain, 6), round(self.gap(), 6)))
        self.trajectory.append((round(self.elapsed(), 3), round(self.gap(), 6)))
        return out

    def run(self) -> RunResult:
        self.t0 = time.perf_counter()
        comps = decompose(self.g)
        self.states = [CompState(c) for c in comps]
        for s in self.states:  # singletons are closed at once: minrk = beta = 1
            if s.n == 1:
                s.offer_acyclic([0], "decompose", 0.0)
                s.mais_ub = 1
        self.trajectory.append((0.0, self.gap()))
        self.policy.attach(self)
        for s in self.states:
            if not s.closed and self.left() > 0:
                self.run_tool(Arm(s.comp.index, "quick"), self.quick_slice)
                self.policy.update(Arm(s.comp.index, "quick"), self.steps[-1])
        stopped = "closed"
        while self.open_states():
            if self.left() <= 0.05:
                stopped = "budget"
                break
            arms = self.arms()
            if not arms:
                stopped = "no_useful_tool"
                break
            choice = self.policy.choose(arms)
            if choice is None:
                stopped = "policy_stop"
                break
            arm, slice_s = choice
            if self.tools[arm.tool].once:  # one-shot tools run to completion: give them the time left as a limit
                slice_s = self.left()
            slice_s = max(0.05, min(slice_s, self.left()))
            self.run_tool(arm, slice_s)
            self.policy.update(arm, self.steps[-1])
        seconds = self.elapsed()
        bundle = make_bundle(self.g, self.states, {"policy": self.policy.name, "budget": self.budget})
        verified = verify_bundle(self.g, bundle)
        llm = getattr(self.policy, "llm_usage", None)
        return RunResult(self.g.name, self.policy.name, self.budget, seconds, totals(self.states), verified,
                         self.steps, self.trajectory, llm() if callable(llm) else None, stopped, bundle)


def main(argv: list[str] | None = None) -> int:
    from .llm import make_backend
    from .policies import make_policy

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--policy", default="index", help="index | roundrobin | llm")
    ap.add_argument("--llm", default="none", help="anthropic[:model[:effort]] | local[:url|model] (policy llm)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lns", action="store_true", help="also offer the block-LNS tool")
    ap.add_argument("--evolved", default=None, help="FunSearch program to offer as the mais_evolved tool")
    ap.add_argument("--out", default=None, help="write the certificate bundle here")
    a = ap.parse_args(argv)
    g = Graph.load(a.graph)
    pol = make_policy(a.policy, make_backend(a.llm))
    res = Pincer(g, a.budget, pol, a.seed, default_tools(a.lns, a.evolved)).run()
    print(json.dumps(res.summary()))
    for st in res.steps:
        print(f"  t={st.t:7.2f}s comp {st.comp:3d} {st.tool:13s} slice {st.slice:6.2f}s -> {st.status:9s} "
              f"gain {st.gain:+.3f} gap {st.gap_after:.3f}", file=sys.stderr)
    if a.out:
        Path(a.out).write_text(json.dumps(res.bundle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
