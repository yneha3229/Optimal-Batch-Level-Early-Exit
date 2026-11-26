"""The three LLM agents: prompts, reply schemas, validation, deterministic defaults.

The agents never write numbers to the bound board. They choose and order tools, argue, and explain.
"""
from __future__ import annotations

import json
import re

from . import llm as L

TOOLS_DOC = """Tools you can put in a plan (args in braces):
- spectral {"top": int}          try the generic code family M = I + cA over F_p, (p, c) pairs ranked from the
                                 integer spectrum (predicted rank = N - total multiplicity of eigenvalues = theta mod p)
- try_code {"p": prime, "c": int} one specific code M = I + cA over F_p; its length is rank_p(M)
- family_rule {"n": int, "k": int} the library's structural rule for a Kneser-type graph K(n,k): names one (p, c)
- independent_set {"restarts": int} more greedy restarts for the lower bound
- binomial_match {}              find (n, k) with N = C(n,k) and degree = C(n-k,k), if any
- hoffman {}                     Hoffman upper bound on alpha from the spectrum (information only)
Costs: one code check is about N^2 * |S| / 1e11 seconds; describe and greedy bounds already ran."""

SOLVER_SYS = """You are the SOLVER agent of an index-coding pipeline. Goal: certify the broadcast rate beta of the
given side-information graph by making the lower bound (an independent set) meet the upper bound (the length of an
explicit linear code). You see a numeric summary of the graph, never the graph itself. Form a hypothesis about the
graph family from the numbers (or say none), then output a short plan of tool calls, cheapest likely-success first.
Every piece of evidence must quote a number from the summary. Reply with ONE JSON object only:
{"family_hypothesis": {"name": str, "params": {}, "confidence": float, "evidence": [str]},
 "plan": [{"tool": str, "args": {}, "why": str}], "notes": str}
""" + TOOLS_DOC

CRITIC_SYS = """You are the CRITIC agent. Challenge the Solver: (1) is the family hypothesis consistent with EVERY
number in the summary (receivers, degree, regularity, common-neighbour counts, eigenvalues and multiplicities)?
(2) is the order of the plan justified by the evidence? (3) is there a cheaper path? Accept only if all three hold.
You may request checks (same tool names as the Solver, plus binomial_match and hoffman). Reply with ONE JSON object:
{"verdict": "accept" | "challenge", "objections": [str], "requested_checks": [{"tool": str, "args": {}}]}
""" + TOOLS_DOC

CHECKPOINT_SYS = SOLVER_SYS + """
CHECKPOINT: your earlier plan did not close the bracket. You now see what was tried and the results. Revise the plan
(do not repeat failed (p, c) pairs) or reply with an empty plan to stop and report the bracket."""

VERIFIER_SYS = """You are the VERIFIER agent. You receive the final bound board, the results of independent
certificate checks (computed by code that shares nothing with the solver), timings, and baseline results if any.
You cannot change the verdict: it is computed from the checks. Write a short, honest analysis (at most 120 words):
what is proven and by which certificate, how the agent compared with the baselines on accuracy and time (state
losses as plainly as wins), and anything suspicious. Optionally request extra checks: "decode_more" (more decoding
trials) or "own_rank" (exact rank by the verifier's own elimination, N <= 2500).
Reply with ONE JSON object: {"checks_requested": [str], "analysis": str}"""

ALLOWED = {"spectral", "try_code", "family_rule", "independent_set", "binomial_match", "hoffman"}


def _numbers(x) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", x if isinstance(x, str) else json.dumps(x)))


def _valid_plan(d, summary=None):
    if not isinstance(d, dict):
        raise ValueError("reply must be a JSON object")
    fh = d.get("family_hypothesis", {"name": "none"})
    if not isinstance(fh, dict) or not isinstance(fh.get("name", ""), str):
        raise ValueError('family_hypothesis must be an object with a string "name"')
    if summary is not None and fh.get("name", "").strip().lower() not in ("", "none", "unknown"):
        # ported from the server version's validator: every evidence item must quote a number of the summary
        ev = fh.get("evidence") or []
        if not isinstance(ev, list) or not ev:
            raise ValueError("a family hypothesis needs evidence: a list of strings quoting numbers from the summary")
        nums = _numbers(summary)
        for e in ev:
            if not isinstance(e, str) or not (_numbers(e) & nums):
                raise ValueError(f"evidence item {e!r} does not quote any number from the graph summary")
    plan = d.get("plan")
    if not isinstance(plan, list):
        raise ValueError("plan must be a list")
    for step in plan:
        if not isinstance(step, dict):
            raise ValueError("each plan step must be an object {\"tool\": ..., \"args\": {...}, \"why\": ...}")
        if step.get("tool") not in ALLOWED:
            raise ValueError(f"unknown tool {step.get('tool')!r}; allowed: {sorted(ALLOWED)}")
        if not isinstance(step.get("args", {}), dict):
            raise ValueError("args must be an object")
    return d


def _valid_critic(d):
    if d.get("verdict") not in ("accept", "challenge"):
        raise ValueError('verdict must be "accept" or "challenge"')
    for c in d.get("requested_checks", []):
        if c.get("tool") not in ALLOWED:
            raise ValueError(f"unknown tool {c.get('tool')!r}")
    return d


def _valid_verifier(d):
    if not isinstance(d.get("analysis"), str):
        raise ValueError("analysis must be a string")
    return d


class Agents:
    def __init__(self, backend, trace):
        self.b, self.trace = backend, trace

    def _ask(self, agent, system, payload, validate, default):
        if self.b is None:
            return default, "default"
        prompt = json.dumps(payload, indent=1)
        err = None
        for attempt in range(2):
            p = prompt if err is None else prompt + f"\n\nYour previous reply was invalid: {err}. Reply again."
            self.trace("llm_request", agent=agent, attempt=attempt)
            try:
                txt = self.b.complete(agent, system, p)
                d = validate(L.parse_json(txt))
                self.trace("llm_reply", agent=agent, reply=d)
                return d, "llm"
            except Exception as e:  # noqa: BLE001  (parse/schema/backend errors -> one retry, then default)
                err = str(e)[:300]
                self.trace("llm_invalid", agent=agent, error=err, raw=str(getattr(self.b, "last_raw", ""))[:1500])
        self.trace("llm_fallback", agent=agent)
        return default, "fallback"

    def solver(self, payload, default):
        return self._ask("solver", SOLVER_SYS, payload, lambda d: _valid_plan(d, payload.get("summary")), default)

    def critic(self, payload):
        return self._ask("critic", CRITIC_SYS, payload, _valid_critic,
                         {"verdict": "accept", "objections": [], "requested_checks": []})

    def checkpoint(self, payload, default):
        return self._ask("solver", CHECKPOINT_SYS, payload, lambda d: _valid_plan(d, payload.get("summary")), default)

    def verifier(self, payload):
        return self._ask("verifier", VERIFIER_SYS, payload, _valid_verifier,
                         {"checks_requested": [], "analysis": "(no LLM: verdict from the checks only)"})
