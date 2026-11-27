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
Rules (replies that break 1-2 are rejected by code and you are asked again):
1. Every evidence item quotes a number you were given (summary, board or check results).
2. Arithmetic is checked: every "C(a,b) = X" you write must be exact, and for a Kneser hypothesis C(n,k) must equal
   the number of receivers. Do not compute binomials in your head: if binomial_match was run, copy its (n, k);
   otherwise put binomial_match first in the plan.
3. If a family rule may apply, query it (family_rule) before relying on it: it may name no field (an exception).
4. Keep one generic step (spectral) in the plan unless a named code already closed the bracket, so a wrong guess
   costs only time. A graph whose degrees are not all equal or whose spectrum is not integral is NOT an exact member
   of a vertex-transitive family; say so (e.g. "perturbed Kneser") instead of claiming the family. A rule that
   names no field does NOT mean the graph is perturbed: it is an exception of the rule (then search generically).
Reply with ONE JSON object only (no prose, no code fences):
{"family_hypothesis": {"name": str, "params": {}, "confidence": float, "evidence": [str]},
 "plan": [{"tool": str, "args": {}, "why": str}], "notes": str}
""" + TOOLS_DOC

CRITIC_SYS = """You are the CRITIC agent. Challenge the Solver: (1) is the family hypothesis consistent with EVERY
number in the summary (receivers, degree, regularity, common-neighbour counts, eigenvalues and multiplicities)?
(2) is the order of the plan justified by the evidence? (3) is there a cheaper path? Accept only if all three hold.
Be concrete: recompute the Solver's arithmetic; compare the hypothesis with check_history (e.g. a binomial_match
result that differs from the hypothesised (n, k) refutes it); if the plan relies on family_rule without having
queried it, request it. Every objection must name a number or a check result. Do NOT request a check whose result
is already in check_history; if nothing new can be learned, accept.
You may request checks (same tool names as the Solver, plus binomial_match and hoffman).
Reply with ONE JSON object only (no prose, no code fences):
{"verdict": "accept" | "challenge", "objections": [str], "requested_checks": [{"tool": str, "args": {}}]}
""" + TOOLS_DOC

CHECKPOINT_SYS = SOLVER_SYS + """
CHECKPOINT: your earlier plan did not close the bracket. You now see what was tried and the results. Revise the plan
(do not repeat failed (p, c) pairs) or reply with an empty plan to stop and report the bracket."""

VERIFIER_SYS = """You are the VERIFIER agent. You receive the final bound board, the results of independent
certificate checks (computed by code that shares nothing with the solver), timings, and baseline results if any.
You cannot change the verdict: it is computed from the checks. Use only numbers present in the input; wall_s
includes LLM time (llm_s), compute_s is the tools' time. Write a short, honest analysis (at most 120 words):
what is proven and by which certificate, how the agent compared with the baselines on accuracy and time (state
losses as plainly as wins), and anything suspicious. Optionally request extra checks: "decode_more" (more decoding
trials) or "own_rank" (exact rank by the verifier's own elimination, N <= 2500).
Reply with ONE JSON object: {"checks_requested": [str], "analysis": str}"""

# v1: branch prompts; v2: + arithmetic/evidence rules, check history, safety net; v3: + argument validation,
# evidence may name a summary field, "a rule naming no field is an exception, not a perturbation"
PROMPT_VERSION = "v4"   # v4: + plan-step results in the check history (9081f70), prompts unchanged from v3

ALLOWED = {"spectral", "try_code", "family_rule", "independent_set", "binomial_match", "hoffman"}


def _numbers(x) -> set[str]:
    """Numbers in a text or object, normalised (455.0 and 455 are the same number)."""
    out = set()
    for t in re.findall(r"-?\d+(?:\.\d+)?", x if isinstance(x, str) else json.dumps(x)):
        v = float(t)
        out.add(str(int(v)) if v.is_integer() else repr(v))
    return out


_BINOM = re.compile(r"C\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)")


def check_arithmetic(fh: dict, receivers=None) -> str | None:
    """Deterministic arithmetic check of a family hypothesis: every 'C(a,b) = X' in the evidence must be exact, and
    a Kneser hypothesis K(n,k) must have C(n,k) = number of receivers. Returns an error text or None."""
    from math import comb
    for e in fh.get("evidence") or []:
        for a, b, x in _BINOM.findall(str(e)):
            a, b, x = int(a), int(b), int(x)
            if a < 200 and comb(a, b) != x:
                return f"arithmetic error in evidence: C({a},{b}) = {comb(a, b)}, not {x}"
    pr = fh.get("params") or {}
    if receivers and "kneser" in str(fh.get("name", "")).lower() and "perturb" not in str(fh.get("name", "")).lower():
        try:
            n, k = int(pr["n"]), int(pr["k"])
        except (KeyError, TypeError, ValueError):
            return None
        if comb(n, k) != receivers:
            return (f"hypothesis K({n},{k}) has C({n},{k}) = {comb(n, k)} vertices, but the graph has {receivers} "
                    f"receivers; use binomial_match")
    return None


def _valid_plan(d, summary=None, context=None):
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
        nums = _numbers(summary) | (_numbers(context) if context is not None else set())
        keys = set(summary) if isinstance(summary, dict) else set()
        for e in ev:
            if not isinstance(e, str) or not (_numbers(e) & nums or any(k in e for k in keys)):
                raise ValueError(f"evidence item {e!r} quotes no number you were given and no summary field")
        err = check_arithmetic(fh, summary.get("receivers") if isinstance(summary, dict) else None)
        if err:
            raise ValueError(err)
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
        _check_args(step)
    return d


NEEDS = {"family_rule": ("n", "k"), "try_code": ("p", "c")}


def _check_args(step):
    for key in NEEDS.get(step.get("tool"), ()):
        v = (step.get("args") or {}).get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
            raise ValueError(f'{step["tool"]} needs integer args {list(NEEDS[step["tool"]])}, e.g. '
                             + ('{"n": 16, "k": 4}' if step["tool"] == "family_rule" else '{"p": 3, "c": 1}'))
    if step.get("tool") == "try_code":
        from .tools import is_prime
        if not is_prime(int(step["args"]["p"])):
            raise ValueError(f'try_code: p = {step["args"]["p"]} is not a prime')


def _valid_critic(d):
    if d.get("verdict") not in ("accept", "challenge"):
        raise ValueError('verdict must be "accept" or "challenge"')
    for c in d.get("requested_checks", []):
        if not isinstance(c, dict) or c.get("tool") not in ALLOWED:
            raise ValueError(f"unknown tool {c.get('tool') if isinstance(c, dict) else c!r}")
        _check_args(c)
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
        return self._ask("solver", SOLVER_SYS, payload, lambda d: _valid_plan(d, payload.get("summary"), payload),
                         default)

    def critic(self, payload):
        return self._ask("critic", CRITIC_SYS, payload, _valid_critic,
                         {"verdict": "accept", "objections": [], "requested_checks": []})

    def checkpoint(self, payload, default):
        return self._ask("solver", CHECKPOINT_SYS, payload, lambda d: _valid_plan(d, payload.get("summary"), payload),
                         default)

    def verifier(self, payload):
        return self._ask("verifier", VERIFIER_SYS, payload, _valid_verifier,
                         {"checks_requested": [], "analysis": "(no LLM: verdict from the checks only)"})
