"""Solver, Critic and Verifier: prompt builders and reply validators. The agents read the summary and the board;
they cannot write to the board."""
from __future__ import annotations

import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
PLAN_TOOLS = {"code_rules", "frac_cover", "clique_cover", "indep_set", "sat_minrank", "alpha_bound"}
CHECK_TOOLS = {"code_rules", "alpha_bound", "indep_set", "clique_cover"}
VERIFIER_CHECKS = {"freivalds_more", "fit_sample", "indep_recheck", "decode_more", "exact_rank"}


def system(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def _numbers(s: str) -> set[str]:
    return set(re.findall(r"-?\d+(?:\.\d+)?", s))


# ------------------------------------------------------------------------------------------------ Solver
def solver_prompt(summary: str, board: str, tools: str, history: list[str], checkpoint: str | None = None) -> str:
    parts = ["GRAPH SUMMARY\n" + summary, "BOARD\n" + board, "TOOLS AND MEASURED COSTS\n" + tools]
    if history:
        parts.append("CRITIC OBJECTIONS AND CHECK RESULTS SO FAR\n" + "\n".join(history))
    if checkpoint:
        parts.append("CHECKPOINT\n" + checkpoint + "\nRevise the plan for the remaining budget.")
    return "\n\n".join(parts)


def validate_solver(obj, summary: str) -> str | None:
    if not isinstance(obj, dict):
        return "reply must be a JSON object"
    fh = obj.get("family_hypothesis")
    if not isinstance(fh, dict) or not isinstance(fh.get("name"), str):
        return "family_hypothesis.name (string) is required"
    if not isinstance(fh.get("params", {}), dict):
        return "family_hypothesis.params must be an object"
    try:
        conf = float(fh.get("confidence", 0))
    except (TypeError, ValueError):
        return "family_hypothesis.confidence must be a number"
    if not 0 <= conf <= 1:
        return "family_hypothesis.confidence must be in [0, 1]"
    ev = fh.get("evidence", [])
    if not isinstance(ev, list):
        return "family_hypothesis.evidence must be a list of strings"
    nums = _numbers(summary)
    if fh["name"].strip().lower() not in ("none", "unknown", ""):
        if not ev:
            return "evidence is required for a family hypothesis"
        for e in ev:
            if not isinstance(e, str) or not (_numbers(e) & nums):
                return f"evidence item {e!r} does not quote any number from the graph summary"
    plan = obj.get("plan")
    if not isinstance(plan, list) or not plan:
        return "plan must be a non-empty list"
    for it in plan:
        if not isinstance(it, dict) or it.get("tool") not in PLAN_TOOLS:
            return f"plan item {it!r}: tool must be one of {sorted(PLAN_TOOLS)}"
        if not isinstance(it.get("args", {}), dict):
            return f"plan item {it!r}: args must be an object"
    if not isinstance(obj.get("settings", {}), dict):
        return "settings must be an object"
    pred = obj.get("predictions", {}) or {}
    if not isinstance(pred, dict):
        return "predictions must be an object"
    return None


# ------------------------------------------------------------------------------------------------ Critic
def critic_prompt(summary: str, board: str, solver_reply: dict) -> str:
    import json

    return "\n\n".join(["GRAPH SUMMARY\n" + summary, "BOARD\n" + board,
                        "SOLVER PROPOSAL\n" + json.dumps(solver_reply)])


def validate_critic(obj) -> str | None:
    if not isinstance(obj, dict):
        return "reply must be a JSON object"
    if obj.get("verdict") not in ("accept", "challenge"):
        return "verdict must be accept or challenge"
    if not isinstance(obj.get("objections", []), list):
        return "objections must be a list"
    rc = obj.get("requested_checks", [])
    if not isinstance(rc, list):
        return "requested_checks must be a list"
    for c in rc:
        if not isinstance(c, dict) or c.get("tool") not in CHECK_TOOLS:
            return f"requested check {c!r}: tool must be one of {sorted(CHECK_TOOLS)}"
    return None


# ------------------------------------------------------------------------------------------------ Verifier
def verifier_prompt(payload: str) -> str:
    return payload


def validate_verifier(obj) -> str | None:
    if not isinstance(obj, dict):
        return "reply must be a JSON object"
    if not isinstance(obj.get("analysis", ""), str):
        return "analysis must be a string"
    cr = obj.get("checks_requested", [])
    if not isinstance(cr, list):
        return "checks_requested must be a list"
    for c in cr:
        if not isinstance(c, dict) or c.get("check") not in VERIFIER_CHECKS:
            return f"check {c!r}: must be one of {sorted(VERIFIER_CHECKS)}"
    return None


def validate_llm_only(obj) -> str | None:
    if not isinstance(obj, dict):
        return "reply must be a JSON object"
    try:
        int(obj["beta"])
        int(obj["field"])
    except (KeyError, TypeError, ValueError):
        return "beta and field must be integers"
    return None
