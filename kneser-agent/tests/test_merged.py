"""Tests for the pieces ported from the server version into kagent/ (rank engine, prompt validation, JSON mode),
and for the rules the merge must keep.   cd kneser-agent && python -m pytest -q tests"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from kagent import graphs, tools, verify
from kagent.rank import SpectralMatrix, rank_factor

ROOT = Path(__file__).resolve().parents[1]


def _graphs():
    yield "K(9,2)", graphs.kneser(9, 2)
    yield "K(10,3)", graphs.kneser(10, 3)
    yield "K(11,3)-del", graphs.delete_edges(graphs.kneser(11, 3), 0.02, 1)
    rng = np.random.default_rng(0)
    B = np.triu(rng.random((150, 150)) < 0.1, 1)
    yield "G(150,0.1)", B | B.T


@pytest.mark.parametrize("name,A", list(_graphs()))
def test_rank_engine_equals_flint(name, A):
    for p in (2, 3, 5, 7, 13):
        for c in range(1, min(p, 4)):
            M = tools.fitting_matrix(A, p, c)
            exact = tools.rank_mod(M, p)
            r = rank_factor(SpectralMatrix(A, p, c))
            assert r.rank == exact, (name, p, c)
            # the factorization is a certificate: M = C . M[S, :] (mod p)
            assert np.array_equal((r.C @ M[r.S]) % p, M)
            # the verifier's own elimination agrees too (independent code)
            if A.shape[0] <= 160:
                assert verify.rank_mod_p(M, p) == exact


def test_cap_stops_early_and_is_honest():
    A = graphs.kneser(10, 3)
    exact = tools.rank_mod(tools.fitting_matrix(A, 5, 1), 5)
    assert rank_factor(SpectralMatrix(A, 5, 1), cap=exact - 1).rank is None
    assert rank_factor(SpectralMatrix(A, 5, 1), cap=exact).rank == exact


def test_try_code_engines_agree():
    A = graphs.kneser(14, 4)
    S = tools.independent_set(A)
    for p, c in ((2, 1), (3, 2), (5, 1), (7, 6), (11, 1)):
        a = tools.try_code(A, p, c, S, engine="flint")
        b = tools.try_code(A, p, c, S)
        assert (a["rank"], a["meets_lower"]) == (b["rank"], b["meets_lower"]), (p, c)
        capped = tools.try_code(A, p, c, S, cap=len(S))
        assert capped["rank"] in (None, len(S)) and (capped["rank"] is None) == (a["rank"] > len(S))


def test_verifier_shares_no_code_with_tools():
    tree = ast.parse((ROOT / "kagent" / "verify.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
    assert not names & {"tools", "kagent.tools", "rank", "kagent.rank", "pipeline", "agents"}, names


def test_solver_evidence_must_quote_summary_numbers():
    from kagent.agents import _valid_plan
    summary = {"receivers": 1820, "degree_max": 495}
    ok = {"family_hypothesis": {"name": "Kneser K(16,4)", "params": {"n": 16, "k": 4}, "confidence": 0.9,
                                "evidence": ["1820 = C(16,4)", "degree 495 = C(12,4)"]},
          "plan": [{"tool": "family_rule", "args": {"n": 16, "k": 4}}]}
    assert _valid_plan(ok, summary) is ok
    bad = json.loads(json.dumps(ok))
    bad["family_hypothesis"]["evidence"] = ["looks like a Kneser graph"]
    with pytest.raises(ValueError):
        _valid_plan(bad, summary)
    none = {"family_hypothesis": {"name": "none", "params": {}, "confidence": 0, "evidence": []},
            "plan": [{"tool": "spectral", "args": {}}]}
    assert _valid_plan(none, summary) is none


def test_parse_json_variants():
    from kagent.llm import parse_json
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure. {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert parse_json('{"a": 1}') == {"a": 1}


def test_arithmetic_check_catches_the_qwen_errors():
    from kagent.agents import _valid_plan, check_arithmetic
    # Qwen 32B on K(14,4) (1001 receivers) claimed K(14,7) and "C(7, 2) = 210"; on K(16,4) it first claimed K(14,5)
    assert "3432" in check_arithmetic({"name": "Kneser", "params": {"n": 14, "k": 7}, "evidence": []}, 1001)
    assert "21" in check_arithmetic({"name": "Kneser", "params": {}, "evidence": ["degree 210 = C(7, 2) = 210"]}, 1001)
    assert check_arithmetic({"name": "Kneser", "params": {"n": 14, "k": 5}, "evidence": []}, 1820)
    assert check_arithmetic({"name": "Kneser", "params": {"n": 16, "k": 4},
                             "evidence": ["1820 = C(16,4)", "C(12, 4) = 495"]}, 1820) is None
    assert check_arithmetic({"name": "perturbed Kneser", "params": {"n": 16, "k": 4, "deleted": 0.01},
                             "evidence": []}, 1820) is None
    # evidence may quote numbers from check results, not only from the summary
    d = {"family_hypothesis": {"name": "Kneser", "params": {"n": 16, "k": 4}, "confidence": 0.9,
                               "evidence": ["binomial_match returned n=16, k=4", "Hoffman bound 455"]},
         "plan": [{"tool": "spectral", "args": {}}]}
    ctx = {"check_history": [{"tool": "binomial_match", "result": [{"n": 16, "k": 4}]},
                             {"tool": "hoffman", "result": 455.0}]}
    assert _valid_plan(d, {"receivers": 1820}, ctx) is d


class _Scripted:
    """Stand-in LLM backend: replies in order (used to test that bad agent replies cannot crash a run)."""

    def __init__(self, replies):
        self.replies, self.calls, self.seconds, self.last_raw = list(replies), 0, 0.0, ""

    def complete(self, agent, system, prompt):
        self.calls += 1
        self.last_raw = self.replies.pop(0) if self.replies else '{"verdict": "accept", "analysis": "ok"}'
        return self.last_raw


def test_bad_agent_args_never_crash_a_run(tmp_path, monkeypatch):
    import kagent.pipeline as P
    from kagent.agents import Agents
    from kagent.pipeline import Run
    (tmp_path / "results" / "codes").mkdir(parents=True)
    monkeypatch.setattr(P, "ROOT", tmp_path)                  # keep the test's outputs out of results/ and traces/
    bad_plan = json.dumps({"family_hypothesis": {"name": "none"}, "plan": [{"tool": "family_rule", "args": {}}]})
    good = json.dumps({"family_hypothesis": {"name": "none"},
                       "plan": [{"tool": "try_code", "args": {"p": 4, "c": 1}}]})       # 4 is not a prime
    critic = json.dumps({"verdict": "challenge", "objections": ["x"],
                         "requested_checks": [{"tool": "family_rule", "args": {"n": "a"}}]})
    run = Run("K(9,2)", llm="none", budget=30, live=False, tag="test_badargs")
    run.backend = _Scripted([bad_plan, bad_plan, critic, critic, good, good] + ['{"verdict": "accept"}'] * 3)
    run.agents = Agents(run.backend, run.trace)
    r = run.run()
    assert r["verdict"] == "proven" and r["bracket"] == [8, 8]
    evs = [e["ev"] for e in run.events]
    assert "llm_fallback" in evs and "llm_invalid" in evs
