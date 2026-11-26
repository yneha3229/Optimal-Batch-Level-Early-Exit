"""Tests for Pincer. Run: python -m pytest pincer/tests -q"""
from __future__ import annotations

import json
import random
from fractions import Fraction
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from minrank.bench import generators as gen
from minrank.core.graph import Graph
from minrank.core.verify import verify
from minrank.exact.lower import is_acyclic_induced
from minrank.exact.sat import minrank_bruteforce

from pincer.certify import make_bundle, verify_bundle
from pincer.decompose import decompose, induced, lift_rows, scc
from pincer.lp import check_frac_cover, check_shannon_certificate, frac_clique_cover, shannon_bound
from pincer.mais import exact_lazy, greedy_fvs, sa_order


def rand_digraph(n: int, p: float, seed: int) -> Graph:
    return gen.gnp(n, p, seed, True)


def mais_brute(g: Graph) -> int:
    for k in range(g.n, 0, -1):
        for S in combinations(range(g.n), k):
            if is_acyclic_induced(g, list(S)):
                return k
    return 0


# ------------------------------------------------------------------------------------------- decomposition
@pytest.mark.parametrize("seed", range(10))
def test_scc_matches_networkx(seed):
    g = rand_digraph(25, 0.08, seed)
    D = nx.DiGraph()
    D.add_nodes_from(range(g.n))
    D.add_edges_from((i, j) for i in range(g.n) for j in range(g.n) if (g.out[i] >> j) & 1)
    ours = sorted(tuple(c) for c in scc(g))
    ref = sorted(tuple(sorted(c)) for c in nx.strongly_connected_components(D))
    assert ours == ref


@pytest.mark.parametrize("seed", range(12))
def test_minrank_and_mais_additive_over_sccs(seed):
    g = rand_digraph(7, 0.25, seed)
    if g.num_edges() > 16:
        pytest.skip("too many edges for brute force")
    comps = decompose(g)
    assert mais_brute(g) == sum(mais_brute(c.graph) for c in comps)
    assert minrank_bruteforce(g) == sum(minrank_bruteforce(c.graph) for c in comps)
    rows = lift_rows(g, comps, [[1 << i for i in range(c.graph.n)] for c in comps])
    assert verify(g, rows) == g.n


# ------------------------------------------------------------------------------------------------ LP bounds
@pytest.mark.parametrize("n", [5, 7, 9])
def test_odd_cycles_pinned(n):
    g = gen.cycle(n)
    s = shannon_bound(g)
    f = frac_clique_cover(g, 5)
    assert s.bound == Fraction(n, 2) == f.value
    assert check_shannon_certificate(g, s.y, s.mu) >= s.bound


@pytest.mark.parametrize("seed", range(10))
def test_shannon_sandwich(seed):
    g = rand_digraph(6, 0.4, seed)
    s = shannon_bound(g)
    assert mais_brute(g) <= s.bound + Fraction(1, 10**6)
    if g.num_edges() <= 16:
        assert s.bound <= minrank_bruteforce(g)
    f = frac_clique_cover(g, 5)
    assert s.bound <= f.value
    assert abs(float(s.bound) - s.lp_value) < 1e-6


@pytest.mark.parametrize("seed", range(6))
def test_shannon_additive_over_sccs(seed):
    """beta(G) = sum over SCCs (Tahmasbi et al.); the LP bound should also add up."""
    g = rand_digraph(8, 0.2, seed)
    comps = decompose(g)
    whole = shannon_bound(g).lp_value
    parts = sum(shannon_bound(c.graph).lp_value for c in comps)
    assert abs(whole - parts) < 1e-6


def test_tampered_certificates_rejected():
    g = gen.cycle(5)
    s = shannon_bound(g)
    y = dict(s.y)
    k = next(iter(y))
    y[k] = y[k] * 3
    assert check_shannon_certificate(g, y, s.mu) < s.bound
    f = frac_clique_cover(g, 5)
    with pytest.raises(ValueError):
        check_frac_cover(g, f.cliques, [w / 2 for w in f.weights])
    with pytest.raises(ValueError):
        check_frac_cover(g, [[0, 2]] + f.cliques, [Fraction(1)] + f.weights)  # 0, 2 are not adjacent in C5


# ------------------------------------------------------------------------------------------------------ MAIS
@pytest.mark.parametrize("seed", range(8))
def test_mais_solvers(seed):
    g = rand_digraph(9, 0.35, seed)
    opt = mais_brute(g)
    a = greedy_fvs(g)
    b = sa_order(g, 0.3, random.Random(seed), init=a)
    c = exact_lazy(g, 10, init=b)
    for s in (a, b, c.verts):
        assert is_acyclic_induced(g, s)
    assert len(a) <= len(b) <= opt
    assert c.proven and c.value == opt and c.upper == opt


def test_sandbox_skeleton_matches_library():
    from pincer.funsearch.evaluate import SEED_PROGRAM
    import pincer.funsearch.sandbox_runner as sr

    ns: dict = {}
    exec(SEED_PROGRAM, ns)
    for seed in range(5):
        g = rand_digraph(40, 0.1, seed)
        assert sorted(sr.greedy(g.n, list(g.out), ns["priority"])) == greedy_fvs(g)


# -------------------------------------------------------------------------------------- agent and bundle
def _run(g, policy="index", budget=15.0):
    from pincer.agent import Pincer
    from pincer.policies import make_policy

    return Pincer(g, budget, make_policy(policy), 0).run()


def test_agent_closes_c5_and_bundle_verifies(tmp_path):
    g = gen.cycle(5)
    r = _run(g)
    assert r.totals["mr"] == [3, 3] and r.totals["beta"] == ["5/2", "5/2"]
    b = json.loads(json.dumps(r.bundle))
    assert verify_bundle(g, b)["beta"] == ["5/2", "5/2"]


def test_bundle_tampering_detected():
    g = rand_digraph(20, 0.2, 3)
    r = _run(g, budget=6.0)
    b = json.loads(json.dumps(r.bundle))
    verify_bundle(g, b)
    bad = json.loads(json.dumps(b))
    bad["totals"]["mr"][0] += 1
    with pytest.raises(ValueError):
        verify_bundle(g, bad)
    bad = json.loads(json.dumps(b))
    bad["acyclic"] = list(range(g.n))
    with pytest.raises(ValueError):
        verify_bundle(g, bad)
    bad = json.loads(json.dumps(b))
    rows = [int(x, 16) for x in bad["rows"]]
    rows[0] |= g.forbidden(0) & -g.forbidden(0) if g.forbidden(0) else 0
    bad["rows"] = [f"{x:x}" for x in rows]
    if g.forbidden(0):
        with pytest.raises(ValueError):
            verify_bundle(g, bad)


def test_roundrobin_and_multi_scc():
    from pincer.bench import clustered

    g = clustered(4, 6, 0.7, 0.2, 0.0, 1)
    r = _run(g, "roundrobin", 20.0)
    assert r.totals["components"] >= 4
    assert r.verified["mr"] == r.totals["mr"]


def test_llm_policy_with_scripted_backend():
    from pincer.agent import Pincer
    from pincer.llm import ScriptedBackend
    from pincer.policies import LLMPolicy

    be = ScriptedBackend(["not json", '{"choice": 99}', '{"choice": 0, "slice": 0.5}'] + ['{"choice": 0}'] * 50)
    pol = LLMPolicy(be)
    r = Pincer(rand_digraph(40, 0.3, 0), 8.0, pol, 0).run()
    assert pol.fallbacks >= 2  # the bad replies fell back to the index choice
    assert r.verified["mr"] == r.totals["mr"]
    assert "candidates" in be.prompts[0]


# ----------------------------------------------------------------------------------------------- FunSearch
def test_static_check_and_crash_feedback():
    from pincer.funsearch.evaluate import evaluate, static_check

    assert static_check("import os\ndef priority(v, ctx):\n    return 0\n")
    assert static_check("def priority(v, ctx):\n    return open('x')\n")
    assert static_check("def priority(v, ctx):\n    ctx.alive = 0\n    return 0\n")
    assert static_check("def priority(v, ctx):\n    return ctx.in_deg(v)\n") is None
    gs = [rand_digraph(15, 0.2, 0)]
    r = evaluate("def priority(v, ctx):\n    return undefined_name + 1\n", graphs=gs, refs=[5])
    assert r.score is None and "NameError" in r.error and "candidate.py" in (r.traceback or "")
    r = evaluate("def priority(v, ctx):\n    while True:\n        pass\n", graphs=gs, refs=[5], timeout=2)
    assert r.score is None and "timeout" in r.error


def test_gp_programs_are_valid():
    from pincer.funsearch.database import ProgramDatabase
    from pincer.funsearch.evaluate import evaluate, static_check
    from pincer.funsearch.gp import GPProposer

    gs = [rand_digraph(20, 0.2, s) for s in range(2)]
    refs = [len(sa_order(g, 0.3, random.Random(0))) for g in gs]
    db = ProgramDatabase(num_islands=2, reset_every=5)
    prop = GPProposer(0)
    from pincer.funsearch.evaluate import SEED_PROGRAM

    r0 = evaluate(SEED_PROGRAM, graphs=gs, refs=refs)
    db.register(SEED_PROGRAM, r0.score, r0.per_graph, None, 0)
    for it in range(8):
        isl, parents = db.sample()
        code = prop.propose(parents)
        assert static_check(code) is None, code
        r = evaluate(code, graphs=gs, refs=refs)
        assert r.score is not None, r.error
        db.register(code, r.score, r.per_graph, isl, it)
    assert db.best().score >= r0.score


def test_extract_llm_reply():
    from pincer.funsearch.loop import extract

    reply = "Here you go:\n```python\ndef priority_v2(v, ctx):\n    x = ctx.in_deg(v)\n    return x * 2\n\nprint(1)\n```"
    code = extract(reply)
    assert "def priority(v, ctx):" in code and "print(1)" not in code
