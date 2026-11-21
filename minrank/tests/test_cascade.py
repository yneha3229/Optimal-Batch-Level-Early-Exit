"""Tests for minrank/cascade (Option D). Run: python -m pytest minrank/tests/test_cascade.py -q"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from minrank.cascade.calibration import Calibration, pav
from minrank.cascade.online import CascadeGate
from minrank.cascade.policies import (Context, CostRule, CostRuleCertified, Full, HoeffdingRace, Hyperband,
                                      OpenEvolveCascade, SuccessiveHalving, clopper_pearson_upper, make_policies)
from minrank.cascade.simulator import SimConfig, default_checkpoints, simulate
from minrank.cascade.traces import FIELDS, Batch, load_traces, save_traces, synthetic_traces, to_batches


@pytest.fixture(scope="module")
def trace() -> pd.DataFrame:
    return synthetic_traces(240, 16, seed=1)


@pytest.fixture(scope="module")
def batches(trace) -> list[Batch]:
    return to_batches(trace, 8)


def fresh_policies() -> list:
    return [Full(), OpenEvolveCascade(), SuccessiveHalving(), Hyperband(), HoeffdingRace(), CostRule(0.003),
            CostRule(0.1), CostRuleCertified((0.003, 0.01, 0.1), 0.2, 0.2)]


# ---- traces --------------------------------------------------------------------------------------------------------
def test_synthetic_contract_and_roundtrip(trace, tmp_path):
    assert list(trace.columns[:len(FIELDS)]) == list(FIELDS) or set(FIELDS) <= set(trace.columns)
    assert set(trace["status"]) <= {"ok", "invalid", "timeout", "error"}
    ok = trace[trace["status"] == "ok"]
    assert np.allclose(ok["score"], (ok["cover_rank"] - ok["rank"]) / ok["cover_rank"])
    assert (trace.loc[trace["status"] != "ok", "score"] == -1.0).all()
    assert trace["score"].between(-1, 1).all()
    p = tmp_path / "t.jsonl"
    save_traces(trace, p)
    back = load_traces(p)
    assert len(back) == len(trace) and np.allclose(back["score"], trace["score"])


def test_batching_shapes(trace, batches):
    assert len(batches) == 30 and all(b.scores.shape == (8, 16) for b in batches)
    g = to_batches(synthetic_traces(50, 6, seed=2, generation_size=10), 8)
    assert [b.size for b in g] == [10] * 5  # generations override batch_size


def test_incomplete_candidates_are_dropped(trace):
    df = trace[~((trace["candidate"] == "c00000") & (trace["order"] == 3))]
    bs = to_batches(df, 8)
    assert "c00000" not in {c for b in bs for c in b.candidates}


def test_validate_rejects_bad_records(trace):
    from minrank.cascade.traces import validate
    with pytest.raises(ValueError):
        validate(trace.drop(columns=["score"]))
    bad = trace.copy()
    bad.loc[bad.index[0], "status"] = "weird"
    with pytest.raises(ValueError):
        validate(bad)


# ---- calibration and statistics ------------------------------------------------------------------------------------
def test_pav_monotone_and_mean_preserving():
    rng = np.random.default_rng(0)
    y, w = rng.random(30), rng.random(30) + 0.1
    z = pav(y, w)
    assert (np.diff(z) >= -1e-12).all()
    assert math.isclose(float((z * w).sum()), float((y * w).sum()), rel_tol=1e-9)


def test_calibration_monotone_in_gap():
    cal = Calibration(2)
    rng = np.random.default_rng(1)
    for _ in range(500):
        full = rng.normal()
        cal.add(np.array([full + rng.normal(0, 0.3), full + rng.normal(0, 0.1)]) * 0.2, full * 0.2, 0.0)
    d = np.linspace(-1, 1, 50)
    for e in range(2):
        assert (np.diff(cal.p_win(e, d)) >= -1e-12).all()
    assert cal.exp_improvement() > 0


def test_clopper_pearson():
    # zero events: bound = 1 - delta^(1/m)
    for m in (5, 22, 100):
        assert math.isclose(clopper_pearson_upper(0, m, 0.1), 1 - 0.1 ** (1 / m), rel_tol=1e-6)
    assert clopper_pearson_upper(0, 22, 0.1) <= 0.1 < clopper_pearson_upper(0, 21, 0.1)
    assert clopper_pearson_upper(3, 3, 0.1) == 1.0


# ---- policies ------------------------------------------------------------------------------------------------------
def _ctx(L=8, B=6, inc=0.0) -> Context:
    return Context(default_checkpoints(L), inc, Calibration(len(default_checkpoints(L)) - 1), 0.05, np.ones(L), B)


def test_successive_halving_keeps_at_least_one():
    sh = SuccessiveHalving()
    ctx = _ctx()
    prefix = np.random.default_rng(0).random((6, 2))
    one = np.zeros(6, dtype=bool)
    one[2] = True
    assert sh.decide(ctx, 1, prefix, one).tolist() == one.tolist()
    keep = sh.decide(ctx, 1, prefix, np.ones(6, dtype=bool))
    assert keep.sum() == 3 and keep[np.argmax(prefix.mean(1))]


def test_policies_never_revive_and_hoeffding_without_incumbent():
    ctx = _ctx(inc=-math.inf)
    prefix = np.random.default_rng(1).normal(size=(6, 4))
    alive = np.array([1, 0, 1, 1, 0, 1], dtype=bool)
    for p in fresh_policies():
        keep = p.decide(ctx, 2, prefix, alive)
        assert not (keep & ~alive).any(), p.label()
    assert (HoeffdingRace().decide(ctx, 2, prefix, alive) == alive).all()


def test_make_policies_all():
    labels = [p.label() for p in make_policies(["all"], [0.01, 0.1])]
    assert labels[0] == "full" and "cost_rule@0.01" in labels and "cost_rule@0.1" in labels
    with pytest.raises(ValueError):
        make_policies(["nope"], [0.1])


# ---- simulator -----------------------------------------------------------------------------------------------------
def test_full_policy_accounting(batches):
    cfg = SimConfig(a=0.05)
    r = simulate(batches, Full(), cfg)
    expected = sum(float(b.seconds.sum()) for b in batches) + 0.05 * batches[0].n_graphs * len(batches)
    assert math.isclose(r.summary["eval_seconds"], expected, rel_tol=1e-12)
    assert math.isclose(r.summary["full_seconds"], expected, rel_tol=1e-12)
    assert r.summary["savings"] == 0.0 and r.summary["lost_winners"] == 0
    assert r.summary["final_incumbent"] == r.summary["full_final_incumbent"]


def test_all_policies_spend_at_most_full(batches):
    for p in fresh_policies():
        s = simulate(batches, p).summary
        assert -1e-12 <= s["savings"] < 1.0, p.label()
        assert s["final_incumbent"] <= s["full_final_incumbent"] + 1e-12


def test_lambda_zero_equals_full(batches):
    a = simulate(batches, Full())
    b = simulate(batches, CostRule(0.0))
    assert math.isclose(a.summary["eval_seconds"], b.summary["eval_seconds"])
    assert all((x == y).all() for da, db in zip(a.decisions, b.decisions) for x, y in zip(da, db))


def test_cost_rule_saves_with_positive_lambda(batches):
    s0 = simulate(batches, CostRule(0.0)).summary["savings"]
    s1 = simulate(batches, CostRule(0.3)).summary["savings"]
    assert s1 > s0


@pytest.mark.parametrize("pidx", range(8))
def test_decisions_are_causal_within_a_batch(batches, pidx):
    """Changing scores of graphs >= checkpoints[e] in batch k never changes decisions up to checkpoint e."""
    cfg = SimConfig()
    cps = default_checkpoints(batches[0].n_graphs)
    k, e = 5, 1  # batch 5 is an ordinary batch (warmup 3, audit at 3, 8, ...)
    base = simulate(batches, fresh_policies()[pidx], cfg)
    rng = np.random.default_rng(7)
    pert = [Batch(b.bid, b.candidates, b.scores.copy(), b.seconds.copy(), b.graphs) for b in batches]
    pert[k].scores[:, cps[e]:] = np.clip(rng.normal(0, 0.5, pert[k].scores[:, cps[e]:].shape), -1, 1)
    other = simulate(pert, fresh_policies()[pidx], cfg)
    for kk in range(k):
        assert all((x == y).all() for x, y in zip(base.decisions[kk], other.decisions[kk]))
    for x, y in zip(base.decisions[k][:e + 1], other.decisions[k][:e + 1]):
        assert (x == y).all()


@pytest.mark.parametrize("pidx", [5, 6, 7])
def test_future_batches_do_not_affect_the_past(batches, pidx):
    """Calibration, cost estimates, incumbent and certification only use earlier batches."""
    k = 12
    rng = np.random.default_rng(3)
    alt = batches[:k] + [Batch(b.bid, b.candidates, np.clip(b.scores + rng.normal(0, 0.3, b.scores.shape), -1, 1),
                               b.seconds * 3.0, b.graphs) for b in batches[k:]]
    a = simulate(batches, fresh_policies()[pidx])
    b = simulate(alt, fresh_policies()[pidx])
    assert a.per_batch[:k] == b.per_batch[:k]
    for kk in range(k):
        assert all((x == y).all() for x, y in zip(a.decisions[kk], b.decisions[kk]))


@pytest.mark.parametrize("seed", range(3))
def test_certified_loss_rate_within_alpha(seed):
    alpha = 0.15
    bs = to_batches(synthetic_traces(960, 12, seed=seed), 8)
    pol = CostRuleCertified((0.001, 0.003, 0.01, 0.03, 0.1), alpha=alpha, delta=0.1)
    r = simulate(bs, pol, SimConfig(warmup=3, audit_every=2, seed=seed))
    normal = [x for x in r.per_batch if x["role"] == "normal"]
    rate = sum(x["lost_winner"] for x in normal) / len(normal)
    assert rate <= alpha + 0.05
    assert pol.history  # certification ran


def test_certified_is_not_vacuous():
    """On this trace certification admits a positive lambda at some point (it does not always fall back to full)."""
    bs = to_batches(synthetic_traces(960, 12, seed=1), 8)
    pol = CostRuleCertified((0.001, 0.003, 0.01, 0.03, 0.1), alpha=0.15, delta=0.1)
    r = simulate(bs, pol, SimConfig(warmup=3, audit_every=2, seed=1))
    assert max(lam for _, lam in pol.history) > 0
    assert r.summary["savings"] > 0


# ---- online gate ---------------------------------------------------------------------------------------------------
def test_online_gate(tmp_path, trace):
    p = tmp_path / "evals.jsonl"
    save_traces(trace, p)
    cps = default_checkpoints(16)
    gate = CascadeGate.from_traces(p, CostRule(0.1), cps)
    full = trace.groupby("candidate")["score"].mean()
    assert math.isclose(gate.incumbent, float(full.max()))
    out = gate.should_continue({"x": [0.9, 0.9], "y": [-1.0, -1.0]}, batch_id=0)
    assert set(out) == {"x", "y"} and out["x"] >= out["y"]
    assert isinstance(gate.gate_single([0.9, 0.9, 0.9, 0.9]), bool)
    assert gate.gate_single([-1.0], lam=0.0) is True
    with pytest.raises(ValueError):
        gate.should_continue({"x": [0.1, 0.2, 0.3]})  # 3 is not a checkpoint
