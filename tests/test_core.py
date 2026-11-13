"""Core checks: simulator vs hand computation, policy validity, rescue, attacks, bandit.
Run: ~/venv/bin/python -m pytest tests -q"""
import numpy as np
import pytest

from bandit.bandit import CeeBertStyle, LabelFreeStopRule
from simulator import policies as P
from simulator.attacks import hardest_pool, victim_experiment
from simulator.rescue import combine, fit_learned_weights, make_rescue
from simulator.simulator import Cost, Traces, full_latency, run_batch, simulate, simulate_lanes


def tiny() -> Traces:
    # 2 samples, exits at layers 2 and 4. Sample 0 confident at exit 0; sample 1 not.
    probs = np.array([[[0.02, 0.95, 0.03], [0.005, 0.99, 0.005]],
                      [[0.40, 0.25, 0.35], [0.05, 0.05, 0.90]]])
    return Traces(conf=probs.max(-1), pred=probs.argmax(-1), label=np.array([1, 2]), exit_layers=[2, 4], probs=probs)


def rand_traces(N=400, E=4, C=3, seed=0) -> Traces:
    rng = np.random.default_rng(seed)
    label = rng.integers(0, C, N)
    logits = rng.normal(size=(N, E, C))
    logits[np.arange(N), :, label] += np.linspace(0.2, 3.0, E)  # deeper exits are better
    p = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    return Traces(p.max(-1), p.argmax(-1), label, [3, 6, 9, 12], probs=p)


C = Cost("t", a=1.0, b=0.5, head=0.1)


def test_full_latency_formula():
    assert np.isclose(full_latency([2, 4], C, 2), 4 * (1 + 1) + 0.1)


def test_threshold_hand_computation():
    # layers 0-2 with n=2: 2*(1+1)=4, head .1; sample 0 exits; layers 2-4 with n=1: 2*(1.5)=3, head .1 -> 7.2
    r = simulate(P.threshold(0.9), tiny(), C, batch_size=2, n_batches=1)
    assert np.isclose(r["rel_latency"], 7.2 / 8.1) and r["accuracy"] == 1.0
    assert np.isclose(r["flops_frac"], (2 * 2 + 2 * 1) / (2 * 4))
    r = simulate(P.threshold(0.9), tiny(), C, batch_size=2, n_batches=1, compact=False)
    assert np.isclose(r["rel_latency"], 8.2 / 8.1)


def test_consensus_and_unified():
    c0 = Cost("t", 1.0, 0.5, 0.0)
    assert np.isclose(simulate(P.consensus(0.9), tiny(), c0, 2, 1)["rel_latency"], 1.0)
    r = simulate(P.consensus(0.3), tiny(), c0, 2, 1)
    assert np.isclose(r["rel_latency"], 0.5) and r["accuracy"] == 0.5
    r = simulate(P.unified_exit(0), tiny(), c0, 2, 1)
    assert np.isclose(r["rel_latency"], 0.5) and r["frac_cut"] == 1.0


def test_async_ready_times():
    o = run_batch(P.threshold(0.9), tiny(), np.array([0, 1]), C)
    assert np.isclose(o.ready[0], 4.1) and np.isclose(o.ready[1], 7.2) and np.isclose(o.latency, 7.2)


def test_rescue_fixes_cut_off_sample():
    tr = tiny()
    # stop the whole batch at exit 0: sample 1 is wrong at exit 0 (pred 0, label 2)
    stop_all = lambda e, c, n, B, l, a, b: (np.zeros(n, bool), True)
    assert simulate(stop_all, tr, C, 2, 1)["accuracy"] == 0.5
    # combine() only uses exits passed, so mean over exit 0 alone = exit 0
    assert combine(tr.probs[[1]], np.array([0]), "mean")[0] == 0
    assert combine(tr.probs[[1]], np.array([1]), "mean")[0] == 2
    assert combine(tr.probs[[1]], np.array([1]), "conf")[0] == 2


def test_learned_weights_not_worse_on_fit_data():
    tr = rand_traces()
    W = fit_learned_weights(tr)
    assert (W >= 0).all() and np.allclose(np.triu(W, 1), 0)
    e = np.full(len(tr), 3)
    acc_l = (combine(tr.probs, e, "learned", W) == tr.label).mean()
    acc_last = (combine(tr.probs, e, "last") == tr.label).mean()
    assert acc_l >= acc_last - 0.02


def test_policy_masks_valid_on_random_traces():
    tr = rand_traces()
    cal = P.Calibration(tr)
    for d in [P.threshold(0.7), P.majority(0.6), P.greedy(0.9), P.param_collective(0.8, 0.2, 0.5),
              P.theory_rule(0.1, cal, tr.exit_layers, C), P.unified_exit(2)]:
        r = simulate(d, tr, C, 32, 20)
        assert 0 <= r["accuracy"] <= 1 and 0 < r["rel_latency"] <= 1.0 + 1e-9 + 3 * C.head / full_latency(tr.exit_layers, C, 32)


def test_theory_rule_regime_switch():
    """Large rho (a >> b n): individual exits save ~nothing, so the rule should prefer collective stops."""
    tr = rand_traces(N=2000)
    cal = P.Calibration(tr)
    hi = Cost("hi", a=10.0, b=0.001)
    lo = Cost("lo", a=0.001, b=1.0)
    r_hi = simulate(P.theory_rule(0.3, cal, tr.exit_layers, hi), tr, hi, 32, 50)
    r_lo = simulate(P.theory_rule(0.3, cal, tr.exit_layers, lo), tr, lo, 32, 50)
    assert r_hi["frac_cut"] > r_lo["frac_cut"]


def test_lanes_runs_and_matches_full_when_no_exits():
    tr = rand_traces()
    r = simulate_lanes(P.never_exit(), tr, Cost("t", 1.0, 0.5, 0.0), 16, 8, lanes=4)
    assert np.isclose(r["rel_latency"], 1.0) and np.isclose(r["flops_frac"], 1.0)


def test_attack_hurts_threshold_not_majority():
    tr = rand_traces(N=1000)
    att = tr.subset(hardest_pool(tr, 0.05))
    att.conf[:] = 0.34                       # attackers: never confident
    tr.conf[:, 1:] = np.maximum(tr.conf[:, 1:], 0.9)  # clean inputs: confident from exit 1 on
    hi = Cost("hi", a=5.0, b=0.01)
    thr = victim_experiment(P.threshold(0.6), tr, att, hi, 32, ks=(0, 1), n_batches=50)
    maj = victim_experiment(P.majority(0.6), tr, att, hi, 32, ks=(0, 1), n_batches=50)
    assert thr[1]["victim_slowdown"] > 1.05
    assert maj[1]["victim_slowdown"] < thr[1]["victim_slowdown"]


def test_bandit_learns_and_threshold_monotone():
    tr = rand_traces(N=3000)
    hi = Cost("hi", a=5.0, b=0.05)
    bd = LabelFreeStopRule(tr.exit_layers, hi, num_classes=3, lam=0.3, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(300):
        rows = rng.choice(len(tr), 32, replace=False)
        if bd.is_probe():
            bd.observe_probe(tr.conf[rows], tr.pred[rows])
        else:
            run_batch(bd.decide, tr, rows, hi)
    assert bd.n_probes > 10
    th = [bd.implied_threshold(1, n, 32) for n in (32, 8, 2)]
    finite = [t for t in th if not np.isnan(t)]
    assert finite and all(x >= y - 1e-9 for x, y in zip(finite, finite[1:]))   # threshold drops as batch drains


def test_ceebert_style_runs():
    tr = rand_traces()
    cb = CeeBertStyle(np.linspace(0.5, 0.95, 5), mu=0.2, exit_layers=tr.exit_layers)
    rng = np.random.default_rng(0)
    for _ in range(20):
        cb.choose()
        rows = rng.choice(len(tr), 16, replace=False)
        o = run_batch(cb.decide, tr, rows, C)
        cb.update(tr.conf[rows, o.exit_of], o.exit_of)
    assert cb.n.sum() == 20


# ---------------------------------------------------------------- robustness theorem, trajectory, decoder costs
from simulator.policies import Calibration2D, _trimmed_sum  # noqa: E402


def test_trimmed_sum_sandwich():
    """TS_m(C) <= TS_m(C u K) <= TS_{m-k}(C) for k <= m, any nonnegative gains."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        C = rng.random(rng.integers(1, 30))
        k = int(rng.integers(0, 5))
        m = int(rng.integers(k, 8))
        K = rng.random(k) * 5
        U = np.concatenate([C, K])
        assert _trimmed_sum(C, m) <= _trimmed_sum(U, m) + 1e-12
        assert _trimmed_sum(U, m) <= _trimmed_sum(C, m - k) + 1e-12


def test_robust_rule_pathwise_depth_bound():
    tr = rand_traces(N=3000)
    att = tr.subset(np.arange(50))
    att.conf[:] = 0.34                                  # never confident: maximal gain estimate
    both = Traces.concat(tr, att)
    cal = P.Calibration(tr)
    hi = Cost("hi", a=5.0, b=0.05)
    B, m = 32, 4
    rule = lambda mm: P.theory_rule(0.3, cal, tr.exit_layers, hi, trim_frac=mm / B)
    rng = np.random.default_rng(1)
    for k in range(1, m + 1):
        fixed = (lambda e, c, n, B_, l, a, b, r=rule(m - k): r(e, c, n, B, l, a, b))
        for _ in range(40):
            C = rng.choice(len(tr), B - k, replace=False)
            K = len(tr) + rng.choice(len(att), k, replace=False)
            d_att = run_batch(rule(m), both, np.concatenate([C, K]), hi).exit_of.max()
            d_cln = run_batch(fixed, both, C, hi).exit_of.max()
            assert d_att <= d_cln


def test_calibration2d_and_history_interface():
    tr = rand_traces(N=2000)
    cal2 = Calibration2D(tr)
    d = P.theory_rule(0.2, cal2, tr.exit_layers, C)
    assert d.history is True
    pc = cal2.p_correct(2, tr.conf[:5, :3])
    assert pc.shape == (5, tr.conf.shape[1]) and np.all((pc >= 0) & (pc <= 1))
    r = simulate(d, tr, C, 32, 10)
    assert 0 < r["accuracy"] <= 1


def test_decoder_cost_terms():
    # stop at exit 0 of [2, 4]: layers 0-2 cost 2*(1+1)=4 +head .1, then skip 2 layers * 0.25 = 0.5
    c = Cost("d", 1.0, 0.5, head=0.1, skip=0.25, exit_pass=0.7)
    stop_all = lambda e, cf, n, B, l, a, b: (np.zeros(n, bool), True)
    assert np.isclose(run_batch(stop_all, tiny(), np.array([0, 1]), c).latency, 4.1 + 0.5)
    # partial exit at exit 0 pays exit_pass once: 4.1 + 0.7 + layers 2-4 with n=1: 2*1.5=3 + head .1
    assert np.isclose(run_batch(P.threshold(0.9), tiny(), np.array([0, 1]), c).latency, 4.1 + 0.7 + 3.1)
