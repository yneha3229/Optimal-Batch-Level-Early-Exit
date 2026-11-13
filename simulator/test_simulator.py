"""Unit tests: simulator must match a hand computation.  Run: ~/venv/bin/python -m pytest simulator -q"""
import numpy as np

from simulator.policies import consensus, never_exit, threshold
from simulator.simulator import Cost, Traces, simulate


def tiny() -> Traces:
    # 2 samples, exits at layers 2 and 4. Sample 0 confident at exit 0; sample 1 not.
    return Traces(conf=np.array([[0.95, 0.99], [0.40, 0.90]]), pred=np.array([[1, 1], [0, 2]]),
                  label=np.array([1, 2]), exit_layers=[2, 4])


def test_never_exit_is_full_cost():
    r = simulate(never_exit(), tiny(), Cost("t", a=1.0, b=0.5, head=0.0), batch_size=2, n_batches=3)
    assert np.isclose(r["rel_latency"], 1.0) and r["accuracy"] == 1.0
    # with head cost, evaluating heads but never exiting is slower than the plain full model
    c = Cost("t", a=1.0, b=0.5, head=0.1)
    r = simulate(never_exit(), tiny(), c, batch_size=2, n_batches=1)
    assert np.isclose(r["rel_latency"], 8.2 / 8.1)
    assert simulate(never_exit(), tiny(), c, batch_size=2, n_batches=1, exits_enabled=False)["rel_latency"] == 1.0


def test_threshold_hand_computation():
    c = Cost("t", a=1.0, b=0.5, head=0.1)
    # layers 0-2 with n=2: 2*(1+1)=4, head .1; sample 0 exits; layers 2-4 with n=1: 2*(1.5)=3, head .1 -> 7.2
    # full = 4*(1+1) + .1 = 8.1
    r = simulate(threshold(0.9), tiny(), c, batch_size=2, n_batches=1)
    assert np.isclose(r["rel_latency"], 7.2 / 8.1) and r["accuracy"] == 1.0
    assert np.isclose(r["flops_frac"], (2 * 2 + 2 * 1) / (2 * 4))
    # without compaction the second segment still pays for 2 samples: 4 + .1 + 4 + .1 = 8.2
    r = simulate(threshold(0.9), tiny(), c, batch_size=2, n_batches=1, compact=False)
    assert np.isclose(r["rel_latency"], 8.2 / 8.1)


def test_consensus_waits_for_all():
    c = Cost("t", a=1.0, b=0.5, head=0.0)
    r = simulate(consensus(0.9), tiny(), c, batch_size=2, n_batches=1)
    assert np.isclose(r["rel_latency"], 1.0)  # sample 1 is not confident at exit 0 -> no stop
    r = simulate(consensus(0.3), tiny(), c, batch_size=2, n_batches=1)
    assert np.isclose(r["rel_latency"], 0.5) and r["accuracy"] == 0.5  # whole batch stops at layer 2
