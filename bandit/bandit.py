"""Label-free, partial-feedback bandit for batch stopping, and a CeeBERT-style per-sample bandit baseline.

LabelFreeStopRule (ours)
------------------------
The theory rule (simulator/policies.theory_rule) needs g_j(e -> e') = P(correct at e' | conf_j at e) - P(correct at
e | conf_j at e), which takes labels. We replace "correct" with "agrees with the final exit" (the full model's
answer), so the target is fidelity to the full model and no label is ever used:

    D[e, k, e'] = P(pred_e' != pred_final | conf at exit e in bin k)       gain(e -> e') = D[e,k,e] - D[e,k,e']

Feedback is censored: once the batch stops, what deeper exits would have said is never seen. A *probe* batch runs
to full depth (still answers, at full cost), which reveals its whole trace, so every table cell its samples touch
is updated. Probes happen with probability p_t = min(1, c * t^(-1/3)), the forced-exploration rate for partial
monitoring with a revealing action (regret O(T^(2/3)), cf. label-efficient prediction). Between probes the rule
uses an optimistic (upper-confidence) gain, so with little data it continues rather than stops.

Context = (exit index e, how many inputs are still active n): n enters through the cost of continuing, a + b*n per
layer, which is exactly why the implied stopping threshold drops as the batch drains (H3).

Optional dual update: with target_disagree = 0.01, lambda is adapted on probe batches so that the estimated
disagreement with the full model stays at 1%, i.e. "accuracy within 1% of the full model" without labels.

CeeBertStyle (baseline)
-----------------------
UCB1 over per-sample confidence thresholds; reward per batch = mean confidence at the exit taken minus
mu * mean depth fraction (CeeBERT's confidence-minus-cost reward). It never stops a batch collectively.
"""
from __future__ import annotations

import numpy as np

from simulator.simulator import Cost, full_latency, run_batch


class LabelFreeStopRule:
    def __init__(self, exit_layers: list[int], cost: Cost, num_classes: int, lam: float = 0.1,
                 n_bins: int = 12, beta: float = 1.0, probe_c: float = 1.0, min_probe: float = 0.0,
                 individual: bool = True, target_disagree: float | None = None, eta: float = 2.0,
                 seed: int = 0) -> None:
        self.L, self.E, self.cost = list(exit_layers), len(exit_layers), cost
        self.lam, self.beta, self.probe_c, self.min_probe = lam, beta, probe_c, min_probe
        self.individual, self.target, self.eta = individual, target_disagree, eta
        self.edges = np.linspace(1.0 / num_classes, 1.0, n_bins + 1)[1:-1]
        self.nb = n_bins
        self.cnt = np.zeros((self.E, n_bins))                  # samples seen at exit e in bin k
        self.dis = np.zeros((self.E, n_bins, self.E))          # ... of which disagree with final at exit e'
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.n_probes = 0
        self.lam_hist: list[float] = []

    # ------------------------------------------------------------------ feedback
    def probe_prob(self) -> float:
        return float(min(1.0, max(self.min_probe, self.probe_c * (self.t + 1) ** (-1 / 3))))

    def is_probe(self) -> bool:
        self.t += 1
        return bool(self.rng.random() < self.probe_prob())

    def _bin(self, conf: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.edges, conf, side="right")

    def observe_probe(self, conf: np.ndarray, pred: np.ndarray) -> None:
        """conf, pred: [B, E] full trace of a probe batch."""
        self.n_probes += 1
        dis = pred != pred[:, -1:]                              # [B, E] disagreement with the final exit
        for e in range(self.E - 1):
            k = self._bin(conf[:, e])
            np.add.at(self.cnt[e], k, 1)
            np.add.at(self.dis[e], k, dis.astype(float))
        if self.target is not None:
            d_hat = self._replay_disagreement(conf, pred)
            self.lam *= float(np.exp(-self.eta * (d_hat - self.target) / max(self.target, 1e-6) * 0.1))
            self.lam = float(np.clip(self.lam, 1e-4, 1e3))
        self.lam_hist.append(self.lam)

    def _replay_disagreement(self, conf: np.ndarray, pred: np.ndarray) -> float:
        """Counterfactual: what fraction of this probe batch would the current rule have answered differently from
        the full model? (A probe reveals the whole trace, so the replay is exact.)"""
        class _T:  # minimal Traces stand-in
            pass
        t = _T()
        t.conf, t.pred, t.exit_layers = conf, pred, self.L
        t.label = pred[:, -1]
        o = run_batch(self.decide, t, np.arange(len(conf)), self.cost)
        return float((o.pred != pred[:, -1]).mean())

    # ------------------------------------------------------------------ decision
    def gain_table(self, e: int, k: np.ndarray, optimistic: bool = True) -> np.ndarray:
        """[n, E] estimated disagreement gain of continuing from exit e to every e' (optimistic = UCB)."""
        c = self.cnt[e, k]
        d = (self.dis[e, k] + 0.5) / (c[:, None] + 1.0)        # Beta(0.5, 0.5)-smoothed rates
        g = np.maximum(d[:, [e]] - d, 0.0)                      # gain(e -> e'), [n, E], noise-level negatives -> 0
        if optimistic:
            # empirical-Bernstein bonus on the log of the number of probes (the information count). A gain is a
            # difference of probabilities, so it is capped at 1: without the cap one input in a sparse bin
            # (5 samples -> "gain" 5.5) blocked every collective stop.
            lt = np.log(self.n_probes + 2)
            var = d[:, [e]] * (1 - d[:, [e]]) + d * (1 - d)
            g = g + self.beta * np.sqrt(2 * var * lt / (c[:, None] + 1.0)) + lt / (c[:, None] + 1.0)
            g = np.minimum(g, 1.0)
            g[:, :e + 1] = 0.0
        return g

    def decide(self, e, confs, n, B, left, a, b):
        L, E = self.L, self.E
        F = full_latency(L, self.cost, B)
        k = self._bin(confs)
        g = self.gain_table(e, k)
        mask = np.zeros(n, bool)
        if self.individual:
            mask = g[:, E - 1] / B < self.lam * b * left / F
        rest = ~mask
        n_rest = int(rest.sum())
        if n_rest == 0:
            return mask, False
        for e2 in range(e + 1, E):
            gain = g[rest, e2].sum() / B
            dcost = (L[e2] - L[e]) * (a + b * n_rest) + (e2 - e) * self.cost.head
            if gain >= self.lam * dcost / F:
                return mask, False
        return mask, True

    def implied_threshold(self, e: int, n: int, B: int) -> float:
        """Smallest confidence c such that a batch remnant of n inputs, all at confidence c, is stopped at exit e.
        (Point estimate, no optimism.) Returns nan if it would never stop."""
        L, E = self.L, self.E
        F = full_latency(L, self.cost, B)
        for k in range(self.nb):
            g = self.gain_table(e, np.array([k]), optimistic=False)[0]
            ok = all(n * g[e2] / B <= self.lam * ((L[e2] - L[e]) * (self.cost.a + self.cost.b * n)
                                                 + (e2 - e) * self.cost.head) / F for e2 in range(e + 1, E))
            if ok:
                lo = 1.0 / (len(self.edges) + 1) if k == 0 else self.edges[k - 1]
                return float(lo)
        return float("nan")


class CeeBertStyle:
    """UCB1 over per-sample thresholds; label-free reward = mean conf at exit - mu * mean depth / L."""

    def __init__(self, taus: np.ndarray, mu: float, exit_layers: list[int], c: float = 1.0) -> None:
        self.taus, self.mu, self.L, self.c = np.asarray(taus), mu, exit_layers, c
        self.n = np.zeros(len(taus))
        self.s = np.zeros(len(taus))
        self.t = 0
        self.arm = 0

    def choose(self) -> int:
        self.t += 1
        if (self.n == 0).any():
            self.arm = int(np.flatnonzero(self.n == 0)[0])
        else:
            ucb = self.s / self.n + self.c * np.sqrt(2 * np.log(self.t) / self.n)
            self.arm = int(ucb.argmax())
        return self.arm

    def decide(self, e, confs, n, B, left, a, b):
        return confs > self.taus[self.arm], False

    def update(self, conf_at_exit: np.ndarray, exit_of: np.ndarray) -> None:
        depth = np.array(self.L)[exit_of] / self.L[-1]
        r = float(conf_at_exit.mean() - self.mu * depth.mean())
        self.n[self.arm] += 1
        self.s[self.arm] += r
