"""Calibration table P(win | partial score, checkpoint) for the cost rule.

Contract
--------
- A candidate *wins* if its full fitness (mean score over all L graphs) is strictly above the incumbent at the time
  its batch started.
- At checkpoint e (after the first checkpoints[e] graphs) the candidate's state is its gap
      d = partial_mean_e - incumbent,
  binned on fixed edges. Per (e, bin) we count records and wins.
- P(win | d, e) = posterior mean with a Beta prior of strength `prior_strength` centred on the checkpoint's base rate
  (Jeffreys-smoothed), then made non-decreasing in d by weighted isotonic regression (pool adjacent violators):
  a higher partial score never lowers the win probability.
- E[improvement | win] = (sum of (full - incumbent) over wins + imp_prior * imp_prior_weight) / (wins + weight).
- Records must come from candidates whose full evaluation was *not* chosen by the stopping rule (warm-up batches,
  audit batches, random exploration), otherwise the table is biased towards survivors. The simulator enforces this;
  `add` cannot check it.
- Strictly causal use: a table is only ever fed with batches that finished before the batch it is applied to.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


def pav(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression by pool adjacent violators."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    if y.shape != w.shape or y.ndim != 1:
        raise ValueError("y and w must be 1-D of equal length")
    if (w <= 0).any():
        raise ValueError("weights must be positive")
    vals: list[float] = []
    wts: list[float] = []
    lens: list[int] = []
    for yi, wi in zip(y, w):
        vals.append(yi)
        wts.append(wi)
        lens.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            tw = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / tw
            ln = lens[-2] + lens[-1]
            vals[-2:], wts[-2:], lens[-2:] = [v], [tw], [ln]
    return np.repeat(vals, lens)


@dataclass
class Calibration:
    n_checkpoints: int
    edges: np.ndarray = field(default_factory=lambda: np.linspace(-0.5, 0.5, 21))
    prior_strength: float = 2.0
    imp_prior: float = 0.02
    imp_prior_weight: float = 1.0
    counts: np.ndarray = field(init=False)
    wins: np.ndarray = field(init=False)
    imp_sum: float = field(init=False, default=0.0)
    imp_n: int = field(init=False, default=0)
    n_records: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.n_checkpoints < 1:
            raise ValueError("need at least one decision checkpoint")
        self.edges = np.asarray(self.edges, dtype=float)
        if self.edges.ndim != 1 or len(self.edges) < 2 or (np.diff(self.edges) <= 0).any():
            raise ValueError("edges must be strictly increasing with at least 2 entries")
        nb = len(self.edges) + 1  # bins: (-inf, e0), [e0, e1), ..., [e_last, inf)
        self.counts = np.zeros((self.n_checkpoints, nb), dtype=float)
        self.wins = np.zeros((self.n_checkpoints, nb), dtype=float)

    def bin_of(self, d: np.ndarray) -> np.ndarray:
        return np.digitize(np.asarray(d, dtype=float), self.edges)  # values in [0, nb-1]

    def add(self, prefix_means: np.ndarray, full: float, incumbent: float) -> None:
        """One candidate: prefix_means[e] = its mean score after checkpoint e (decision checkpoints only)."""
        prefix_means = np.asarray(prefix_means, dtype=float)
        if prefix_means.shape != (self.n_checkpoints,):
            raise ValueError(f"expected {self.n_checkpoints} prefix means, got {prefix_means.shape}")
        if not np.isfinite(incumbent):
            return  # no incumbent yet: "win" is undefined
        win = float(full > incumbent)
        bins = self.bin_of(prefix_means - incumbent)
        rows = np.arange(self.n_checkpoints)
        self.counts[rows, bins] += 1.0
        self.wins[rows, bins] += win
        self.n_records += 1
        if win:
            self.imp_sum += float(full - incumbent)
            self.imp_n += 1

    def table(self, e: int) -> np.ndarray:
        """Monotone P(win) per bin at checkpoint e, shape (nb,)."""
        c, w = self.counts[e], self.wins[e]
        base = (w.sum() + 0.5) / (c.sum() + 1.0)
        post = (w + self.prior_strength * base) / (c + self.prior_strength)
        return pav(post, c + self.prior_strength)

    def p_win(self, e: int, d: np.ndarray) -> np.ndarray:
        return self.table(e)[self.bin_of(d)]

    def exp_improvement(self) -> float:
        return (self.imp_sum + self.imp_prior * self.imp_prior_weight) / (self.imp_n + self.imp_prior_weight)

    def copy(self) -> "Calibration":
        return copy.deepcopy(self)
