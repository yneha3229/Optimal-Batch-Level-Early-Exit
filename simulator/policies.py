"""Exit policies. Each factory returns decide(exit_idx, confs, n_active, batch_size, layers_left, a, b) -> (mask, stop).

Baselines: per-sample threshold (DeeBERT-style), and DREX's whole-batch rules Consensus / Majority / Greedy.
Ours: `theory_rule`, which stops the whole batch when the expected accuracy gain of the remaining samples is
worth less than the batch cost of continuing (a + b*n per layer), and lets a sample exit alone only when its gain
is worth less than its own per-sample cost b.
"""
from __future__ import annotations

import numpy as np

from simulator.simulator import Cost, Traces, full_latency


def never_exit():
    def decide(e, confs, n, B, left, a, b):
        return np.zeros(n, bool), False
    return decide


def threshold(tau: float):
    def decide(e, confs, n, B, left, a, b):
        return confs > tau, False
    return decide


def consensus(tau: float):
    def decide(e, confs, n, B, left, a, b):
        return np.zeros(n, bool), bool((confs > tau).all())
    return decide


def majority(tau: float):
    def decide(e, confs, n, B, left, a, b):
        return np.zeros(n, bool), bool((confs > tau).mean() > 0.5)
    return decide


def greedy(tau: float):
    def decide(e, confs, n, B, left, a, b):
        return np.zeros(n, bool), bool((confs > tau).any())
    return decide


def unified_exit(exit_idx: int):
    """SkipDecode-style: one data-independent exit shared by the whole batch."""
    def decide(e, confs, n, B, left, a, b):
        return np.zeros(n, bool), e >= exit_idx
    return decide


def param_collective(tau: float, frac_cut: float, conf_floor: float):
    """Hand-parameterized collective rule (random-search family): per-sample threshold plus
    'stop the batch when few remain and they are confident enough'."""
    def decide(e, confs, n, B, left, a, b):
        mask = confs > tau
        rest = confs[~mask]
        stop = (len(rest) == 0) or ((len(rest) / B) < frac_cut and rest.mean() > conf_floor)
        return mask, bool(stop)
    return decide


class Calibration:
    """P(correct at exit e' | conf bin at exit e), estimated on the evolve split (labels used offline only)."""

    def __init__(self, tr: Traces, n_bins: int = 20, prior: float = 30.0, monotone: bool = True) -> None:
        """prior: Beta-style shrinkage of each bin towards the exit-level rate (pseudo-count);
        monotone: P(correct at e' | conf_e) made non-decreasing in conf_e (isotonic, count-weighted)."""
        from sklearn.isotonic import IsotonicRegression
        E = tr.conf.shape[1]
        self.edges = [np.unique(np.quantile(tr.conf[:, e], np.linspace(0, 1, n_bins + 1))) for e in range(E)]
        self.table = []  # table[e][bin, e'] = P(correct at e' | bin of conf_e)
        corr = tr.correct.astype(float)
        base = corr.mean(0)
        for e in range(E):
            bins = self._bin(e, tr.conf[:, e])
            nb = len(self.edges[e]) - 1
            cnt = np.bincount(bins, minlength=nb).astype(float)
            t = np.stack([corr[bins == k].sum(0) for k in range(nb)])        # [nb, E] correct counts
            t = (t + prior * base) / (cnt[:, None] + prior)
            if monotone and nb > 1:
                for e2 in range(E):
                    t[:, e2] = IsotonicRegression().fit_transform(np.arange(nb), t[:, e2], sample_weight=cnt + 1e-9)
            self.table.append(t)

    def _bin(self, e: int, c: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges[e], c, side="right") - 1, 0, len(self.edges[e]) - 2)

    def p_correct(self, e: int, confs: np.ndarray) -> np.ndarray:
        """[n, E]: probability each sample is correct if it exits at each exit e' (columns < e unused)."""
        return self.table[e][self._bin(e, confs)]


class Calibration2D:
    """P(correct at e' | conf bin at e, trend bin at e): the trend is conf_e - conf_{e-1} (terciles: falling, flat,
    rising). Each cell is shrunk towards the 1-D table, and made monotone in conf within each trend bin.
    An input whose confidence is stuck low is not the same as one that is climbing; DeepSloth-style slowdown
    inputs are the extreme case (confidence ~0.04 at every exit)."""

    def __init__(self, tr: Traces, n_bins: int = 10, n_trend: int = 3, prior: float = 30.0) -> None:
        from sklearn.isotonic import IsotonicRegression
        self.c1 = Calibration(tr, n_bins=n_bins, prior=prior)
        E = tr.conf.shape[1]
        corr = tr.correct.astype(float)
        self.tedges, self.table = [None], [None]
        for e in range(1, E):
            d = tr.conf[:, e] - tr.conf[:, e - 1]
            te = np.quantile(d, np.linspace(0, 1, n_trend + 1))[1:-1]
            kc, kd = self.c1._bin(e, tr.conf[:, e]), np.searchsorted(te, d, side="right")
            nb = len(self.c1.edges[e]) - 1
            base = self.c1.table[e]                                    # [nb, E]
            t = np.zeros((nb, n_trend, E))
            for j in range(n_trend):
                for k in range(nb):
                    sel = (kc == k) & (kd == j)
                    t[k, j] = (corr[sel].sum(0) + prior * base[k]) / (sel.sum() + prior)
                cnt = np.array([((kc == k) & (kd == j)).sum() for k in range(nb)], float)
                for e2 in range(E):
                    t[:, j, e2] = IsotonicRegression().fit_transform(np.arange(nb), t[:, j, e2], sample_weight=cnt + 1e-9)
            self.tedges.append(te)
            self.table.append(t)

    def p_correct(self, e: int, hist: np.ndarray) -> np.ndarray:
        if e == 0:
            return self.c1.p_correct(0, hist[:, -1])
        kc = self.c1._bin(e, hist[:, -1])
        kd = np.searchsorted(self.tedges[e], hist[:, -1] - hist[:, -2], side="right")
        return self.table[e][kc, kd]


def _trimmed_sum(x: np.ndarray, m: int) -> float:
    """Sum of x without its m largest entries (0 if len(x) <= m)."""
    if m <= 0:
        return float(x.sum())
    if len(x) <= m:
        return 0.0
    return float(np.partition(x, len(x) - m)[: len(x) - m].sum())


def theory_rule(lam: float, cal, exit_layers: list[int], cost: Cost, collective: bool = True,
                individual: bool = True, trim_frac: float = 0.0):
    """Optimal-stopping rule for score = accuracy - lam * relative_latency, one-step plan lookahead.

    At exit e with n active samples (batch size B, full-model latency F):
      - sample j exits alone iff  g_j / B  <=  lam * b * layers_left / F
        (an individual exit only saves the per-sample cost b on each remaining layer)
      - the rest stop collectively iff for every later exit e':
            sum_j g_j(e->e') / B  <=  lam * [(l_e' - l_e)(a + b n') + heads] / F
        (stopping the batch saves the fixed cost a too, and a is what dominates when rho = a/(b n) is large)
    g_j(e->e') = P(correct at e' | conf_j at e) - P(correct at e | conf_j at e), clipped at 0: a negative estimate
    (overthinking) is mostly calibration noise, and acting on it makes inputs exit even at lam = 0.
    Strict inequalities, so lam = 0 is exactly the full model.

    trim_frac > 0 makes the collective test robust: the m = floor(trim_frac * B) largest gains are left out of the
    sum. With k <= m slowdown inputs in the batch, the trimmed sum is sandwiched between the clean remnant's
    m-trimmed and (m-k)-trimmed sums, so the batch stops no later than the clean batch would under the rule trimmed
    by m - k: victims' latency is bounded by a clean-batch latency, without detecting who the attackers are.
    cal = Calibration2D makes the rule trajectory-aware (decide then receives confs at exits 0..e).
    Stopping early also pays cost.skip per skipped layer (decoder KV propagation), which the cost terms include.
    """
    L, E = exit_layers, len(exit_layers)
    hist = isinstance(cal, Calibration2D)

    def decide(e, confs, n, B, left, a, b):
        F = full_latency(L, cost, B)
        pc = cal.p_correct(e, confs)              # [n, E]
        g = np.maximum(pc - pc[:, [e]], 0.0)       # gain of continuing to each e'
        mask = np.zeros(n, bool)
        if individual:
            mask = g[:, E - 1] / B < lam * b * left / F
        if not collective:
            return mask, False
        rest = ~mask
        n_rest = int(rest.sum())
        if n_rest == 0:
            return mask, False
        m = int(trim_frac * B)
        for e2 in range(e + 1, E):
            gain = _trimmed_sum(g[rest, e2], m) / B
            dcost = (L[e2] - L[e]) * (a + b * n_rest - cost.skip) + (e2 - e) * cost.head
            if gain >= lam * dcost / F:
                return mask, False             # some continuation is worth its cost
        return mask, True
    decide.history = hist
    return decide
