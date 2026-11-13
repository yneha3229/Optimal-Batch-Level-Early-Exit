"""Straggler rescue: an input cut off by a collective stop at exit e gets a prediction that combines every exit it
passed (0..e) instead of exit e alone (BEEM-style "exits as experts").

    last      argmax p_e                                   (no rescue)
    mean      argmax mean_k p_k
    conf      argmax sum_k conf_k * p_k                    (confidence-weighted vote)
    learned   argmax sum_k w[e, k] * log p_k               (w fitted per stop exit on the evolve split; uses labels)

A rescue function has the signature  rescue(tr, rows, exit_of) -> predictions  and is passed to simulate().
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from simulator.simulator import Traces


def _need_probs(tr: Traces) -> np.ndarray:
    if tr.probs is None:
        raise ValueError("rescue needs full probability vectors: load_traces(..., with_probs=True)")
    return tr.probs


def combine(probs: np.ndarray, exit_of: np.ndarray, method: str, weights: np.ndarray | None = None) -> np.ndarray:
    """probs [n, E, C]; exit_of [n]. Returns predictions [n] using exits 0..exit_of[i] of each row."""
    n, E, C = probs.shape
    upto = np.arange(E)[None, :] <= exit_of[:, None]                    # [n, E]
    if method == "last":
        return probs[np.arange(n), exit_of].argmax(-1)
    if method == "mean":
        return (probs * upto[..., None]).sum(1).argmax(-1)
    if method == "conf":
        w = probs.max(-1) * upto
        return (probs * w[..., None]).sum(1).argmax(-1)
    if method == "learned":
        assert weights is not None
        w = weights[exit_of] * upto                                       # [n, E]
        return (np.log(np.clip(probs, 1e-8, 1)) * w[..., None]).sum(1).argmax(-1)
    raise ValueError(method)


def fit_learned_weights(tr: Traces, l2: float = 1e-3) -> np.ndarray:
    """w[e, k] >= 0 for k <= e: log-linear pooling of exits 0..e, fitted by NLL on all rows of tr (evolve split)."""
    P = np.log(np.clip(_need_probs(tr), 1e-8, 1))                         # [N, E, C]
    N, E, C = P.shape
    W = np.zeros((E, E))
    for e in range(E):
        X = P[:, :e + 1]                                                  # [N, e+1, C]

        def nll(w):
            z = np.einsum("k,nkc->nc", w, X)
            z = z - z.max(1, keepdims=True)
            lp = z - np.log(np.exp(z).sum(1, keepdims=True))
            return -lp[np.arange(N), tr.label].mean() + l2 * (w ** 2).sum()

        w0 = np.zeros(e + 1)
        w0[-1] = 1.0
        res = minimize(nll, w0, method="L-BFGS-B", bounds=[(0, None)] * (e + 1))
        W[e, :e + 1] = res.x
    return W


def make_rescue(method: str, weights: np.ndarray | None = None):
    if method == "last":
        return None

    def rescue(tr: Traces, rows: np.ndarray, exit_of: np.ndarray) -> np.ndarray:
        return combine(_need_probs(tr)[rows], exit_of, method, weights)
    return rescue
