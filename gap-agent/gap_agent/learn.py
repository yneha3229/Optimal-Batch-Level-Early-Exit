"""Offline training of the learned agent: batch reinforcement learning with fitted Q-iteration.

Data: transitions (state, strategy, reward, next state, done) from real runs on training graphs. Round 0 uses random
switching (each strategy kept for a while with prob. 0.5, so long runs of one strategy are seen too); later rounds
use the current learned agent with 15 % random choices, so the data covers the states the agent itself reaches.

Reward of a slice: (gap closed in the slice / initial gap) x (time left after the slice / budget). The sum over a run
equals the drop in the gap integral, so maximising return = closing the gap early.

Q(state, strategy) is a gradient-boosted regression model over the state features plus a one-hot of the strategy;
fitted Q-iteration repeats   y = r + gamma * max_a' Q(s', a')   and refits.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .agent import A

SCALE = 100.0


def _xa(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    return np.hstack([obs, np.eye(A, dtype=np.float32)[actions]])


def fitted_q(transitions, iters: int = 12, gamma: float = 0.97, log=print):
    obs = np.stack([t[0] for t in transitions])
    act = np.array([t[1] for t in transitions])
    rew = np.array([t[2] for t in transitions]) * SCALE
    nxt = np.stack([t[3] for t in transitions])
    done = np.array([t[4] for t in transitions], dtype=bool)
    X = _xa(obs, act)
    Xn = np.vstack([_xa(nxt, np.full(len(nxt), a)) for a in range(A)])
    y = rew.copy()
    model = None
    for k in range(iters):
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=40, l2_regularization=1.0, random_state=k)
        model.fit(X, y)
        qn = model.predict(Xn).reshape(A, -1).max(0)
        y_new = rew + gamma * np.where(done, 0.0, qn)
        log(f"  FQI iteration {k}: mean target {y_new.mean():.3f}, change {np.abs(y_new - y).mean():.4f}")
        y = y_new
    return model


def rate_model(transitions):
    """Supervised model of the gain rate of a strategy in the next slice (gap closed per second / initial gap),
    from the state before the slice. The target is read from the next state's 'recent rate' feature."""
    obs = np.stack([t[0] for t in transitions])
    act = np.array([t[1] for t in transitions])
    nxt = np.stack([t[3] for t in transitions])
    y = nxt[np.arange(len(act)), nxt.shape[1] - A + act] * SCALE
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)
    model.fit(_xa(obs, act), y)
    return model
