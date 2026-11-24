"""The control loop and the controllers that choose a strategy for each time slice.

Every 0.25 s the controller sees the state of the search (graph features, bounds, time left, how each strategy has
done so far) and picks the strategy that runs for the next slice.  The run is scored by the *gap integral*: the area
under gap(t) / gap(0) over the time budget (0 = closed at once, 1 = nothing gained).  Gaining early counts more
than gaining late.

Controllers
  RandomSwitch    a uniformly random strategy each slice (keeps it for a while with probability `stay`)
  RoundRobin      every strategy in turn
  Bandit          sliding-window UCB on gap closed per second (adaptive, learns online within one run only)
  LearnedAgent    a Q-function trained offline with fitted Q-iteration on many runs (see learn.py)
"""
from __future__ import annotations

import math
import time

import numpy as np

from .graph import Graph
from .strategies import LOWER, STRATEGIES, Search, warm_start

SLICE = 0.25
A = len(STRATEGIES)


class Stats:
    def __init__(self):
        self.used = np.zeros(A)            # seconds spent
        self.last_gain = np.full(A, -1.0)  # clock time of the last gain, -1 = never
        self.fails = np.zeros(A)           # slices in a row without gain
        self.gain = np.zeros(A)            # bound units gained
        self.recent = np.zeros(A)          # gain per second the last time each strategy ran
        self.last_action = -1
        self.streak = 0
        self.last_lo_gain = 0.0
        self.last_hi_gain = 0.0


def observe(s: Search, st: Stats, T: float, gap0: int) -> np.ndarray:
    t = s.clock()
    f = s.g.features
    g = [math.log(f["n"]) / 8, f["arc_density"], f["mutual_density"], t / T, (s.hi - s.lo) / max(1, gap0),
         s.lo / f["n"], s.hi / f["n"], (t - st.last_lo_gain) / T, (t - st.last_hi_gain) / T, min(st.streak, 20) / 20]
    last = np.zeros(A)
    if st.last_action >= 0:
        last[st.last_action] = 1
    since = np.where(st.last_gain < 0, t, t - st.last_gain) / T
    per = np.concatenate([st.used / T, since, np.minimum(st.fails, 20) / 20, st.gain / max(1, gap0),
                          st.recent / max(1, gap0)])
    return np.concatenate([g, last, per]).astype(np.float32)


def gap_integral(events, T: float, lo0: int, hi0: int, gap0: int, until: float | None = None) -> float:
    """Area under gap(t)/gap0 on [0, T] from the improvement events (bounds before the first event: lo0, hi0)."""
    lo, hi, t_prev, area = lo0, hi0, 0.0, 0.0
    for t, side, v, _ in sorted(events):
        t = min(t, T)
        area += (hi - lo) / gap0 * (t - t_prev)
        t_prev = t
        if side == "lo":
            lo = max(lo, v)
        else:
            hi = min(hi, v)
    area += (hi - lo) / gap0 * (T - t_prev)
    return area / T


def run(g: Graph, controller, T: float = 30.0, seed: int = 0, record: list | None = None, slice_s: float = SLICE):
    """One run of `controller` on g for T seconds. Returns a summary; appends transitions to `record` if given."""
    s = Search(g, seed)
    warm_start(s)
    gap0 = max(1, s.hi - s.lo)
    lo0, hi0 = s.lo, s.hi
    st = Stats()
    if hasattr(controller, "reset"):
        controller.reset(g)
    decide_time, choices = 0.0, []
    while True:
        t = s.clock()
        if t >= T or s.hi <= s.lo:
            break
        obs = observe(s, st, T, gap0)
        t1 = time.perf_counter()
        a = controller.choose(obs, st, s)
        decide_time += time.perf_counter() - t1
        lo, hi = s.lo, s.hi
        s.run(STRATEGIES[a], min(slice_s, T - s.clock()))
        t_end = min(s.clock(), T)
        gained = (s.lo - lo) + (hi - s.hi)
        reward = gained / gap0 * (T - t_end) / T
        st.used[a] += slice_s
        st.recent[a] = gained / max(1e-3, t_end - t)
        st.streak = st.streak + 1 if a == st.last_action else 1
        st.last_action = a
        if gained:
            st.last_gain[a] = t_end
            st.fails[a] = 0
            st.gain[a] += gained
            if s.lo > lo:
                st.last_lo_gain = t_end
            if s.hi < hi:
                st.last_hi_gain = t_end
        else:
            st.fails[a] += 1
        if hasattr(controller, "feedback"):
            controller.feedback(a, gained / gap0, t_end - t)
        choices.append(a)
        if record is not None:
            done = t_end >= T or s.hi <= s.lo
            record.append((obs, a, reward, observe(s, st, T, gap0), done))
    return {"graph": g.name, "n": g.n, "controller": getattr(controller, "name", "?"), "T": T,
            "lo0": lo0, "hi0": hi0, "lo": s.lo, "hi": s.hi, "gap0": gap0, "gap": s.hi - s.lo,
            "integral": gap_integral(s.events, T, lo0, hi0, gap0),
            "events": [(round(t, 3), side, int(v), who) for t, side, v, who in s.events],
            "decide_seconds": round(decide_time, 3), "decisions": len(choices),
            "share": {STRATEGIES[i]: round(choices.count(i) / max(1, len(choices)), 3) for i in range(A)},
            "acyclic_set": s.acyclic_set(), "cover": s.colour.tolist()}


# ------------------------------------------------------------------------------------------------- controllers
class RandomSwitch:
    name = "random switching"

    def __init__(self, stay: float = 0.0, seed: int = 0):
        self.stay, self.rng = stay, np.random.default_rng(seed)

    def choose(self, obs, st, s):
        if st.last_action >= 0 and self.rng.random() < self.stay:
            return st.last_action
        return int(self.rng.integers(A))


class RoundRobin:
    name = "round robin"

    def choose(self, obs, st, s):
        return (st.last_action + 1) % A


class Bandit:
    """Sliding-window UCB on gap closed per second; untried strategies first."""
    name = "online bandit (UCB)"

    def __init__(self, window: int = 12, c: float = 0.5):
        self.window, self.c = window, c

    def reset(self, g):
        self.hist = [[] for _ in range(A)]
        self.n = 0

    def feedback(self, a, gain, dt):
        self.hist[a] = (self.hist[a] + [gain / max(dt, 1e-3)])[-self.window:]
        self.n += 1

    def choose(self, obs, st, s):
        for a in range(A):
            if not self.hist[a]:
                return a
        scale = max(1e-9, max(max(h) for h in self.hist))
        ucb = [np.mean(h) / scale + self.c * math.sqrt(math.log(self.n + 1) / len(h)) for h in self.hist]
        return int(np.argmax(ucb))


class LearnedAgent:
    """Greedy policy of a learned Q-function Q(state, strategy); epsilon > 0 only while collecting data."""
    name = "learned agent"

    def __init__(self, model, epsilon: float = 0.0, seed: int = 0):
        self.model, self.eps, self.rng = model, epsilon, np.random.default_rng(seed)
        self.eye = np.eye(A, dtype=np.float32)

    def q(self, obs):
        X = np.hstack([np.repeat(obs[None], A, 0), self.eye])
        return self.model.predict(X)

    def choose(self, obs, st, s):
        if self.eps and self.rng.random() < self.eps:
            return int(self.rng.integers(A))
        return int(np.argmax(self.q(obs)))


def is_lower(a: int) -> bool:
    return STRATEGIES[a] in LOWER
