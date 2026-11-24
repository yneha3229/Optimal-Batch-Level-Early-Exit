"""The strategies the agent can switch between, all working on one shared search state.

Lower bound (largest acyclic set found so far):
  lo_greedy   fresh randomised greedy sets (restarts); replaces the current set when larger
  lo_anneal   simulated annealing on the current set at a high temperature (explores, accepts losses)
  lo_descent  the same moves at a low temperature (intensifies around the current set)
  lo_kick     drop a random fifth of the current set and refill greedily (escapes a local optimum)
Upper bound (smallest clique cover found so far):
  hi_dsatur   randomised DSatur restarts
  hi_greedy   Culberson's iterated greedy on the best cover (cheap, never worse)
  hi_tabu     TabuCol trying to fit the receivers into one clique fewer than the best cover

State persists between calls: a strategy stopped now resumes where it left off when chosen again (the tabu search
keeps its colouring, conflict table and tabu list; the annealing keeps its current set).
"""
from __future__ import annotations

import time

import numpy as np

from . import kernels as K
from .graph import Graph

STRATEGIES = ["lo_greedy", "lo_anneal", "lo_descent", "lo_kick", "hi_dsatur", "hi_greedy", "hi_tabu"]
LOWER = {s for s in STRATEGIES if s.startswith("lo_")}


class Search:
    def __init__(self, g: Graph, seed: int = 0):
        self.g = g
        K.seed(seed)
        self.rng = np.random.default_rng(seed)
        n = g.n
        self.order = np.empty(n, dtype=np.int32)
        self.pos = -np.ones(n, dtype=np.int32)
        self.m = 0
        self.best_order = np.empty(n, dtype=np.int32)
        self.lo = 0
        self.colour = np.arange(n, dtype=np.int32)
        self.hi = n
        self.tabu = None                       # (k, colour, gamma, tabu, conflicts, iteration)
        self.iters = {s: 64 for s in STRATEGIES}
        self.t0 = time.perf_counter()
        self.events = []                       # (seconds, "lo"/"hi", value, strategy)
        self.current = ""

    # ----------------------------------------------------------------------------------------------- bookkeeping
    def clock(self) -> float:
        return time.perf_counter() - self.t0

    def _new_lo(self, m, order):
        if m > self.lo:
            self.lo = m
            self.best_order[:m] = order[:m]
            self.events.append((self.clock(), "lo", m, self.current))

    def _new_hi(self, k, colour):
        if k < self.hi:
            self.hi = k
            self.colour = colour.copy()
            self.events.append((self.clock(), "hi", k, self.current))

    def acyclic_set(self):
        return sorted(int(v) for v in self.best_order[: self.lo])

    # ------------------------------------------------------------------------------------------------ strategies
    def run(self, name: str, seconds: float) -> None:
        """Run one strategy for about `seconds` (in short compiled chunks)."""
        self.current = name
        end = time.perf_counter() + seconds
        step = getattr(self, "_" + name)
        while True:
            t = time.perf_counter()
            if t >= end:
                return
            asked = self.iters[name]
            done = step(asked)
            done = asked if done is None else max(1, done)
            dt = time.perf_counter() - t
            target = min(0.02, max(0.002, end - time.perf_counter()))
            if done >= asked and dt < target / 2:      # only grow a chunk that was used in full
                self.iters[name] = min(asked * 2, 1 << 22)
            elif dt > target * 2 and asked > 1:
                self.iters[name] = max(1, asked // 2)

    def _lo_greedy(self, iters):
        g = self.g
        for _ in range(max(1, iters // 64)):
            order, m = K.acyclic_greedy(g.succ_ptr, g.succ, g.pred_ptr, g.pred, 0.6)
            if m > self.m:
                self.order[:] = order
                self.m = m
                self.pos[:] = -1
                self.pos[order[:m]] = np.arange(m, dtype=np.int32)
            self._new_lo(m, order)

    def _anneal(self, iters, temp):
        g = self.g
        self.m, best = K.acyclic_anneal(g.succ_ptr, g.succ, g.pred_ptr, g.pred, self.order, self.pos, self.m,
                                        self.best_order, self.lo, temp, iters)
        if best > self.lo:  # the kernel already copied the set into best_order
            self.lo = best
            self.events.append((self.clock(), "lo", best, self.current))

    def _lo_anneal(self, iters):
        self._anneal(iters, 0.6)

    def _lo_descent(self, iters):
        self._anneal(iters, 0.15)

    def _lo_kick(self, iters):
        g = self.g
        for _ in range(max(1, iters // 256)):
            keep = self.rng.random(self.m) >= 0.2
            kept = self.order[: self.m][keep].copy()
            self.pos[:] = -1
            self.m = len(kept)
            self.order[: self.m] = kept
            self.pos[kept] = np.arange(self.m, dtype=np.int32)
            cand = self.rng.permutation(g.n).astype(np.int32)
            self.m = K.acyclic_fill(g.succ_ptr, g.succ, g.pred_ptr, g.pred, self.order, self.pos, self.m, cand)
            self._new_lo(self.m, self.order)

    def _hi_dsatur(self, iters):
        g = self.g
        for _ in range(max(1, iters // 64)):
            col, k = K.dsatur(g.conf_ptr, g.conf, 0.5)
            self._new_hi(k, col)

    def _hi_greedy(self, iters):
        g = self.g
        col = self.colour.copy()
        k = K.iterated_greedy(g.conf_ptr, g.conf, col, self.hi, max(1, iters // 16))
        self._new_hi(k, col)

    def _hi_tabu(self, iters):
        g = self.g
        if self.hi <= 1:
            return
        if self.tabu is None or self.tabu[0] >= self.hi:
            self._tabu_start(self.hi - 1)
        k, col, gamma, tabu, f, it = self.tabu
        it0 = it
        f, it = K.tabu_search(g.conf_ptr, g.conf, col, gamma, tabu, k, f, it, iters)
        self.tabu = (k, col, gamma, tabu, f, it)
        if f == 0:
            self._new_hi(k, col)
            if k > 1:
                self._tabu_start(k - 1)
        return it - it0

    def _tabu_start(self, k):
        """Start a k-colouring attempt from the best cover: empty its smallest class into random other classes."""
        g = self.g
        col = np.unique(self.colour, return_inverse=True)[1].astype(np.int32)
        while int(col.max()) + 1 > k:
            kk = int(col.max()) + 1
            drop = int(np.argmin(np.bincount(col, minlength=kk)))
            idx = np.nonzero(col == drop)[0]
            others = np.array([c for c in range(kk) if c != drop])
            col[idx] = others[self.rng.integers(0, len(others), len(idx))]
            col = np.unique(col, return_inverse=True)[1].astype(np.int32)
        gamma, f = K.tabu_init(g.conf_ptr, g.conf, col, k)
        self.tabu = (k, col, gamma, np.zeros((g.n, k), dtype=np.int64), int(f), 0)


def warm_start(search: Search) -> None:
    """Identical opening for every controller: one greedy acyclic set and one DSatur cover."""
    search.current = "warm start"
    g = search.g
    order, m = K.acyclic_greedy(g.succ_ptr, g.succ, g.pred_ptr, g.pred, 0.0)
    search.order[:] = order
    search.m = m
    search.pos[order[:m]] = np.arange(m, dtype=np.int32)
    search._new_lo(m, order)
    col, k = K.dsatur(g.conf_ptr, g.conf, 0.0)
    search._new_hi(k, col)


def compile_all() -> None:
    """Trigger numba compilation (cached on disk) outside any timed run."""
    from .graph import random_digraph

    g = random_digraph(30, 0.5, 0.5, np.random.default_rng(0))
    s = Search(g)
    warm_start(s)
    for name in STRATEGIES:
        s.run(name, 0.01)
