"""Clique-cover upper bounds (baselines B1, B2, B8).

A partition of the vertices into cliques of the *mutual* graph H (i ~ j iff i knows x_j and j knows x_i) gives a
fitting matrix of rank = number of cliques: row i is the indicator of i's clique (broadcast the XOR of each clique).
A clique partition of H is a proper colouring of the complement of H, so all methods here colour comp(H).

- B1 `dsatur`:        DSatur greedy colouring.
- B8 `random_greedy`: greedy colouring in random vertex orders, best of many restarts within the budget.
- B2 `tabucol`:       TabuCol (Hertz & de Werra 1987; Galinier & Hertz 2006) with decreasing k, started from DSatur.
"""
from __future__ import annotations

import time

import numpy as np

from ..core.graph import Graph, iter_bits


def conflict_graph(g: Graph) -> np.ndarray:
    """Boolean (n, n) adjacency of comp(mutual(g)): an edge means the two vertices cannot share a clique."""
    h = g.mutual().adjacency()  # (n, n)
    c = ~h
    np.fill_diagonal(c, False)
    return c


def colors_to_rows(colors: np.ndarray) -> list[int]:
    """Row i = indicator mask of the colour class of i."""
    colors = np.asarray(colors, dtype=np.int64)
    if colors.ndim != 1 or (colors < 0).any():
        raise ValueError("colours must be a 1-D array of non-negative ints")
    cls: dict[int, int] = {}
    for v, c in enumerate(colors.tolist()):
        cls[c] = cls.get(c, 0) | (1 << v)
    return [cls[c] for c in colors.tolist()]


def is_proper(conf: np.ndarray, colors: np.ndarray) -> bool:
    same = colors[:, None] == colors[None, :]  # (n, n)
    return not (conf & same).any()


def num_colors(colors: np.ndarray) -> int:
    return int(len(np.unique(colors)))


def _relabel(colors: np.ndarray) -> np.ndarray:
    _, inv = np.unique(colors, return_inverse=True)
    return inv.astype(np.int64)


def dsatur(conf: np.ndarray) -> np.ndarray:
    """DSatur greedy colouring of the graph with boolean adjacency `conf` (n, n)."""
    n = conf.shape[0]
    colors = -np.ones(n, dtype=np.int64)
    sat: list[set[int]] = [set() for _ in range(n)]
    deg = conf.sum(1)
    nbrs = [np.flatnonzero(conf[v]) for v in range(n)]
    for _ in range(n):
        unc = np.flatnonzero(colors < 0)
        # max saturation, then max degree, then lowest index
        v = int(max(unc, key=lambda u: (len(sat[u]), deg[u], -u)))
        c = 0
        while c in sat[v]:
            c += 1
        colors[v] = c
        for u in nbrs[v]:
            sat[u].add(c)
    return colors


def greedy_order(conf: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = conf.shape[0]
    colors = -np.ones(n, dtype=np.int64)
    for v in order:
        used = set(colors[conf[v] & (colors >= 0)].tolist())
        c = 0
        while c in used:
            c += 1
        colors[v] = c
    return colors


def random_greedy(conf: np.ndarray, time_limit: float, rng: np.random.Generator) -> np.ndarray:
    """B8: best greedy colouring over random orders within the time limit (at least one try)."""
    deadline = time.perf_counter() + time_limit
    best = greedy_order(conf, rng.permutation(conf.shape[0]))
    while time.perf_counter() < deadline:
        c = greedy_order(conf, rng.permutation(conf.shape[0]))
        if num_colors(c) < num_colors(best):
            best = c
    return best


def tabucol_k(conf: np.ndarray, k: int, init: np.ndarray, deadline: float, rng: np.random.Generator,
              max_iters: int = 10 ** 9) -> np.ndarray | None:
    """Search for a proper k-colouring starting from `init` (values in [0, k)). None if the deadline passes first.

    Moves recolour one conflicting vertex. gamma[v, c] = number of neighbours of v with colour c.
    Tabu tenure = 0.6 * conflicts + U{0..9} (Galinier & Hertz). Aspiration: a tabu move is allowed if it reaches a new
    best conflict count.
    """
    n = conf.shape[0]
    col = np.asarray(init, dtype=np.int64).copy()
    if col.min() < 0 or col.max() >= k:
        raise ValueError("init colours outside [0, k)")
    A = conf.astype(np.int32)  # (n, n)
    onehot = np.zeros((n, k), dtype=np.int32)
    onehot[np.arange(n), col] = 1
    gamma = A @ onehot  # (n, k)
    f = int(gamma[np.arange(n), col].sum()) // 2
    best_f = f
    tabu = np.zeros((n, k), dtype=np.int64)
    it = 0
    while f > 0:
        if it >= max_iters or (it & 63) == 0 and time.perf_counter() > deadline:
            return None
        it += 1
        cv = np.flatnonzero(gamma[np.arange(n), col] > 0)  # conflicting vertices
        own = gamma[cv, col[cv]]  # (m,)
        delta = (gamma[cv] - own[:, None]).astype(np.float64)  # (m, k)
        delta[np.arange(len(cv)), col[cv]] = np.inf
        allowed = (tabu[cv] <= it) | (f + delta < best_f)
        delta[~allowed] = np.inf
        if not np.isfinite(delta).any():
            continue
        delta = delta + rng.random(delta.shape) * 1e-3  # random tie-break
        idx = int(np.argmin(delta))
        vi, c_new = divmod(idx, k)
        v = int(cv[vi])
        c_old = int(col[v])
        f += int(gamma[v, c_new] - gamma[v, c_old])
        nb = A[v]  # (n,)
        gamma[:, c_old] -= nb
        gamma[:, c_new] += nb
        col[v] = c_new
        tabu[v, c_old] = it + int(0.6 * f) + int(rng.integers(0, 10))
        best_f = min(best_f, f)
    return col


def tabucol(conf: np.ndarray, time_limit: float, rng: np.random.Generator,
            start: np.ndarray | None = None) -> np.ndarray:
    """B2: start from DSatur (or `start`), then repeatedly try k - 1 colours with TabuCol until the deadline."""
    deadline = time.perf_counter() + time_limit
    best = _relabel(dsatur(conf) if start is None else start)
    while time.perf_counter() < deadline:
        k = num_colors(best) - 1
        if k < 1:
            break
        drop = int(np.argmin(np.bincount(best)))  # remove the smallest class
        moved = best == drop
        init = np.where(best > drop, best - 1, best)  # labels now 0..k-1 for kept vertices
        init[moved] = rng.integers(0, k, int(moved.sum()))  # recolour the removed class at random
        res = tabucol_k(conf, k, init, deadline, rng)
        if res is None:
            break
        best = _relabel(res)
    return best


def clique_cover_rows(g: Graph, method: str, time_limit: float, rng: np.random.Generator) -> list[int]:
    conf = conflict_graph(g)
    if method == "dsatur":
        c = dsatur(conf)
    elif method == "random_greedy":
        c = random_greedy(conf, time_limit, rng)
    elif method == "tabucol":
        c = tabucol(conf, time_limit, rng)
    else:
        raise ValueError(f"unknown clique-cover method {method}")
    if not is_proper(conf, c):
        raise RuntimeError(f"{method} returned an improper colouring on {g.name}")
    return colors_to_rows(c)


__all__ = ["conflict_graph", "colors_to_rows", "dsatur", "random_greedy", "tabucol", "tabucol_k",
           "clique_cover_rows", "is_proper", "num_colors", "iter_bits"]
