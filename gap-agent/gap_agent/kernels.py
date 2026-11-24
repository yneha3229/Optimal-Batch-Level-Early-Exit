"""Compiled inner loops (numba). Every kernel does a bounded amount of work and returns, so the caller can stop or
switch strategies between calls.

Acyclic sets are kept as a *sequence*: every arc inside the set points forward in the sequence, which proves the set
is acyclic.  Inserting a receiver v either right after its last in-set predecessor or right before its first in-set
successor keeps that property once the in-set receivers that would point backwards are removed (the move used by
Galinier, Lemamou & Bouzidi's simulated annealing for the feedback vertex set problem).

Clique covers are colourings of the conflict graph (two receivers conflict unless they are mutual).
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def seed(s):
    np.random.seed(s)


# ------------------------------------------------------------------------------------------------- acyclic sets
@njit(cache=True)
def _best_insert(v, sp, s, pp, p, pos, m):
    """Cheaper of the two insert positions for v: returns (conflicts, position, option)."""
    lp = -1
    for e in range(pp[v], pp[v + 1]):
        u = p[e]
        if pos[u] > lp:
            lp = pos[u]
    p1 = lp + 1
    c1 = 0
    fs = m
    for e in range(sp[v], sp[v + 1]):
        w = s[e]
        q = pos[w]
        if q >= 0:
            if q < p1:
                c1 += 1
            if q < fs:
                fs = q
    c2 = 0
    for e in range(pp[v], pp[v + 1]):
        q = pos[p[e]]
        if q >= fs:
            c2 += 1
    if c1 < c2 or (c1 == c2 and np.random.random() < 0.5):
        return c1, p1, 1
    return c2, fs, 2


@njit(cache=True)
def _apply_insert(v, at, opt, sp, s, pp, p, order, pos, m, tmp):
    """Insert v at sequence index `at`, dropping the receivers that would point backwards. Returns the new size."""
    if opt == 1:
        for e in range(sp[v], sp[v + 1]):
            w = s[e]
            if 0 <= pos[w] < at:
                pos[w] = -2
    else:
        for e in range(pp[v], pp[v + 1]):
            u = p[e]
            if pos[u] >= at:
                pos[u] = -2
    k = 0
    for i in range(m):
        if i == at:
            tmp[k] = v
            k += 1
        x = order[i]
        if pos[x] == -2:
            pos[x] = -1
        else:
            tmp[k] = x
            k += 1
    if at >= m:
        tmp[k] = v
        k += 1
    for i in range(k):
        order[i] = tmp[i]
        pos[tmp[i]] = i
    return k


@njit(cache=True)
def acyclic_anneal(sp, s, pp, p, order, pos, m, best_order, best_m, temp, iters):
    """Simulated annealing at a fixed temperature. Returns (size, best size)."""
    n = pos.shape[0]
    tmp = np.empty(n, dtype=np.int32)
    for _ in range(iters):
        v = np.random.randint(n)
        if pos[v] >= 0:
            continue
        c, at, opt = _best_insert(v, sp, s, pp, p, pos, m)
        d = c - 1
        if d > 0 and np.random.random() >= np.exp(-d / temp):
            continue
        m = _apply_insert(v, at, opt, sp, s, pp, p, order, pos, m, tmp)
        if m > best_m:
            best_m = m
            best_order[:m] = order[:m]
    return m, best_m


@njit(cache=True)
def acyclic_fill(sp, s, pp, p, order, pos, m, cand):
    """Insert every candidate that fits with no conflict, in the given order. Returns the new size."""
    tmp = np.empty(pos.shape[0], dtype=np.int32)
    for v in cand:
        if pos[v] >= 0:
            continue
        c, at, opt = _best_insert(v, sp, s, pp, p, pos, m)
        if c == 0:
            m = _apply_insert(v, at, opt, sp, s, pp, p, order, pos, m, tmp)
    return m


@njit(cache=True)
def acyclic_greedy(sp, s, pp, p, noise):
    """Fresh greedy set: low-degree receivers first (randomised by `noise`). Returns (order, size)."""
    n = pp.shape[0] - 1
    key = np.empty(n)
    for v in range(n):
        key[v] = (sp[v + 1] - sp[v] + pp[v + 1] - pp[v] + 1.0) * (1.0 + noise * np.random.random())
    cand = np.argsort(key).astype(np.int32)
    order = np.empty(n, dtype=np.int32)
    pos = -np.ones(n, dtype=np.int32)
    m = acyclic_fill(sp, s, pp, p, order, pos, 0, cand)
    return order, m


# ------------------------------------------------------------------------------------------------ clique covers
@njit(cache=True)
def dsatur(cp, c, noise):
    """Randomised DSatur colouring of the conflict graph. Returns (colour, number of colours)."""
    n = cp.shape[0] - 1
    col = -np.ones(n, dtype=np.int32)
    sat = np.zeros(n, dtype=np.int32)
    seen = np.zeros((n, n + 1), dtype=np.bool_)
    deg = np.empty(n)
    for v in range(n):
        deg[v] = (cp[v + 1] - cp[v]) * (1.0 + noise * np.random.random())
    k = 0
    for _ in range(n):
        bv, bs, bd = -1, -1, -1.0
        for v in range(n):
            if col[v] < 0 and (sat[v] > bs or (sat[v] == bs and deg[v] > bd)):
                bv, bs, bd = v, sat[v], deg[v]
        cc = 0
        while seen[bv, cc]:
            cc += 1
        col[bv] = cc
        if cc + 1 > k:
            k = cc + 1
        for e in range(cp[bv], cp[bv + 1]):
            w = c[e]
            if not seen[w, cc]:
                seen[w, cc] = True
                sat[w] += 1
    return col, k


@njit(cache=True)
def iterated_greedy(cp, c, col, k, rounds):
    """Culberson's iterated greedy: recolour class by class in a new class order (never uses more colours)."""
    n = col.shape[0]
    stamp = -np.ones(n + 1, dtype=np.int64)
    tick = 0
    for _ in range(rounds):
        size = np.zeros(k, dtype=np.int64)
        for v in range(n):
            size[col[v]] += 1
        r = np.random.randint(4)
        if r == 0:
            key = -size.astype(np.float64)                    # largest class first
        elif r == 1:
            key = -np.arange(k).astype(np.float64)            # reverse order
        elif r == 2:
            key = size.astype(np.float64)                     # smallest first
        else:
            key = np.random.random(k)                         # random
        rank = np.empty(k, dtype=np.int64)
        rank[np.argsort(key)] = np.arange(k)
        vkey = np.empty(n)
        for v in range(n):
            vkey[v] = rank[col[v]] + 0.5 * np.random.random()
        vo = np.argsort(vkey)
        new = -np.ones(n, dtype=np.int32)
        nk = 0
        for v in vo:
            tick += 1
            for e in range(cp[v], cp[v + 1]):
                w = c[e]
                if new[w] >= 0:
                    stamp[new[w]] = tick
            cc = 0
            while stamp[cc] == tick:
                cc += 1
            new[v] = cc
            if cc + 1 > nk:
                nk = cc + 1
        col[:] = new
        k = nk
    return k


@njit(cache=True)
def tabu_init(cp, c, col, k):
    """Conflict table gamma[v, colour] and the number of conflicting pairs for a k-colouring attempt."""
    n = col.shape[0]
    gamma = np.zeros((n, k), dtype=np.int32)
    f = 0
    for v in range(n):
        for e in range(cp[v], cp[v + 1]):
            gamma[v, col[c[e]]] += 1
    for v in range(n):
        f += gamma[v, col[v]]
    return gamma, f // 2


@njit(cache=True)
def tabu_search(cp, c, col, gamma, tabu, k, f, it, iters):
    """TabuCol (Hertz & de Werra) on a fixed number of colours k. Returns (conflicts, iteration counter)."""
    n = col.shape[0]
    best_f = f
    for _ in range(iters):
        if f == 0:
            break
        it += 1
        bd = 1 << 30
        bv, bc, ties, nconf = -1, -1, 0, 0
        for v in range(n):
            cv = col[v]
            gv = gamma[v, cv]
            if gv == 0:
                continue
            nconf += 1
            for cc in range(k):
                if cc == cv:
                    continue
                d = gamma[v, cc] - gv
                if tabu[v, cc] > it and f + d >= best_f:
                    continue
                if d < bd:
                    bd, bv, bc, ties = d, v, cc, 1
                elif d == bd:
                    ties += 1
                    if np.random.randint(ties) == 0:
                        bv, bc = v, cc
        if bv < 0:  # everything tabu: random conflicting move
            while True:
                bv = np.random.randint(n)
                if gamma[bv, col[bv]] > 0:
                    break
            bc = (col[bv] + 1 + np.random.randint(k - 1)) % k
            bd = gamma[bv, bc] - gamma[bv, col[bv]]
        old = col[bv]
        col[bv] = bc
        for e in range(cp[bv], cp[bv + 1]):
            w = c[e]
            gamma[w, old] -= 1
            gamma[w, bc] += 1
        f += bd
        tabu[bv, old] = it + int(0.6 * nconf) + np.random.randint(10)
        if f < best_f:
            best_f = f
    return f, it
