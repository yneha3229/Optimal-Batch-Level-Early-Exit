"""lambda_2 upper bound on very large graphs: block codes.

Partition the receivers into blocks and code each block on its own with a scalar linear code over GF(2): a block B
with a fitting matrix of GF(2) rank r costs r transmissions, so  lambda_2(G) <= sum of the block ranks.
A clique cover is the special case where every block is a clique (rank 1). Starting from the best clique cover,
the search repeatedly picks a few cliques whose receivers know many of each other's messages (at most MAX_UNION
receivers in total) and asks a SAT solver: "is there a fitting matrix of the union with rank one less than the
cliques cost together?" A yes merges them into one cheaper block. Example: three receivers in a directed triangle
are three cliques (cost 3) but one block of rank 2 (send x1+x2, x2+x3).

Blocks never cross SCC boundaries, so the cost is reported per SCC and combines with the other pieces' bounds.
Every block code is a matrix that check_block re-verifies (ones on the diagonal, zeros where a receiver lacks the
message, GF(2) rank).
"""
from __future__ import annotations

import numpy as np
from pysat.solvers import Solver

MAX_UNION = 48
MAX_BLOCKS = 6
CONF_BUDGET = 20_000


def rank_gf2(rows) -> int:
    rows, r = [int(x) for x in rows], 0
    while rows:
        p = max(rows)
        if p == 0:
            break
        rows = [x ^ p if x ^ p < x else x for x in rows if x != p]
        r += 1
    return r


def check_block(adj: np.ndarray, verts, rows, cost: int) -> bool:
    """rows[i] = bitmask over positions in `verts` of row i of the fitting matrix."""
    if rows is None:  # a clique block: the all-ones matrix (rank 1)
        sub = adj[np.ix_(verts, verts)]
        return cost >= 1 and bool((sub | sub.T | np.eye(len(verts), dtype=bool)).all() and (sub & sub.T).sum()
                                  == len(verts) * (len(verts) - 1))
    m = len(verts)
    for i in range(m):
        if not (rows[i] >> i) & 1:
            return False
        for j in range(m):
            if j != i and (rows[i] >> j) & 1 and not adj[verts[i], verts[j]]:
                return False
    return rank_gf2(rows) <= cost


def _encode(N: int, out: list[int], r: int):
    """CNF for: U, V in GF(2)^{N x r} with (U V^T)[p][p] = 1 and (U V^T)[p][q] = 0 where p lacks q."""
    u = lambda p, k: 1 + p * r + k                       # noqa: E731
    v = lambda q, k: 1 + N * r + q * r + k               # noqa: E731
    nxt = [1 + 2 * N * r]
    cnf = []

    def new():
        nxt[0] += 1
        return nxt[0] - 1

    for p in range(N):
        for q in range(N):
            if q != p and (out[p] >> q) & 1:
                continue
            prev = None
            for k in range(r):
                z = new()
                a, b = u(p, k), v(q, k)
                cnf += [[-z, a], [-z, b], [z, -a, -b]]
                if prev is None:
                    prev = z
                    continue
                x = new()
                cnf += [[-x, prev, z], [-x, -prev, -z], [x, -prev, z], [x, prev, -z]]
                prev = x
            cnf.append([prev] if p == q else [-prev])
    return cnf, u, v


def fit_rank(adj: np.ndarray, verts, r: int, budget: int = CONF_BUDGET):
    """A fitting matrix of rank <= r for the receivers `verts` (rows as bitmasks), False if none, None if unknown."""
    N = len(verts)
    sub = adj[np.ix_(verts, verts)]
    out = [int(sum(1 << q for q in np.nonzero(sub[p])[0])) for p in range(N)]
    cnf, u, v = _encode(N, out, r)
    with Solver(name="glucose4", bootstrap_with=cnf) as s:
        s.conf_budget(budget)
        ok = s.solve_limited()
        if ok is None:
            return None
        if not ok:
            return False
        model = set(x for x in s.get_model() if x > 0)
    rows = []
    for p in range(N):
        row = 0
        for q in range(N):
            if sum(1 for k in range(r) if u(p, k) in model and v(q, k) in model) & 1:
                row |= 1 << q
        rows.append(row)
    return rows


class BlockSearch:
    """Block-code search over a whole graph, blocks kept inside SCCs."""

    def __init__(self, adj: np.ndarray, scc_label: np.ndarray, colour: np.ndarray, seed: int = 0):
        self.adj, self.label = adj, scc_label
        self.n = adj.shape[0]
        self.rng = np.random.default_rng(seed)
        self.scc_count = int(scc_label.max()) + 1
        self.blocks = {}         # id -> [verts (np array), cost, rows or None]
        self.block_of = np.zeros(self.n, dtype=np.int64)
        self.next_id = 0
        self.scc_cost = np.zeros(self.scc_count, dtype=np.int64)
        self.tried = set()
        self.merges = 0
        self.rebase(colour, force=True)

    def _new_block(self, verts, cost, rows):
        b = self.next_id
        self.next_id += 1
        self.blocks[b] = [np.asarray(verts), cost, rows]
        self.block_of[verts] = b
        return b

    def rebase(self, colour: np.ndarray, force: bool = False) -> list[int]:
        """Restart the SCCs where the given clique cover is cheaper than the current blocks. Returns those SCCs."""
        colour = np.asarray(colour)
        per = np.zeros(self.scc_count, dtype=np.int64)
        pairs = np.unique(np.stack([self.label, colour], 1), axis=0)
        np.add.at(per, pairs[:, 0], 1)
        changed = [s for s in range(self.scc_count) if force or per[s] < self.scc_cost[s]]
        if not changed:
            return []
        ch = np.zeros(self.scc_count, dtype=bool)
        ch[changed] = True
        for b in [b for b, (vs, _, _) in self.blocks.items() if ch[self.label[vs[0]]]]:
            del self.blocks[b]
        for s, c in pairs[ch[pairs[:, 0]]]:
            vs = np.nonzero((self.label == s) & (colour == c))[0]
            self._new_block(vs, 1, None)
        self.scc_cost[ch] = per[ch]
        return changed

    def step(self) -> int | None:
        """One merge attempt. Returns the SCC whose cost dropped, or None."""
        ids = [b for b, (vs, _, _) in self.blocks.items() if len(vs) <= MAX_UNION // 3]
        if not ids:
            return None
        b0 = ids[self.rng.integers(len(ids))]
        v0 = self.blocks[b0][0]
        s = int(self.label[v0[0]])
        w = self.adj[v0].sum(0).astype(np.float64) + self.adj[:, v0].sum(1)
        w[self.label != s] = 0
        score = np.bincount(self.block_of, weights=w, minlength=self.next_id)
        size = np.bincount(self.block_of, minlength=self.next_id).astype(np.float64)
        score = np.where(size > 0, score / np.maximum(size, 1), 0)
        score[b0] = 0
        order = np.argsort(-score)[:12]
        order = [b for b in order if score[b] > 0 and b in self.blocks]
        head = order[:6]
        order = [head[k] for k in self.rng.permutation(len(head))] + order[6:]
        chosen, total = [b0], len(v0)
        for b in order:
            if len(chosen) >= MAX_BLOCKS:
                break
            if total + len(self.blocks[b][0]) <= MAX_UNION:
                chosen.append(b)
                total += len(self.blocks[b][0])
        if len(chosen) < 2:
            return None
        key = frozenset(chosen)
        if key in self.tried:
            return None
        self.tried.add(key)
        verts = np.concatenate([self.blocks[b][0] for b in chosen])
        cost = sum(self.blocks[b][1] for b in chosen)
        rows = fit_rank(self.adj, verts, cost - 1)
        if not rows:
            return None
        if not check_block(self.adj, verts, rows, cost - 1):
            raise RuntimeError("bad block code")
        for b in chosen:
            del self.blocks[b]
        self._new_block(verts, cost - 1, rows)
        self.scc_cost[s] -= 1
        self.merges += 1
        return s

    def _full_successors(self, b):
        """Clique blocks c in the same SCC such that every receiver of block b has every message of block c."""
        vs = self.blocks[b][0]
        w = self.adj[vs].sum(0).astype(np.int64)
        cnt = np.bincount(self.block_of, weights=w, minlength=self.next_id)
        size = np.bincount(self.block_of, minlength=self.next_id)
        cand = np.nonzero((cnt == len(vs) * size) & (size > 0))[0]
        s = self.label[vs[0]]
        return [int(c) for c in cand if c != b and c in self.blocks and self.blocks[c][2] is None
                and self.label[self.blocks[c][0][0]] == s]

    def cycle_step(self, max_nodes: int = 300, max_len: int = 12) -> int | None:
        """Find a directed cycle of clique blocks A1 -> A2 -> ... -> Ak -> A1 (every receiver of Ai has every message
        of A(i+1)) and code it with k - 1 transmissions: (sum of Ai) + (sum of A(i+1)) for i = 1..k-1."""
        ids = [b for b, (_, _, r) in self.blocks.items() if r is None]
        if not ids:
            return None
        b0 = ids[self.rng.integers(len(ids))]
        parent, frontier, seen = {b0: None}, [b0], 1
        for depth in range(max_len):
            nxt = []
            for b in frontier:
                for c in self._full_successors(b):
                    if c == b0 and depth >= 1:
                        cyc = [b]
                        while parent[cyc[-1]] is not None:
                            cyc.append(parent[cyc[-1]])
                        return self._merge_cycle(cyc[::-1])
                    if c not in parent:
                        parent[c] = b
                        nxt.append(c)
                        seen += 1
                        if seen > max_nodes:
                            return None
            frontier = nxt
            if not frontier:
                return None
        return None

    def _merge_cycle(self, cyc):
        k = len(cyc)
        parts = [self.blocks[b][0] for b in cyc]
        verts = np.concatenate(parts)
        start = np.cumsum([0] + [len(p) for p in parts])
        mask = [sum(1 << q for q in range(start[i], start[i + 1])) for i in range(k)]
        rows = []
        for i in range(k):
            row = mask[i] | mask[(i + 1) % k]
            rows += [row] * len(parts[i])
        if not check_block(self.adj, verts, rows, k - 1):
            raise RuntimeError("bad cycle code")
        s = int(self.label[verts[0]])
        for b in cyc:
            del self.blocks[b]
        self._new_block(verts, k - 1, rows)
        self.scc_cost[s] -= 1
        self.merges += 1
        return s

    def cycle_cover_step(self, max_len: int = 10, max_nodes: int = 3000, max_degree: float = 30.0) -> int | None:
        """Carve a directed cycle of single receivers out of the clique cover and re-cover what is left.

        Find a shortest directed cycle c0 -> c1 -> ... -> c(k-1) -> c0 through receivers that sit in clique blocks;
        the cycle is coded with k - 1 transmissions (x_ci + x_c(i+1)). The other members of the cliques it touches,
        together with the cliques of their mutual neighbours, are re-covered exactly (CP-SAT). Kept only when the
        total cost drops. Meant for sparse pieces, where the clique cover is mostly pairs and singletons."""
        from .exact import cover_exact

        if not hasattr(self, "_outdeg"):
            self._outdeg = self.adj.sum(1)
            self._scc_deg = np.bincount(self.label, weights=self._outdeg) / np.maximum(1, np.bincount(self.label))
            self._succ = None
            if self._outdeg.mean() <= max_degree:
                i, j = np.nonzero(self.adj)
                ptr = np.searchsorted(i, np.arange(self.n + 1))
                self._succ = [j[ptr[u]:ptr[u + 1]] for u in range(self.n)]
        free = [v for v in range(self.n) if self.blocks[self.block_of[v]][2] is None]
        if not free:
            return None
        v = int(free[self.rng.integers(len(free))])
        s = int(self.label[v])
        if self._scc_deg[s] > max_degree:
            return None
        parent, frontier, cyc = {v: -1}, [v], None
        for depth in range(max_len):
            nxt = []
            for u in frontier:
                for w in (self._succ[u] if self._succ is not None else np.nonzero(self.adj[u])[0]):
                    w = int(w)
                    if w == v and depth >= 1:
                        cyc = [u]
                        while parent[cyc[-1]] != -1:
                            cyc.append(parent[cyc[-1]])
                        cyc = cyc[::-1]
                        break
                    if w not in parent and self.label[w] == s and self.blocks[self.block_of[w]][2] is None:
                        parent[w] = u
                        nxt.append(w)
                if cyc or len(parent) > max_nodes:
                    break
            if cyc or len(parent) > max_nodes or not nxt:
                break
            frontier = nxt
        if not cyc:
            return None
        k = len(cyc)
        touched = {int(self.block_of[c]) for c in cyc}
        rest = np.setdiff1d(np.concatenate([self.blocks[b][0] for b in touched]), cyc)
        mutual = self.adj[rest] & self.adj[:, rest].T if len(rest) else np.zeros((0, self.n), dtype=bool)
        nb = {int(self.block_of[w]) for w in np.nonzero(mutual.any(0))[0]} - touched if len(rest) else set()
        nb = {b for b in nb if self.blocks[b][2] is None and self.label[self.blocks[b][0][0]] == s}
        pool = np.concatenate([rest] + [self.blocks[b][0] for b in nb]) if (len(rest) or nb) else np.zeros(0, int)
        old = len(touched) + len(nb)
        if len(pool):
            kk, col, _ = cover_exact(self.adj[np.ix_(pool, pool)], 0.3)
        else:
            kk, col = 0, np.zeros(0, dtype=np.int32)
        if (k - 1) + kk >= old:
            return None
        cyc = np.asarray(cyc)
        rows = [(1 << i) | (1 << ((i + 1) % k)) for i in range(k)]
        if not check_block(self.adj, cyc, rows, k - 1):
            raise RuntimeError("bad cycle code")
        for b in touched | nb:
            del self.blocks[b]
        self._new_block(cyc, k - 1, rows)
        for c in np.unique(col):
            self._new_block(pool[col == c], 1, None)
        self.scc_cost[s] -= old - (k - 1) - kk
        self.merges += 1
        return s

    def certificate(self, s: int):
        return [(vs.tolist(), c, r) for vs, c, r in self.blocks.values() if self.label[vs[0]] == s]
