"""B3: Least-Difference Greedy (LDG), Birk & Kol (ISCOD, INFOCOM 1998; IEEE Trans. Inf. Theory 2006).

What is implemented (unicast, one message per receiver; the setting of this project)
------------------------------------------------------------------------------------
Each receiver i is a *specification row* over the n messages with three states per position:
    1  at position i            (x_i is wanted),
    *  at every j in S_i        (known: the coefficient there is free),
    0  everywhere else          (unknown: the coefficient must be 0).
Two rows (or merged groups) are *compatible* iff no position holds 1 in one and 0 in the other. Merging takes
position-wise: 1 if either is 1, else 0 if either is 0, else *. The merged row is one transmission: the XOR of the
messages at its 1-positions. Every member i decodes it, because all other 1-positions are members j != i, and
compatibility forces position j to be * (known) in i's row.

LDG repeatedly merges the compatible pair with the least *difference*, the number of positions whose states differ
(ties: lowest index pair), until no compatible pair is left. The result is one row per group.

Fitting matrix: row i of A = indicator of the 1-positions of i's group. A[i, i] = 1, and every other 1 is a member
j that i knows, so A fits; rows of different groups have disjoint supports, so rank(A) = number of groups.

With one wanted message per receiver, compatibility of a group is exactly "every member knows every other member",
i.e. the groups are cliques of the mutual graph. LDG is therefore a clique cover with Birk-Kol's merge order
(least difference first), which differs from DSatur/TabuCol: it tends to merge receivers with similar side
information. Birk & Kol's general ISCOD version (several wanted blocks per receiver) reduces to this one here.

Implementation: rows are one-hot over the 3 states, S in {0,1}^(g x 3n). Positions agree iff the one-hot vectors
match, so difference(a, b) = n - <S_a, S_b>; compatibility(a, b) iff <ones_a, zeros_b> = <ones_b, zeros_a> = 0.
After a merge only the merged row changes, so one O(g n) update per step; total O(n^2 * n) = O(n^3).
"""
from __future__ import annotations

import numpy as np

from ..core.graph import Graph
from ..core.verify import fit_error


def ldg_groups(g: Graph) -> list[list[int]]:
    """Run LDG; return the groups (lists of receivers)."""
    n = g.n
    adj = g.adjacency()  # (n, n) bool: adj[i, j] = j in S_i
    ones = np.eye(n, dtype=bool)  # (n, n): state 1
    zeros = ~adj & ~ones  # (n, n): state 0 (unknown, not wanted)
    alive = np.ones(n, dtype=bool)
    members: list[list[int]] = [[i] for i in range(n)]

    def stack(o: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.concatenate([o, z, ~o & ~z], axis=-1).astype(np.int32)  # (..., 3n) one-hot states

    S = stack(ones, zeros)  # (n, 3n)
    O, Z = ones.astype(np.int32), zeros.astype(np.int32)
    agree = S @ S.T  # (n, n) positions with equal state
    conflict = (O @ Z.T) + (Z @ O.T)  # (n, n) > 0 iff incompatible
    big = np.iinfo(np.int64).max
    while True:
        diff = np.where((conflict == 0) & alive[:, None] & alive[None, :], (n - agree).astype(np.int64), big)
        np.fill_diagonal(diff, big)
        k = int(np.argmin(diff))  # row-major argmin: least difference, then lowest (a, b)
        a, b = divmod(k, n)
        if diff[a, b] == big:
            break
        a, b = min(a, b), max(a, b)
        ones[a] |= ones[b]
        zeros[a] |= zeros[b]
        alive[b] = False
        members[a] += members[b]
        members[b] = []
        # refresh row/column a of the pairwise tables (O(n * 3n))
        O[a], Z[a] = ones[a], zeros[a]
        S[a] = stack(ones[a], zeros[a])
        agree[a, :] = agree[:, a] = S @ S[a]
        c = O @ Z[a] + Z @ O[a]  # (n,)
        conflict[a, :] = conflict[:, a] = c
    return [sorted(m) for m in members if m]


def ldg_rows(g: Graph) -> list[int]:
    """Verified fitting rows from LDG: row i = indicator of i's group."""
    rows = [0] * g.n
    for grp in ldg_groups(g):
        mask = sum(1 << v for v in grp)
        for v in grp:
            rows[v] = mask
    err = fit_error(g, rows)
    if err:
        raise RuntimeError(f"LDG produced a non-fitting matrix on {g.name}: {err}")
    return rows
