"""Structured undirected graphs for index coding, and cheap structural features.

For an undirected side-information graph the maximum acyclic induced subgraph is just the maximum independent set,
so alpha(G) <= beta(G) <= minrk_F(G) for every field F.  The gap closes only when structure lets an explicit code
reach alpha; structure shows up in the spectrum (eigenvalues of the adjacency matrix with high multiplicity) and in
the automorphism group.  Random graphs have none of it, so their gap stays open -- that is a fact about the graph,
not a failure of the solver.

Families (all vertex-transitive or close to it, where the gap can close):
  kneser(n,k)      k-subsets of [n], adjacent iff disjoint          alpha = C(n-1,k-1) (Erdos-Ko-Rado)
  johnson(n,k)     k-subsets, adjacent iff they meet in k-1
  paley(q)         F_q (q = 1 mod 4 prime), adjacent iff difference is a quadratic residue
  cycle_power(n,r) vertices 0..n-1, adjacent iff circular distance <= r
  comp_cycle(n)    complement of the n-cycle
  cayley_sum(n,S)  circulant / Cayley graph on Z_n with connection set S
  random_gnp(n,p)  Erdos-Renyi (the control: no structure, gap stays open)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np


@dataclass
class Graph:
    name: str
    A: np.ndarray            # 0/1 symmetric adjacency, zero diagonal
    family: str
    alpha_known: int | None = None     # a known independence number, when there is a formula

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def density(self):
        n = self.n
        return float(self.A.sum()) / max(1, n * (n - 1))


def _sym(n, edges, name, family, alpha_known=None):
    A = np.zeros((n, n), dtype=np.int64)
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    np.fill_diagonal(A, 0)
    return Graph(name, A, family, alpha_known)


def kneser(n, k):
    V = list(itertools.combinations(range(n), k))
    S = [set(v) for v in V]
    edges = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V)) if S[i].isdisjoint(S[j])]
    from math import comb
    return _sym(len(V), edges, f"K({n},{k})", "kneser", comb(n - 1, k - 1))


def johnson(n, k):
    V = list(itertools.combinations(range(n), k))
    S = [set(v) for v in V]
    edges = [(i, j) for i in range(len(V)) for j in range(i + 1, len(V)) if len(S[i] & S[j]) == k - 1]
    return _sym(len(V), edges, f"J({n},{k})", "johnson")


def paley(q):
    qr = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (i - j) % q in qr]
    return _sym(q, edges, f"Paley{q}", "paley")


def cycle_power(n, r):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if min((i - j) % n, (j - i) % n) <= r]
    return _sym(n, edges, f"C{n}^{r}", "cycle_power")


def comp_cycle(n):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if min((i - j) % n, (j - i) % n) > 1]
    return _sym(n, edges, f"compC{n}", "comp_cycle", 2)


def cayley_sum(n, S):
    S = set(s % n for s in S) | set((-s) % n for s in S)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i - j) % n in S]
    return _sym(n, edges, f"Cay{n}:{sorted(S)}", "cayley")


def random_gnp(n, p, seed):
    rng = np.random.default_rng(seed)
    up = np.triu(rng.random((n, n)) < p, 1)
    A = (up | up.T).astype(np.int64)
    return Graph(f"gnp{n}_{p:.2f}_{seed}", A, "random")


def eigen_multiplicity_profile(A, fields=(2, 3, 5, 7)):
    """For each field, the largest nullity of (A - lambda I) over nonzero lambda -- how much a single spectral
    construction I + cA can remove.  A cheap structure signal: large here means the gap is likely closable."""
    import flint

    n = A.shape[0]
    out = {}
    flat = A.reshape(-1).tolist()
    for p in fields:
        best = 0
        base = [int(x) % p for x in flat]
        for lam in range(1, p):
            ent = base.copy()
            for i in range(n):
                ent[i * n + i] = (-lam) % p
            M = flint.nmod_mat(n, n, ent, p)
            best = max(best, n - M.rank())
        out[p] = best
    return out
