"""Side-information graphs for index coding.

Convention (used everywhere in `minrank/`):
- Vertices are 0..n-1. Receiver i wants message i and knows the messages in S_i.
- `out[i]` is a bitmask of S_i: bit j is set iff i knows x_j (directed edge i -> j). Never contains bit i.
- An undirected graph is a directed graph with symmetric `out` (j in S_i  <=>  i in S_j).
- A matrix A over F_2 is stored as rows: `rows[i]` is an int whose bit j is A[i, j].
- A fits G iff A[i, i] = 1 for all i and A[i, j] = 0 for every j not in S_i u {i}.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Graph:
    """Side-information graph. `out[i]` is the bitmask of S_i (no self loops)."""

    n: int
    out: tuple[int, ...]
    directed: bool
    name: str = ""
    family: str = ""
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("graph needs at least one vertex")
        if len(self.out) != self.n:
            raise ValueError(f"out has {len(self.out)} masks, expected {self.n}")
        full = (1 << self.n) - 1
        for i, m in enumerate(self.out):
            if m < 0 or m & ~full:
                raise ValueError(f"vertex {i}: mask has bits outside 0..n-1")
            if (m >> i) & 1:
                raise ValueError(f"vertex {i}: self loop")
        if not self.directed:
            for i in range(self.n):
                for j in iter_bits(self.out[i]):
                    if not (self.out[j] >> i) & 1:
                        raise ValueError(f"undirected graph is not symmetric at ({i}, {j})")

    # ---- construction -------------------------------------------------------------------------------------------
    @staticmethod
    def from_edges(n: int, edges: Iterable[tuple[int, int]], directed: bool, name: str = "", family: str = "",
                   meta: dict | None = None) -> "Graph":
        """Build from (i, j) pairs. Undirected edges are added in both directions; self loops are rejected."""
        out = [0] * n
        for i, j in edges:
            if i == j:
                raise ValueError("self loop")
            out[i] |= 1 << j
            if not directed:
                out[j] |= 1 << i
        return Graph(n, tuple(out), directed, name, family, dict(meta or {}))

    @staticmethod
    def from_adjacency(adj: np.ndarray, directed: bool, name: str = "", family: str = "",
                       meta: dict | None = None) -> "Graph":
        """Build from a boolean (n, n) adjacency matrix; adj[i, j] = True iff j in S_i. Diagonal must be False."""
        adj = np.asarray(adj, dtype=bool)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adjacency must be square, got {adj.shape}")
        if adj.diagonal().any():
            raise ValueError("adjacency has self loops")
        n = adj.shape[0]
        out = tuple(int(sum(1 << int(j) for j in np.flatnonzero(adj[i]))) for i in range(n))
        return Graph(n, out, directed, name, family, dict(meta or {}))

    # ---- views --------------------------------------------------------------------------------------------------
    @property
    def full(self) -> int:
        return (1 << self.n) - 1

    def allowed(self, i: int) -> int:
        """Columns where row i of a fitting matrix may be nonzero: S_i u {i}."""
        return self.out[i] | (1 << i)

    def forbidden(self, i: int) -> int:
        """Columns where row i of a fitting matrix must be zero."""
        return self.full & ~self.allowed(i)

    def adjacency(self) -> np.ndarray:
        """Boolean (n, n) matrix, adj[i, j] = j in S_i."""
        a = np.zeros((self.n, self.n), dtype=bool)
        for i in range(self.n):
            for j in iter_bits(self.out[i]):
                a[i, j] = True
        return a

    def mutual(self) -> "Graph":
        """Undirected graph of bidirected pairs (i knows j and j knows i). Cliques of it give valid clique covers."""
        inc = [0] * self.n  # inc[i]: bitmask of j with i in S_j
        for j in range(self.n):
            for i in iter_bits(self.out[j]):
                inc[i] |= 1 << j
        out = tuple(self.out[i] & inc[i] for i in range(self.n))
        return Graph(self.n, out, False, self.name + "_mutual", self.family, dict(self.meta))

    def complement(self) -> "Graph":
        """Complement (same directedness)."""
        out = tuple(self.full & ~self.out[i] & ~(1 << i) for i in range(self.n))
        return Graph(self.n, out, self.directed, self.name + "_c", self.family, dict(self.meta))

    def num_edges(self) -> int:
        """Directed edge count (an undirected edge counts twice)."""
        return sum(m.bit_count() for m in self.out)

    def density(self) -> float:
        return self.num_edges() / (self.n * (self.n - 1)) if self.n > 1 else 0.0

    def digest(self) -> str:
        h = hashlib.sha1(f"{self.n}|{int(self.directed)}|".encode())
        h.update(",".join(f"{m:x}" for m in self.out).encode())
        return h.hexdigest()[:16]

    # ---- I/O ----------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        edges = [(i, j) for i in range(self.n) for j in iter_bits(self.out[i]) if self.directed or i < j]
        return {"name": self.name, "family": self.family, "n": self.n, "directed": self.directed,
                "digest": self.digest(), "meta": self.meta, "edges": edges}

    @staticmethod
    def from_dict(d: dict) -> "Graph":
        g = Graph.from_edges(d["n"], [tuple(e) for e in d["edges"]], d["directed"], d.get("name", ""),
                             d.get("family", ""), d.get("meta", {}))
        if "digest" in d and d["digest"] != g.digest():
            raise ValueError(f"digest mismatch for {g.name}: file {d['digest']}, rebuilt {g.digest()}")
        return g

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @staticmethod
    def load(path: str | Path) -> "Graph":
        return Graph.from_dict(json.loads(Path(path).read_text()))


def iter_bits(mask: int) -> Iterable[int]:
    """Indices of set bits, ascending."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def masks_to_bool(masks: Sequence[int], n: int) -> np.ndarray:
    """(len(masks), n) boolean array; out[i, j] = bit j of masks[i]."""
    a = np.zeros((len(masks), n), dtype=bool)
    for i, m in enumerate(masks):
        for j in iter_bits(m):
            a[i, j] = True
    return a


def bool_to_masks(a: np.ndarray) -> list[int]:
    """Inverse of masks_to_bool for a 2-D boolean/0-1 array."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"expected 2-D array, got shape {a.shape}")
    weights = [1 << j for j in range(a.shape[1])]
    return [sum(w for w, b in zip(weights, row) if b) for row in a.astype(bool).tolist()]
