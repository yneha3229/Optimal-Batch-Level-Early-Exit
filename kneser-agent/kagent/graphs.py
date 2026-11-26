"""Graph inputs: Kneser graphs K(n,k), random relabelling, compact save/load.

The agent only ever sees the relabelled adjacency matrix. The ground truth (family, n, k, permutation) is written
to a separate file that only the evaluation step reads.
"""
from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRAPHS = ROOT / "bench" / "graphs"
TRUTH = ROOT / "bench" / "truth.json"


def kneser(n: int, k: int) -> np.ndarray:
    """Adjacency (bool, N x N) of K(n,k): k-subsets of [n], adjacent iff disjoint. Built in row blocks."""
    masks = np.array([sum(1 << i for i in c) for c in combinations(range(n), k)], dtype=np.int64)
    N = len(masks)
    A = np.zeros((N, N), dtype=bool)
    for s in range(0, N, 1024):
        A[s:s + 1024] = (masks[s:s + 1024, None] & masks[None, :]) == 0
    np.fill_diagonal(A, False)
    return A


def delete_edges(A: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    B = A.copy()
    i, j = np.nonzero(np.triu(B, 1))
    pick = rng.random(len(i)) < frac
    B[i[pick], j[pick]] = False
    B[j[pick], i[pick]] = False
    return B


def random_regular(N: int, d: int, seed: int) -> np.ndarray:
    import networkx as nx
    g = nx.random_regular_graph(d, N, seed=seed)
    A = np.zeros((N, N), dtype=bool)
    for u, v in g.edges():
        A[u, v] = A[v, u] = True
    return A


def relabel(A: np.ndarray, seed: int):
    perm = np.random.default_rng(seed).permutation(A.shape[0])
    return A[np.ix_(perm, perm)], perm


def save(name: str, A: np.ndarray, truth: dict) -> Path:
    GRAPHS.mkdir(parents=True, exist_ok=True)
    path = GRAPHS / f"{name}.npz"
    np.savez_compressed(path, bits=np.packbits(A, axis=1), n=A.shape[0])
    t = json.loads(TRUTH.read_text()) if TRUTH.exists() else {}
    t[name] = truth
    TRUTH.write_text(json.dumps(t, indent=1, sort_keys=True))
    return path


def load(path) -> np.ndarray:
    d = np.load(path)
    N = int(d["n"])
    return np.unpackbits(d["bits"], axis=1, count=N).astype(bool)


def make(spec: str, seed: int = 0) -> Path:
    """spec: 'K(n,k)', 'K(n,k)-del<frac>' or 'reg(N,d)'. Returns the saved path (cached)."""
    name = spec.replace("(", "_").replace(")", "").replace(",", "_") + f"_s{seed}"
    path = GRAPHS / f"{name}.npz"
    if path.exists():
        return path
    if spec.startswith("reg("):
        N, d = map(int, spec[4:-1].split(","))
        A, truth = random_regular(N, d, seed), {"family": "random regular", "N": N, "d": d}
    else:
        base, _, dele = spec.partition("-del")
        n, k = map(int, base[2:-1].split(","))
        A = kneser(n, k)
        truth = {"family": "kneser", "n": n, "k": k, "N": comb(n, k)}
        if dele:
            A = delete_edges(A, float(dele), seed)
            truth = {"family": "kneser with deleted edges", "n": n, "k": k, "deleted_frac": float(dele)}
    B, perm = relabel(A, seed)
    truth["seed"] = seed
    return save(name, B, truth)
