"""Benchmark graphs: generators, random relabelling, and the on-disk format.

On disk a graph is two files that carry no family name and no parameters:
  <id>.npz   the adjacency matrix, bit-packed by rows (np.packbits), compressed
  <id>.json  {"n": vertices, "m": edges, "seed": relabelling seed}
The ground truth (family, parameters, original labels) lives only in bench/truth.json.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
GRAPH_DIR = BENCH / "graphs"


# ------------------------------------------------------------------------------------------------ generators
def kneser(n: int, k: int) -> np.ndarray:
    """Adjacency of the Kneser graph K(n, k): k-subsets of [n], adjacent iff disjoint. Vertex order = lexicographic."""
    subs = list(itertools.combinations(range(n), k))
    N = len(subs)
    inc = np.zeros((N, n), dtype=np.float32)
    for i, s in enumerate(subs):
        inc[i, list(s)] = 1.0
    A = np.zeros((N, N), dtype=bool)
    for a in range(0, N, 2048):                               # blocked, so the float product stays small
        A[a:a + 2048] = (inc[a:a + 2048] @ inc.T) == 0
    np.fill_diagonal(A, False)
    return A


def kneser_labels(n: int, k: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(n), k))


def delete_edges(A: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Delete a fraction of the edges uniformly at random (control graphs)."""
    rng = np.random.default_rng(seed)
    iu, ju = np.nonzero(np.triu(A, 1))
    kill = rng.choice(len(iu), size=int(round(frac * len(iu))), replace=False)
    B = A.copy()
    B[iu[kill], ju[kill]] = False
    B[ju[kill], iu[kill]] = False
    return B


def random_regular(n: int, d: int, seed: int) -> np.ndarray:
    import networkx as nx

    g = nx.random_regular_graph(d, n, seed=seed)
    A = np.zeros((n, n), dtype=bool)
    for u, v in g.edges():
        A[u, v] = A[v, u] = True
    return A


def relabel_perm(N: int, seed: int) -> np.ndarray:
    return np.random.default_rng(1_000_003 + seed).permutation(N)


def relabel(A: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """New vertex i is old vertex perm[i]."""
    perm = relabel_perm(A.shape[0], seed)
    return A[np.ix_(perm, perm)], perm


# ------------------------------------------------------------------------------------------------ file format
def save_graph(stem: Path, A: np.ndarray, seed: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), bits=np.packbits(A, axis=1), n=np.int64(A.shape[0]))
    stem.with_suffix(".json").write_text(json.dumps({"n": int(A.shape[0]), "m": int(A.sum() // 2), "seed": seed}))


def load_graph(path: str | Path) -> np.ndarray:
    path = Path(path)
    d = np.load(path.with_suffix(".npz"))
    n = int(d["n"])
    A = np.unpackbits(d["bits"], axis=1, count=n).astype(bool)
    if A.shape != (n, n) or (A != A.T).any() or A.diagonal().any():
        raise ValueError(f"{path}: not a simple undirected graph")
    return A


# ------------------------------------------------------------------------------------------------ benchmark set
K2 = (21, 22, 25, 29, 33, 37, 41, 45, 53, 61, 69, 81, 89, 99, 109, 119, 129, 139, 140)
K3 = (12, 13, 14, 15, 16, 17, 19, 21, 24, 27, 30, 33, 36, 40)
K4 = (12, 13, 15, 16, 17, 18, 20, 22, 23)
K5 = (15, 17, 18)
HARD = ((10, 4), (11, 4), (14, 4), (11, 5), (12, 5), (13, 5), (14, 5), (16, 5), (13, 6), (14, 6), (15, 6), (16, 6),
        (15, 7))


def bench_specs() -> list[dict]:
    out = []
    for k, ns in ((2, K2), (3, K3), (4, K4), (5, K5)):
        out += [{"name": f"K({n},{k})", "family": "kneser", "n": n, "k": k, "group": "main"} for n in ns]
    out += [{"name": f"K({n},{k})", "family": "kneser", "n": n, "k": k, "group": "hard"} for n, k in HARD]
    out += [{"name": "K(16,4)-1%", "family": "kneser_minus_1pct", "n": 16, "k": 4, "group": "control"},
            {"name": "K(30,3)-1%", "family": "kneser_minus_1pct", "n": 30, "k": 3, "group": "control"},
            {"name": "RR(1000,6)", "family": "random_regular", "n": 1000, "d": 6, "group": "control"}]
    for s in out:
        s["receivers"] = comb(s["n"], s["k"]) if "k" in s else s["n"]
    return out


def graph_id(name: str, seed: int) -> str:
    return "g" + hashlib.sha1(f"{name}|{seed}".encode()).hexdigest()[:6]


def build_adjacency(spec: dict) -> np.ndarray:
    if spec["family"] == "kneser":
        return kneser(spec["n"], spec["k"])
    if spec["family"] == "kneser_minus_1pct":
        return delete_edges(kneser(spec["n"], spec["k"]), 0.01, seed=7)
    if spec["family"] == "random_regular":
        return random_regular(spec["n"], spec["d"], seed=7)
    raise ValueError(spec["family"])


def build_bench(seeds=(0,), extra_seed_names: tuple[str, ...] = ()) -> None:
    """Write bench/graphs/<id>.{npz,json}, bench/graphs.json (ids and files only) and bench/truth.json."""
    truth_path = BENCH / "truth.json"
    truth = json.loads(truth_path.read_text()) if truth_path.exists() else {}
    listing_path = BENCH / "graphs.json"
    listing = json.loads(listing_path.read_text()) if listing_path.exists() else []
    have = {e["id"] for e in listing}
    for spec in bench_specs():
        for seed in seeds:
            if seed != 0 and spec["name"] not in extra_seed_names:
                continue
            gid = graph_id(spec["name"], seed)
            if gid in have and (GRAPH_DIR / f"{gid}.npz").exists():
                continue
            A0 = build_adjacency(spec)
            A, perm = relabel(A0, seed)
            save_graph(GRAPH_DIR / gid, A, seed)
            truth[gid] = {**spec, "seed": seed, "perm_seed": 1_000_003 + seed,
                          "note": "new vertex i = original vertex perm[i]; perm = relabel_perm(N, seed); "
                                  "original vertex order = itertools.combinations(range(n), k)"}
            listing.append({"id": gid, "file": f"bench/graphs/{gid}", "n": int(A.shape[0]), "seed": seed})
            have.add(gid)
            print(gid, spec["name"], seed, A.shape[0], flush=True)
    listing.sort(key=lambda e: (e["n"], e["id"]))
    listing_path.write_text(json.dumps(listing, indent=1))
    truth_path.write_text(json.dumps(truth, indent=1))


def name_to_id(name: str, seed: int = 0) -> str:
    return graph_id(name, seed)


if __name__ == "__main__":
    import sys

    extra = tuple(sys.argv[1:])
    build_bench(seeds=(0, 1, 2), extra_seed_names=extra)
