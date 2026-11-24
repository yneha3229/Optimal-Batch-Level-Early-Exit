"""Which fixed independent set makes "minrk_2 <= r?" fast? Sample distinct maximum acyclic sets, time one SAT call
per set, record simple set features. Output: JSON lines (one per set) for analysis.

    python -m pincer.fixset_study GRAPH.json R --sets 20 --limit 60 --out pincer/results/fixset_<name>.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from minrank.core.graph import Graph, iter_bits
from minrank.exact.symmetry import rank_at_most_sym

from .mais import greedy_fvs, sa_order


def features(g: Graph, I: list[int]) -> dict:
    mask = sum(1 << v for v in I)
    deg = [g.out[v].bit_count() for v in range(g.n)]
    nbr_count = [(g.out[u] & mask).bit_count() for u in range(g.n) if not (mask >> u) & 1]  # I-neighbours of others
    return {"deg_mean": sum(deg[v] for v in I) / len(I), "deg_max": max(deg[v] for v in I),
            "deg_min": min(deg[v] for v in I),
            "covered": sum(1 for c in nbr_count if c > 0), "private": sum(1 for c in nbr_count if c == 1),
            "nbr_mean": sum(nbr_count) / max(1, len(nbr_count)), "nbr_max": max(nbr_count, default=0)}


def sample_sets(g: Graph, k: int, seed: int = 0) -> list[list[int]]:
    rng = random.Random(seed)
    found: dict[tuple, None] = {}
    best = 0
    tries = 0
    while len(found) < 4 * k and tries < 30 * k:
        tries += 1
        order = list(range(g.n))
        rng.shuffle(order)
        start = greedy_fvs(g, lambda v, ctx, o={v: i for i, v in enumerate(order)}: -o[v] + 1000 * ctx.in_deg(v))
        S = tuple(sorted(sa_order(g, 0.15, random.Random(rng.randrange(2**31)), init=start)))
        best = max(best, len(S))
        found[S] = None
    sets = [list(S) for S in found if len(S) == best]
    rng.shuffle(sets)
    return sets[:k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("r", type=int)
    ap.add_argument("--sets", type=int, default=20)
    ap.add_argument("--limit", type=float, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = Graph.load(a.graph)
    sets = sample_sets(g, a.sets)
    with Path(a.out).open("a") as f:
        for I in sets:
            t = time.perf_counter()
            res = rank_at_most_sym(g, a.r, time_limit=a.limit, mode="indep_cols", indep=I[: a.r])
            sec = time.perf_counter() - t
            st = "timeout" if res is None else ("UNSAT" if res is False else "SAT")
            row = {"graph": g.name, "r": a.r, "size": len(I), "status": st, "seconds": round(sec, 2), "set": I,
                   **features(g, I)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(st, round(sec, 1), flush=True)


if __name__ == "__main__":
    main()
