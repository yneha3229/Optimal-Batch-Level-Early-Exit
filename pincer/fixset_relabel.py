"""Is a fixed set fast because of the set, or by luck? Re-run "minrk <= r?" for the same set under random vertex
relabellings (isomorphic graph, same set mapped along, different CNF variable order).
    python -m pincer.fixset_relabel STUDY.jsonl GRAPH.json --perms 4 --limit 60"""
from __future__ import annotations

import argparse
import json
import random
import time

from minrank.core.graph import Graph, iter_bits
from minrank.exact.symmetry import rank_at_most_sym


def relabel(g: Graph, perm: list[int]) -> Graph:
    edges = [(perm[i], perm[j]) for i in range(g.n) for j in iter_bits(g.out[i])]
    return Graph.from_edges(g.n, edges, True, g.name + "_perm", g.family)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study")
    ap.add_argument("graph")
    ap.add_argument("--perms", type=int, default=4)
    ap.add_argument("--limit", type=float, default=60)
    a = ap.parse_args()
    rows = sorted((json.loads(line) for line in open(a.study)), key=lambda r: r["seconds"])
    g = Graph.load(a.graph)
    rng = random.Random(0)
    for tag, row in (("fastest", rows[0]), ("slowest", rows[-1]), ("median", rows[len(rows) // 2])):
        times = []
        for _ in range(a.perms):
            perm = list(range(g.n))
            rng.shuffle(perm)
            h = relabel(g, perm)
            I = sorted(perm[v] for v in row["set"])[: row["r"]]
            t = time.perf_counter()
            res = rank_at_most_sym(h, row["r"], time_limit=a.limit, mode="indep_cols", indep=I)
            times.append(round(time.perf_counter() - t, 1) if res is not None else f">{a.limit:.0f}")
        print(f"{tag:8s} set (original {row['seconds']} s): relabelled {times}", flush=True)


if __name__ == "__main__":
    main()
