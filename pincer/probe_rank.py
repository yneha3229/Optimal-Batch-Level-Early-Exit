"""Ask SAT one question, minrk_2(G) <= r, with the independent-set symmetry fix; save a verified certificate if SAT.

    python -m pincer.probe_rank GRAPH.json R --limit 600 [--solver cadical153] [--out CERT.json]
"""
from __future__ import annotations

import argparse
import json
import time

from minrank.core.graph import Graph
from minrank.core.verify import save_certificate
from minrank.exact.lower import mais
from minrank.exact.symmetry import rank_at_most_sym


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("r", type=int)
    ap.add_argument("--limit", type=float, default=600)
    ap.add_argument("--solver", default="cadical153")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    g = Graph.load(a.graph)
    I = mais(g, 10).vertices[: a.r]
    t = time.perf_counter()
    res = rank_at_most_sym(g, a.r, time_limit=a.limit, mode="indep_cols", indep=I, solver=a.solver)
    sec = time.perf_counter() - t
    status = "timeout" if res is None else ("UNSAT" if res is False else "SAT")
    if status == "SAT" and a.out:
        save_certificate(a.out, g, res, f"pincer.probe_rank r={a.r} {a.solver}")
    print(json.dumps({"graph": g.name, "r": a.r, "status": status, "seconds": round(sec, 1), "solver": a.solver,
                      "fixed_set": len(I)}), flush=True)


if __name__ == "__main__":
    main()
