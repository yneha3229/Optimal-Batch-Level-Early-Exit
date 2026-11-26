"""Export the CNF of a Pincer UNSAT step ("minrk_2(G) <= r" with the independent-set fix) for an independent DRAT check
with scripts/minrank/10_drat_check.sh (CaDiCaL + drat-trim). The CNF is re-solved here first, so only confirmed-UNSAT
formulas are written. Header comments record the graph digest, r, the mode and the fixed set.

    python -m pincer.export_unsat_cnf GRAPH.json R --out-dir pincer/certificates/unsat_cnf [--limit 600]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from minrank.core.graph import Graph
from minrank.exact.families import write_dimacs
from minrank.exact.lower import mais
from minrank.exact.symmetry import check_indep, encode_sym, solve_timed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("r", type=int)
    ap.add_argument("--out-dir", default="pincer/certificates/unsat_cnf")
    ap.add_argument("--limit", type=float, default=600)
    a = ap.parse_args()
    g = Graph.load(a.graph)
    I = sorted(mais(g, 10).vertices[: a.r])
    check_indep(g, I)
    t = time.perf_counter()
    res = solve_timed(encode_sym, (g, a.r, "indep_cols", None, I), a.limit)
    sec = time.perf_counter() - t
    status = "timeout" if res is None else ("UNSAT" if res is False else "SAT")
    out = None
    if status == "UNSAT":
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        out = Path(a.out_dir) / f"{g.name}_r{a.r}.cnf.gz"
        write_dimacs(out, encode_sym(g, a.r, "indep_cols", None, I),
                     [f"graph {g.name} digest {g.digest()} n {g.n}", f"claim: minrk_2 <= {a.r} is UNSAT",
                      "mode indep_cols (independent-set fix + column lex, minrank/exact/symmetry.py)",
                      f"fixed set {' '.join(map(str, I))}", f"exported by pincer.export_unsat_cnf (cadical153 {sec:.1f} s)"])
    print(json.dumps({"graph": g.name, "r": a.r, "status": status, "seconds": round(sec, 1),
                      "file": str(out) if out else None}), flush=True)


if __name__ == "__main__":
    main()
