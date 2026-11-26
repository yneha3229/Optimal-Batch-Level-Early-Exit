"""Field-separation SAT calls: is minrk_{F_q}(K(m,k)) <= r? (one-hot F_q encoding of sat_fq.py)

For each `--case m,k,q,r` the script solves the CNF `encode_fq(kneser(m,k), r, q)` with a wall-clock limit and
appends one row to `--out` (default minrank/results/field_sep/fq.csv): graph,q,r,result,seconds with result
SAT, UNSAT or TIMEOUT.
- SAT: the matrix is re-verified with `verify_fq` (fits the graph, rank <= r over F_q) and saved as JSON under
  `--certs` (default minrank/certificates/field_sep/<graph>_F<q>_r<r>.json).
- UNSAT: the exact CNF that was solved (encode_fq is deterministic) is written as gzipped DIMACS under
  `--dimacs-dir` (default minrank/results/unsat_cnf/field_sep/<graph>_F<q>_r<r>.cnf.gz) for the DRAT check
  (scripts/minrank/10_drat_check.sh). Only a VERIFIED DRAT row makes an UNSAT a proof.

Usage:
  python -m minrank.exact.field_sep --case 7,2,3,6 --time-limit 600
  python -m minrank.exact.field_sep --case 9,2,3,8 --case 11,2,5,10 --time-limit 36000
The `__main__` guard is required: the solver runs in a forkserver child (symmetry.solve_timed).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from ..bench.generators import kneser
from .families import write_dimacs
from .sat_fq import encode_fq, rank_at_most_fq, verify_fq

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["graph", "q", "r", "result", "seconds"]


def parse_case(s: str) -> tuple[int, int, int, int]:
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"--case needs m,k,q,r, got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


def append_row(path: Path, row: dict) -> None:
    """Append under an exclusive lock, so several processes can share one CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0, 2)
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_case(m: int, k: int, q: int, r: int, time_limit: float, solver: str, out: Path, certs: Path,
             dimacs_dir: Path) -> dict:
    g = kneser(m, k)
    t0 = time.perf_counter()
    res = rank_at_most_fq(g, r, q, time_limit, solver=solver)
    secs = round(time.perf_counter() - t0, 2)
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if res is None:
        result = "TIMEOUT"
    elif res is False:
        result = "UNSAT"
        dimacs_dir.mkdir(parents=True, exist_ok=True)
        write_dimacs(dimacs_dir / f"{g.name}_F{q}_r{r}.cnf.gz", encode_fq(g, r, q),
                     [f"minrk_{q}({g.name}) <= {r} is UNSAT", f"field F_{q}, one-hot encoding (minrank/exact/sat_fq.py)",
                      "symmetry breaking: u_0 = e_1", f"graph digest {g.digest()}", f"solver {solver}, {secs} s, {stamp}"])
    else:
        rank = verify_fq(g, res, q)  # raises if the matrix does not fit
        if rank > r:
            raise RuntimeError(f"{g.name}: SAT matrix has rank {rank} > {r} over F_{q}")
        result = "SAT"
        certs.mkdir(parents=True, exist_ok=True)
        (certs / f"{g.name}_F{q}_r{r}.json").write_text(json.dumps(
            {"graph": g.name, "digest": g.digest(), "field": q, "r": r, "rank": rank, "solver": solver,
             "seconds": secs, "at": stamp, "rows": res}) + "\n")
    row = {"graph": g.name, "q": q, "r": r, "result": result, "seconds": secs}
    append_row(out, row)
    print(f"{stamp} {g.name} F_{q} r={r}: {result} in {secs} s", flush=True)
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", type=parse_case, action="append", required=True, help="m,k,q,r (repeatable)")
    ap.add_argument("--time-limit", type=float, required=True, help="seconds per case (encoding included)")
    ap.add_argument("--solver", default="cadical153")
    ap.add_argument("--out", type=Path, default=ROOT / "results/field_sep/fq.csv")
    ap.add_argument("--certs", type=Path, default=ROOT / "certificates/field_sep")
    ap.add_argument("--dimacs-dir", type=Path, default=ROOT / "results/unsat_cnf/field_sep")
    a = ap.parse_args(argv)
    for m, k, q, r in a.case:
        run_case(m, k, q, r, a.time_limit, a.solver, a.out, a.certs, a.dimacs_dir)


if __name__ == "__main__":
    main()
