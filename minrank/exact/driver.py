"""Option B driver: exact minrk_2 for benchmark graphs by SAT descent with symmetry breaking.

For each selected graph:
  1. warm start: the stored certificate (minrank/certificates/<g>.json) if it is for this graph and verifies,
     else a TabuCol clique cover (`--cover-time` s);
  2. lower bound: MAIS / alpha by CP-SAT (`--lb-time` s); if it meets the warm start, the graph is proven optimal;
  3. descend: SAT for rank r-1, r-2, ... with `--mode` symmetry breaking (default "cols": u_0 = e_1 plus column
     lex order) and `--time-per-call` s per call, until UNSAT (proven optimal) or a timeout.
     Measured (CPU, 30 s limit): "cols"/"u0" prove Mycielski-4 r<=5 UNSAT in 1.5 s and Paley-13 r<=4 in 0.2 s;
     "aut" (automorphism lex-leader, which cannot be combined with u_0 = e_1) times out on Mycielski-4 and takes
     0.8 s on Paley-13; "none" takes 16 s on Paley-13. The GL(r) fix matters most; "aut" is kept as an option.
Improved certificates are saved with run_baselines.keep_best (in the main process). One CSV row per attempt is
appended to minrank/results/exact.csv; graphs already proven optimal there are skipped (resumable). Graphs not
proven get a new row on each run; readers should take the last row per graph.
`--rerun-from CSV` restricts the run to graphs whose last row in CSV has a status in `--rerun-status` (default
call_timeout) and is not proven; combine it with `--out` so the earlier results are not appended to.

Usage:
  python -m minrank.exact.driver --family kneser paley mycielski orthogonality cycle_power circulant \
      --max-n 60 --time-per-call 600 --workers 8
  python -m minrank.exact.driver --rerun-from minrank/results/exact.csv --time-per-call 3600 \
      --out minrank/results/exact_long.csv --workers 16
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.clique_cover import clique_cover_rows
from ..bench.instances import load_index
from ..core.f2 import rank_f2
from ..core.graph import Graph
from ..core.verify import load_certificate, verify
from ..run_baselines import keep_best, stored_rank
from .lower import mais
from .symmetry import automorphisms, rank_at_most_sym

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["graph", "n", "family", "lb", "lb_proven", "start_rank", "best_rank", "proven_optimal", "seconds",
          "mode", "n_automorphisms", "calls", "status"]


def solve_graph(graph_path: str, cert_dir: str, family: str, lb_time: float, cover_time: float,
                time_per_call: float, total_time: float | None, mode: str, seed: int) -> dict:
    """Run steps 1-3 for one graph. Returns the CSV row plus '_rows' (best verified rows)."""
    t0 = time.perf_counter()
    g = Graph.load(graph_path)
    cert = Path(cert_dir) / f"{g.name}.json"
    if stored_rank(cert, g) is not None:
        rows = load_certificate(cert)["rows"]
    else:
        rows = clique_cover_rows(g, "tabucol", cover_time, np.random.default_rng(seed))
    start = best = verify(g, rows)
    lb = mais(g, lb_time)
    gens = automorphisms(g) if mode == "aut" else None
    use = mode
    proven, calls = best <= lb.value, 0
    status = "lb_meets_ub" if proven else "not_started"
    while not proven and best > lb.value:
        left = None if total_time is None else total_time - (time.perf_counter() - t0)
        if left is not None and left <= 0:
            status = "total_timeout"
            break
        limit = time_per_call if left is None else min(time_per_call, left)
        res = rank_at_most_sym(g, best - 1, limit, mode=use, gens=gens if use == "aut" else None)
        calls += 1
        if res is None:
            status = "call_timeout"
            break
        if res is False:
            proven, status = True, "unsat"
            break
        rows, best = res, rank_f2(res)
    if best <= lb.value:
        proven, status = True, ("lb_meets_ub" if calls == 0 else "reached_lb")
    row = {"graph": g.name, "n": g.n, "family": family, "lb": lb.value, "lb_proven": lb.proven_max,
           "start_rank": start, "best_rank": best, "proven_optimal": proven,
           "seconds": round(time.perf_counter() - t0, 2), "mode": use,
           "n_automorphisms": len(gens) if gens else 0, "calls": calls, "status": status, "_rows": rows}
    return row


def proven_graphs(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != FIELDS:
            raise SystemExit(f"{path} has header {rd.fieldnames}, expected {FIELDS}; use another --out")
        return {r["graph"] for r in rd if r["proven_optimal"] == "True"}


def graphs_with_status(path: Path, statuses: list[str]) -> set[str]:
    """Graphs whose last row in an exact CSV has a status in `statuses` and is not proven optimal."""
    with path.open(newline="") as f:
        last = {r["graph"]: r for r in csv.DictReader(f)}
    return {g for g, r in last.items() if r["status"] in statuses and r["proven_optimal"] != "True"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, default=ROOT / "bench/graphs")
    ap.add_argument("--family", nargs="*", default=None, help="restrict to these families (index.csv 'family')")
    ap.add_argument("--names", nargs="*", default=None, help="restrict to these graph names")
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--max-n", type=int, default=60)
    ap.add_argument("--time-per-call", type=float, default=600.0)
    ap.add_argument("--total-time", type=float, default=None, help="cap per graph (s); default: no cap")
    ap.add_argument("--lb-time", type=float, default=60.0)
    ap.add_argument("--cover-time", type=float, default=5.0)
    ap.add_argument("--mode", default="cols", choices=["none", "u0", "cols", "aut"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=Path, default=ROOT / "results/exact.csv")
    ap.add_argument("--certs", type=Path, default=ROOT / "certificates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rerun-from", type=Path, default=None,
                    help="only graphs whose last row in this exact CSV has a status in --rerun-status")
    ap.add_argument("--rerun-status", nargs="+", default=["call_timeout"])
    a = ap.parse_args(argv)
    rerun = graphs_with_status(a.rerun_from, a.rerun_status) if a.rerun_from else None

    idx = [r for r in load_index(a.graphs) if r["split"] in a.split and int(r["n"]) <= a.max_n
           and (not a.family or r["family"] in a.family) and (not a.names or r["name"] in a.names)
           and (rerun is None or r["name"] in rerun)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.certs.mkdir(parents=True, exist_ok=True)
    done = proven_graphs(a.out)
    todo = [r for r in idx if r["name"] not in done]
    print(f"{len(idx)} graphs selected, {len(idx) - len(todo)} already proven, {len(todo)} to run", flush=True)
    new = not a.out.exists() or a.out.stat().st_size == 0
    graphs = {r["name"]: Graph.load(a.graphs / r["path"]) for r in todo}
    with a.out.open("a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        # spawn: workers must not inherit a forkserver started by this process (waitpid on a non-child fails)
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        futs = [ex.submit(solve_graph, str(a.graphs / r["path"]), str(a.certs), r["family"], a.lb_time,
                          a.cover_time, a.time_per_call, a.total_time, a.mode, a.seed) for r in todo]
        for fu in as_completed(futs):
            row = fu.result()
            rows = row.pop("_rows")
            keep_best(a.certs, graphs[row["graph"]], rows, f"exact_sat[{row['mode']}]")
            w.writerow(row)
            f.flush()
            print(f"{row['graph']}: lb={row['lb']} best={row['best_rank']} proven={row['proven_optimal']} "
                  f"({row['status']}, {row['seconds']}s)", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
