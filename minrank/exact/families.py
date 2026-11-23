"""Exact minrk_2 of named graph families with the independent-set symmetry breaking (symmetry.py, mode "indep").

For each spec `name:a,b,...` the graph is `minrank.bench.generators.<name>(a, b, ...)` (integer or float arguments),
so new generators need no change here. Per graph:
  1. lower bound: an acyclic set I by CP-SAT (`lower.mais`, `--lb-time` s); alpha = |I| <= minrk;
  2. upper bound: the best of a TabuCol clique cover, I + A(G) over F_2 (minrank/baselines/identity_adj.py) and the
     stored certificate (if it is for this graph and verifies);
  3. ascend: SAT for r = alpha, alpha + 1, ... with I fixed. The first SAT r is minrk (every smaller r is UNSAT or
     below the L1 bound), so the result is proven;
  4. on a timeout at r, minrk >= r is known; descend from the upper bound (upper - 1, upper - 2, ...) with the same
     per-call limit to improve the upper bound, stopping at UNSAT (upper is then optimal) or a timeout.
`--dimacs-dir` writes the CNF of every UNSAT call (gzipped DIMACS, the exact clauses solved, with the mode and I in
the header) for an independent DRAT check (WORK.md, task T4). One CSV row per graph; graphs already proven in `--out`
are skipped (resumable).

Usage:
  python -m minrank.exact.families --spec kneser:8,3 mycielski:6 paley:29 --time-per-call 900 --workers 7
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.clique_cover import clique_cover_rows
from ..baselines.identity_adj import identity_plus_adjacency_rows
from ..bench import generators
from ..core.graph import Graph
from ..core.verify import load_certificate, verify
from ..run_baselines import keep_best, stored_rank
from .lower import mais
from .symmetry import INDEP_MODES, encode_sym, rank_at_most_sym

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["spec", "graph", "n", "family", "directed", "alpha", "alpha_proven", "cover", "lower", "upper", "proven",
          "seconds", "mode", "solver", "time_per_call", "indep", "log"]


def parse_spec(spec: str) -> tuple[str, list]:
    name, _, args = spec.partition(":")
    vals = []
    for a in filter(None, args.split(",")):
        vals.append(float(a) if any(c in a for c in ".e") else int(a))
    return name, vals


def build(spec: str) -> Graph:
    name, args = parse_spec(spec)
    fn = getattr(generators, name, None)
    if fn is None or name.startswith("_"):
        raise ValueError(f"no generator {name!r} in minrank.bench.generators")
    return fn(*args)


def write_dimacs(path: Path, cnf: list[list[int]], comments: list[str]) -> None:
    nvars = max((abs(lit) for c in cnf for lit in c), default=0)
    with gzip.open(path, "wt") as f:
        for c in comments:
            f.write(f"c {c}\n")
        f.write(f"p cnf {nvars} {len(cnf)}\n")
        for c in cnf:
            f.write(" ".join(map(str, c)) + " 0\n")


def solve_family(spec: str, cert_dir: str, lb_time: float, cover_time: float, time_per_call: float, mode: str,
                 solver: str, dimacs_dir: str | None, seed: int) -> dict:
    t0 = time.perf_counter()
    g = build(spec)
    lb = mais(g, lb_time)
    I = sorted(int(v) for v in lb.vertices)
    cover_rows = clique_cover_rows(g, "tabucol", cover_time, np.random.default_rng(seed))
    cover = verify(g, cover_rows)
    rows, upper = cover_rows, cover
    ia = identity_plus_adjacency_rows(g)            # I + A(G): fits every graph, optimal on several Kneser graphs
    if verify(g, ia) < upper:
        rows, upper = ia, verify(g, ia)
    cert = Path(cert_dir) / f"{g.name}.json"
    stored = stored_rank(cert, g)
    if stored is not None and stored < upper:
        rows, upper = load_certificate(cert)["rows"], stored
    log: list[tuple[int, str, float]] = []

    def call(r: int):
        t = time.perf_counter()
        res = rank_at_most_sym(g, r, time_per_call, mode=mode, indep=I, solver=solver)
        st = "timeout" if res is None else ("unsat" if res is False else "sat")
        log.append((r, st, round(time.perf_counter() - t, 2)))
        if st == "unsat" and dimacs_dir and len(I) <= r:
            write_dimacs(Path(dimacs_dir) / f"{g.name}_r{r}_{mode}.cnf.gz", encode_sym(g, r, mode, indep=I),
                         [f"minrk_2({g.name}) <= {r} is UNSAT", f"mode {mode}", f"indep {' '.join(map(str, I))}",
                          f"graph digest {g.digest()}"])
        return st, res

    lower, proven = lb.value, upper <= lb.value
    r = lb.value
    while not proven and r < upper:
        st, res = call(r)
        if st == "sat":
            rows, upper, proven = res, verify(g, res), True
            break
        if st == "timeout":
            break
        lower = r + 1
        proven = lower >= upper
        r += 1
    if not proven:  # ascent timed out at r = lower: try to lower the upper bound from above
        while upper - 1 > lower:
            st, res = call(upper - 1)
            if st == "sat":
                rows, upper = res, verify(g, res)
                continue
            if st == "unsat":
                lower = upper
            break
        proven = lower >= upper
    return {"spec": spec, "graph": g.name, "n": g.n, "family": g.family, "directed": g.directed,
            "alpha": lb.value, "alpha_proven": lb.proven_max, "cover": cover, "lower": lower, "upper": upper,
            "proven": proven, "seconds": round(time.perf_counter() - t0, 1), "mode": mode, "solver": solver,
            "time_per_call": time_per_call, "indep": " ".join(map(str, I)), "log": json.dumps(log), "_rows": rows}


def proven_specs(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != FIELDS:
            raise SystemExit(f"{path} has header {rd.fieldnames}, expected {FIELDS}; use another --out")
        return {r["spec"] for r in rd if r["proven"] == "True"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", nargs="+", required=True, help="generator specs, e.g. kneser:8,3 mycielski:6")
    ap.add_argument("--time-per-call", type=float, default=900.0)
    ap.add_argument("--lb-time", type=float, default=60.0)
    ap.add_argument("--cover-time", type=float, default=10.0)
    ap.add_argument("--mode", default="indep", choices=list(INDEP_MODES))
    ap.add_argument("--solver", default="cadical153")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "results/families.csv")
    ap.add_argument("--certs", type=Path, default=ROOT / "certificates")
    ap.add_argument("--dimacs-dir", type=Path, default=None, help="write the CNF of every UNSAT call here")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    for s in a.spec:  # fail fast on a bad spec, before starting workers
        build(s)
    done = proven_specs(a.out)
    todo = [s for s in a.spec if s not in done]
    print(f"{len(a.spec)} specs, {len(a.spec) - len(todo)} already proven, {len(todo)} to run", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.certs.mkdir(parents=True, exist_ok=True)
    if a.dimacs_dir:
        a.dimacs_dir.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists() or a.out.stat().st_size == 0
    with a.out.open("a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        futs = {ex.submit(solve_family, s, str(a.certs), a.lb_time, a.cover_time, a.time_per_call, a.mode, a.solver,
                          str(a.dimacs_dir) if a.dimacs_dir else None, a.seed): s for s in todo}
        for fu in as_completed(futs):
            row = fu.result()
            rows = row.pop("_rows")
            keep_best(a.certs, build(row["spec"]), rows, f"families[{row['mode']}]")
            w.writerow(row)
            f.flush()
            print(f"{row['graph']}: alpha={row['alpha']} cover={row['cover']} minrk in [{row['lower']}, {row['upper']}]"
                  f" proven={row['proven']} ({row['seconds']} s) {row['log']}", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
