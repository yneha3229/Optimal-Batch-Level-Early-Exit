"""Baseline sweep: every (graph, method, budget, seed) -> one CSV row; best certificate per graph is kept.

Usage (server, one process per core):
  python -m minrank.bench.instances                       # once: writes minrank/bench/graphs/
  python -m minrank.run_baselines --split train val test --budgets 1 10 60 --seeds 0 1 2 --workers 32
  python -m minrank.run_baselines --lower-only --lb-time 60 --workers 32   # certified lower bounds only

Outputs:
  minrank/results/baselines.csv     graph, split, family, n, density, method, budget, seed, rank, seconds, verified
  minrank/results/lower_bounds.csv  graph, lb, proven_max, mais_upper, seconds
  minrank/certificates/<graph>.json best verified matrix found so far (any method)
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

from .baselines.alt_f2 import alt_search
from .baselines.clique_cover import clique_cover_rows
from .baselines.factor_search import factor_search
from .bench.instances import load_index
from .core.graph import Graph
from .core.verify import load_certificate, save_certificate, verify
from .exact.lower import mais
from .exact.sat import sat_descend

ROOT = Path(__file__).resolve().parent
METHODS = ("dsatur", "random_greedy", "tabucol", "factor_tabu", "factor_anneal", "alt_isd", "sat", "ldg",
           "partial_cover", "identity_adj")
BUDGET_FREE = ("dsatur", "ldg", "identity_adj")  # deterministic: one row per graph at budget 0
OPTIONAL_METHODS = ("sdp",)  # needs cvxpy; run only when named with --methods
# Pool workers are spawned, never forked: a forked worker would inherit a SAT forkserver started by its parent
# and fail on its next SAT call (ChildProcessError).
SPAWN = mp.get_context("spawn")
RESULT_FIELDS = ["graph", "split", "family", "n", "density", "method", "budget", "seed", "rank", "seconds", "verified"]


def run_method(g: Graph, method: str, budget: float, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    if method in ("dsatur", "random_greedy", "tabucol"):
        return clique_cover_rows(g, method, budget, rng)
    if method == "factor_tabu":
        return factor_search(g, budget, rng, mode="tabu").rows
    if method == "factor_anneal":
        return factor_search(g, budget, rng, mode="anneal").rows
    if method == "sat":
        return sat_descend(g, budget, rng)
    if method == "alt_isd":
        return alt_search(g, budget, rng).rows
    if method == "ldg":
        from .baselines.ldg import ldg_rows
        return ldg_rows(g)
    if method == "partial_cover":
        from .baselines.partial_cover import partial_cover_rows
        return partial_cover_rows(g, budget, rng)
    if method == "identity_adj":
        from .baselines.identity_adj import identity_plus_adjacency_rows
        return identity_plus_adjacency_rows(g)
    if method == "sdp":
        from .baselines.sdp import sdp_rows
        return sdp_rows(g, budget, rng)
    raise ValueError(f"unknown method {method}")


def _job(graph_path: str, meta: dict, method: str, budget: float, seed: int) -> dict:
    g = Graph.load(graph_path)
    t = time.perf_counter()
    rows = run_method(g, method, budget, seed)
    sec = time.perf_counter() - t
    r = verify(g, rows)  # raises on a non-fitting matrix
    return {**meta, "method": method, "budget": budget, "seed": seed, "rank": r, "seconds": round(sec, 3),
            "verified": True, "_rows": rows}


def _lb_job(graph_path: str, name: str, lb_time: float) -> dict:
    g = Graph.load(graph_path)
    t = time.perf_counter()
    lb = mais(g, lb_time)
    return {"graph": name, "lb": lb.value, "proven_max": lb.proven_max, "mais_upper": lb.upper,
            "seconds": round(time.perf_counter() - t, 2)}


def stored_rank(path: Path, g: Graph) -> int | None:
    """Rank of the stored certificate if it is for this exact graph and still verifies; else None (treat as absent).

    A certificate for a different graph with the same name (benchmark regenerated) or one that no longer fits is
    never allowed to block a valid new result.
    """
    if not path.exists():
        return None
    c = load_certificate(path)
    if c.get("digest") != g.digest():
        print(f"warning: {path.name} is for another graph (digest {c.get('digest')} != {g.digest()}); replacing",
              flush=True)
        return None
    try:
        return verify(g, c["rows"])
    except ValueError as e:
        print(f"warning: {path.name} does not verify ({e}); replacing", flush=True)
        return None


def keep_best(cert_dir: Path, g: Graph, rows: list[int], method: str) -> bool:
    """Save rows as the graph's certificate if strictly better than a valid stored one. Returns True if saved."""
    path = cert_dir / f"{g.name}.json"
    r = verify(g, rows)
    old = stored_rank(path, g)
    if old is not None and old <= r:
        return False
    save_certificate(path, g, rows, method)
    return True


def job_key(graph: str, method: str, budget: float, seed: int) -> tuple[str, str, float, int]:
    return graph, method, float(budget), int(seed)


def existing_keys(path: Path) -> set[tuple[str, str, float, int]]:
    """Keys of rows already in the results CSV (so reruns skip them). Refuses a file with a different header."""
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != RESULT_FIELDS:
            raise SystemExit(f"{path} has header {rd.fieldnames}, expected {RESULT_FIELDS}; use another --out")
        return {job_key(r["graph"], r["method"], r["budget"], r["seed"]) for r in rd}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, default=ROOT / "bench/graphs")
    ap.add_argument("--split", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--family", nargs="*", default=None, help="restrict to these families")
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS + OPTIONAL_METHODS)
    ap.add_argument("--budgets", nargs="+", type=float, default=[1.0, 10.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=ROOT / "results/baselines.csv")
    ap.add_argument("--certs", type=Path, default=ROOT / "certificates")
    ap.add_argument("--lower-only", action="store_true")
    ap.add_argument("--lb-time", type=float, default=30.0)
    a = ap.parse_args()

    idx = [r for r in load_index(a.graphs) if r["split"] in a.split and int(r["n"]) <= a.max_n
           and (not a.family or r["family"] in a.family)]
    print(f"{len(idx)} graphs", flush=True)

    if a.lower_only:
        out = a.out.parent / "lower_bounds.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=SPAWN) as ex:
            w = csv.DictWriter(f, ["graph", "lb", "proven_max", "mais_upper", "seconds"])
            w.writeheader()
            futs = [ex.submit(_lb_job, str(a.graphs / r["path"]), r["name"], a.lb_time) for r in idx]
            for fu in as_completed(futs):
                w.writerow(fu.result())
                f.flush()
        print(f"wrote {out}")
        return

    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(a.out)
    new = not done and (not a.out.exists() or a.out.stat().st_size == 0)
    graphs = {r["name"]: Graph.load(a.graphs / r["path"]) for r in idx}
    with a.out.open("a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=SPAWN) as ex:
        w = csv.DictWriter(f, RESULT_FIELDS)
        if new:
            w.writeheader()
        futs = []
        skipped = 0
        for r in idx:
            meta = {"graph": r["name"], "split": r["split"], "family": r["family"], "n": int(r["n"]),
                    "density": float(r["density"])}
            for m in a.methods:
                budgets = [0.0] if m in BUDGET_FREE else a.budgets  # deterministic, budget-free
                seeds = [0] if m in BUDGET_FREE else a.seeds
                for b in budgets:
                    for s in seeds:
                        if job_key(r["name"], m, b, s) in done:
                            skipped += 1
                            continue
                        done.add(job_key(r["name"], m, b, s))  # also dedups repeated --budgets/--seeds values
                        futs.append(ex.submit(_job, str(a.graphs / r["path"]), meta, m, b, s))
        print(f"{len(futs)} jobs on {a.workers} workers ({skipped} already in {a.out.name}, skipped)", flush=True)
        for k, fu in enumerate(as_completed(futs), 1):
            row = fu.result()
            rows = row.pop("_rows")
            keep_best(a.certs, graphs[row["graph"]], rows, f"{row['method']}@{row['budget']}s/seed{row['seed']}")
            w.writerow(row)
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(futs)} done", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
