"""SAT reference at the evaluation budget, for the re-targeted ("hard graphs") evolution.

reference_sat.json maps "<graph>@<budget>s<seed>" -> {"graph", "sat_rank", "sat_proven", "budget", "seed", "seconds",
"digest", "split", "family", "n"}:
- sat_rank: rank found by the SAT baseline (minrank.exact.sat.sat_descend, method "sat" of the baseline sweep) with
  the same wall-clock budget per graph as the evolved program gets. The evaluator scores a program on a graph as
  (sat_rank - rank) / sat_rank when MINRANK_EVO_TARGET=sat, so 0 = as good as SAT at equal time, > 0 = better.
- sat_proven: the descent ended with UNSAT inside the budget (sat_rank is the exact minrank; nothing to gain there).
SAT is wall-clock limited, so a value depends on the machine and its load: it is computed once and cached, and every
candidate is compared with the same cached number. Entries whose digest no longer matches the graph are recomputed.
Each entry runs in a spawned worker process (the SAT calls use a forkserver that must not be inherited).

Usage:
  python -m minrank.evolve.reference_sat --split train val --family planted planted_d gnp --min-n 50 --max-n 100 \
      --budget 10 --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from minrank.evolve.reference import GRAPH_DIR, _save_merged, load_reference, select

ROOT = Path(__file__).resolve().parents[2]
SAT_REFERENCE_PATH = Path(os.environ.get("MINRANK_SAT_REFERENCE", ROOT / "minrank/evolve/reference_sat.json"))


def sat_key(name: str, budget: float, seed: int) -> str:
    return f"{name}@{float(budget):g}s{int(seed)}"


def compute_sat_entry(graph_path: str, split: str, budget: float, seed: int) -> tuple[str, dict]:
    """SAT reference entry for one graph (runs sat_descend for `budget` seconds; CPU only)."""
    from minrank.core.graph import Graph
    from minrank.core.verify import verify
    from minrank.exact.sat import sat_descend

    g = Graph.load(graph_path)
    info: dict = {}
    t = time.perf_counter()
    rows = sat_descend(g, budget, np.random.default_rng(seed), info=info)
    sec = time.perf_counter() - t
    return sat_key(g.name, budget, seed), {
        "graph": g.name, "sat_rank": int(verify(g, rows)), "sat_proven": bool(info.get("proven", False)),
        "budget": float(budget), "seed": int(seed), "seconds": round(sec, 3), "digest": g.digest(), "split": split,
        "family": g.family, "n": g.n}


def ensure_sat_reference(index_rows: list[dict], budget: float, seed: int = 0, graph_dir: Path = GRAPH_DIR,
                         path: Path = SAT_REFERENCE_PATH, workers: int = 1) -> dict:
    """Return the SAT reference file's entries, computing missing or stale ones for `index_rows` at (budget, seed)."""
    ref = load_reference(path)
    todo = [r for r in index_rows
            if ref.get(sat_key(r["name"], budget, seed), {}).get("digest") != r["digest"]]
    if not todo:
        return ref
    new: dict = {}
    with ProcessPoolExecutor(max(1, workers), mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(compute_sat_entry, str(graph_dir / r["path"]), r["split"], budget, seed) for r in todo]
        for fu in as_completed(futs):
            key, e = fu.result()
            new[key] = e
    return _save_merged(path, new)


def sat_entry(ref: dict, name: str, budget: float, seed: int) -> dict:
    return ref[sat_key(name, budget, seed)]


def main() -> None:
    from minrank.bench.instances import load_index

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, default=GRAPH_DIR)
    ap.add_argument("--out", type=Path, default=SAT_REFERENCE_PATH)
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--family", nargs="*", default=None)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--budget", type=float, required=True, help="s per graph; must equal MINRANK_EVO_BUDGET")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    a = ap.parse_args()
    rows = [r for r in select(load_index(a.graphs), a.split, a.family, a.max_n) if int(r["n"]) >= a.min_n]
    ref = ensure_sat_reference(rows, a.budget, a.seed, a.graphs, a.out, a.workers)
    for r in rows:
        e = sat_entry(ref, r["name"], a.budget, a.seed)
        print(f"{r['name']}: sat_rank={e['sat_rank']} proven={e['sat_proven']} ({e['seconds']}s)")
    print(f"{len(rows)} graphs selected; {len(ref)} entries in {a.out}")


if __name__ == "__main__":
    main()
