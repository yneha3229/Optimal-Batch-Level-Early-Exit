"""Per-graph reference values used to score evolved programs.

reference.json maps graph name -> {"cover_rank", "lb", "digest", "split", "family", "n", "cover_time", "lb_time"}:
- cover_rank: best clique cover found by TabuCol (seed 0) in `cover_time` seconds. Score of a program on a graph is
  (cover_rank - rank) / cover_rank, so 0 = as good as a strong clique cover, > 0 = below it.
- lb: certified lower bound (size of a verified acyclic induced subgraph, minrank.exact.lower.mais).
Entries whose digest no longer matches the graph file are recomputed.

Usage:
  python -m minrank.evolve.reference --cover-time 10 --lb-time 30 --workers 32
  python -m minrank.evolve.reference --family planted planted_d gnp --max-n 50 --split train val
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRAPH_DIR = Path(os.environ.get("MINRANK_GRAPHS", ROOT / "minrank/bench/graphs"))
REFERENCE_PATH = Path(os.environ.get("MINRANK_REFERENCE", ROOT / "minrank/evolve/reference.json"))


def compute_entry(graph_path: str, split: str, cover_time: float, lb_time: float) -> tuple[str, dict]:
    """Reference entry for one graph (runs TabuCol and CP-SAT; CPU only)."""
    from minrank.baselines.clique_cover import clique_cover_rows
    from minrank.core.graph import Graph
    from minrank.core.verify import verify
    from minrank.exact.lower import mais

    g = Graph.load(graph_path)
    cover = verify(g, clique_cover_rows(g, "tabucol", cover_time, np.random.default_rng(0)))
    lb = mais(g, lb_time).value
    if lb > cover:
        raise RuntimeError(f"{g.name}: lower bound {lb} > clique cover {cover}; a bound is wrong")
    return g.name, {"cover_rank": int(cover), "lb": int(lb), "digest": g.digest(), "split": split,
                    "family": g.family, "n": g.n, "cover_time": cover_time, "lb_time": lb_time}


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text())


def _save_merged(path: Path, new: dict) -> dict:
    """Merge `new` into the file under an exclusive lock (several evaluator processes may write) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(path.suffix + ".lock")
    with lock.open("a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            cur = load_reference(path)
            cur.update(new)
            tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(cur, indent=1, sort_keys=True))
            tmp.replace(path)  # atomic on POSIX
            return cur
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def ensure_reference(index_rows: list[dict], graph_dir: Path = GRAPH_DIR, path: Path = REFERENCE_PATH,
                     cover_time: float = 1.0, lb_time: float = 5.0, workers: int = 1) -> dict:
    """Return reference entries for `index_rows` (rows of bench/graphs/index.csv), computing missing or stale ones."""
    ref = load_reference(path)
    todo = [r for r in index_rows if r["name"] not in ref or ref[r["name"]].get("digest") != r["digest"]]
    if not todo:
        return ref
    new: dict = {}
    if workers <= 1:
        for r in todo:
            name, e = compute_entry(str(graph_dir / r["path"]), r["split"], cover_time, lb_time)
            new[name] = e
    else:
        with ProcessPoolExecutor(workers) as ex:
            futs = [ex.submit(compute_entry, str(graph_dir / r["path"]), r["split"], cover_time, lb_time) for r in todo]
            for fu in as_completed(futs):
                name, e = fu.result()
                new[name] = e
    return _save_merged(path, new)


def select(index_rows: list[dict], splits: list[str] | None = None, families: list[str] | None = None,
           max_n: int | None = None) -> list[dict]:
    return [r for r in index_rows if (not splits or r["split"] in splits)
            and (not families or r["family"] in families) and (max_n is None or int(r["n"]) <= max_n)]


def main() -> None:
    from minrank.bench.instances import load_index

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", type=Path, default=GRAPH_DIR)
    ap.add_argument("--out", type=Path, default=REFERENCE_PATH)
    ap.add_argument("--split", nargs="*", default=None)
    ap.add_argument("--family", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=None)
    ap.add_argument("--cover-time", type=float, default=10.0)
    ap.add_argument("--lb-time", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--force", action="store_true", help="recompute every selected entry")
    a = ap.parse_args()
    rows = select(load_index(a.graphs), a.split, a.family, a.max_n)
    if a.force and a.out.exists():
        ref = load_reference(a.out)
        for r in rows:
            ref.pop(r["name"], None)
        a.out.write_text(json.dumps(ref, indent=1, sort_keys=True))
    ref = ensure_reference(rows, a.graphs, a.out, a.cover_time, a.lb_time, a.workers)
    print(f"{len(rows)} graphs selected; {len(ref)} entries in {a.out}")


if __name__ == "__main__":
    main()
