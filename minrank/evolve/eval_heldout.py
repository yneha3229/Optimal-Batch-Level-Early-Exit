"""Score an evolved program on held-out graphs, in the same CSV format as the baseline sweep.

Rows use minrank.run_baselines.RESULT_FIELDS with method "evolved:<candidate id>", so the aggregator compares the
evolved heuristic with every baseline at the same budgets. The answer of every run is verified in this process.
A run whose answer is missing or does not fit is written with rank empty and verified=False (and counted as a
failure by the aggregator). Better certificates are saved with minrank.run_baselines.keep_best.

Usage:
  python -m minrank.evolve.eval_heldout --run-dir minrank/evolve/runs/RUN --split val test --budgets 1 10 60
  python -m minrank.evolve.eval_heldout --program path/to/program.py --split test --seeds 0 1 2 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from minrank.bench.instances import load_index
from minrank.core.f2 import rank_f2
from minrank.core.graph import Graph
from minrank.core.verify import fit_error
from minrank.evolve.evaluator import candidate_id, run_one, static_check
from minrank.run_baselines import RESULT_FIELDS, existing_keys, job_key, keep_best

ROOT = Path(__file__).resolve().parents[2]


def find_best_program(run_dir: Path) -> Path:
    """OpenEvolve output: <run>/best/best_program.py, else the newest checkpoint's best_program.py."""
    best = run_dir / "best" / "best_program.py"
    if best.exists():
        return best
    cps = sorted((run_dir / "checkpoints").glob("checkpoint_*/best_program.py"),
                 key=lambda p: int(p.parent.name.split("_")[-1]))
    if not cps:
        raise FileNotFoundError(f"no best_program.py under {run_dir}")
    return cps[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--program", type=Path)
    src.add_argument("--run-dir", type=Path)
    ap.add_argument("--graphs", type=Path, default=ROOT / "minrank/bench/graphs")
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    ap.add_argument("--family", nargs="*", default=None)
    ap.add_argument("--max-n", type=int, default=10 ** 9)
    ap.add_argument("--budgets", nargs="+", type=float, default=[1.0, 10.0, 60.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=ROOT / "minrank/results/evolved.csv")
    ap.add_argument("--certs", type=Path, default=ROOT / "minrank/certificates")
    a = ap.parse_args()

    prog = a.program if a.program else find_best_program(a.run_dir)
    code = prog.read_text()
    err = static_check(code)
    if err:
        raise SystemExit(f"{prog} rejected by the static check: {err}")
    method = f"evolved:{candidate_id(code)}"
    idx = [r for r in load_index(a.graphs) if r["split"] in a.split and int(r["n"]) <= a.max_n
           and (not a.family or r["family"] in a.family)]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(a.out)
    write_header = not a.out.exists() or a.out.stat().st_size == 0
    graphs = {r["name"]: Graph.load(a.graphs / r["path"]) for r in idx}
    jobs = []
    for r in idx:
        for b in a.budgets:
            for s in a.seeds:
                k = job_key(r["name"], method, b, s)
                if k in done:
                    continue
                done.add(k)
                jobs.append((r, b, s))
    print(f"{prog} as {method}: {len(jobs)} jobs on {a.workers} workers", flush=True)

    def job(r: dict, b: float, s: int) -> tuple[dict, list[int] | None]:
        g = graphs[r["name"]]
        res = run_one(str(prog), str(a.graphs / r["path"]), b, s)
        rows = res["rows"] if res["status"] == "ok" and fit_error(g, res["rows"]) is None else None
        row = {"graph": r["name"], "split": r["split"], "family": r["family"], "n": int(r["n"]),
               "density": float(r["density"]), "method": method, "budget": b, "seed": s,
               "rank": rank_f2(rows) if rows is not None else "", "seconds": round(res["seconds"], 3),
               "verified": rows is not None}
        if rows is None:
            print(f"warning: {r['name']} budget {b} seed {s}: {res['status']} {res['error'] or 'does not fit'}",
                  flush=True)
        return row, rows

    with a.out.open("a", newline="") as f, ThreadPoolExecutor(max(1, a.workers)) as ex:
        w = csv.DictWriter(f, RESULT_FIELDS)
        if write_header:
            w.writeheader()
        futs = [ex.submit(job, r, b, s) for r, b, s in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            row, rows = fu.result()
            if rows is not None:
                keep_best(a.certs, graphs[row["graph"]], rows, f"{method}@{row['budget']}s/seed{row['seed']}")
            w.writerow(row)
            f.flush()
            if k % 50 == 0:
                print(f"{k}/{len(futs)} done", flush=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
