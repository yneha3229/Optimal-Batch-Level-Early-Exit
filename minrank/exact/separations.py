"""Option B: search for small explicit separations.

(a) minrk_2(G) < clique cover of the mutual graph (linear codes beat the trivial cover), and
(b) minrk_2(G) != minrk_3(G) (field dependence).

Candidates: structured families (Kneser, Paley, Mycielski, cycle powers, circulants, and complements), random
G(n, p) / D(n, p) with small n, and planted / orthogonality graphs. For each candidate:
  cover = TabuCol clique cover (valid over every field), lb = MAIS by CP-SAT,
  minrk_2 by SAT descent with symmetry breaking, minrk_3 by the one-hot F_3 encoding (descending from the cover).
Every value is backed by a verified matrix; results go to `--out` (CSV) and, for every separation found, the graph
and both certificates go to `--out-dir`.

Day-1 note: no gap (a) was found on random or structured graphs with n <= 15; planted / orthogonality graphs have
gaps by construction, so they are included as positive controls.

Usage:
  python -m minrank.exact.separations --random-n 8 9 10 11 12 --random-count 50 --time-per-call 120 --workers 8
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ..baselines.clique_cover import clique_cover_rows
from ..bench import generators as gen
from ..core.f2 import rank_f2
from ..core.graph import Graph
from ..core.verify import verify
from .lower import mais
from .sat_fq import exact_minrank_fq, verify_fq
from .symmetry import rank_at_most_sym

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["graph", "n", "family", "directed", "lb", "cover", "minrk2", "proven2", "minrk3", "proven3",
          "gap_cover_vs_f2", "f2_ne_f3", "seconds"]


Spec = tuple  # (generator function name in minrank.bench.generators, args tuple, take complement: bool)


def build(spec: Spec) -> Graph:
    """Build a candidate from a picklable spec."""
    fn, args, comp = spec
    g = getattr(gen, fn)(*args)
    return g.complement() if comp else g


def candidates(random_n: list[int], random_count: int, seed: int, directed: bool, structured: bool,
               planted: bool) -> list[tuple[str, Spec]]:
    out: list[tuple[str, Spec]] = []
    if structured:
        base: list[tuple[str, tuple]] = [
            ("kneser", (5, 2)), ("kneser", (6, 2)), ("kneser", (7, 2)), ("paley", (13,)), ("paley", (17,)),
            ("mycielski", (4,)), ("mycielski", (5,)), ("cycle_power", (11, 2)), ("cycle_power", (13, 3)),
            ("cycle_power", (15, 2)),
        ] + [("cycle", (n,)) for n in (5, 7, 9, 11)]
        rng = np.random.default_rng(seed)
        for n in (9, 10, 11, 12, 13, 14, 15):
            for _ in range(3):
                jumps = tuple(sorted(set(int(x) for x in rng.integers(1, n // 2 + 1, size=2))))
                base.append(("circulant", (n, jumps)))
        seen = set()
        for fn, args in base:
            name = build((fn, args, False)).name
            if name in seen:
                continue
            seen.add(name)
            out.append((name, (fn, args, False)))
            out.append((name + "_c", (fn, args, True)))
    for n in random_n:
        for k in range(random_count):
            for p in (0.3, 0.5, 0.7):
                s = seed * 100_000 + n * 1000 + k
                out.append((f"{'dnp' if directed else 'gnp'}_n{n}_p{p}_s{s}", ("gnp", (n, p, s, directed), False)))
    if planted:
        out += [(f"orth_{k}", ("orthogonality", (k,), False)) for k in (4, 5)]
        out += [(f"planted_n{n}_r{r}_s{seed}", ("planted", (n, r, 0.0, seed, False), False))
                for n, r in ((12, 4), (16, 5))]
    return out


def analyse(name: str, spec: Spec, time_per_call: float, lb_time: float, cover_time: float,
            seed: int, do_f3: bool) -> dict:
    t0 = time.perf_counter()
    g = build(spec)
    cover_rows = clique_cover_rows(g, "tabucol", cover_time, np.random.default_rng(seed))
    cover = verify(g, cover_rows)
    lb = mais(g, lb_time).value
    best2, rows2, proven2 = cover, cover_rows, cover <= lb
    while not proven2:
        res = rank_at_most_sym(g, best2 - 1, time_per_call, mode="cols")
        if res is None:
            break
        if res is False:
            proven2 = True
            break
        rows2, best2 = res, rank_f2(res)
        proven2 = best2 <= lb
    best3, proven3, A3 = (None, False, None)
    if do_f3:
        best3, proven3, A3 = exact_minrank_fq(g, 3, cover, lb, time_per_call)
        if A3 is not None:
            assert verify_fq(g, A3, 3) == best3
    return {"graph": name, "n": g.n, "family": g.family or name.split("_")[0], "directed": g.directed, "lb": lb,
            "cover": cover, "minrk2": best2, "proven2": proven2, "minrk3": best3, "proven3": proven3,
            "gap_cover_vs_f2": cover - best2,
            "f2_ne_f3": (best3 is not None and proven2 and proven3 and best2 != best3),
            "seconds": round(time.perf_counter() - t0, 2),
            "_g": g.to_dict(), "_rows2": [f"{x:x}" for x in rows2], "_A3": A3}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random-n", nargs="*", type=int, default=[8, 9, 10])
    ap.add_argument("--random-count", type=int, default=20)
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--no-structured", action="store_true")
    ap.add_argument("--no-planted", action="store_true")
    ap.add_argument("--no-f3", action="store_true")
    ap.add_argument("--time-per-call", type=float, default=120.0)
    ap.add_argument("--lb-time", type=float, default=10.0)
    ap.add_argument("--cover-time", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--out", type=Path, default=ROOT / "results/separations.csv")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results/separations")
    a = ap.parse_args(argv)

    cands = candidates(a.random_n, a.random_count, a.seed, a.directed, not a.no_structured, not a.no_planted)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if a.out.exists() and a.out.stat().st_size:
        with a.out.open(newline="") as f:
            done = {r["graph"] for r in csv.DictReader(f)}
    todo = [(n, b) for n, b in cands if n not in done]
    print(f"{len(cands)} candidates, {len(cands) - len(todo)} already in {a.out.name}", flush=True)
    new = not a.out.exists() or a.out.stat().st_size == 0
    found = 0
    with a.out.open("a", newline="") as f, ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as ex:
        # spawn: workers must not inherit a forkserver started by this process (waitpid on a non-child fails)
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        futs = [ex.submit(analyse, n, b, a.time_per_call, a.lb_time, a.cover_time, a.seed, not a.no_f3)
                for n, b in todo]
        for fu in as_completed(futs):
            row = fu.result()
            gdict, rows2, A3 = row.pop("_g"), row.pop("_rows2"), row.pop("_A3")
            w.writerow(row)
            f.flush()
            if row["gap_cover_vs_f2"] > 0 or row["f2_ne_f3"]:
                found += 1
                a.out_dir.mkdir(parents=True, exist_ok=True)
                (a.out_dir / f"{row['graph']}.json").write_text(json.dumps(
                    {"graph": gdict, "minrk2": row["minrk2"], "rows_f2_hex": rows2, "minrk3": row["minrk3"],
                     "matrix_f3": A3, "summary": row}))
                print(f"SEPARATION {row['graph']}: cover={row['cover']} minrk2={row['minrk2']} "
                      f"minrk3={row['minrk3']}", flush=True)
    print(f"wrote {a.out}; {found} separations")


if __name__ == "__main__":
    main()
