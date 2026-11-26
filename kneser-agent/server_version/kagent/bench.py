"""Benchmark driver.

    python -m kagent.bench run --mode no_llm --names "K(21,2)" "K(13,3)" --cores 32-39
    python -m kagent.bench run --mode full --group main --max-receivers 1000 --cores 32-39
    python -m kagent.bench run --mode no_llm --seed 1 --names ...        (relabelling seed 1 graphs)

Graphs are addressed by their opaque id; names are mapped through bench/truth.json by this driver only (the agent
itself never sees them). Results: results/runs/<tag>/<id>.json (tag = mode, or mode_s<seed> for seeds > 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .graphs import graph_id
from .run import parse_cores, run_one

ROOT = Path(__file__).resolve().parents[1]


def select(names=None, group=None, max_receivers=None, min_receivers=None, seed=0) -> list[tuple[str, dict]]:
    truth = json.loads((ROOT / "bench" / "truth.json").read_text())
    out = []
    for gid, t in truth.items():
        if t["seed"] != seed:
            continue
        if names and t["name"] not in names:
            continue
        if group and t["group"] != group:
            continue
        if max_receivers and t["receivers"] > max_receivers:
            continue
        if min_receivers and t["receivers"] <= min_receivers:
            continue
        out.append((gid, t))
    if names:
        order = {n: i for i, n in enumerate(names)}
        out.sort(key=lambda x: order[x[1]["name"]])
    else:
        out.sort(key=lambda x: x[1]["receivers"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "list"])
    ap.add_argument("--mode", default="no_llm")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--group", default=None)
    ap.add_argument("--max-receivers", type=int, default=None)
    ap.add_argument("--min-receivers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--cores", default="32-39")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    sel = select(a.names, a.group, a.max_receivers, a.min_receivers, a.seed)
    if a.cmd == "list":
        for gid, t in sel:
            print(gid, t["name"], t["receivers"], t["group"])
        return 0
    cores = parse_cores(a.cores)
    os.sched_setaffinity(0, cores)
    tag = a.tag or (a.mode if a.seed == 0 else f"{a.mode}_s{a.seed}")
    for gid, t in sel:
        out = ROOT / "results" / "runs" / tag / f"{gid}.json"
        if a.skip_done and out.exists():
            continue
        t0 = time.perf_counter()
        res = run_one(f"bench/graphs/{gid}", a.mode, a.budget, cores, model=a.model, seed=a.seed, tag=tag)
        v = res.get("verification", {})
        print(json.dumps({"id": gid, "name": t["name"], "N": t["receivers"], "L": res["L"], "U": res["U"],
                          "proven": res["proven"], "field": res["field"], "t_first": res["t_first_bound"],
                          "t_proof": res["t_proof"], "wall": res["wall"], "verdict": v.get("verdict"),
                          "llm_calls": (res.get("llm") or {}).get("calls"),
                          "total_s": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
