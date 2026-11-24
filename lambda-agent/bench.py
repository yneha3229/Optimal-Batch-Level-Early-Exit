"""Benchmark: how tightly does each method bound beta, and how fast?

    python lambda-agent/bench.py suite                  write suite.json (fixed graphs)
    python lambda-agent/bench.py run [--limit 60]       results.jsonl (resumable, 3 graphs in parallel)
    python lambda-agent/bench.py report                 RESULTS.md

Every graph gets ONE certified lower bound LB = max(acyclic set, Shannon LP) (per component, summed). Each method
produces an upper bound UB from explicit codes; it pins beta exactly when UB = LB (then lambda_2 = beta too).
Methods (time limit per graph each; LB time is added to every method, since a certificate needs both bounds):
  clique_cover     the classic XOR-per-clique code (CP-SAT, strengthened model)
  frac_cover       time sharing over cliques: the standard vector-linear bound (Blasiak et al.)
  scalar_minrank   the standard minrank method: plain SAT, descend from the clique cover (block length 1)
  vector_plain     the straightforward vector search: block length t = 1..4, descend from t * cover for each t,
                   plain SAT (no symmetry fix, no LP guidance), the budget split evenly over t
  vector_plain_stop  the same plain search, but it is told the lower bound and stops as soon as it reaches it
                   (isolates what the agent adds beyond knowing when it is done)
  agent            the lambda agent (decomposition, symmetry fix, LP-guided targets, doubling slices, early stop)
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lambda_agent.agent import LambdaAgent  # noqa: E402
from lambda_agent.graph import Digraph, blowup, induced, scc  # noqa: E402
from lambda_agent.lower import SHANNON_MAX_N, mais, shannon  # noqa: E402
from lambda_agent.upper import clique_cover, frac_cover, minrank_sat  # noqa: E402

METHODS = ["clique_cover", "frac_cover", "scalar_minrank", "vector_plain", "vector_plain_stop", "agent"]


# ----------------------------------------------------------------------------------------------- suite
def make_suite() -> list[dict]:
    gs = []

    def rnd(n, p, two, seed, undirected=False):
        rng = random.Random(seed * 1000003 + n * 101 + int(p * 100) + int(two * 10))
        arcs = []
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() >= p:
                    continue
                if undirected or rng.random() < two:
                    arcs += [(i, j), (j, i)]
                else:
                    arcs.append((i, j) if rng.random() < 0.5 else (j, i))
        return arcs

    for n in (6, 7, 8, 9, 10):
        for p in (0.3, 0.5):
            for two in (0.3, 0.6):
                for seed in (1, 2, 3, 4):
                    gs.append(Digraph.from_arcs(n, rnd(n, p, two, seed), f"digraph_n{n}_p{p}_two{two}_s{seed}"))
    for n in (7, 8, 9, 10):
        for p in (0.3, 0.5):
            for seed in (1, 2, 3):
                gs.append(Digraph.from_arcs(n, rnd(n, p, 1.0, seed, True), f"undirected_n{n}_p{p}_s{seed}"))
    ring = lambda n: [(i, (i + 1) % n) for i in range(n)] + [((i + 1) % n, i) for i in range(n)]  # noqa: E731
    comp = lambda n, arcs: [(i, j) for i in range(n) for j in range(n) if i != j and (i, j) not in set(arcs)]  # noqa: E731
    for n in (5, 7, 9):
        gs.append(Digraph.from_arcs(n, ring(n), f"odd_cycle_{n}"))
        gs.append(Digraph.from_arcs(n, comp(n, ring(n)), f"odd_cycle_{n}_complement"))
    for n in (5, 6, 7, 8, 9, 10):
        gs.append(Digraph.from_arcs(n, [(i, (i + d) % n) for i in range(n) for d in (1, 2)], f"directed_circulant_{n}_12"))
    for n in (7, 9):
        gs.append(Digraph.from_arcs(n, [(i, (i + d) % n) for i in range(n) for d in (1, 3)], f"directed_circulant_{n}_13"))
    pet = [(i, (i + 1) % 5) for i in range(5)] + [(5 + i, 5 + (i + 2) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)]
    gs.append(Digraph.from_arcs(10, pet + [(j, i) for i, j in pet], "petersen"))
    return [g.to_json() for g in gs]


# ------------------------------------------------------------------------------------------ the methods
def lower_bound(g: Digraph) -> tuple[Fraction, list[tuple[list[int], Digraph, Fraction, list[int]]]]:
    parts, total = [], Fraction(0)
    for c in scc(g):
        h = induced(g, c)
        a, _ = mais(h, 10)
        lb = Fraction(len(a))
        if 1 < h.n <= SHANNON_MAX_N:
            lb = max(lb, shannon(h))
        parts.append((c, h, lb, a))
        total += lb
    return total, parts


def descend(h: Digraph, start: int, t: int, deadline: float, floor_r: int) -> Fraction:
    """Plain SAT descent: r = start-1, start-2, ... until UNSAT or time is up; returns the best rate r/t found."""
    best = Fraction(start, t)
    r = start - 1
    while r >= max(1, floor_r) and time.perf_counter() < deadline:
        res = minrank_sat(blowup(h, t), r, deadline - time.perf_counter(), None)
        if not res:
            break
        best = Fraction(r, t)
        r -= 1
    return best


def run_method(m: str, g: Digraph, parts, limit: float) -> tuple[Fraction, float]:
    t0 = time.perf_counter()
    if m == "agent":
        a = LambdaAgent(g, budget=limit).run()
        return a.hi, time.perf_counter() - t0
    ub = Fraction(0)
    for c, h, lb, _ in parts:
        if h.n == 1:
            ub += 1
            continue
        cov = len(clique_cover(h, 10)[0])
        if m == "clique_cover":
            ub += cov
        elif m == "frac_cover":
            ub += min(Fraction(cov), frac_cover(h)[0])
        elif m == "scalar_minrank":
            ub += descend(h, cov, 1, t0 + limit * h.n / g.n, 1)
        elif m in ("vector_plain", "vector_plain_stop"):
            share = limit * h.n / g.n / 4
            best = Fraction(cov)
            for t in range(1, 5):
                if m == "vector_plain_stop" and best <= lb:
                    break
                floor_r = math.ceil(lb * t) if m == "vector_plain_stop" else 1
                best = min(best, descend(h, cov * t, t, time.perf_counter() + share, floor_r))
            ub += best
    return ub, time.perf_counter() - t0


def solve_graph(d: dict, limit: float, methods=None) -> list[dict]:
    g = Digraph.from_json(d)
    t0 = time.perf_counter()
    lb, parts = lower_bound(g)
    t_lb = time.perf_counter() - t0
    rows = []
    for m in methods or METHODS:
        ub, sec = run_method(m, g, parts, limit)
        if m != "agent":
            sec += t_lb  # the agent computes its own lower bound inside its budget
        rows.append({"graph": g.name, "n": g.n, "method": m, "lb": str(lb), "ub": str(ub), "pinned": ub == lb,
                     "gap": float(ub - lb), "seconds": round(sec, 3)})
    return rows


# ---------------------------------------------------------------------------------------------- CLI
def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    suite_p, res_p = HERE / "suite.json", HERE / "results.jsonl"
    if cmd == "suite":
        s = make_suite()
        suite_p.write_text(json.dumps(s))
        print(len(s), "graphs")
    elif cmd == "run":
        limit = float(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 60.0
        methods = sys.argv[sys.argv.index("--methods") + 1].split(",") if "--methods" in sys.argv else METHODS
        suite = json.loads(suite_p.read_text())
        done = set()
        if res_p.exists():
            done = {json.loads(line)["graph"] for line in res_p.read_text().splitlines()
                    if line.strip() and json.loads(line)["method"] in methods}
        todo = [d for d in suite if d["name"] not in done]
        print(len(todo), "graphs to run", flush=True)
        with ProcessPoolExecutor(3) as ex, res_p.open("a") as f:
            futs = {ex.submit(solve_graph, d, limit, methods): d["name"] for d in todo}
            for fut in as_completed(futs):
                rows = fut.result()
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                f.flush()
                print(futs[fut], " | ".join(f"{r['method']} {r['ub']}" for r in rows), "| LB", rows[0]["lb"], flush=True)
    elif cmd == "report":
        report(res_p)


def report(res_p: Path) -> None:
    import statistics as st

    rows = [json.loads(line) for line in res_p.read_text().splitlines() if line.strip()]
    fam = lambda name: "random digraphs" if name.startswith("digraph") else "random undirected" if name.startswith("undirected") else "structured"  # noqa: E731
    names = {"clique_cover": "Clique cover", "frac_cover": "Fractional clique cover", "scalar_minrank": "Scalar minrank (plain SAT)",
             "vector_plain": "Vector-linear, plain search", "vector_plain_stop": "Vector-linear, plain search + stop at LB",
             "agent": "Lambda agent"}
    graphs = sorted({r["graph"] for r in rows})
    by = {(r["graph"], r["method"]): r for r in rows}
    lines = ["# Lambda agent benchmark", "",
             f"{len(graphs)} graphs. One certified lower bound per graph (acyclic set, Shannon LP); each method's upper "
             "bound comes from explicit codes. \"β pinned\" = upper bound equals the lower bound, which proves β = λ₂. "
             "60 s per graph per method, one core; times include the lower bound.", ""]
    groups = {"all": graphs}
    for gname in graphs:
        groups.setdefault(fam(gname), []).append(gname)
    for k, gs in groups.items():
        lines += [f"## {k} ({len(gs)} graphs)", "",
                  "| method | β pinned | mean gap to lower bound | median time | 90th pct time | max time |", "|---|---|---|---|---|---|"]
        for m in METHODS:
            rs = [by[(x, m)] for x in gs if (x, m) in by]
            ts = [r["seconds"] for r in rs]
            fmt = lambda x: f"{x * 1000:.0f} ms" if x < 1 else f"{x:.1f} s"  # noqa: E731
            p90 = sorted(ts)[int(0.9 * (len(ts) - 1))]
            lines.append(f"| {names[m]} | {sum(r['pinned'] for r in rs)}/{len(rs)} | {st.mean(r['gap'] for r in rs):.3f} | "
                         f"{fmt(st.median(ts))} | {fmt(p90)} | {fmt(max(ts))} |")
        lines.append("")
    lines += ["## Per graph", "", "| graph | lower bound | " + " | ".join(names[m] for m in METHODS) + " |",
              "|---|---|" + "---|" * len(METHODS)]
    for x in graphs:
        lb = by[(x, "agent")]["lb"]
        lines.append(f"| {x} | {lb} | " + " | ".join(f"{by[(x, m)]['ub']} ({by[(x, m)]['seconds']:.1f}s)" for m in METHODS) + " |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    # table for the demo page (beta-agent/index.html shows it next to the bound-only benchmark)
    data = {"graphs": len(graphs), "limit_s": 60, "groups": {}}
    for k, gs in groups.items():
        data["groups"][k] = {}
        for m in METHODS:
            rs = [by[(x, m)] for x in gs if (x, m) in by]
            ts = sorted(r["seconds"] for r in rs)
            data["groups"][k][m] = {"name": names[m], "n": len(rs), "pinned": sum(r["pinned"] for r in rs),
                                    "gap": round(st.mean(r["gap"] for r in rs), 3), "median_s": round(st.median(ts), 4),
                                    "p90_s": round(ts[int(0.9 * (len(ts) - 1))], 4), "max_s": round(max(ts), 3)}
    data["examples"] = [{"graph": x, "lb": by[(x, "agent")]["lb"],
                         "ub": {m: by[(x, m)]["ub"] for m in METHODS if (x, m) in by}}
                        for x in ["odd_cycle_5", "odd_cycle_9_complement", "directed_circulant_8_12", "petersen",
                                  "digraph_n6_p0.3_two0.3_s2"] if (x, "agent") in by]
    out = HERE.parent / "beta-agent" / "bench" / "lambda-data.js"
    out.write_text("window.LAMBDA_BENCH = " + json.dumps(data) + ";\n")
    print("\n".join(lines[:4 + 10 * len(groups)]))


if __name__ == "__main__":
    main()
