"""Benchmark of the parallel agent (v2) on unseen very large graphs.

    python gap-agent/bench2.py run       -> results/bench2.jsonl   (each job in its own process, with a watchdog)
    python gap-agent/bench2.py report    -> RESULTS2.md

Graphs: 3000, 6000 and 10000 receivers; two structured families (sparse side information; clustered "communities")
and the three dense families of bench.py; two graphs each (30 graphs, none seen in training).
Three set-ups, 30 s each, same seed:
  v1 (1 core)          the learned agent from bench.py (learned prior + online switching) on the whole graph
  v1 portfolio (3)     three copies of v1 with different seeds at the same time; best bound per side at every moment
  v2 (3 workers)       search + structure/exact + lambda workers (gap_agent/parallel.py)
Gap integrals use the same reference (the v1 opening on that graph) for all three.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gap_agent.agent import LearnedBandit, gap_integral, run  # noqa: E402
from gap_agent.graph import FAMILIES, STRUCTURED, is_acyclic_set, is_clique_cover, make_graph, make_structured  # noqa: E402

T = 30.0
SIZES = (3000, 6000, 10000)
FAMS = STRUCTURED + FAMILIES
SPECS = [(f, n, r) for f in FAMS for n in SIZES for r in range(2)]
OUT = HERE / "results" / "bench2.jsonl"


def graph(i):
    f, n, r = SPECS[i]
    rng = np.random.default_rng(700_000 + 1000 * n + 10 * FAMS.index(f) + r)
    g = make_structured(f, n, rng) if f in STRUCTURED else make_graph(f, n, rng)
    g.name = f"{f}_n{n}_{r}"
    return g


def _model():
    return pickle.loads((HERE / "models" / "rate.pkl").read_bytes())


def _v1(args):
    g, seed, q = args
    r = run(g, LearnedBandit(_model(), k0=1.0), T=T, seed=seed)
    ok = is_acyclic_set(g, r["acyclic_set"]) and is_clique_cover(g, np.array(r["cover"]))
    q.put({k: r[k] for k in ("lo0", "hi0", "lo", "hi", "events", "integral")} | {"seed": seed, "verified": ok})


def job(kind, i):
    from gap_agent.strategies import compile_all

    compile_all()
    g = graph(i)
    base = {"graph": g.name, "family": SPECS[i][0], "n": g.n, "kind": kind}
    if kind == "portfolio":
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        ps = [ctx.Process(target=_v1, args=((g, s, q),)) for s in range(3)]
        for p in ps:
            p.start()
        runs = sorted([q.get() for _ in ps], key=lambda r: r["seed"])
        for p in ps:
            p.join()
        return base | {"runs": runs}
    from gap_agent.parallel import solve

    r = solve(g, _model(), T=T, seed=0)
    return base | {k: v for k, v in r.items() if k not in ("graph", "n")}


def cmd_run():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {(json.loads(x)["graph"], json.loads(x)["kind"]) for x in OUT.read_text().splitlines() if x.strip()}
    for i in range(len(SPECS)):
        for kind in ("portfolio", "v2"):
            f, n, r = SPECS[i]
            if (f"{f}_n{n}_{r}", kind) in done:
                continue
            try:
                p = subprocess.run([sys.executable, __file__, "job", kind, str(i)], capture_output=True, text=True,
                                   timeout=T + 90)
                line = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else None
                res = json.loads(line) if line else {"graph": f"{f}_n{n}_{r}", "kind": kind, "error": p.stderr[-500:]}
            except subprocess.TimeoutExpired:
                res = {"graph": f"{f}_n{n}_{r}", "kind": kind, "error": "watchdog timeout"}
            with open(OUT, "a") as fh:
                fh.write(json.dumps(res, default=str) + "\n")
            print(res["graph"], kind, "error" if "error" in res else "ok", flush=True)


def _combine(runs):
    """Best lower and best upper bound over the portfolio's runs at every moment, as one event list."""
    ev = sorted((t, side, v, k) for k, r in enumerate(runs) for t, side, v, _ in r["events"])
    lo, hi, out = max(r["lo0"] for r in runs), min(r["hi0"] for r in runs), []
    for t, side, v, _ in ev:
        if side == "lo" and v > lo:
            lo = v
            out.append((t, "lo", v, ""))
        elif side == "hi" and v < hi:
            hi = v
            out.append((t, "hi", v, ""))
    return out, lo, hi


def _time_pinned(ev, lo0, hi0):
    lo, hi = lo0, hi0
    for t, side, v, _ in sorted(ev):
        lo, hi = (max(lo, v), hi) if side == "lo" else (lo, min(hi, v))
        if hi <= lo:
            return t
    return None


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    by = {(r["graph"], r["kind"]): r for r in rows}
    graphs = [f"{f}_n{n}_{r}" for f, n, r in SPECS]
    table = []
    for gname in graphs:
        pf, v2 = by.get((gname, "portfolio")), by.get((gname, "v2"))
        if not pf or "error" in pf or not v2 or "error" in v2:
            table.append((gname, None))
            continue
        r0 = pf["runs"][0]
        lo0, hi0 = r0["lo0"], r0["hi0"]
        gap0 = max(1, hi0 - lo0)
        ev_pf, lo_pf, hi_pf = _combine(pf["runs"])
        row = {"family": pf["family"], "n": pf["n"],
               "v1": (gap_integral(r0["events"], T, lo0, hi0, gap0), r0["lo"], r0["hi"],
                      _time_pinned(r0["events"], lo0, hi0), r0["verified"]),
               "pf": (gap_integral(ev_pf, T, lo0, hi0, gap0), lo_pf, hi_pf, _time_pinned(ev_pf, lo0, hi0),
                      all(r["verified"] for r in pf["runs"])),
               "v2": (gap_integral([tuple(e) for e in v2["events"]], T, lo0, hi0, gap0), v2["lo"], v2["hi"],
                      _time_pinned([tuple(e) for e in v2["events"]], lo0, hi0), v2["verified"]),
               "v2info": v2}
        table.append((gname, row))
    ok = [(gn, r) for gn, r in table if r]
    lines = ["# Gap agent v2 (parallel: search + structure/exact + lambda_2 block codes)", "",
             f"{len(ok)} unseen graphs (3000-10000 receivers), 30 s each. Gap integral = area under gap(t)/gap(0), "
             "same reference for all three set-ups (lower = closed sooner). 'pinned' = lower bound = upper bound, "
             "i.e. beta = lambda_2 proven. All certificates re-checked.", "",
             "| family | set-up | mean gap integral | final gap (sum) | pinned | median time to pin | verified |",
             "|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        rs = [r for _, r in ok if r["family"] == fam]
        if not rs:
            continue
        for key, name in (("v1", "v1, 1 core"), ("pf", "v1 portfolio, 3 cores"), ("v2", "**v2, 3 workers**")):
            pins = [r[key][3] for r in rs if r[key][3] is not None]
            lines.append(f"| {fam} | {name} | {np.mean([r[key][0] for r in rs]):.3f} | "
                         f"{sum(r[key][2] - r[key][1] for r in rs):.1f} | {len(pins)}/{len(rs)} | "
                         f"{(f'{np.median(pins):.1f} s' if pins else '-')} | "
                         f"{sum(bool(r[key][4]) for r in rs)}/{len(rs)} |")
    lines += ["", "## What v2's structure and lambda workers contributed", "",
              "| graph | SCCs | SCCs closed (receivers) | acyclic kernel: receivers before -> after | block-code merges |"
              " pieces through lambda-agent | SCCs whose upper bound beats the clique cover | v1 final | v2 final |",
              "|---|---|---|---|---|---|---|---|---|"]
    for gname, r in ok:
        i = r["v2info"]
        kt = i.get("kernel_total", [0, 0])
        better = i["upper_from"]["block codes"] + i["upper_from"]["lambda-agent codes"]
        lines.append(f"| {gname} | {i['sccs']} | {i['sccs_closed']} ({i['receivers_in_closed']}) | {kt[0]} -> {kt[1]} | "
                     f"{i['lambda_merges']} | {i['lambda_agent_pieces']} | {better} | "
                     f"{r['v1'][1]:g} .. {r['v1'][2]:g} | {i['lo_exact']} .. {i['hi_exact']} |")
    missing = [gn for gn, r in table if not r]
    if missing:
        lines += ["", "Missing or failed runs: " + ", ".join(missing)]
    (HERE / "RESULTS2.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(sys.argv[2], int(sys.argv[3])), default=str))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
