"""Test the learned agent on unseen large graphs.

    python gap-agent/bench.py run        all controllers on the test graphs -> results/bench.jsonl
    python gap-agent/bench.py cpsat      the standard exact solver (CP-SAT) on the smallest test graphs
    python gap-agent/bench.py report     -> RESULTS.md

Test graphs: 3000, 6000 and 10000 receivers, three families, three graphs each (27 graphs; none used in training,
training never saw more than 8000 receivers). Every controller gets the same 30 s, the same opening (one greedy acyclic set,
one DSatur cover) and the same random seed. One core each. Every final bound is re-checked: the acyclic set by a
topological sort, the cover clique by clique.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gap_agent.agent import Bandit, LearnedAgent, RandomSwitch, RoundRobin, gap_integral, run  # noqa: E402
from gap_agent.graph import FAMILIES, is_acyclic_set, is_clique_cover, make_graph  # noqa: E402
from gap_agent.strategies import STRATEGIES, compile_all  # noqa: E402

T_TEST = 30.0
SIZES = (3000, 6000, 10000)
OUT = HERE / "results"


class FixedPair:
    """Reference: the two strongest single strategies, alternated (annealing for the set, tabu for the cover)."""
    name = "fixed pair (annealing + tabu)"

    def choose(self, obs, st, s):
        return STRATEGIES.index("hi_tabu") if st.last_action == STRATEGIES.index("lo_descent") \
            else STRATEGIES.index("lo_descent")


SPECS = [(n, f, r) for n in SIZES for f in FAMILIES for r in range(3)]


def test_graph(gi):
    n, f, r = SPECS[gi]
    g = make_graph(f, n, np.random.default_rng(900_000 + 1000 * n + 10 * FAMILIES.index(f) + r))
    g.name = f"{f}_n{n}_{r}"
    return g


def controllers():
    model = pickle.loads((HERE / "models" / "agent.pkl").read_bytes())
    return {"learned": lambda: LearnedAgent(model), "bandit": Bandit, "random": lambda: RandomSwitch(0.0, 1),
            "roundrobin": RoundRobin, "fixedpair": FixedPair}


def _init():
    compile_all()


def _job(args):
    gi, key = args
    g = test_graph(gi)
    res = run(g, controllers()[key](), T=T_TEST, seed=gi)
    res["family"] = g.name.split("_")[0]
    res["key"] = key
    res["verified"] = bool(is_acyclic_set(g, res["acyclic_set"]) and len(res["acyclic_set"]) == res["lo"]
                           and is_clique_cover(g, np.array(res["cover"])) and len(set(res["cover"])) <= res["hi"])
    del res["acyclic_set"], res["cover"]
    return res


def cmd_run(workers=3):
    OUT.mkdir(exist_ok=True)
    jobs = [(gi, k) for gi in range(len(SPECS)) for k in controllers()]
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool, open(OUT / "bench.jsonl", "w") as f:
        for res in pool.imap_unordered(_job, jobs):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['graph']:22s} {res['key']:10s} gap {res['gap0']}->{res['gap']} integral "
                  f"{res['integral']:.3f} ok {res['verified']}", flush=True)


# ------------------------------------------------------------------------------- standard exact solver (CP-SAT)
def cpsat(g, T, lo0, hi0):
    """CP-SAT on the textbook models, half the time each: largest acyclic set (ordering model), smallest clique
    cover (colouring of the conflict graph with hi0 colours, DSatur as a hint). Model building counts as time."""
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    events = []
    n = g.n
    for side in ("lo", "hi"):
        end = t0 + (T / 2 if side == "lo" else T)
        m = cp_model.CpModel()
        if side == "lo":
            y = [m.NewBoolVar("") for _ in range(n)]
            pos = [m.NewIntVar(0, n - 1, "") for _ in range(n)]
            for u in range(n):
                for e in range(g.succ_ptr[u], g.succ_ptr[u + 1]):
                    v = int(g.succ[e])
                    if g.adj[v, u]:
                        if u < v:
                            m.AddBoolOr([y[u].Not(), y[v].Not()])
                    else:
                        m.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
                if time.perf_counter() > end:
                    break
            m.Maximize(sum(y))
            val = lambda cb: sum(cb.Value(v) for v in y)  # noqa: E731
        else:
            k = hi0
            x = [[m.NewBoolVar("") for _ in range(k)] for _ in range(n)]
            used = [m.NewBoolVar("") for _ in range(k)]
            for v in range(n):
                m.AddExactlyOne(x[v])
                for c in range(k):
                    m.AddImplication(x[v][c], used[c])
            for u in range(n):
                for e in range(g.conf_ptr[u], g.conf_ptr[u + 1]):
                    v = int(g.conf[e])
                    if u < v:
                        for c in range(k):
                            m.AddBoolOr([x[u][c].Not(), x[v][c].Not()])
                if time.perf_counter() > end:
                    break
            m.Minimize(sum(used))
            val = lambda cb: sum(cb.Value(v) for v in used)  # noqa: E731
        left = end - time.perf_counter()
        if left <= 0.5:
            events.append((time.perf_counter() - t0, side, None, "model not built in time"))
            continue

        class CB(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                events.append((time.perf_counter() - t0, side, int(val(self)), "cpsat"))

        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = left
        s.parameters.num_workers = 1
        s.Solve(m, CB())
    ev = [e for e in events if e[2] is not None]
    lo = max([lo0] + [v for _, sd, v, _ in ev if sd == "lo"])
    hi = min([hi0] + [v for _, sd, v, _ in ev if sd == "hi"])
    gap0 = max(1, hi0 - lo0)
    return {"lo": lo, "hi": hi, "gap0": gap0, "gap": hi - lo, "integral": gap_integral(ev, T, lo0, hi0, gap0),
            "notes": sorted({e[3] for e in events if e[2] is None})}


def cmd_cpsat():
    from gap_agent.strategies import Search, warm_start

    rows = []
    for gi in range(len(SPECS)):
        if SPECS[gi][0] != SIZES[0]:
            continue
        g = test_graph(gi)
        s = Search(g, gi)
        warm_start(s)
        r = cpsat(g, T_TEST, s.lo, s.hi)
        r.update(graph=g.name, n=g.n, key="cpsat", family=g.name.split("_")[0])
        rows.append(r)
        print(r, flush=True)
    (OUT / "cpsat.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def cmd_report():
    rows = [json.loads(x) for x in (OUT / "bench.jsonl").read_text().splitlines()]
    names = {"learned": "**learned agent**", "bandit": "online bandit (UCB)", "random": "random switching",
             "roundrobin": "round robin", "fixedpair": "fixed pair (annealing + tabu)"}
    keys = list(names)
    by = {(r["graph"], r["key"]): r for r in rows}
    graphs = sorted({r["graph"] for r in rows}, key=lambda x: (int(x.split("_n")[1].split("_")[0]), x))
    lines = ["# Gap agent: results on unseen large graphs", "",
             f"{len(graphs)} test graphs, 30 s each, one core, same opening and seed for every controller. "
             "Gap integral = area under gap(t)/gap(0) over the 30 s (lower is better: the gap closed sooner). "
             "All final bounds re-checked.", ""]
    lines += ["| controller | mean gap integral | final gap (sum) | wins (best integral) | lower bound (sum) | "
              "upper bound (sum) | decision time |", "|---|---|---|---|---|---|---|"]
    best = {gname: min(by[(gname, k)]["integral"] for k in keys if (gname, k) in by) for gname in graphs}
    for k in keys:
        rs = [by[(gname, k)] for gname in graphs if (gname, k) in by]
        if not rs:
            continue
        wins = sum(1 for r in rs if r["integral"] <= best[r["graph"]] + 1e-12)
        lines.append(f"| {names[k]} | {np.mean([r['integral'] for r in rs]):.3f} | {sum(r['gap'] for r in rs)} | "
                     f"{wins}/{len(rs)} | {sum(r['lo'] for r in rs)} | {sum(r['hi'] for r in rs)} | "
                     f"{np.mean([r['decide_seconds'] for r in rs]):.2f} s |")
    lines += ["", "## By size and family (mean gap integral)", "",
              "| graphs | " + " | ".join(names[k].strip("*") for k in keys) + " |",
              "|---|" + "---|" * len(keys)]
    for n in SIZES:
        for f in FAMILIES:
            gs = [x for x in graphs if x.startswith(f"{f}_n{n}_")]
            cells = [f"{np.mean([by[(x, k)]['integral'] for x in gs if (x, k) in by]):.3f}" for k in keys]
            lines.append(f"| {f}, n = {n} | " + " | ".join(cells) + " |")
    lines += ["", "## How the learned agent spends its time (share of slices)", "",
              "| graphs | " + " | ".join(STRATEGIES) + " |", "|---|" + "---|" * len(STRATEGIES)]
    for f in FAMILIES:
        rs = [by[(x, "learned")] for x in graphs if x.startswith(f + "_") and (x, "learned") in by]
        if rs:
            lines.append(f"| {f} | " + " | ".join(f"{np.mean([r['share'][s] for r in rs]):.2f}" for s in STRATEGIES)
                         + " |")
    cp = OUT / "cpsat.jsonl"
    if cp.exists():
        cr = [json.loads(x) for x in cp.read_text().splitlines() if x.strip()]
        lines += ["", "## Standard exact solver (CP-SAT, textbook models) on the smallest test graphs", "",
                  "| graph | CP-SAT gap integral | CP-SAT final gap | learned agent gap integral | learned final gap |"
                  " notes |", "|---|---|---|---|---|---|"]
        for r in cr:
            la = by.get((r["graph"], "learned"), {})
            lines.append(f"| {r['graph']} | {r['integral']:.3f} | {r['gap']} | {la.get('integral', float('nan')):.3f}"
                         f" | {la.get('gap', '')} | {', '.join(r['notes'])} |")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    {"run": cmd_run, "cpsat": cmd_cpsat, "report": cmd_report}[sys.argv[1]]()
