"""Standard-solver baseline: Google OR-Tools CP-SAT solves both bounds exactly with textbook models.

    python beta-agent/bench/cpsat.py [--limit 30]        -> bench/results_cpsat.jsonl

Models (the usual ones in the literature):
  lower bound  maximum acyclic induced subgraph: keep y_v; for a two-way pair not both kept; for a one-way arc u -> v
               with both kept, position(u) < position(v). Maximise the number kept.
  upper bound  minimum clique cover = colouring of the conflict graph: x[v][c] one colour per receiver, conflicting
               receivers (not both arcs) never share a colour, minimise colours used (used[c] >= used[c+1]).
               Strengthened as is standard practice: at most as many colours as a DSatur colouring needs, and a
               greedy clique of the conflict graph fixed to distinct colours (symmetry breaking). The plain model
               with n symmetric colours could not even prove n = 15 covers in 15 s.
Single worker (one CPU core, like the JavaScript methods). The per-graph limit is split evenly between the two models.
"""
import argparse
import json
import time
from pathlib import Path

from ortools.sat.python import cp_model

HERE = Path(__file__).resolve().parent


def mais(n, adj, limit):
    m = cp_model.CpModel()
    y = [m.NewBoolVar(f"y{v}") for v in range(n)]
    pos = [m.NewIntVar(0, max(0, n - 1), f"p{v}") for v in range(n)]
    for u in range(n):
        for v in range(n):
            if u < v and adj[u][v] and adj[v][u]:
                m.AddBoolOr([y[u].Not(), y[v].Not()])
            elif adj[u][v] and not adj[v][u]:
                m.Add(pos[u] < pos[v]).OnlyEnforceIf([y[u], y[v]])
    m.Maximize(sum(y))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return 0, False
    return int(round(s.ObjectiveValue())), st == cp_model.OPTIMAL


def dsatur(n, conf):
    color = [-1] * n
    for _ in range(n):
        best, key = -1, None
        for v in range(n):
            if color[v] >= 0:
                continue
            sat = len({color[u] for u in range(n) if conf[v][u] and color[u] >= 0})
            k = (sat, sum(conf[v]))
            if key is None or k > key:
                best, key = v, k
        used = {color[u] for u in range(n) if conf[best][u] and color[u] >= 0}
        c = 0
        while c in used:
            c += 1
        color[best] = c
    return max(color) + 1


def greedy_clique(n, conf):
    order = sorted(range(n), key=lambda v: -sum(conf[v]))
    best = []
    for s in order[: min(n, 10)]:
        cl = [s]
        for v in order:
            if v != s and all(conf[v][u] for u in cl):
                cl.append(v)
        if len(cl) > len(best):
            best = cl
    return best


def cover(n, adj, limit):
    m = cp_model.CpModel()
    conf = [[u != v and not (adj[u][v] and adj[v][u]) for v in range(n)] for u in range(n)]
    k = dsatur(n, conf)
    x = [[m.NewBoolVar(f"x{v}_{c}") for c in range(k)] for v in range(n)]
    used = [m.NewBoolVar(f"u{c}") for c in range(k)]
    for v in range(n):
        m.AddExactlyOne(x[v])
        for c in range(k):
            m.AddImplication(x[v][c], used[c])
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u][v] and adj[v][u]):
                for c in range(k):
                    m.AddBoolOr([x[u][c].Not(), x[v][c].Not()])
    for c in range(k - 1):
        m.AddImplication(used[c + 1], used[c])
    for c, v in enumerate(greedy_clique(n, conf)):  # pairwise-conflicting receivers: fix their colours
        m.Add(x[v][c] == 1)
    m.Minimize(sum(used))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return n, False
    return int(round(s.ObjectiveValue())), st == cp_model.OPTIMAL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=float, default=30)
    a = ap.parse_args()
    suite = json.loads((HERE / "suite.json").read_text())
    out = HERE / "results_cpsat.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["graph"] for line in out.read_text().splitlines() if line.strip()}
    for item in suite:
        if item["name"] in done:
            continue
        n = item["n"]
        adj = [[False] * n for _ in range(n)]
        for i, j in item["arcs"]:
            adj[i][j] = True
        t0 = time.perf_counter()
        L, lp = mais(n, adj, a.limit / 2)
        U, up = cover(n, adj, a.limit / 2)
        row = {"graph": item["name"], "family": item["family"], "n": n, "arcs": len(item["arcs"]), "method": "cpsat",
               "L": L, "U": U, "lower_optimal_proven": lp or L == U, "upper_optimal_proven": up or L == U,
               "beta_exact": L == U, "seconds": round(time.perf_counter() - t0, 3)}
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"cpsat {item['name']:24s} [{L}, {U}] {row['seconds']:.2f} s", flush=True)


if __name__ == "__main__":
    main()
