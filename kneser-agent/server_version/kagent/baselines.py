"""Baselines ("standard methods", section 6). Each method runs alone in a fresh process, pinned to the same number of
cores as the agent, with LIMIT = 120 s; the driver kills it (by PID) at LIMIT + GRACE. Anytime methods record the
time at which their best bound was reached.

  indep_cpsat        CP-SAT maximum independent set                                  lower bound
  clique_cover_best  best of DSatur + TabuCol (gap-agent kernels) and CP-SAT colouring of the complement   upper
  frac_cover         fractional clique cover by column generation                    upper bound on beta
  sat_minrank_f2     Pincer's F_2 minrank SAT with the independent-set fix (rank_at_most_sym, "indep_cols"),
                     r = |I|, |I|+1, ... until SAT                                    upper bound (and minrk_2 lower)
  lambda_search      lambda-agent vector-linear search on G^(2) (n * t <= 150 only) upper bound
  pincer             Pincer, index policy, budget 120 s (Pincer(...).run(), as python -m pincer.agent does)
  minrank_agent      the scripted multi-field agent (minrank-agent/kneser_bench.py: agent())

"standard best" = best lower and best upper over the first five, time = when the last of those two was reached.
Where a method cannot start (memory, size) the record is status "n/a (size)".

    python -m kagent.baselines all --names "K(21,2)" ... --cores 40-47 [--parallel 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]   # server_version -> kneser-agent -> repository root
LIMIT = 120.0
GRACE = 30.0
METHODS = ["indep_cpsat", "clique_cover_best", "frac_cover", "sat_minrank_f2", "lambda_search", "pincer",
           "minrank_agent"]
STANDARD = METHODS[:5]
CPSAT_MAX_CLAUSES = 3_000_000
SAT_MAX_UNITS = 1_000_000          # Pincer's own limit (pincer/tools.py)


def _pin(cores):
    os.sched_setaffinity(0, cores)
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(len(cores))
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ methods
def m_indep_cpsat(A, limit, cores, t0):
    from ortools.sat.python import cp_model

    n = A.shape[0]
    m_edges = int(A.sum() // 2)
    if m_edges > CPSAT_MAX_CLAUSES:
        return {"status": "n/a (size)", "why": f"{m_edges} edge clauses > {CPSAT_MAX_CLAUSES}"}
    m = cp_model.CpModel()
    x = [m.NewBoolVar("") for _ in range(n)]
    iu, ju = np.nonzero(np.triu(A, 1))
    for i, j in zip(iu.tolist(), ju.tolist()):
        m.AddBoolOr([x[i].Not(), x[j].Not()])
    m.Maximize(sum(x))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(1.0, limit - (time.perf_counter() - t0))
    s.parameters.num_workers = len(cores)
    hist = []

    class CB(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            hist.append((round(time.perf_counter() - t0, 2), int(round(self.ObjectiveValue()))))

    st = s.Solve(m, CB())
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": "no solution", "time": round(time.perf_counter() - t0, 2)}
    S = [i for i in range(n) if s.Value(x[i])]
    assert not A[np.ix_(S, S)].any()
    best_t = next(t for t, v in hist if v == len(S)) if hist else round(time.perf_counter() - t0, 2)
    return {"status": "ok", "lower": len(S), "time": best_t, "alpha_proven": st == cp_model.OPTIMAL,
            "alpha_upper": int(np.floor(s.BestObjectiveBound() + 1e-6)), "history": hist[-10:]}


def m_clique_cover_best(A, limit, cores, t0):
    sys.path.insert(0, str(ROOT))
    from kagent.tools.covers import clique_cover, clique_cover_cpsat

    hist = []
    best = [A.shape[0] + 1]

    def rec(cl):
        if len(cl) < best[0]:
            best[0] = len(cl)
            hist.append((round(time.perf_counter() - t0, 2), len(cl), "heuristic"))

    cl = clique_cover(A, limit / 2, on_improve=rec)
    comp_edges = int((~A).sum() - A.shape[0]) // 2
    used_cpsat = False
    if comp_edges <= CPSAT_MAX_CLAUSES:
        used_cpsat = True

        def rec2(c2):
            if len(c2) < best[0]:
                best[0] = len(c2)
                hist.append((round(time.perf_counter() - t0, 2), len(c2), "cpsat"))

        cl2, opt = clique_cover_cpsat(A, max(1.0, limit - (time.perf_counter() - t0) - 2), workers=len(cores),
                                      hint=cl, on_improve=rec2)
    else:
        opt = False
    t_best = hist[-1][0] if hist else None
    return {"status": "ok", "upper": best[0], "time": t_best, "proven_optimal_cover": bool(opt),
            "cpsat_used": used_cpsat, "history": hist[-10:]}


def m_frac_cover(A, limit, cores, t0):
    sys.path.insert(0, str(ROOT))
    from kagent.tools.covers import clique_cover, frac_cover

    init = clique_cover(A, min(10.0, limit / 6))
    hist = []

    def rec(cl, ws, val):
        hist.append((round(time.perf_counter() - t0, 2), str(val)))

    cl, ws, val, proven = frac_cover(A, max(1.0, limit - (time.perf_counter() - t0)), init_cliques=init,
                                     on_improve=rec)
    best_t = next(t for t, v in hist if Fraction(v) == val) if hist else None
    return {"status": "ok", "upper": str(val), "upper_float": float(val), "time": best_t, "lp_optimal": proven,
            "history": hist[-10:]}


def m_sat_minrank_f2(A, limit, cores, t0):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ROOT))
    from kagent.tools.indep import indep_ils
    from minrank.exact.symmetry import rank_at_most_sym
    from minrank.core.graph import Graph

    n = A.shape[0]
    I = indep_ils(A, min(5.0, limit / 10))
    pairs = int((~A).sum() - n) + n
    if pairs * len(I) > SAT_MAX_UNITS:
        return {"status": "n/a (size)", "why": f"CNF units {pairs * len(I)} > {SAT_MAX_UNITS} (Pincer's limit)"}
    g = Graph(n, tuple(int.from_bytes(np.packbits(A[i][::-1]).tobytes(), "big") >> ((-n) % 8) for i in range(n)),
              False)
    r = len(I)
    lower = None
    while r <= n:
        left = limit - (time.perf_counter() - t0)
        if left <= 0.5:
            return {"status": "timeout", "minrk2_lower": lower, "time": round(time.perf_counter() - t0, 2)}
        res = rank_at_most_sym(g, r, time_limit=left, mode="indep_cols", indep=sorted(I)[:r])
        if res is None:
            return {"status": "timeout", "minrk2_lower": lower, "asked_r": r,
                    "time": round(time.perf_counter() - t0, 2)}
        if res is False:
            lower = r + 1
            r += 1
            continue
        return {"status": "ok", "upper": r, "minrk2_lower": lower if lower is not None else len(I),
                "exact_minrk2": True, "time": round(time.perf_counter() - t0, 2)}
    return {"status": "ok", "upper": n, "time": round(time.perf_counter() - t0, 2)}


def m_lambda_search(A, limit, cores, t0):
    n = A.shape[0]
    if 2 * n > 150:
        return {"status": "n/a (size)", "why": f"n * t = {2 * n} > 150 for t = 2"}
    sys.path.insert(0, str(REPO / "lambda-agent"))
    from lambda_agent.graph import Digraph, blowup
    from lambda_agent.upper import minrank_sat

    sys.path.insert(0, str(ROOT))
    from kagent.tools.indep import indep_ils

    I = indep_ils(A, 2.0)
    i, j = np.nonzero(A)
    g = Digraph.from_arcs(n, list(zip(i.tolist(), j.tolist())), "g")
    h = blowup(g, 2)
    r = 2 * len(I)
    while time.perf_counter() - t0 < limit:
        res = minrank_sat(h, r, limit - (time.perf_counter() - t0))
        if res is None:
            return {"status": "timeout", "time": round(time.perf_counter() - t0, 2)}
        if res is False:
            r += 1
            continue
        return {"status": "ok", "upper": str(Fraction(r, 2)), "upper_float": r / 2,
                "time": round(time.perf_counter() - t0, 2)}
    return {"status": "timeout"}


def m_pincer(A, limit, cores, t0):
    sys.path.insert(0, str(REPO))
    from minrank.core.graph import Graph
    from pincer.agent import Pincer
    from pincer.policies import make_policy
    from pincer.tools import default_tools

    n = A.shape[0]
    g = Graph(n, tuple(int.from_bytes(np.packbits(A[i][::-1]).tobytes(), "big") >> ((-n) % 8) for i in range(n)),
              False, name="g")
    t_build = time.perf_counter() - t0
    res = Pincer(g, limit, make_policy("index", None), 0, default_tools()).run()
    s = res.summary()
    lo, hi = Fraction(s["beta"][0]), Fraction(s["beta"][1])
    return {"status": "ok", "lower": str(lo), "upper": str(hi), "lower_float": float(lo), "upper_float": float(hi),
            "mr": s["mr"], "time": round(time.perf_counter() - t0, 2), "graph_build_s": round(t_build, 2),
            "verified": True, "steps": s["steps"], "stopped": s["stopped"]}


def m_minrank_agent(A, limit, cores, t0):
    sys.path.insert(0, str(REPO / "minrank-agent"))
    sys.path.insert(0, str(REPO / "lambda-agent"))
    import kneser_bench as KB

    r = KB.agent(A.astype(np.int64), limit)
    return {"status": "ok", "lower": r["lower"], "upper": float(r["upper"]), "field": r.get("field"),
            "exact": bool(r["exact"]), "time": round(r["s"], 2), "fields_tried": r.get("fields_tried")}


FUNCS = {k: globals()[f"m_{k}"] for k in METHODS}


# ------------------------------------------------------------------------------------------------ drivers
def one(method: str, gid: str, cores: list[int], limit: float) -> dict:
    _pin(cores)
    sys.path.insert(0, str(ROOT))
    from kagent.graphs import load_graph

    A = load_graph(ROOT / "bench" / "graphs" / gid)
    t0 = time.perf_counter()
    out = FUNCS[method](A, limit, cores, t0)
    out["wall"] = round(time.perf_counter() - t0, 2)
    return out


def run_method(method: str, gid: str, cores: list[int], limit: float = LIMIT) -> dict:
    """Fresh process, killed by PID at limit + GRACE."""
    cmd = [sys.executable, "-m", "kagent.baselines", "one", method, gid, "--cores",
           ",".join(map(str, cores)), "--limit", str(limit)]
    env = dict(os.environ, OMP_NUM_THREADS=str(len(cores)), OPENBLAS_NUM_THREADS=str(len(cores)),
               PYTHONPATH=str(ROOT))
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                         start_new_session=True)
    try:
        so, se = p.communicate(timeout=limit + GRACE)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, 9)
        p.communicate()
        return {"status": "timeout (killed, no result)", "wall": round(time.perf_counter() - t0, 2)}
    lines = [l for l in so.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        err = se.strip().splitlines()[-3:] if se else []
        mem = any("MemoryError" in l or "Killed" in l for l in err)
        return {"status": "n/a (memory)" if mem else "error", "error": err, "rc": p.returncode,
                "wall": round(time.perf_counter() - t0, 2)}
    return json.loads(lines[-1])


def standard_best(res: dict) -> dict:
    lows = [(Fraction(str(r["lower"])), r.get("time") or r.get("wall"), m) for m, r in res.items()
            if m in STANDARD and r.get("status") == "ok" and r.get("lower") is not None]
    ups = [(Fraction(str(r["upper"])), r.get("time") or r.get("wall"), m) for m, r in res.items()
           if m in STANDARD and r.get("status") == "ok" and r.get("upper") is not None]
    if not lows or not ups:
        L = max(lows)[0] if lows else None
        U = min(ups)[0] if ups else None
        return {"lower": str(L) if L is not None else None, "upper": str(U) if U is not None else None,
                "exact": False, "time": None}
    L = max(v for v, _, _ in lows)
    U = min(v for v, _, _ in ups)
    tL = min(t for v, t, _ in lows if v == L)
    tU = min(t for v, t, _ in ups if v == U)
    return {"lower": str(L), "upper": str(U), "lower_by": next(m for v, t, m in lows if v == L and t == tL),
            "upper_by": next(m for v, t, m in ups if v == U and t == tU), "exact": L >= U, "time": max(tL, tU),
            "lower_float": float(L), "upper_float": float(U)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["one", "all"])
    ap.add_argument("method", nargs="?")
    ap.add_argument("gid", nargs="?")
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--group", default=None)
    ap.add_argument("--max-receivers", type=int, default=None)
    ap.add_argument("--min-receivers", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    ap.add_argument("--cores", default="40-47")
    ap.add_argument("--limit", type=float, default=LIMIT)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    from .run import parse_cores

    cores = parse_cores(a.cores)
    if a.cmd == "one":
        print(json.dumps(one(a.method, a.gid, cores, a.limit), default=str), flush=True)
        return 0
    from .bench import select

    out_dir = ROOT / "results" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    for gid, t in select(a.names, a.group, a.max_receivers, a.min_receivers, 0):
        f = out_dir / f"{gid}.json"
        res = json.loads(f.read_text()) if f.exists() else {}
        for m in a.methods:
            if a.skip_done and m in res and not str(res[m].get("status", "")).startswith("error"):
                continue
            res[m] = run_method(m, gid, cores, a.limit)
            res["standard_best"] = standard_best(res)
            f.write_text(json.dumps(res, indent=1, default=str))
            print(json.dumps({"id": gid, "name": t["name"], "method": m,
                              **{k: res[m].get(k) for k in ("status", "lower", "upper", "time", "wall")}}),
                  flush=True)
        print(json.dumps({"id": gid, "name": t["name"], "standard_best": res["standard_best"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
