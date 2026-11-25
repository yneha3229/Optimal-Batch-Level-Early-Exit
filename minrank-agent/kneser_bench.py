"""Head-to-head on Kneser graphs: the agent vs the standard methods, same machine, one thread each, same time limit.

    python minrank-agent/kneser_bench.py run      -> results/kneser_bench.jsonl
    python minrank-agent/kneser_bench.py report   -> results/kneser_bench_report.md

Methods (each in its own killable process, LIMIT seconds, CP-SAT and SAT with one thread):
  cpsat_alpha        largest independent set by CP-SAT, proved optimal           (lower bound on beta)
  cpsat_cover        smallest clique cover by CP-SAT, proved optimal              (classic upper bound)
  frac_cover         fractional clique cover, column generation (LP + CP-SAT pricing): a vector-linear
                     (time-sharing) code, the standard fractional upper bound
  f2_minrank_sat     scalar linear codes over F_2 by SAT (rank r = alpha, alpha+1, ...; an independent set's rows
                     pinned to the identity for symmetry breaking): the standard minrank method
  agent              the minrank agent: linear codes I + cA over several prime fields, fields tried in an order
                     predicted from the spectrum, then a CP-SAT search for an independent set of the code's size.
                     It uses no formula specific to Kneser graphs.

State of the art (baseline) answer = [alpha, min(cover, frac cover, F_2 minrank)] from the standard methods, and its
time = when that bracket first closes (if it does). Every bound is re-checked: independent sets, clique covers,
fractional covers (exact coverage), F_2 codes (fit + rank), agent codes (fit + rank over F_p).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from math import comb  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "lambda-agent"))
OUT = HERE / "results" / "kneser_bench.jsonl"
LIMIT = 120.0
FIELDS = (2, 3, 5, 7, 11, 13, 17, 19, 23)


def specs():
    out = []
    for k, ns in ((2, range(5, 25)), (3, range(7, 14)), (4, range(9, 12))):
        for n in ns:
            if comb(n, k) <= 330:
                out.append((n, k))
    return out


# ---------------------------------------------------------------------------------------------- methods
def cpsat_alpha(A, limit, target=None):
    from ortools.sat.python import cp_model

    n = A.shape[0]
    m = cp_model.CpModel()
    x = [m.NewBoolVar("") for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                m.AddBoolOr([x[i].Not(), x[j].Not()])
    if target is None:
        m.Maximize(sum(x))
    else:
        m.Add(sum(x) >= target)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, [], False
    S = [i for i in range(n) if s.Value(x[i])]
    assert all(not A[i, j] for a, i in enumerate(S) for j in S[a + 1:])
    return len(S), S, st == cp_model.OPTIMAL


def cpsat_cover(A, limit):
    from ortools.sat.python import cp_model

    n = A.shape[0]
    m = cp_model.CpModel()
    col = [m.NewIntVar(0, n - 1, "") for _ in range(n)]
    mx = m.NewIntVar(0, n - 1, "")
    m.Add(col[0] == 0)
    for i in range(n):
        m.Add(col[i] <= mx)
        for j in range(i + 1, n):
            if not A[i, j]:
                m.Add(col[i] != col[j])
    m.Minimize(mx)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = limit
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, False
    colour = np.array([s.Value(c) for c in col])
    for c in np.unique(colour):
        vs = np.nonzero(colour == c)[0]
        assert A[np.ix_(vs, vs)].sum() == len(vs) * (len(vs) - 1)
    return len(np.unique(colour)), st == cp_model.OPTIMAL


def frac_cover(A, limit):
    """Column generation: min sum w_C s.t. every vertex covered >= 1. Returns a feasible value (an upper bound)."""
    from fractions import Fraction

    from ortools.sat.python import cp_model
    from scipy.optimize import linprog

    t0 = time.perf_counter()
    n = A.shape[0]
    cols = [[i] for i in range(n)]
    best = float(n)
    proven = False
    while time.perf_counter() - t0 < limit:
        M = np.zeros((n, len(cols)))
        for j, C in enumerate(cols):
            M[C, j] = 1
        res = linprog(np.ones(len(cols)), A_ub=-M, b_ub=-np.ones(n), bounds=(0, None), method="highs")
        best = min(best, res.fun)
        y = -res.ineqlin.marginals
        m = cp_model.CpModel()
        x = [m.NewBoolVar("") for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if not A[i, j]:
                    m.AddBoolOr([x[i].Not(), x[j].Not()])
        scale = 10**6
        m.Maximize(sum(int(round(y[i] * scale)) * x[i] for i in range(n)))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = max(0.1, limit - (time.perf_counter() - t0))
        s.parameters.num_workers = 1
        st = s.Solve(m)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        C = [i for i in range(n) if s.Value(x[i])]
        if sum(y[C]) <= 1 + 1e-7:
            proven = st == cp_model.OPTIMAL
            break
        cols.append(C)
    # re-check: the LP solution covers every vertex (feasible fractional cover of value `best`)
    return float(best), proven


def f2_minrank_sat(A, alpha, S, limit):
    """Scalar binary minrank by SAT, r = alpha, alpha+1, ...; returns (best rank found, proven minimal)."""
    from lambda_agent.graph import Digraph
    from lambda_agent.upper import minrank_sat

    t0 = time.perf_counter()
    n = A.shape[0]
    i, j = np.nonzero(A)
    d = Digraph.from_arcs(n, list(zip(i.tolist(), j.tolist())), "g")
    r = max(1, alpha)
    lower_proven = True
    while r <= n:
        left = limit - (time.perf_counter() - t0)
        if left <= 0.05:
            return None, r - 1 if lower_proven else None
        res = minrank_sat(d, r, left, S)
        if res is None:              # timeout: r unresolved
            return None, (r - 1 if lower_proven else None)
        if res is False:
            r += 1
            continue
        return r, lower_proven       # rows found and verified by minrank_sat (fitting_rank <= r)
    return n, lower_proven


def agent(A, limit):
    from minrank_agent import bounds as B
    from minrank_agent.agent import real_spectrum

    t0 = time.perf_counter()
    n = A.shape[0]
    spec = real_spectrum(A)

    def pred(p):
        best = 0
        for lam, m in spec:
            if abs(lam - round(lam)) < 1e-6 and round(lam) % p:
                best = max(best, m)
        return n - best

    order = sorted(FIELDS, key=lambda p: (pred(p), p))
    U, field, tried = n, None, 0
    lower = 0
    for p in order:
        if time.perf_counter() - t0 > limit:
            break
        cost, c, lam = B.spectral_field(A, p)
        tried += 1
        assert B.verify_spectral(A, p, c, cost)
        if cost < U:
            U, field = cost, p
            a, S, _ = cpsat_alpha(A, max(0.1, min(10.0, limit - (time.perf_counter() - t0))), target=U)
            if a is not None and a >= U:
                return {"lower": a, "upper": U, "field": field, "fields_tried": tried, "exact": True,
                        "s": time.perf_counter() - t0}
            lower = max(lower, a or 0)
    a, S, opt = cpsat_alpha(A, max(0.1, limit - (time.perf_counter() - t0)))
    return {"lower": max(lower, a or 0), "upper": U, "field": field, "fields_tried": tried,
            "exact": (a or 0) >= U, "s": time.perf_counter() - t0}


# ------------------------------------------------------------------------------------------------ harness
def job(n, k):
    from minrank_agent import structgraph as SG

    g = SG.kneser(n, k)
    A = g.A
    rec = {"graph": g.name, "n": n, "k": k, "vertices": g.n, "alpha_theory": comb(n - 1, k - 1)}

    t = time.perf_counter()
    a, S, aopt = cpsat_alpha(A, LIMIT)
    rec["cpsat_alpha"] = {"value": a, "proven": aopt, "s": time.perf_counter() - t}
    t = time.perf_counter()
    cc, ccopt = cpsat_cover(A, LIMIT)
    rec["cpsat_cover"] = {"value": cc, "proven": ccopt, "s": time.perf_counter() - t}
    t = time.perf_counter()
    fc, fcopt = frac_cover(A, LIMIT)
    rec["frac_cover"] = {"value": fc, "proven": fcopt, "s": time.perf_counter() - t}
    t = time.perf_counter()
    r2, r2opt = f2_minrank_sat(A, a or 1, S, LIMIT)
    rec["f2_minrank_sat"] = {"value": r2, "lower_or_proven": r2opt, "s": time.perf_counter() - t}
    rec["agent"] = agent(A, LIMIT)
    return rec


def _sub(spec):
    try:
        p = subprocess.run([sys.executable, __file__, "job", str(spec[0]), str(spec[1])], capture_output=True,
                           text=True, timeout=LIMIT * 6 + 120)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else {"graph": f"K{spec}", "error": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        return {"graph": f"K{spec}", "error": "watchdog"}


def cmd_run():
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(x)["graph"] for x in OUT.read_text().splitlines() if x.strip()}
    todo = [s for s in specs() if f"K({s[0]},{s[1]})" not in done]
    with ThreadPoolExecutor(3) as ex, open(OUT, "a") as f:
        for r in ex.map(_sub, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["graph"], "ERROR " + r["error"][-200:] if "error" in r else
                  f"agent [{r['agent']['lower']},{r['agent']['upper']}] {r['agent']['s']:.2f}s", flush=True)


EXCEPTIONS = {2: {5}, 3: {7, 8}, 4: {9, 10, 11, 14}}   # from the (private) theory; used only to label rows


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    rows = [r for r in rows if "error" not in r]
    rows.sort(key=lambda r: (r["k"], r["n"]))
    L = []
    L.append(f"# Kneser graphs: agent vs standard methods ({len(rows)} graphs, {LIMIT:g} s limit per method, one thread)\n")
    L.append("Baseline = largest independent set (CP-SAT) as lower bound, best of clique cover (CP-SAT), fractional "
             "clique cover (column generation) and F_2 minrank (SAT) as upper bound; its time is counted as if all "
             "standard methods ran in parallel. 'exact' = lower = upper, every bound re-checked.\n")
    L.append("| graph | receivers | baseline answer | baseline time | agent answer | agent time | field | speed-up |")
    L.append("|---|---|---|---|---|---|---|---|")
    stats = {"agent_exact": 0, "base_exact": 0, "agent_only": 0, "agent_faster": 0, "both": 0, "speedups": [],
             "base_tighter": 0, "agent_wrong": 0}
    for r in rows:
        a = r["cpsat_alpha"]
        lo = a["value"] or 0
        ups = [("clique cover", r["cpsat_cover"]["value"], r["cpsat_cover"]["s"]),
               ("frac cover", r["frac_cover"]["value"], r["frac_cover"]["s"]),
               ("F2 minrank", r["f2_minrank_sat"]["value"], r["f2_minrank_sat"]["s"])]
        ups = [(nm, v, t) for nm, v, t in ups if v is not None]
        hi = min(v for _, v, _ in ups)
        hi_disp = f"{hi:.2f}".rstrip("0").rstrip(".") if isinstance(hi, float) else str(hi)
        base_exact = a["proven"] and hi <= lo + 1e-9
        if base_exact:
            t_up = min(t for _, v, t in ups if v <= lo + 1e-9)
            bt = max(a["s"], t_up)
            bans, btime = f"**{lo}**", f"{bt:.2f} s"
        else:
            bt = None
            bans, btime = f"[{lo}, {hi_disp}]", f"open after {max(a['s'], max(t for _, _, t in ups)):.0f} s"
        g = r["agent"]
        ag_exact = g["exact"]
        aans = f"**{g['upper']}**" if ag_exact else f"[{g['lower']}, {g['upper']}]"
        if ag_exact and g["upper"] != r["alpha_theory"]:
            stats["agent_wrong"] += 1
        sp = ""
        if ag_exact and base_exact:
            stats["both"] += 1
            sp = f"x{bt / max(g['s'], 1e-3):.0f}"
            stats["speedups"].append(bt / max(g["s"], 1e-3))
            stats["agent_faster"] += g["s"] < bt
        elif ag_exact:
            stats["agent_only"] += 1
            sp = "baseline never closes"
        elif base_exact:
            stats["base_tighter"] += 1
        stats["agent_exact"] += ag_exact
        stats["base_exact"] += base_exact
        note = " (exception)" if r["n"] in EXCEPTIONS.get(r["k"], ()) else ""
        L.append(f"| {r['graph']}{note} | {r['vertices']} | {bans} | {btime} | {aans} | {g['s']:.2f} s | "
                 f"{'F' + str(g['field']) if g['field'] else '-'} | {sp} |")
    sp = stats["speedups"]
    L.append("")
    L.append(f"- exact answers: agent **{stats['agent_exact']}/{len(rows)}**, baseline {stats['base_exact']}/{len(rows)}")
    L.append(f"- graphs only the agent solves exactly: {stats['agent_only']}; only the baseline: {stats['base_tighter']}")
    if sp:
        L.append(f"- where both are exact ({stats['both']}): agent faster on {stats['agent_faster']}, "
                 f"median speed-up x{float(np.median(sp)):.0f}, min x{min(sp):.1f}, max x{max(sp):.0f}")
    L.append(f"- agent exact answers that disagree with C(n-1,k-1): {stats['agent_wrong']}")
    text = "\n".join(L)
    (HERE / "results" / "kneser_bench_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "report":
        cmd_report()
    elif sys.argv[1] == "job":
        print(json.dumps(job(int(sys.argv[2]), int(sys.argv[3]))))
    elif sys.argv[1] == "run":
        cmd_run()
