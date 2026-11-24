"""Headroom study: how much could ANY adaptive combination of the existing LP methods gain?

    python lp-agent/headroom.py run      every method on every instance, two accuracy targets -> results/headroom.jsonl
    python lp-agent/headroom.py report   best single method vs per-instance oracle

Instances: Netlib (103), Meszaros (76) and the LP relaxations of MIPLIB models (495), from the
optimatika/optimisation-models collection. Infeasible or unbounded LPs are skipped (dual simplex decides, 30 s).
A method "solves" an instance at target eps when the independent check (problem.kkt) certifies relative KKT error
<= eps. Tolerance-based methods (interior point, PDLP) are re-run with 100x tighter internal tolerance while the
check fails, until the 30 s limit; all time counts. Simplex and crossover give vertices: one run serves both targets.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = Path("/home/user/optimatika/optimisation-models/src/main/resources/optimisation")
OUT = HERE / "results" / "headroom.jsonl"
LIMIT = 30.0
TARGETS = (1e-4, 1e-8)
VERTEX = ("dual_simplex", "primal_simplex", "ipm_crossover")


def instances():
    out = []
    for d, pat in (("netlib", "*.SIF"), ("meszaros", "*.mps"), ("MIPLIB", "*.mps")):
        out += sorted(str(p) for p in (ROOT / d).glob(pat))
    return out


def job(path):
    from lp_agent.methods import METHODS
    from lp_agent.problem import kkt, load_mps

    t = time.perf_counter()
    lp = load_mps(path, Path(path).parent.name + "/" + Path(path).stem)
    load_s = time.perf_counter() - t
    rec = {"name": lp.name, "rows": lp.shape[0], "cols": lp.shape[1], "nnz": lp.nnz, "load_s": round(load_s, 2),
           "runs": {}}
    x, y, st, s = METHODS["dual_simplex"](lp, 1e-8, LIMIT)
    if st != "Optimal":
        rec["skip"] = st
        return rec
    for name, fn in METHODS.items():
        for target in TARGETS:
            if name in VERTEX and target != TARGETS[0]:
                rec["runs"][f"{name}@{target:g}"] = rec["runs"][f"{name}@{TARGETS[0]:g}"]
                continue
            used, tol, best = 0.0, target, None
            while used < LIMIT:
                if name == "dual_simplex" and best is None and used == 0.0:
                    xx, yy, st2, s2 = x, y, st, s            # reuse the screening run
                else:
                    xx, yy, st2, s2 = fn(lp, tol, LIMIT - used)
                used += s2
                k = kkt(lp, xx, yy) if xx is not None and yy is not None else {"kkt": float("inf")}
                best = k["kkt"] if best is None else min(best, k["kkt"])
                if k["kkt"] <= target or name in VERTEX or tol < target * 1e-5:
                    break
                tol /= 100
            ok = best is not None and best <= target
            rec["runs"][f"{name}@{target:g}"] = {"ok": bool(ok), "s": round(used, 4), "kkt": best}
    return rec


def _one(path):
    try:
        p = subprocess.run([sys.executable, __file__, "job", path], capture_output=True, text=True,
                           timeout=LIMIT * 14 + 120)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else {"name": path, "error": p.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {"name": path, "error": "watchdog"}


def cmd_run(workers=3):
    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(x).get("path") for x in OUT.read_text().splitlines() if x.strip()}
    todo = [p for p in instances() if p not in done]
    with ThreadPoolExecutor(workers) as ex, open(OUT, "a") as f:
        for p, r in zip(todo, ex.map(_one, todo)):
            r["path"] = p
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["name"], r.get("skip") or r.get("error", "")[:60] or "ok", flush=True)


def sgm(ts, shift=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(ts) + shift))) - shift)


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    good = [r for r in rows if r.get("runs") and "skip" not in r and "error" not in r]
    print(f"{len(rows)} instances: {len(good)} solved LPs, "
          f"{sum('skip' in r for r in rows)} skipped (infeasible/unbounded/limit), {sum('error' in r for r in rows)} errors")
    methods = sorted({k.split('@')[0] for r in good for k in r["runs"]})
    for target in TARGETS:
        key = lambda m: f"{m}@{target:g}"  # noqa: E731
        pen = lambda run: run["s"] if run["ok"] else 2 * LIMIT  # noqa: E731  (PAR2: failures count double)
        print(f"\n== target relative KKT <= {target:g}  (shifted geometric mean of time, failures = {2 * LIMIT:g} s)")
        res = {}
        for m in methods:
            ts = [pen(r["runs"][key(m)]) for r in good]
            res[m] = sgm(ts)
            print(f"  {m:15s} solved {sum(r['runs'][key(m)]['ok'] for r in good):4d}/{len(good)}  sgm {res[m]:.3f} s")
        oracle = [min(pen(r["runs"][key(m)]) for m in methods) for r in good]
        best = min(res, key=res.get)
        print(f"  best single method: {best} ({res[best]:.3f} s); per-instance oracle: {sgm(oracle):.3f} s "
              f"-> headroom x{res[best] / max(sgm(oracle), 1e-9):.2f}; oracle solves {sum(o < 2 * LIMIT for o in oracle)}")
        wins = {}
        for r in good:
            w = min(methods, key=lambda m: pen(r["runs"][key(m)]))
            wins[w] = wins.get(w, 0) + 1
        print("  fastest method per instance:", dict(sorted(wins.items(), key=lambda kv: -kv[1])))
        big = [r for r in good if r["nnz"] >= 100_000]
        if big:
            ob = [min(pen(r["runs"][key(m)]) for m in methods) for r in big]
            bb = {m: sgm([pen(r["runs"][key(m)]) for r in big]) for m in methods}
            b = min(bb, key=bb.get)
            print(f"  on the {len(big)} instances with >= 100k nonzeros: best single {b} {bb[b]:.2f} s, "
                  f"oracle {sgm(ob):.2f} s -> x{bb[b] / max(sgm(ob), 1e-9):.2f}")


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(sys.argv[2])))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
