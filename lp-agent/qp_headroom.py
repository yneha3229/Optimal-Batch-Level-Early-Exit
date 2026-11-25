"""Headroom study for convex QP: how much could ANY adaptive combination of the existing QP solvers gain?

    python lp-agent/qp_headroom.py run      every solver on every Maros-Meszaros QP, two targets -> results/qp_headroom.jsonl
    python lp-agent/qp_headroom.py report   best single solver vs per-instance oracle, by difficulty

A solver "solves" a QP at target eps when the independent check (qp.kkt) certifies relative KKT error <= eps. While
the check fails, the solver is re-run with a 100x tighter internal tolerance until the 30 s limit; all time counts.
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
ROOT = Path("/home/user/optimatika/optimisation-models/src/main/resources/optimisation/marosmeszaros")
OUT = HERE / "results" / "qp_headroom.jsonl"
LIMIT = 30.0
TARGETS = (1e-4, 1e-8)


def job(path):
    from lp_agent.qp import METHODS, kkt, load_qps, run

    t = time.perf_counter()
    qp = load_qps(path, "mm/" + Path(path).stem)
    rec = {"name": qp.name, "rows": qp.shape[0], "cols": qp.shape[1], "nnz": int(qp.A.nnz), "qnnz": int(qp.Q.nnz),
           "load_s": round(time.perf_counter() - t, 2), "runs": {}}
    for name in METHODS:
        for target in TARGETS:
            used, tol, best, status = 0.0, target, None, ""
            while used < LIMIT:
                x, y, status, s = run(name, qp, tol, LIMIT - used)
                used += s
                k = kkt(qp, x, y)
                best = k["kkt"] if best is None else min(best, k["kkt"])
                if k["kkt"] <= target or tol < target * 1e-5 or "killed" in status:
                    break
                tol /= 100
            rec["runs"][f"{name}@{target:g}"] = {"ok": bool(best is not None and best <= target),
                                                  "s": round(min(used, LIMIT), 4), "kkt": best, "status": status[:40]}
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
    todo = [str(p) for p in sorted(ROOT.glob("*.SIF")) if str(p) not in done]
    with ThreadPoolExecutor(workers) as ex, open(OUT, "a") as f:
        for p, r in zip(todo, ex.map(_one, todo)):
            r["path"] = p
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["name"], r.get("error", "")[:60] or "ok", flush=True)


def sgm(ts, shift=1.0):
    return float(np.exp(np.mean(np.log(np.asarray(ts) + shift))) - shift)


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    good = [r for r in rows if r.get("runs")]
    pen = lambda run: run["s"] if run["ok"] else 2 * LIMIT  # noqa: E731
    methods = sorted({k.split("@")[0] for r in good for k in r["runs"]})
    lines = [f"{len(good)} QPs ({len(rows) - len(good)} errors)"]
    for target in TARGETS:
        key = lambda m: f"{m}@{target:g}"  # noqa: E731
        oracle = {r["name"]: min(pen(r["runs"][key(m)]) for m in methods) for r in good}
        lines.append(f"\n== target relative KKT <= {target:g}  (shifted geometric mean, shift 1 s; failures = {2 * LIMIT:g} s)")
        res = {}
        for m in methods:
            res[m] = sgm([pen(r["runs"][key(m)]) for r in good])
            lines.append(f"  {m:9s} solved {sum(r['runs'][key(m)]['ok'] for r in good):3d}/{len(good)}  sgm {res[m]:.3f} s")
        b = min(res, key=res.get)
        lines.append(f"  best single: {b} {res[b]:.3f} s; oracle {sgm(list(oracle.values())):.3f} s "
                     f"(x{res[b] / max(sgm(list(oracle.values())), 1e-9):.2f}); oracle solves "
                     f"{sum(v < 2 * LIMIT for v in oracle.values())}/{len(good)}")
        wins = {}
        for r in good:
            w = min(methods, key=lambda m: pen(r["runs"][key(m)]))
            wins[w] = wins.get(w, 0) + 1
        lines.append("  fastest per instance: " + str(dict(sorted(wins.items(), key=lambda kv: -kv[1]))))
        for lo, hi in ((0, 0.1), (0.1, 1), (1, 1e9)):
            sel = [r for r in good if lo <= oracle[r["name"]] < hi]
            if not sel:
                continue
            rr = {m: sgm([pen(r["runs"][key(m)]) for r in sel]) for m in methods}
            bb = min(rr, key=rr.get)
            tot = {m: sum(pen(r["runs"][key(m)]) for r in sel) for m in methods}
            bt = min(tot, key=tot.get)
            lines.append(f"  oracle time in [{lo:g},{hi:g}) s: {len(sel):3d} QPs | best single {bb} {rr[bb]:.3f}s vs oracle "
                         f"{sgm([oracle[r['name']] for r in sel]):.3f}s | total: best single {bt} {tot[bt]:.1f}s vs oracle "
                         f"{sum(oracle[r['name']] for r in sel):.1f}s")
    text = "\n".join(lines)
    (HERE / "results" / "qp_headroom_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(sys.argv[2])))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
