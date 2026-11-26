"""Baselines on exactly the same (relabelled) graphs the agent sees.

    python bench/baselines.py run [--limit 60] [--par 2] [--out results/X.jsonl] [specs...]   -> results/baselines.jsonl
Methods (each in its own forked process, one thread, LIMIT seconds; same rules as minrank-agent/kneser_bench.py):
  standard methods  cpsat_alpha (lower), cpsat_cover, frac_cover, f2_minrank_sat (upper)
  kushagra_agent    Kushagra's theory-guided script (recognise (n,k) -> rule prime -> one rank -> greedy star),
                    falling back to the general minrank agent when it does not certify
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
ROOT = HERE.parent
REPO = ROOT.parent
sys.path[:0] = [str(ROOT), str(HERE), str(REPO / "minrank-agent"), str(REPO / "lambda-agent")]
OUT = ROOT / "results" / "baselines.jsonl"


def _isolated(fn, args, limit):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    a, b = ctx.Pipe(duplex=False)

    def child(conn):
        try:
            conn.send(("ok", fn(*args)))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)[:200]))

    p = ctx.Process(target=child, args=(b,), daemon=True)
    p.start()
    got = a.poll(limit + 30)
    res = a.recv() if got else None
    p.join(5)
    if p.is_alive():
        p.kill()
        p.join()
    if res is None:
        return None, "timeout or crash"
    return (res[1], "") if res[0] == "ok" else (None, "error: " + str(res[1]))


def job(spec, seed, limit):
    import kneser_bench as KB
    from kagent import graphs
    from kushagra_kneser_agent import solve

    A = graphs.load(graphs.make(spec, seed)).astype(np.int64)
    rec = {"spec": spec, "seed": seed, "receivers": int(A.shape[0]), "limit": limit}
    t = time.perf_counter()
    res, err = _isolated(KB.cpsat_alpha, (A, limit), limit)
    a, S, aopt = res if res else (None, [], False)
    rec["cpsat_alpha"] = {"value": a, "proven": aopt, "s": time.perf_counter() - t, "note": err}
    for name, fn in (("cpsat_cover", KB.cpsat_cover), ("frac_cover", KB.frac_cover)):
        t = time.perf_counter()
        res, err = _isolated(fn, (A, limit), limit)
        v, opt = res if res else (None, False)
        rec[name] = {"value": v, "proven": opt, "s": time.perf_counter() - t, "note": err}
    t = time.perf_counter()
    res, err = _isolated(KB.f2_minrank_sat, (A, a or 1, S, limit), limit)
    v, opt = res if res else (None, None)
    rec["f2_minrank_sat"] = {"value": v, "lower_or_proven": opt, "s": time.perf_counter() - t, "note": err}
    t = time.perf_counter()
    res, err = _isolated(solve, (A, KB.agent, limit), 2 * limit + 30)
    r = res or {}
    rec["kushagra_agent"] = {"lower": r.get("lower"), "upper": r.get("upper"), "exact": bool(r.get("exact")),
                             "field": r.get("field"), "path": r.get("path"), "s": time.perf_counter() - t, "note": err}
    return rec


def standard_best(r):
    """[best lower, best upper] of the standard methods, and when it closed (as if run in parallel) or None."""
    a = r["cpsat_alpha"]
    lo = a["value"]
    ups = [(r[m]["value"], r[m]["s"]) for m in ("cpsat_cover", "frac_cover", "f2_minrank_sat")
           if r.get(m) and r[m]["value"] is not None]
    hi = min(v for v, _ in ups) if ups else None
    if lo is not None and a["proven"] and hi is not None and hi <= lo + 1e-9:
        return lo, hi, max(a["s"], min(t for v, t in ups if v <= lo + 1e-9))
    return lo, hi, None


def _sub(spec, seed, limit):
    try:
        p = subprocess.run([sys.executable, __file__, "job", spec, str(seed), str(limit)], capture_output=True,
                           text=True, timeout=limit * 8 + 300)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else {"spec": spec, "error": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        return {"spec": spec, "error": "watchdog"}


def cmd_run(argv):
    global OUT
    limit, par, specs = 60.0, 2, []
    it = iter(argv)
    for x in it:
        if x == "--limit":
            limit = float(next(it))
        elif x == "--par":
            par = int(next(it))
        elif x == "--out":
            OUT = ROOT / next(it)
        else:
            specs.append(x)
    specs = specs or json.loads((HERE / "specs.json").read_text())["benchmark"]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["spec"] for l in OUT.read_text().splitlines() if l.strip()}
    todo = [s for s in specs if s not in done]
    with open(OUT, "a") as f, ThreadPoolExecutor(par) as ex:
        for r in ex.map(lambda s: _sub(s, 0, limit), todo):
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(r["spec"], standard_best(r) if "error" not in r else r["error"][:80], flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(sys.argv[2], int(sys.argv[3]), float(sys.argv[4])), default=str))
    else:
        cmd_run(sys.argv[2:])
