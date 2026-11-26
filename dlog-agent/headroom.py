"""Headroom study for discrete logarithm: best single method vs the per-instance oracle.

    python dlog-agent/headroom.py run      every method on every instance -> results/headroom.jsonl
    python dlog-agent/headroom.py report   by structure family

A method "solves" an instance when its returned x passes group.check (g^x = h, in the interval if one is given).
Each method runs in its own process with a hard time limit; an unfinished sqrt(n) loop is just killed. The question
is how far the single best method is from choosing (and, for Pohlig-Hellman, composing) the right method per instance.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "headroom.jsonl"
LIMIT = 60.0


def _load(d):
    from dlog_agent.group import EC, Fp, Instance

    kind = d["kind"]
    g = tuple(json.loads(d["g"])) if kind == "EC" and json.loads(d["g"]) is not None else json.loads(d["g"])
    h = tuple(json.loads(d["h"])) if kind == "EC" and json.loads(d["h"]) is not None else json.loads(d["h"])
    n = int(d["n"])
    if kind == "Fp":
        G = Fp(int(d["p"]), n)
    else:
        G = EC(int(d["a"]), int(d["b"]), int(d["p"]), n)
    interval = None if d["interval"] is None else int(d["interval"])
    return Instance(d["name"], d["family"], G, g, h, n, d["bits"], interval)


def _run_one(name, inst, limit):
    """Run one method in a forked process, killed at the limit. Returns (solved, seconds)."""
    import multiprocessing as mp

    from dlog_agent.group import check
    from dlog_agent.methods import solve

    ctx = mp.get_context("fork")
    a, b = ctx.Pipe(duplex=False)

    def child(conn):
        try:
            t = time.perf_counter()
            x = solve(name, inst, limit)
            conn.send((check(inst, x), time.perf_counter() - t))
        except Exception as e:  # noqa: BLE001
            conn.send((False, repr(e)[:150]))

    p = ctx.Process(target=child, args=(b,), daemon=True)
    t = time.perf_counter()
    p.start()
    if a.poll(limit + 2):
        solved, info = a.recv()
    else:
        solved, info = False, time.perf_counter() - t
    if p.is_alive():
        p.kill()
    p.join()
    secs = info if isinstance(info, float) else (time.perf_counter() - t)
    return bool(solved), round(min(secs, LIMIT), 4), ("" if isinstance(info, float) else str(info))


def job(d):
    from dlog_agent.methods import METHODS

    inst = _load(d)
    runs = {}
    for m in METHODS:
        solved, secs, err = _run_one(m, inst, LIMIT)
        runs[m] = {"ok": solved, "s": secs, **({"err": err} if err else {})}
    return {"name": d["name"], "family": d["family"], "kind": d["kind"], "bits": d["bits"], "runs": runs}


def cmd_run():
    import subprocess
    from concurrent.futures import ThreadPoolExecutor

    insts = json.loads((HERE / "instances.json").read_text())
    done = set()
    if OUT.exists():
        done = {json.loads(x)["name"] for x in OUT.read_text().splitlines() if x.strip()}
    todo = [d for d in insts if d["name"] not in done]
    OUT.parent.mkdir(exist_ok=True)

    def sub(d):
        try:
            p = subprocess.run([sys.executable, __file__, "job", json.dumps(d)], capture_output=True, text=True,
                               timeout=LIMIT * 6 + 60)
            line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
            return json.loads(line) if line.startswith("{") else {"name": d["name"], "error": p.stderr[-300:]}
        except subprocess.TimeoutExpired:
            return {"name": d["name"], "error": "watchdog"}

    with ThreadPoolExecutor(3) as ex, open(OUT, "a") as f:
        for r in ex.map(sub, todo):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if "error" in r:
                print(r["name"], "ERROR", r["error"][-150:], flush=True)
            else:
                print(r["name"], {m: (v["s"] if v["ok"] else "x") for m, v in r["runs"].items()}, flush=True)


def sgm(ts, shift=0.1):
    return float(np.exp(np.mean(np.log(np.asarray(ts) + shift))) - shift)


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip() and "runs" in json.loads(x)]
    methods = list(rows[0]["runs"])
    pen = lambda r: r["s"] if r["ok"] else 2 * LIMIT  # noqa: E731
    lines = [f"{len(rows)} discrete-log instances, {LIMIT:g}s per method (failure = {2 * LIMIT:g}s); "
             "shifted geometric mean of time, shift 0.1s. A run counts only if g^x = h was verified."]
    groups = [("all", rows)]
    for kind in ("Fp", "EC"):
        for fam in sorted({r["family"] for r in rows if r["kind"] == kind}):
            groups.append((f"{kind}/{fam}", [r for r in rows if r["kind"] == kind and r["family"] == fam]))
    for label, sel in groups:
        if not sel:
            continue
        res = {m: sgm([pen(r["runs"][m]) for r in sel]) for m in methods}
        solved = {m: sum(r["runs"][m]["ok"] for r in sel) for m in methods}
        orc = sgm([min(pen(r["runs"][m]) for m in methods) for r in sel])
        orc_solved = sum(any(r["runs"][m]["ok"] for m in methods) for r in sel)
        b = min(res, key=res.get)
        wins = {}
        for r in sel:
            cands = [m for m in methods if r["runs"][m]["ok"]]
            if cands:
                w = min(cands, key=lambda m: r["runs"][m]["s"])
                wins[w] = wins.get(w, 0) + 1
        lines.append(f"\n{label} ({len(sel)}):")
        for m in methods:
            lines.append(f"    {m:15s} solved {solved[m]:2d}/{len(sel)}  sgm {res[m]:7.3f}s")
        lines.append(f"  best single: {b} {res[b]:.3f}s (solves {solved[b]}) | oracle {orc:.3f}s solves {orc_solved}"
                     f" | speedup x{res[b] / max(orc, 1e-9):.1f} | fastest-correct: {wins}")
    text = "\n".join(lines)
    (HERE / "results" / "headroom_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(json.loads(sys.argv[2]))))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
