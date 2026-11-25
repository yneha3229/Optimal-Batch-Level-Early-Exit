"""Headroom study for certified primality: best single method vs per-number oracle.

    python prime-agent/headroom.py run      -> results/headroom.jsonl
    python prime-agent/headroom.py report
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "headroom.jsonl"
LIMIT = 60.0


def one(rec):
    from prime_agent.methods import METHODS, run

    n = int(rec["n"])
    res = {}
    for m in METHODS:
        ok, s, info, err = run(m, n, LIMIT)
        res[m] = {"ok": ok, "s": round(s, 4), **info, **({"err": err} if err else {})}
    return {"name": rec["name"], "family": rec["family"], "digits": rec["digits"], "runs": res}


def _sub(rec):
    import subprocess

    try:
        p = subprocess.run([sys.executable, __file__, "job", json.dumps(rec)], capture_output=True, text=True,
                           timeout=LIMIT * 6 + 60)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else {**rec, "error": p.stderr[-300:]}
    except subprocess.TimeoutExpired:
        return {**rec, "error": "watchdog"}


def cmd_run():
    nums = json.loads((HERE / "numbers.json").read_text())
    done = set()
    if OUT.exists():
        done = {json.loads(x)["name"] for x in OUT.read_text().splitlines() if x.strip()}
    todo = [r for r in nums if r["name"] not in done]
    with ThreadPoolExecutor(3) as ex, open(OUT, "a") as f:
        for r in ex.map(_sub, todo):
            if "error" in r:
                print(r["name"], "ERROR", r["error"][-200:], flush=True)
                continue
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(r["name"], r["digits"], {m: (v["s"] if v["ok"] else "x") for m, v in r["runs"].items()}, flush=True)


def sgm(ts, shift=0.1):
    return float(np.exp(np.mean(np.log(np.asarray(ts) + shift))) - shift)


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    pen = lambda r: r["s"] if r["ok"] else 2 * LIMIT  # noqa: E731
    methods = list(rows[0]["runs"])
    lines = [f"{len(rows)} primes, limit {LIMIT:g} s per method (failures count {2 * LIMIT:g} s); "
             "shifted geometric mean, shift 0.1 s"]
    for fam in ["all"] + sorted({r["family"] for r in rows}):
        sel = [r for r in rows if fam == "all" or r["family"] == fam]
        res = {m: sgm([pen(r["runs"][m]) for r in sel]) for m in methods}
        orc = sgm([min(pen(r["runs"][m]) for m in methods) for r in sel])
        b = min(res, key=res.get)
        solved = {m: sum(r["runs"][m]["ok"] for r in sel) for m in methods}
        wins = {}
        for r in sel:
            w = min(methods, key=lambda m: pen(r["runs"][m]))
            wins[w] = wins.get(w, 0) + 1
        lines.append(f"\n{fam} ({len(sel)}): " + ", ".join(f"{m} {res[m]:.3f}s ({solved[m]})" for m in methods))
        lines.append(f"   best single {b} {res[b]:.3f}s | oracle {orc:.3f}s | x{res[b] / max(orc, 1e-9):.2f} | "
                     f"total time best single {sum(pen(r['runs'][b]) for r in sel):.1f}s vs oracle "
                     f"{sum(min(pen(r['runs'][m]) for m in methods) for r in sel):.1f}s | fastest: {wins}")
    text = "\n".join(lines)
    (HERE / "results" / "headroom_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(one(json.loads(sys.argv[2]))))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
