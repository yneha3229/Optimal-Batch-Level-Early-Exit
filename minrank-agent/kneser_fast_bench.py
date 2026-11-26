"""The theory-guided Kneser agent vs the general agent vs the standard methods.

    python minrank-agent/kneser_fast_bench.py run      -> results/kneser_fast.jsonl
    python minrank-agent/kneser_fast_bench.py report   -> results/kneser_fast_report.md

Part 1: the 30 graphs of kneser_bench.py (standard methods and general agent results re-used from kneser_bench.jsonl).
Part 2: larger graphs (406 - 3003 receivers): standard methods re-run here with the same 120 s / one thread rules.
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
OUT = HERE / "results" / "kneser_fast.jsonl"
LIMIT = 120.0
LARGE = [(29, 2), (31, 2), (40, 2), (16, 3), (17, 3), (18, 3), (13, 4), (14, 4), (15, 4), (13, 5), (17, 4), (15, 5)]


def _isolated(fn, args, limit):
    """Run fn(*args) in a forked child; a crash (e.g. out of memory) is reported instead of killing the row."""
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
        return None, ("crashed (exit code %s)" % p.exitcode) if p.exitcode not in (None, 0, -9) else "timeout"
    if res[0] == "ok":
        return res[1], ""
    return None, "error: " + str(res[1])


def job(n, k, part):
    import kneser_bench as KB
    from minrank_agent import structgraph as SG
    from minrank_agent.kneser_agent import solve

    g = SG.kneser(n, k)
    A = g.A
    rec = {"graph": g.name, "n": n, "k": k, "vertices": g.n, "part": part}
    t = time.perf_counter()
    r = solve(A, fallback=KB.agent, fallback_limit=LIMIT)
    rec["fast_agent"] = {kk: (v if not isinstance(v, (np.integer,)) else int(v)) for kk, v in r.items()}
    rec["fast_agent"]["s"] = time.perf_counter() - t
    if part == 2:                                   # standard methods, same rules, each isolated
        t = time.perf_counter()
        res, err = _isolated(KB.cpsat_alpha, (A, LIMIT), LIMIT)
        a, S, aopt = res if res else (None, [], False)
        rec["cpsat_alpha"] = {"value": a, "proven": aopt, "s": time.perf_counter() - t, "note": err}
        t = time.perf_counter()
        res, err = _isolated(KB.cpsat_cover, (A, LIMIT), LIMIT)
        cc, ccopt = res if res else (None, False)
        rec["cpsat_cover"] = {"value": cc, "proven": ccopt, "s": time.perf_counter() - t, "note": err}
        t = time.perf_counter()
        res, err = _isolated(KB.frac_cover, (A, LIMIT), LIMIT)
        fc, fcopt = res if res else (None, False)
        rec["frac_cover"] = {"value": fc, "proven": fcopt, "s": time.perf_counter() - t, "note": err}
        t = time.perf_counter()
        res, err = _isolated(KB.f2_minrank_sat, (A, a or 1, S, LIMIT), LIMIT)
        r2, r2opt = res if res else (None, None)
        rec["f2_minrank_sat"] = {"value": r2, "lower_or_proven": r2opt, "s": time.perf_counter() - t, "note": err}
    return rec


def _sub(spec):
    n, k, part = spec
    try:
        p = subprocess.run([sys.executable, __file__, "job", str(n), str(k), str(part)], capture_output=True,
                           text=True, timeout=LIMIT * 8 + 300)
        line = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
        return json.loads(line) if line.startswith("{") else {"graph": f"K({n},{k})", "part": part,
                                                              "error": p.stderr[-400:]}
    except subprocess.TimeoutExpired:
        return {"graph": f"K({n},{k})", "part": part, "error": "watchdog"}


def cmd_run():
    import kneser_bench as KB

    OUT.parent.mkdir(exist_ok=True)
    done = set()
    if OUT.exists():
        done = {json.loads(x)["graph"] for x in OUT.read_text().splitlines() if x.strip()}
    todo = [(n, k, 1) for n, k in KB.specs()] + [(n, k, 2) for n, k in LARGE]
    todo = [s for s in todo if f"K({s[0]},{s[1]})" not in done]
    # part 1 is quick: run serially for clean timings; part 2 three at a time
    with open(OUT, "a") as f:
        for s in [s for s in todo if s[2] == 1]:
            r = _sub(s)
            f.write(json.dumps(r, default=str) + "\n")
            f.flush()
            print(r["graph"], r.get("fast_agent", {}).get("path", r.get("error", ""))[:60],
                  f"{r.get('fast_agent', {}).get('s', 0):.3f}s", flush=True)
        with ThreadPoolExecutor(3) as ex:
            for r in ex.map(_sub, [s for s in todo if s[2] == 2]):
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                print(r["graph"], r.get("fast_agent", {}).get("path", r.get("error", ""))[:60],
                      f"{r.get('fast_agent', {}).get('s', 0):.3f}s", flush=True)


def _baseline(r):
    """Standard-methods bracket and the time it first closes (as if all ran in parallel), or None."""
    a = r["cpsat_alpha"]
    lo = a["value"] or 0
    if a["value"] is None:
        lo = None
    ups = [(r[m]["value"], r[m]["s"]) for m in ("cpsat_cover", "frac_cover", "f2_minrank_sat")
           if r.get(m) and r[m]["value"] is not None]
    hi = min(v for v, _ in ups) if ups else None
    if lo is not None and a["proven"] and hi is not None and hi <= lo + 1e-9:
        return lo, hi, max(a["s"], min(t for v, t in ups if v <= lo + 1e-9))
    return lo, hi, None


def _fmt(v):
    return "-" if v is None else (f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    old = {json.loads(x)["graph"]: json.loads(x) for x in (HERE / "results" / "kneser_bench.jsonl").read_text()
           .splitlines() if x.strip()}
    L = ["# Theory-guided Kneser agent vs general agent vs standard methods", "",
         f"Same machine, one thread, {LIMIT:g} s per standard method. 'exact' = certified lower = upper.", ""]
    for part, title in ((1, "Part 1: the 30 benchmark graphs (10-330 receivers)"),
                        (2, "Part 2: larger graphs (406-3003 receivers)")):
        L += [f"## {title}", "",
              "| graph | receivers | standard methods | std time | general agent | gen time | **Kneser agent** | time | field |",
              "|---|---|---|---|---|---|---|---|---|"]
        sp_std, sp_gen, ex_fast, ex_std, ex_gen, nrows = [], [], 0, 0, 0, 0
        for r in sorted([r for r in rows if r.get("part") == part and "error" not in r], key=lambda r: (r["k"], r["n"])):
            nrows += 1
            f = r["fast_agent"]
            src = old.get(r["graph"], {}) if part == 1 else r
            if part == 1 and not src:
                continue
            lo, hi, bt = _baseline(src)
            std = f"**{lo}**" if bt is not None else f"[{_fmt(lo)}, {_fmt(hi)}]"
            notes = [m for m in ("cpsat_alpha", "cpsat_cover", "frac_cover", "f2_minrank_sat")
                     if part == 2 and src.get(m, {}).get("note")]
            if notes:
                std += " (" + ", ".join(m + " " + src[m]["note"].split(":")[0] for m in notes) + ")"
            stdt = f"{bt:.2f} s" if bt is not None else "open"
            if part == 1:
                g = src["agent"]
                gen = f"**{g['upper']}**" if g["exact"] else f"[{_fmt(g['lower'])}, {_fmt(g['upper'])}]"
                gent = f"{g['s']:.2f} s"
                ex_gen += g["exact"]
                if g["exact"] and f["exact"]:
                    sp_gen.append(g["s"] / max(f["s"], 1e-4))
            else:
                gen, gent = "-", "-"
            fa = f"**{f['upper']}**" if f["exact"] else f"[{_fmt(f['lower'])}, {_fmt(f['upper'])}]"
            ex_fast += f["exact"]
            ex_std += bt is not None
            if bt is not None and f["exact"]:
                sp_std.append(bt / max(f["s"], 1e-4))
            L.append(f"| {r['graph']} | {r['vertices']} | {std} | {stdt} | {gen} | {gent} | {fa} | {f['s']:.3f} s | "
                     f"{'F' + str(f['field']) if f.get('field') else '-'} |")
        L += ["", f"- exact: Kneser agent {ex_fast}/{nrows}, standard methods {ex_std}/{nrows}"
              + (f", general agent {ex_gen}/{nrows}" if part == 1 else "")]
        if sp_std:
            L.append(f"- vs standard methods where both exact ({len(sp_std)}): median speed-up "
                     f"x{np.median(sp_std):.0f} (min x{min(sp_std):.0f}, max x{max(sp_std):.0f})")
        if sp_gen:
            L.append(f"- vs the general agent where both exact ({len(sp_gen)}): median speed-up x{np.median(sp_gen):.0f}")
        L.append("")
    text = "\n".join(L)
    (HERE / "results" / "kneser_fast_report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if sys.argv[1] == "job":
        print(json.dumps(job(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])), default=str))
    else:
        {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
