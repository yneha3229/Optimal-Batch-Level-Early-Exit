"""Live terminal view of one run (rich), or a replay of a recorded trace at real speed.

    python -m kagent.demo --graph "K(16,4)" --seed 0 --live [--cores 32-39 --baseline-cores 40-47]
    python -m kagent.demo --replay traces/demo/<id>.jsonl [--speed 1.0]

Panels: agent dialogue, bound board (L and U as they change), active workers, baseline timers (the standard methods
run in parallel on their own cores), and the final explicit-code summary. The baselines' events are written into the
same trace, so a replay shows everything.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

from rich.console import Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]


class View:
    def __init__(self, title: str) -> None:
        self.title = title
        self.t = 0.0
        self.n = None
        self.mode = ""
        self.budget = 300
        self.dialogue: list[Text] = []
        self.board: list[tuple] = []
        self.L, self.U = None, None
        self.workers: dict[str, dict] = {}
        self.done_tools = 0
        self.baselines: dict[str, dict] = {}
        self.final: list[str] = []
        self.events: list[str] = []
        self.lock = threading.Lock()

    # ------------------------------------------------------------------------------------------------ records
    def feed(self, r: dict) -> None:
        with self.lock:
            self._feed(r)

    def _feed(self, r: dict) -> None:
        k = r["kind"]
        self.t = max(self.t, r.get("t", 0.0))
        t = f"{r.get('t', 0):6.1f}s"
        if k == "run_start":
            self.n, self.mode, self.budget = r["n"], r["mode"], r["budget"]
        elif k == "describe":
            first = r["summary"].splitlines()
            self.dialogue.append(Text(f"{t} describe: " + "; ".join(first[:2]), style="dim"))
            spec = [s for s in first if s.startswith("spectrum")]
            if spec:
                self.dialogue.append(Text(f"        {spec[0][:150]}", style="dim"))
        elif k == "llm_call":
            rep = r.get("reply")
            who = r["agent"].upper()
            style = {"SOLVER": "bold cyan", "CRITIC": "bold magenta", "VERIFIER": "bold green"}.get(who, "white")
            head = f"{t} {who} ({r['latency']:.1f} s, {r.get('tokens_out', 0)} tok)"
            if isinstance(rep, dict):
                if who == "SOLVER":
                    fh = rep.get("family_hypothesis", {})
                    plan = rep.get("plan", [])
                    steps = ", ".join(f"{p.get('tool')}{json.dumps(p.get('args', {}))}" for p in plan[:4])
                    body = (f"hypothesis: {fh.get('name')} {json.dumps(fh.get('params', {}))} "
                            f"(conf {fh.get('confidence')})\n  evidence: {'; '.join(fh.get('evidence', [])[:2])}\n"
                            f"  plan: {steps}")
                elif who == "CRITIC":
                    body = f"verdict: {rep.get('verdict')}  " + " | ".join(rep.get("objections", [])[:2])
                    if rep.get("requested_checks"):
                        body += f"\n  checks: {json.dumps(rep['requested_checks'])[:160]}"
                else:
                    body = str(rep.get("analysis") or rep.get("checks_requested"))
                self.dialogue.append(Text(head, style=style))
                self.dialogue.append(Text("  " + body[:420]))
            else:
                self.dialogue.append(Text(head + "  (unparsed reply, retry)", style=style))
        elif k == "llm_fallback":
            self.dialogue.append(Text(f"{t} {r['agent']}: fallback to scheduler default ({r.get('reason')})",
                                      style="yellow"))
        elif k == "plan_accepted":
            self.dialogue.append(Text(f"{t} plan accepted ({r['source']})", style="bold"))
        elif k == "critic_check":
            pass
        elif k == "board_update" and r["quantity"] == "beta":
            if r["bound"] == "lower":
                self.L = r["value"]
            else:
                self.U = r["value"]
            self.board.append((t, r["bound"], r["value"], r["tool"]))
        elif k == "event" and r["name"] in ("NEAR_MISS", "STALL", "CONTRADICTION", "NEW_PHASE", "CLOSED",
                                            "BUDGET"):
            self.events.append(f"{t} {r['name']} {json.dumps(r.get('detail'))[:90]}")
        elif k == "tool_start":
            self.workers[r["task"]] = {"tool": r["tool"], "t0": r["t"], "args": r.get("args", {})}
        elif k == "tool_stop":
            self.workers.pop(r["task"], None)
            self.done_tools += 1
        elif k == "tool_report" and r.get("tool") == "spectral":
            rk = r.get("rank")
            self.events.append(f"{t} F_{r['p']} theta={r.get('theta')}: " +
                               (f"rank {rk}" if rk is not None else f"rank > {r.get('cap')}"))
        elif k == "baseline_start":
            self.baselines[r["method"]] = {"t0": r["t"], "status": "running"}
        elif k == "baseline_result":
            self.baselines[r["method"]] = {"t0": self.baselines.get(r["method"], {}).get("t0", r["t"]),
                                           "status": r.get("status"), "lower": r.get("lower"),
                                           "upper": r.get("upper"), "time": r.get("time"), "end": r["t"]}
        elif k == "verify_check":
            self.final.append(f"{'PASS' if r.get('ok') else 'FAIL'} {r.get('check')}")
        elif k == "verdict":
            self.final.append(f"VERDICT: {r['verdict']}")
            if r.get("analysis"):
                self.dialogue.append(Text(f"{t} VERIFIER analysis", style="bold green"))
                self.dialogue.append(Text("  " + r["analysis"][:500]))
        elif k == "code_summary":
            self.final.append(r["text"])

    # ------------------------------------------------------------------------------------------------ render
    def render(self):
        with self.lock:
            lay = Layout()
            lay.split_column(Layout(name="top", size=3), Layout(name="mid", ratio=3), Layout(name="bot", ratio=2))
            lay["mid"].split_row(Layout(name="dlg", ratio=3), Layout(name="board", ratio=2))
            lay["bot"].split_row(Layout(name="workers"), Layout(name="base"), Layout(name="final"))
            gap = "" if self.L is None or self.U is None else f"   gap {self._gap()}"
            lay["top"].update(Panel(Text(f"{self.title}   n = {self.n}   mode {self.mode}   t = {self.t:5.1f} s / "
                                         f"{self.budget:.0f} s   beta in [{self.L}, {self.U}]{gap}",
                                         style="bold"), title="kneser-agent"))
            lay["dlg"].update(Panel(Group(*self.dialogue[-16:]), title="agent dialogue"))
            bt = Table.grid(padding=(0, 1))
            for row in self.board[-10:]:
                bt.add_row(row[0], ("L" if row[1] == "lower" else "U"), str(row[2]), row[3])
            lay["board"].update(Panel(Group(bt, Text("\n" + "\n".join(self.events[-8:]), style="dim")),
                                      title="bound board"))
            wt = Table.grid(padding=(0, 1))
            for tid, w in list(self.workers.items())[:10]:
                a = w["args"]
                extra = f"p={a['p']} th={a.get('theta')}" if w["tool"] == "spectral" else ""
                wt.add_row(w["tool"], extra, f"{self.t - w['t0']:5.1f}s")
            lay["workers"].update(Panel(wt, title=f"workers ({len(self.workers)} active, {self.done_tools} done)"))
            bl = Table.grid(padding=(0, 1))
            for m, b in self.baselines.items():
                if b["status"] == "running":
                    bl.add_row(m, f"running {self.t - b['t0']:5.1f}s")
                else:
                    bl.add_row(m, f"{b['status']}", f"[{b.get('lower')}, {b.get('upper')}]", f"t={b.get('time')}")
            lay["base"].update(Panel(bl, title="baselines (120 s each)"))
            lay["final"].update(Panel(Text("\n".join(self.final[-8:])), title="verification / code"))
            return lay

    def _gap(self):
        from fractions import Fraction

        try:
            return str(Fraction(str(self.U)) - Fraction(str(self.L)))
        except (ValueError, ZeroDivisionError):
            return "?"


def replay(path: Path, speed: float = 1.0) -> None:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    v = View(path.stem)
    t0 = time.perf_counter()
    with Live(v.render(), refresh_per_second=8, screen=False) as live:
        for r in recs:
            while (time.perf_counter() - t0) * speed < r.get("t", 0):
                v.t = (time.perf_counter() - t0) * speed
                live.update(v.render())
                time.sleep(0.05)
            v.feed(r)
            live.update(v.render())
        live.update(v.render())


def live_run(name: str, seed: int, cores: list[int], bcores: list[int], budget: float, mode: str,
             baselines: bool) -> dict:
    from .baselines import STANDARD, run_method
    from .graphs import graph_id
    from .run import run_one

    gid = graph_id(name, seed)
    v = View(f"{gid}")
    holder = {}

    def listener(r):
        v.feed(r)
        holder.setdefault("trace", None)

    def run_baselines(trace_log):
        per = max(1, len(bcores) // 2)
        groups = [bcores[i * per:(i + 1) * per] for i in range(2)]

        def worker(methods, cs):
            for m in methods:
                trace_log("baseline_start", method=m)
                r = run_method(m, gid, cs)
                trace_log("baseline_result", method=m, **{k: r.get(k) for k in ("status", "lower", "upper",
                                                                                     "time", "wall")})
        half = (len(STANDARD) + 1) // 2
        ths = [threading.Thread(target=worker, args=(STANDARD[:half], groups[0])),
               threading.Thread(target=worker, args=(STANDARD[half:], groups[1]))]
        for th in ths:
            th.start()
        return ths

    result = {}

    def agent():
        def lst(r):
            v.feed(r)
            if r["kind"] == "run_start" and baselines and "ths" not in holder:
                tr = holder["tr"]
                holder["ths"] = run_baselines(lambda kind, **d: tr.log(kind, **d))

        import kagent.orchestrator as O
        orig = O.Trace.__init__

        def patched(self, *a, **k):
            orig(self, *a, **k)
            holder["tr"] = self
        O.Trace.__init__ = patched
        try:
            result["res"] = run_one(f"bench/graphs/{gid}", mode, budget, cores, seed=seed, tag="demo",
                                    listener=lst)
        finally:
            O.Trace.__init__ = orig

    th = threading.Thread(target=agent)
    th.start()
    with Live(v.render(), refresh_per_second=6, screen=False) as live:
        while th.is_alive() or any(t.is_alive() for t in holder.get("ths", [])):
            live.update(v.render())
            time.sleep(0.15)
            if not th.is_alive() and "code_done" not in holder:
                holder["code_done"] = True
                res = result.get("res", {})
                code = ROOT / "results" / "codes" / f"{gid}.json"
                if code.exists():
                    d = json.loads(code.read_text())
                    txt = (f"explicit code: {d['length']} transmissions over {d['field']} for {d['receivers']} "
                           f"receivers; decode simulation: {d['simulation'] and d['simulation'].get('ok')}")
                else:
                    txt = f"final bracket [{res.get('L')}, {res.get('U')}] (no proven code)"
                v.feed({"kind": "code_summary", "t": v.t, "text": txt})
                if holder.get("tr") is not None:
                    pass
        live.update(v.render())
    return result.get("res", {})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--cores", default="32-39")
    ap.add_argument("--baseline-cores", default="40-47")
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--mode", default="full")
    ap.add_argument("--no-baselines", action="store_true")
    a = ap.parse_args(argv)
    if a.replay:
        replay(Path(a.replay), a.speed)
        return 0
    from .run import parse_cores

    os.sched_setaffinity(0, parse_cores(a.cores) + parse_cores(a.baseline_cores))
    res = live_run(a.graph, a.seed, parse_cores(a.cores), parse_cores(a.baseline_cores), a.budget, a.mode,
                   not a.no_baselines)
    print(json.dumps({k: res.get(k) for k in ("graph", "L", "U", "proven", "field", "t_proof", "wall")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
