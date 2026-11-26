"""Terminal view of a run trace (rich): agent dialogue, bound board, events, verdict. A light port of the server
version's demo.py (server_version/kagent/demo.py) to this pipeline's trace format.

    python -m kagent.view traces/K_16_4_s3_Qwen2.5-Coder-32B-Instruct-AWQ.jsonl            # final picture
    python -m kagent.view traces/K_16_4_s3_Qwen2.5-Coder-32B-Instruct-AWQ.jsonl --replay 1  # replay at real speed
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

COL = {"solver": "cyan", "critic": "magenta", "verifier": "green"}


def _say(e) -> Text | None:
    ev = e["ev"]
    t = Text(f"{e['t']:7.1f}s ", style="dim")
    if ev == "llm_reply":
        r, a = e["reply"], e["agent"]
        t.append(f"{a.upper():9s}", style="bold " + COL.get(a, "white"))
        if a == "critic":
            t.append(r.get("verdict", "?") + ": ", style="bold")
            t.append(" | ".join(r.get("objections", []))[:400])
            if r.get("requested_checks"):
                t.append("  checks: " + ", ".join(c.get("tool", "?") for c in r["requested_checks"]), style="yellow")
        elif a == "verifier":
            t.append(r.get("analysis", "")[:500])
        else:
            h = r.get("family_hypothesis") or {}
            t.append(f"{h.get('name')} {h.get('params', '')} conf {h.get('confidence')}; plan: ", style="bold")
            t.append(", ".join(f"{s.get('tool')}{s.get('args') or ''}" for s in r.get("plan", [])))
        return t
    if ev in ("llm_invalid", "llm_fallback"):
        t.append(f"{e['agent'].upper():9s}", style="bold red")
        t.append(f"{ev}: {e.get('error', 'default plan used')}"[:300], style="red")
        return t
    if ev == "checks":
        t.append("TOOLS    ", style="bold yellow")
        t.append("; ".join(f"{c.get('tool')} -> {json.dumps(c.get('result'))[:80]}" for c in e["results"]))
        return t
    return None


def render(events, upto=None):
    evs = [e for e in events if upto is None or e["t"] <= upto]
    head = next((e for e in evs if e["ev"] == "input"), {})
    dialog = [x for x in (_say(e) for e in evs) if x is not None][-14:]
    board = Table(show_header=True, header_style="bold", expand=True)
    for c in ("t", "side", "value", "tool", "bracket"):
        board.add_column(c)
    for e in [e for e in evs if e["ev"] == "board"][-8:]:
        board.add_row(f"{e['t']:.2f}", e["side"], str(e["value"]), e["tool"], str(e["bracket"]))
    ev = Text()
    for e in [e for e in evs if e["ev"] in ("event", "rule", "plan_accepted", "plan_item_ignored", "tool_error")][-8:]:
        name = e.get("name") or e["ev"]
        ev.append(f"{e['t']:7.1f}s {name} ", style="bold")
        ev.append(json.dumps({k: v for k, v in e.items() if k not in ("t", "ev", "name", "hypothesis")})[:150] + "\n")
    v = next((e for e in evs if e["ev"] == "verdict"), None)
    vt = Text("running ..." if v is None else f"{v['verdict'].upper()}  bracket {v['bracket']}  checks {v['checks']}",
              style="bold green" if v and v["verdict"] == "proven" else "bold yellow")
    title = f"{head.get('graph', '?')}  N = {head.get('receivers', '?')}  llm = {head.get('llm', '?')}"
    return Panel(Group(Panel(Group(*dialog) if dialog else Text("(no LLM)"), title="agent dialogue"),
                       Panel(board, title="bound board (certified entries only)"), Panel(ev, title="scheduler events"),
                       Panel(vt, title="verdict (computed from the checks)")), title=title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--replay", type=float, default=0, help="replay speed (1 = real time); 0 = final picture only")
    a = ap.parse_args()
    events = [json.loads(line) for line in Path(a.trace).read_text().splitlines() if line.strip()]
    if not a.replay:
        Console().print(render(events))
        return
    t0, end = time.perf_counter(), events[-1]["t"]
    with Live(render(events, 0), refresh_per_second=8) as live:
        while (now := (time.perf_counter() - t0) * a.replay) < end:
            live.update(render(events, now))
            time.sleep(0.1)
        live.update(render(events))


if __name__ == "__main__":
    main()
