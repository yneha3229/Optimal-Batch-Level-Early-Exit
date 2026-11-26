"""results/baselines.jsonl + results/runs.jsonl -> RESULTS.md"""
import json
from pathlib import Path

import numpy as np

from baselines import standard_best

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
specs = json.loads((HERE / "specs.json").read_text())
base = {}
for l in (ROOT / "results" / "baselines.jsonl").read_text().splitlines():
    if l.strip():
        r = json.loads(l)
        base[r["spec"]] = r
runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if l.strip()]


def pick(spec, llm, fam):
    m = [r for r in runs if r["spec"] == spec and r["llm"] == llm and r["family_rule_allowed"] == fam
         and r["truth"].get("seed", 0) == 0]
    return m[-1] if m else None


def fmt(lo, hi, exact):
    f = lambda v: "-" if v is None else (f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))
    return f"**{f(hi)}**" if exact else f"[{f(lo)}, {f(hi)}]"


L = ["# Kneser agent benchmark", "",
     f"Container with 4 cores. Standard methods: one thread, {next(iter(base.values()))['limit']:g} s each, in "
     "separate processes; 'time' = when their bracket first closed, counting them as if they ran in parallel. "
     "Agent: 3 worker processes, 120 s budget, no LLM (deterministic plan). Every agent bound is re-checked by the "
     "independent verifier. Bold = proven exact (lower = upper).", "",
     "| graph | receivers | standard methods | time | Kushagra's script | time | **agent: generic + rule** | time | agent: generic only | time |",
     "|---|---|---|---|---|---|---|---|---|---|"]
cnt = {"std": 0, "kus": 0, "ag": 0, "gen": 0}
sp_std, sp_kus, n = [], [], 0
for spec in specs["benchmark"]:
    b, a, g = base.get(spec), pick(spec, "none", True), pick(spec, "none", False)
    if not b or "error" in b:
        continue
    n += 1
    lo, hi, bt = standard_best(b)
    k = b["kushagra_agent"]
    tag = " (exception)" if spec in specs["exceptions"] else " (control)" if spec in specs["controls"] else ""
    row = [spec + tag, str(b["receivers"]), fmt(lo, hi, bt is not None), f"{bt:.2f} s" if bt else "open",
           fmt(k["lower"], k["upper"], k["exact"]), f"{k['s']:.2f} s"]
    cnt["std"] += bt is not None
    cnt["kus"] += k["exact"]
    for r, key in ((a, "ag"), (g, "gen")):
        if r is None:
            row += ["-", "-"]
            continue
        ex = r["verdict"] == "proven"
        cnt[key] += ex
        row += [fmt(*r["bracket"], ex), f"{r['wall_s']:.2f} s"]
    if a and a["verdict"] == "proven":
        if bt:
            sp_std.append(bt / a["wall_s"])
        if k["exact"]:
            sp_kus.append(k["s"] / a["wall_s"])
    L.append("| " + " | ".join(row) + " |")
L += ["", f"- proven exact ({n} graphs): standard methods {cnt['std']}, Kushagra's script {cnt['kus']}, "
      f"agent generic + rule {cnt['ag']}, agent generic only {cnt['gen']}"]
if sp_std:
    L.append(f"- agent vs standard methods where both exact ({len(sp_std)}): median speed-up x{np.median(sp_std):.1f} "
             f"(min x{min(sp_std):.1f}, max x{max(sp_std):.1f})")
if sp_kus:
    L.append(f"- agent vs Kushagra's script where both exact ({len(sp_kus)}): median time ratio "
             f"x{np.median(sp_kus):.2f} (>1 means the agent is faster)")
chats = [r for r in runs if r["llm"] == "chat"]
if chats:
    L += ["", "## Runs with an LLM in the loop (chat backend: the assistant in the conversation answered as the agents)", "",
          "| graph | verdict | bracket | wall | of which LLM | compute | LLM calls | critic rounds | hypothesis | truth |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in chats:
        h = r.get("hypothesis") or {}
        L.append(f"| {r['graph']} | {r['verdict']} | {r['bracket']} | {r['wall_s']:.1f} s | {r['llm_s']:.1f} s | "
                 f"{r['compute_s']:.2f} s | {r['llm_calls']} | {r['critic_rounds']} | {h.get('name', '-')} {h.get('params', '')} | "
                 f"{r['truth'].get('family')} {r['truth'].get('n', '')},{r['truth'].get('k', '')} |")
(ROOT / "RESULTS.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
