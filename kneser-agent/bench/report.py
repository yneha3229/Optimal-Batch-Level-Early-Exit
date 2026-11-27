"""results/baselines.jsonl + results/runs.jsonl + traces/ -> RESULTS.md

    cd kneser-agent/bench && python report.py

RESULTS.md = bench/analysis.md (hand-written analysis, if present) + the computed summary + the tables below.
Every number in the summary and the tables is computed here from the result files; nothing is typed in by hand.
"""
import json
from pathlib import Path

import numpy as np

from baselines import standard_best

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
specs = json.loads((HERE / "specs.json").read_text())
SERVER = (ROOT / "results" / "baselines.jsonl").exists()
bfile = ROOT / "results" / ("baselines.jsonl" if SERVER else "baselines_container.jsonl")
base = {}
for line in bfile.read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        base[r["spec"]] = r
runs = [json.loads(line) for line in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if line.strip()]

LLMS = [("none", "agent, no LLM"), ("none+fast", "agent, no LLM, fast path"),
        ("Qwen2.5-Coder-32B-Instruct-AWQ", "agent + Qwen 32B"), ("Qwen3-4B-Instruct-2507", "agent + Qwen3 4B")]
LLMS = [(k, v) for k, v in LLMS if any(r["llm"] == k.split("+")[0] and (k != "none+fast" or r.get("fast_path"))
                                        for r in runs)]
LAB = dict(LLMS)
BENCH = specs["benchmark"]
LARGE = specs.get("large", [])


def pick(spec, llm, fam):
    """Last benchmark run (seed 0; for LLM runs v4 (after the plan-results fix) if present, else v3) of this graph / LLM / rule setting."""
    fast = llm == "none+fast"                       # the opt-in deterministic fast path is its own configuration
    base_llm = "none" if fast else llm
    m = [r for r in runs if r["spec"] == spec and r["llm"] == base_llm and r["family_rule_allowed"] == fam
         and bool(r.get("fast_path")) == fast and r["truth"].get("seed", 0) == 0
         and (base_llm.startswith("none") or r.get("prompt_version") in ("v3", "v4"))
         and not r["graph"].endswith("_nojsonmode")]
    m.sort(key=lambda r: r.get("prompt_version") or "")         # stable: v4 (after the plan-results fix) wins
    return m[-1] if m else None


def f(v):
    return "-" if v is None else (f"{v:.2f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))


def fmt(lo, hi, exact):
    return f"**{f(hi)}**" if exact else f"[{f(lo)}, {f(hi)}]"


def trace_counts(r):
    p = ROOT / "traces" / (r.get("trace") or f"{r['graph']}_{r['llm']}.jsonl")
    c = {"llm_request": 0, "llm_invalid": 0, "llm_fallback": 0, "challenge": 0, "safety_net": 0, "hyps": []}
    if not p.exists():
        return c
    for line in p.read_text().splitlines():
        e = json.loads(line)
        if e["ev"] in ("llm_request", "llm_invalid", "llm_fallback"):
            c[e["ev"]] += 1
        if e["ev"] == "event" and e.get("name") == "safety_net":
            c["safety_net"] += 1
        if e["ev"] == "llm_reply" and e["agent"] == "critic" and e["reply"].get("verdict") == "challenge":
            c["challenge"] += 1
        if e["ev"] == "llm_reply" and e["agent"] == "solver":
            c["hyps"].append(e["reply"].get("family_hypothesis") or {})
    return c


def hyp_ok(h, truth):
    """Does a family hypothesis match the hidden truth? Kneser: the name says Kneser (not perturbed) and (n, k)
    match. Kneser with deleted edges: the name says perturbed/deleted. Random regular: no family claimed."""
    name = str((h or {}).get("name", "")).lower()
    pr = (h or {}).get("params") or {}
    pert = any(w in name for w in ("perturb", "delet", "noisy", "near", "not an exact"))
    fam = truth.get("family", "")
    if fam == "kneser":
        try:
            return "kneser" in name and not pert and (int(pr.get("n")), int(pr.get("k"))) == (truth["n"], truth["k"])
        except (TypeError, ValueError):
            return False
    if fam == "kneser with deleted edges":
        return pert
    return name.strip() in ("", "none", "unknown") or "random" in name


# ------------------------------------------------------------------------------------------------ main table
heads = ["graph", "receivers", "standard methods", "time", "Kushagra's script", "time"]
for k, lab in LLMS:
    heads += [f"{lab}" + (" (rule)" if not k.startswith("none") else ""), "time"]
    if not k.startswith("none"):
        heads += [f"{lab} (generic only)", "time"]
L = ["| " + " | ".join(heads) + " |", "|" + "---|" * len(heads)]
cnt = {"std": 0, "kus": 0}
exact = {}
speed = {}
losses = []
n = 0
for spec in BENCH + LARGE:
    b = base.get(spec)
    if (not b or "error" in b) and not any(pick(spec, k, True) for k, _ in LLMS):
        continue
    n += spec in BENCH
    tag = " (exception)" if spec in specs["exceptions"] else " (control)" if spec in specs["controls"] else (
        " (large)" if spec in LARGE else "")
    bt = hi = None
    k = {"exact": False, "s": None}
    if b and "error" not in b:
        lo, hi, bt = standard_best(b)
        k = b["kushagra_agent"]
        row = [spec + tag, str(b["receivers"]),
               fmt(lo, hi, bt is not None) if (lo is not None or hi is not None) else "n/a",
               f"{bt:.2f} s" if bt else ("n/a" if lo is None and hi is None else "open"),
               fmt(k["lower"], k["upper"], k["exact"]) if k["upper"] is not None else "n/a",
               f"{k['s']:.2f} s" if k["exact"] else ("n/a" if k["upper"] is None else "not exact")]
        cnt["std"] += bt is not None and spec in BENCH
        cnt["kus"] += bool(k["exact"]) and spec in BENCH
    else:
        row = [spec + tag, "-", "n/a", "n/a", "n/a", "n/a"]
    for llm, _ in LLMS:
        for fam in ((True,) if llm.startswith("none") else (True, False)):
            r = pick(spec, llm, fam)
            if r is None:
                row += ["-", "-"]
                continue
            if row[1] == "-":
                row[1] = str(r["receivers"])
            ex = r["verdict"] == "proven"
            if spec in BENCH:
                exact[(llm, fam)] = exact.get((llm, fam), 0) + ex
            row += [fmt(*r["bracket"], ex) + (" INVALID" if r["verdict"] == "invalid" else ""),
                    f"{r['wall_s']:.2f} s" + (f" ({r['compute_s']:.2f})" if not llm.startswith("none") else "")]
            if not fam or spec not in BENCH:
                continue
            agent_t = r["compute_s"] if not llm.startswith("none") else r["wall_s"]
            if ex and bt:
                speed.setdefault((llm, "std"), []).append(bt / agent_t)
            if ex and k["exact"]:
                speed.setdefault((llm, "kus"), []).append(k["s"] / agent_t)
                speed.setdefault((llm, "kus_wall"), []).append(k["s"] / r["wall_s"])
            who = [w for w, ok in (("standard methods", bt), ("Kushagra's script", k["exact"])) if ok]
            if not ex and who:
                losses.append((spec, llm, f"not proven (bracket {r['bracket']}); {' and '.join(who)} proved it"))
            elif not ex and hi is not None and r["bracket"][1] > hi + 1e-9:
                losses.append((spec, llm, f"upper bound {r['bracket'][1]} vs {f(hi)} from the standard methods"))
            if ex and bt and bt < agent_t:
                losses.append((spec, llm, f"slower than the standard methods ({agent_t:.2f} s vs {bt:.2f} s)"))
            if ex and k["exact"] and k["s"] < agent_t:
                losses.append((spec, llm, f"slower than Kushagra's script ({agent_t:.2f} s vs {k['s']:.2f} s)"))
    L.append("| " + " | ".join(row) + " |")

# ------------------------------------------------------------------------------------------------ summary block
S = ["## Computed summary (benchmark graphs only; generated by bench/report.py)", "",
     f"- {n} graphs. Proven exact: standard methods {cnt['std']}, Kushagra's script {cnt['kus']}, "
     + ", ".join(f"{lab} {exact.get((k, True), 0)}" + ("" if k.startswith("none") else f" (generic only "
                 f"{exact.get((k, False), 0)})") for k, lab in LLMS)]
WHAT = {"std": "standard methods", "kus": "Kushagra's script", "kus_wall": "Kushagra's script, LLM seconds included"}
for (llm, other), v in sorted(speed.items()):
    if other == "kus_wall" and llm.startswith("none"):
        continue
    S.append(f"- {LAB[llm]} vs {WHAT[other]}, both exact ({len(v)} graphs): median time ratio x{np.median(v):.2f} "
             f"(min x{min(v):.2f}, max x{max(v):.2f}; > 1 = the agent is faster"
             + ("; agent time = compute time without LLM seconds" if not llm.startswith("none") and other != "kus_wall" else "")
             + ")")
for llm, lab in LLMS:
    if llm.startswith("none"):
        continue
    rs = [r for s in BENCH for fam in (True, False) for r in [pick(s, llm, fam)] if r]
    if not rs:
        continue
    tc = [trace_counts(r) for r in rs]
    req, inv, fb = (sum(c[x] for c in tc) for x in ("llm_request", "llm_invalid", "llm_fallback"))
    fin = [hyp_ok(r.get("hypothesis"), r["truth"]) for r in rs]
    first = [hyp_ok(c["hyps"][0] if c["hyps"] else None, r["truth"]) for r, c in zip(rs, tc)]
    S.append(f"- {lab}: {len(rs)} runs (both rule settings); LLM seconds per run median "
             f"{np.median([r['llm_s'] for r in rs]):.1f} (max {max(r['llm_s'] for r in rs):.1f}); LLM calls "
             f"{sum(r['llm_calls'] for r in rs)}; invalid replies {inv} of {req} requests, fallbacks to the default "
             f"{fb}; Critic challenges {sum(c['challenge'] for c in tc)}; safety net ran in "
             f"{sum(c['safety_net'] > 0 for c in tc)} runs")
    S.append(f"  - family guess vs bench/truth.json: first Solver reply right in {sum(first)}/{len(rs)}, final plan "
             f"right in {sum(fin)}/{len(rs)} (wrong -> right during the dialogue: "
             f"{sum(1 for a, b in zip(first, fin) if b and not a)}; right -> wrong: "
             f"{sum(1 for a, b in zip(first, fin) if a and not b)})")
    for fam, flab in ((True, "rule allowed"), (False, "generic only")):
        none_ex = {s for s in BENCH if (r := pick(s, "none", True)) and r["verdict"] == "proven"}
        llm_ex = {s for s in BENCH if (r := pick(s, llm, fam)) and r["verdict"] == "proven"}
        both = sorted(none_ex & llm_ex)
        if not both:
            continue
        ratio = [pick(s, "none", True)["wall_s"] / pick(s, llm, fam)["compute_s"] for s in both]
        S.append(f"  - {flab} vs no LLM: proven only with the LLM: {sorted(llm_ex - none_ex) or 'none'}; only "
                 f"without: {sorted(none_ex - llm_ex) or 'none'}; both proven ({len(both)}): no-LLM wall / LLM-run "
                 f"compute time, median x{np.median(ratio):.2f} (> 1 = the LLM run computes less)")
if losses:
    S += ["", "### Every loss of an agent run (rule allowed) to a baseline", ""]
    S += [f"- {s} | {LAB[lm]}: {w}" for s, lm, w in losses]

# ------------------------------------------------------------------------------------------------ LLM table
X = []
llm_runs = [r for r in runs if r["llm"] != "none"]
if llm_runs:
    X += ["", "## Every run with an LLM in the loop", "",
          "llm = chat: an assistant answered as the agents through the file-based chat backend (reference runs); "
          "otherwise the named model on the server (vLLM, GPU 4). prompts: v1 = branch prompts; v2, v3 = after the "
          "fixes described in MERGE_NOTES.md and the commit messages.", "",
          "| graph | llm | prompts | rule | verdict | bracket | wall | LLM s | compute | calls | invalid | Critic challenges | final hypothesis | truth |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in llm_runs:
        h = r.get("hypothesis") or {}
        c = trace_counts(r)
        X.append(f"| {r['graph']} | {r['llm']} | {r.get('prompt_version') or '-'} | "
                 f"{'yes' if r['family_rule_allowed'] else 'no'} | {r['verdict']} | {r['bracket']} | "
                 f"{r['wall_s']:.1f} s | {r['llm_s']:.1f} | {r['compute_s']:.2f} s | {r['llm_calls']} | "
                 f"{c['llm_invalid']} | {c['challenge']} | {h.get('name', '-')} {h.get('params', '')} | "
                 f"{r['truth'].get('family')} {r['truth'].get('n', '')},{r['truth'].get('k', '')} |")

where = ("GPU server (64 cores; baselines pinned to cores 16-27, 3 graphs at a time; agent runs pinned to cores "
         "28-31, 3 worker processes; other users' jobs were running on the machine)." if SERVER
         else "Container with 4 cores.")
lim = next(iter(base.values()))["limit"] if base else None
head = ["# Kneser agent benchmark", "",
        f"{where} Standard methods: one thread each, {f(lim)} s each, in separate processes; 'time' = when their "
        "bracket first closed, counting them as if they ran in parallel. Kushagra's script: recognise (n, k), rule "
        "field, one rank, greedy star; falls back to the general minrank agent. Agent: budget 300 s; time = wall time "
        "of the run (graph loading and the one-time numba warm-up excluded); for LLM runs the compute time without "
        "LLM seconds is in brackets. Every agent bound is re-checked by the independent verifier. Bold = proven exact "
        "(lower = upper). '(rule)' = the family rule may be used (only an LLM hypothesis can invoke it, so without an "
        "LLM it makes no difference); '(generic only)' = --no-family-rule.", ""]
# ------------------------------------------------------------------------------------------------ model ablation
A = []
sc = HERE / "ablation" / "scores.jsonl"
if sc.exists():
    rows = [json.loads(line) for line in sc.read_text().splitlines() if line.strip()]
    A += ["", "## Model ablation: blind agents on the three reference cases (bench/ablation/)", "",
          "Each run gives the Solver, Critic and Verifier roles to one model answering blind (bench/ablation/README.md). "
          "Claude rows: fixed code (9081f70). Qwen rows: GPU server, prompts v3, code before 9081f70. One run per cell; "
          "false objections labelled by hand (bench/ablation/false_objections.json).", "",
          "![model ablation](bench/ablation/model_ablation.png)", "",
          "| graph | model | verdict | bracket | LLM calls | invalid | fallbacks | (n, k) in first reply | Critic objections | false | safety net | codes tried |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        A.append(f"| {r['graph']} | {r['config']} | {r['verdict']} | {r['bracket']} | {r['llm_calls']} | {r['invalid']} | "
                 f"{r['fallback']} | {'yes' if r['first_reply_names_nk'] else 'no'} | {r['critic_objections']} | "
                 f"{r['false_objections']} | {'yes' if r['safety_net'] else 'no'} | {r['codes_tried']} |")
analysis = (HERE / "analysis.md").read_text().strip().splitlines() + [""] if (HERE / "analysis.md").exists() else []
(ROOT / "RESULTS.md").write_text("\n".join(head + analysis + S + ["", "## Per-graph results", ""] + L + A + X) + "\n")
print("\n".join(S))
