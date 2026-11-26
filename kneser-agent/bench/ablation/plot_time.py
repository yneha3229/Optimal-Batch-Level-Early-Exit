"""Time to prove β per graph, by method -> time_to_prove.png (LLM agents: compute + LLM seconds).

One chart; each legend entry says in brackets where it ran. Container, one core per pipeline: standard methods +
Kushagra's Kneser agent, no LLM (minrank-agent kneser_agent.py; results/baselines_ctr.jsonl), the agent without LLM (tag "ctr"), the blind Claude agents (tag
"<config>-full"). GPU server: agent + Qwen 32B (vLLM, prompts v3), drawn dashed. Agent times are compute only (wall minus LLM seconds): the
Claude models answer through Claude Code agents, so their LLM seconds include agent overhead; Qwen's are vLLM.
Only graphs a method proves (bracket closed, all checks passed) are drawn.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "bench"))
from baselines import standard_best  # noqa: E402

specs = json.loads((ROOT / "bench" / "specs.json").read_text())
SPECS = specs["benchmark"] + specs["large"]
runs = [json.loads(l) for l in (ROOT / "results" / "runs.jsonl").read_text().splitlines() if l.strip()]


def base_name(spec):
    return spec.replace("(", "_").replace(")", "").replace(",", "_") + "_s0"


def proven(r):
    return r["verdict"] == "proven" and r["bracket"][0] == r["bracket"][1] and all(r["checks"].values())


def agent_pts(pred, field="compute_s", only_proven=True):
    """[(receivers, seconds)] over the full set, last matching row per spec wins."""
    pts = {}
    for r in runs:
        if r["spec"] in SPECS and pred(r) and (proven(r) or not only_proven):
            pts[r["spec"]] = (r["receivers"], max(r[field], 1e-3))
    return sorted(pts.values())


def baseline_pts(path):
    std, kus = {}, {}
    if not path.exists():
        return [], []
    for line in path.read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("spec") not in SPECS or "error" in r:
            continue
        lo, hi, t = standard_best(r)
        if t is not None:
            std[r["spec"]] = (r["receivers"], t)
        k = r["kushagra_agent"]
        if k.get("exact"):
            kus[r["spec"]] = (r["receivers"], max(k["s"], 1e-3))
    return sorted(std.values()), sorted(kus.values())


def tagged(tag):
    return lambda r: r["graph"] == base_name(r["spec"]) + "_" + tag


def server_none(r):
    return r["llm"] == "none" and r["graph"] == base_name(r["spec"]) and r["family_rule_allowed"] \
        and not r.get("fast_path")


def server_qwen(r):
    return r["llm"].startswith("Qwen2.5-Coder-32B-Instruct-AWQ") and r.get("prompt_version") == "v3" \
        and r["graph"] == base_name(r["spec"])


CLAUDE = [("sonnet-med", "agent + Sonnet 5.5 medium"), ("opus-low", "agent + Opus 5.5 low"),
          ("opus-med", "agent + Opus 5.5 medium"), ("opus-high", "agent + Opus 5.5 high")]
# fixed categorical order (blue, orange, aqua, yellow, magenta); Opus efforts share one violet ramp, light -> dark
COL = {"std": "#257de0", "kus": "#f47230", "none": "#00b894", "qwen": "#f2a600", "sonnet-med": "#ee7fac",
       "opus-low": "#b3a5f2", "opus-med": "#8a70ea", "opus-high": "#4b2fc0"}
MARK = {"kus": "D", "none": "o", "qwen": "^", "sonnet-med": "P", "opus-low": "v", "opus-med": "s", "opus-high": "X"}

std_c, kus_c = baseline_pts(ROOT / "results" / "baselines_ctr.jsonl")
CTR, GPU = "container, 1 core", "GPU server"
series = [("std", f"standard methods [{CTR}]", std_c), ("kus", f"Kneser agent (no LLM) [{CTR}]", kus_c),
          ("none", f"multi-agent pipeline, no LLM [{CTR}]", agent_pts(tagged("ctr")))]
# LLM agents: total time = compute + LLM seconds (wall_s); the label says what the LLM part is
series += [(c, f"{lab} (compute + LLM) [{CTR}]", agent_pts(tagged(c + "-full"), "wall_s"))
           for c, lab in CLAUDE]
series += [("qwen", f"agent + Qwen 32B (compute + LLM) [{GPU}]", agent_pts(server_qwen, "wall_s"))]

TXT, SUB, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": SUB, "ytick.color": SUB})
fig, ax = plt.subplots(figsize=(12.5, 7.6), dpi=200)
for key, lab, pts in series:
    med = sorted(v for _, v in pts)[len(pts) // 2] if pts else 0
    lab = f"{lab}: {len(pts)}/{len(SPECS)} proven, median {med:.2g} s"
    print(lab)
    if not pts:
        ax.plot([], [], color=COL[key], label=lab)
        continue
    x, y = zip(*pts)
    if key == "std":
        ax.scatter(x, y, marker="s", s=48, color=COL[key], edgecolor="white", linewidth=1, zorder=4, label=lab)
    else:
        ax.plot(x, y, color=COL[key], lw=2, marker=MARK[key], ms=7, mec="white", mew=1, zorder=3, label=lab,
                ls="--" if key == "qwen" else "-")
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid(True, which="major", color=GRID, zorder=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.set_xlabel("receivers (vertices of the side-information graph)", color=SUB)
ax.set_ylabel("seconds to prove β (log)", color=SUB)
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_ylim(top=150)
ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=TXT, ncol=1)
ax.set_title("Time to prove β, LLM seconds included (only graphs each method proves)", loc="left", color=TXT,
             fontsize=13, fontweight="bold", pad=12)
note = ("Where each line ran is in brackets. Graphs: the 20 benchmark graphs plus K(20,4), K(17,5), K(23,4), seed 0. "
        "LLM agents: total time = compute + LLM seconds (Solver, Critic, Verifier);\nmethods without an LLM: compute "
        "only. Claude LLM seconds are wall time of a blind Claude Code agent (one fresh agent per graph, tool calls and "
        "file writes included), so they overstate pure model time;\nQwen 32B LLM seconds are vLLM on the GPU server "
        "(prompts v3; dashed: different machine). Standard methods: best certified pair as if run in parallel, 60 s per "
        "method (120 s on the large graphs).")
fig.text(0.01, 0.01, note, color=SUB, fontsize=7.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(HERE / "time_to_prove.png", facecolor="white")
print("wrote", HERE / "time_to_prove.png")
