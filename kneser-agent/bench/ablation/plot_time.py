"""Time to prove β per graph, by method -> time_to_prove.png (two panels, one per machine).

Left: everything run in this container, one core per pipeline (standard methods + Kushagra's script from
results/baselines_ctr.jsonl, the agent without LLM tagged "ctr", the blind Claude agents tagged "<config>-full").
Right: the earlier GPU-server runs (results/baselines.jsonl, agent without LLM, agent + Qwen 32B, prompts v3).
Timings from different machines never share a panel. Agent times are compute only (wall minus LLM seconds): the
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


def agent_pts(pred):
    """{receivers: compute seconds} over the full set, last matching row per spec wins."""
    pts = {}
    for r in runs:
        if r["spec"] in SPECS and pred(r) and proven(r):
            pts[r["spec"]] = (r["receivers"], max(r["compute_s"], 1e-3))
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
std_s, kus_s = baseline_pts(ROOT / "results" / "baselines.jsonl")
left = [("std", "standard methods", std_c), ("kus", "Kushagra's script", kus_c),
        ("none", "agent, no LLM", agent_pts(tagged("ctr")))]
left += [(c, lab, agent_pts(tagged(c + "-full"))) for c, lab in CLAUDE]
right = [("std", "standard methods", std_s), ("kus", "Kushagra's script", kus_s),
         ("none", "agent, no LLM", agent_pts(server_none)), ("qwen", "agent + Qwen 32B", agent_pts(server_qwen))]

TXT, SUB, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": SUB, "ytick.color": SUB})
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), dpi=200, sharex=True, sharey=True)
counts = {}
for ax, series, title in ((axes[0], left, "This container, one core per pipeline"),
                          (axes[1], right, "GPU server (earlier runs, different machine)")):
    for key, lab, pts in series:
        n = len(pts)
        counts[(title, key)] = n
        lab = f"{lab} ({n}/{len(SPECS)})"
        if not pts:
            ax.plot([], [], color=COL[key], label=lab)
            continue
        x, y = zip(*pts)
        if key == "std":
            ax.scatter(x, y, marker="s", s=48, color=COL[key], edgecolor="white", linewidth=1, zorder=4, label=lab)
        else:
            ax.plot(x, y, color=COL[key], lw=2, marker=MARK[key], ms=7, mec="white", mew=1, zorder=3, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="major", color=GRID, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, color=TXT, fontsize=11, loc="left")
    ax.set_xlabel("receivers (vertices of the side-information graph)", color=SUB)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=TXT)
axes[0].set_ylabel("seconds to prove β (log)", color=SUB)
fig.suptitle("Time to prove β (only graphs each method proves)", x=0.01, ha="left", color=TXT, fontsize=13,
             fontweight="bold")
note = ("Graphs: the 20 benchmark graphs plus K(20,4), K(17,5), K(23,4), seed 0; legend counts = graphs proven out "
        "of 23. Agent lines: compute time only (wall minus LLM seconds). Claude agents answer blind (one fresh agent "
        "per graph,\nsame isolation prompt); their LLM seconds include Claude Code agent overhead and are not drawn. "
        "Standard methods: best certified pair as if run in parallel, 60 s per method (120 s on the large graphs). "
        "Panels come from different machines: compare within a panel.")
fig.text(0.01, 0.01, note, color=SUB, fontsize=7.5)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
fig.savefig(HERE / "time_to_prove.png", facecolor="white")
print("wrote", HERE / "time_to_prove.png")
for k, v in counts.items():
    print(k, v)
