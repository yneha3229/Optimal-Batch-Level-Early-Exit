"""Regenerate three figures of the report from committed data (writes only into report/fig/).

  fig/speedup.png                      Part I summary chart (Table 4 values), over-budget bars hatched
  fig/time_to_prove_beamer.png         Figure 11: Claude agents (range) vs Kneser script vs our pipeline without LLM
  fig/time_to_prove_models_beamer.png  Figure 12: per model; Qwen counted over the 23 graphs it ran on

The two agent charts use the same data and definitions as kneser-agent/bench/ablation/plot_deck.py (time = wall
seconds until the certified bracket first closes, LLM seconds included; only proven graphs drawn).
Usage: ~/venv/bin/python report/make_figs.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
KA = HERE.parent / "kneser-agent"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(KA / "bench"))
from make_tables import T4  # noqa: E402
from baselines import standard_best  # noqa: E402


# ------------------------------------------------------------------ Part I summary chart (Figure 6)
def speedup_chart():
    methods = [("Per-sample threshold", "#AFC0DF", "per-sample threshold"),
               ("Unified exit", "#E8A25E", "unified exit"),
               ("Majority vote", "#D9CA9F", "majority vote"),
               ("Random search (150 rules)", "#7191C9", "random search"),
               ("Cost rule + exit plan (ours)", "#0F3F91", "cost rule + exit plan (ours)")]
    cols = [(0, "len 16, B = 8"), (2, "len 16, B = 64"), (8, "len 128, B = 64")]
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=250)
    w = 0.16
    for i, (row, color, _) in enumerate(methods):
        vals = T4[row].split()
        for j, (c, _) in enumerate(cols):
            v = vals[c]
            over = v.endswith("*")
            x = j + (i - 2) * w
            ax.bar(x, float(v.rstrip("*")), width=w * 0.92, color=color, hatch="////" if over else None,
                   edgecolor="white" if over else color, linewidth=0, zorder=3)
            if i == len(methods) - 1:
                ax.text(x, float(v.rstrip("*")) + 0.04, f"{float(v.rstrip('*')):.2f}×", ha="center", va="bottom", fontsize=10.5,
                        color=color, fontweight="bold")
            elif over:
                ax.text(x, float(v.rstrip("*")) + 0.04, "*", ha="center", va="bottom", fontsize=12, color="#555555")
    ax.set_xticks(range(len(cols)), [lab for _, lab in cols])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylim(0, 2.95)
    ax.grid(axis="y", color="#dcdde3", zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Patch(facecolor=c, label=lab) for _, c, lab in methods]
    handles.append(Patch(facecolor="#bbbbbb", hatch="////", edgecolor="white",
                         label="* held-out drop above one point"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "speedup.png", facecolor="white")
    plt.close(fig)
    print("wrote fig/speedup.png")


# ------------------------------------------------------------------ agent data (as in plot_time.py)
specs = json.loads((KA / "bench" / "specs.json").read_text())
SPECS = specs["benchmark"] + specs["large"] + specs.get("xlarge", [])
runs = [json.loads(l) for l in (KA / "results" / "runs.jsonl").read_text().splitlines() if l.strip()]


def base_name(spec):
    return spec.replace("(", "_").replace(")", "").replace(",", "_") + "_s0"


def proven(r):
    return r["verdict"] == "proven" and r["bracket"][0] == r["bracket"][1] and all(r["checks"].values())


def cert_seconds(r):
    f = KA / "traces" / (r.get("trace") or "")
    if r.get("trace") and f.is_file():
        for line in f.open():
            e = json.loads(line)
            if e.get("ev") == "event" and e.get("name") == "bracket_closed":
                return e["t"]
    return r["wall_s"]


def agent_pts(pred):
    pts = {}
    for r in runs:
        if r["spec"] in SPECS and pred(r) and proven(r):
            pts[r["spec"]] = (r["receivers"], max(cert_seconds(r), 1e-3))
    return sorted(pts.values())


def tagged(tag):
    return lambda r: r["graph"] == base_name(r["spec"]) + "_" + tag


def server_qwen(r):
    return r["llm"].startswith("Qwen2.5-Coder-32B-Instruct-AWQ") and r.get("prompt_version") == "v3" \
        and r["graph"] == base_name(r["spec"])


def qwen_ran():
    return len({r["spec"] for r in runs if r["spec"] in SPECS and server_qwen(r)})


def baseline_pts(path):
    std, kus = {}, {}
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


std_c, kus_c = baseline_pts(KA / "results" / "baselines_ctr.jsonl")
_s, _k = baseline_pts(KA / "results" / "baselines_ctr_xl.jsonl")
KUS = sorted(dict((x, (x, y)) for x, y in kus_c + _k).values())
STD = sorted(std_c + _s)
NOLLM = agent_pts(tagged("ctr"))          # our pipeline, no LLM, default plan, same container
CFG = ["sonnet-med", "opus-low", "opus-med", "opus-high"]
NAMES = {"sonnet-med": "Sonnet 5.5 medium", "opus-low": "Opus 5.5 low", "opus-med": "Opus 5.5 medium",
         "opus-high": "Opus 5.5 high"}
NOLLM_COL, KUS_COL = "#00a383", "#e0631a"


def style(ax, sub="#555555"):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(170, 19000)
    ax.grid(True, which="major", color="#e3e3e3", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("receivers (vertices of the graph)", color=sub)
    ax.set_ylabel("seconds to prove β (log), LLM time included", color=sub)


def ttp_chart():
    per = {}
    for r in runs:
        for c in CFG:
            if r["spec"] in SPECS and tagged(c + "-full")(r) and proven(r):
                per.setdefault(r["spec"], {})[c] = (r["receivers"], cert_seconds(r))
    band = sorted((v[CFG[0]][0], sorted(t for _, t in v.values())) for v in per.values() if len(v) == len(CFG))
    X = [b[0] for b in band]
    LO, HI = [b[1][0] for b in band], [b[1][-1] for b in band]
    MED = [sum(b[1][1:3]) / 2 for b in band]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=300)
    ax.fill_between(X, LO, HI, color="#d4d4f2", lw=0, zorder=1)
    ax.plot(X, MED, color="#3333b3", lw=2.2, marker="o", ms=4, mec="white", mew=0.7, zorder=3,
            label=f"our pipeline + Claude (4 settings: median, range shaded): {len(X)}/{len(SPECS)}")
    x, y = zip(*NOLLM)
    ax.plot(x, y, color=NOLLM_COL, lw=1.8, marker="o", ms=4, mec="white", mew=0.7, zorder=4,
            label=f"our pipeline, no LLM (default plan): {len(NOLLM)}/{len(SPECS)}")
    x, y = zip(*KUS)
    ax.plot(x, y, color=KUS_COL, lw=2.2, marker="D", ms=4.4, mec="white", mew=0.7, zorder=4,
            label=f"Kneser script, no LLM: {len(KUS)}/{len(SPECS)}")
    sx, sy = zip(*STD)
    ax.scatter(sx, sy, marker="s", s=36, color="#8a8a8a", edgecolor="white", lw=0.8, zorder=5,
               label=f"standard methods (60 s each): {len(STD)}/{len(SPECS)}")
    style(ax)
    ax.set_ylim(0.01, 1500)
    kd = dict(KUS)
    i24 = X.index(10626)
    ax.annotate(f"10,626 receivers\nClaude agents {LO[i24]:.0f}–{HI[i24]:.0f} s vs {kd[10626]:.0f} s",
                xy=(10626, MED[i24]), xytext=(2600, 900), fontsize=8, va="top", color="#191919",
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8))
    ax.legend(loc="lower right", frameon=False, fontsize=7.2, title="proven (of 25):", title_fontsize=7.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "time_to_prove_beamer.png", facecolor="white")
    plt.close(fig)
    print("wrote fig/time_to_prove_beamer.png")


def models_chart():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=300)
    col = {"sonnet-med": "#ee7fac", "opus-low": "#b3a5f2", "opus-med": "#8a70ea", "opus-high": "#3333b3"}
    for c in CFG:
        pts = agent_pts(tagged(c + "-full"))
        x, y = zip(*pts)
        ax.plot(x, y, color=col[c], lw=1.6, marker="o", ms=3, label=f"pipeline + {NAMES[c]} (container): {len(pts)}/25")
    q = agent_pts(server_qwen)
    x, y = zip(*q)
    ax.plot(x, y, color="#f2a600", lw=1.4, ls="--", marker="^", ms=3,
            label=f"pipeline + Qwen2.5-Coder-32B (GPU server, prompts v3): {len(q)}/{qwen_ran()}")
    x, y = zip(*NOLLM)
    ax.plot(x, y, color=NOLLM_COL, lw=1.4, marker="o", ms=3, label=f"pipeline, no LLM, default plan (container): {len(NOLLM)}/25")
    x, y = zip(*KUS)
    ax.plot(x, y, color=KUS_COL, lw=1.8, marker="D", ms=3, label=f"Kneser script, no LLM (container): {len(KUS)}/25")
    style(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=7, title="proven / graphs run",
              title_fontsize=7)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "time_to_prove_models_beamer.png", facecolor="white")
    plt.close(fig)
    print("wrote fig/time_to_prove_models_beamer.png")


if __name__ == "__main__":
    speedup_chart()
    ttp_chart()
    models_chart()
