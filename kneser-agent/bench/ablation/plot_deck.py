"""Presentation chart -> time_to_prove_deck.png: time to certified beta vs receivers (container, one core).

Kneser agent (no LLM) against the LLM agent with the four blind Claude configurations drawn as one range (min-max,
median line), plus the part of the agent's time spent waiting for the LLM (median), which stays roughly flat while
compute grows. Standard methods are drawn as points (they prove only small graphs). Only proven graphs are drawn.
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
sys.path.insert(0, str(HERE))
import plot_time as P  # noqa: E402  (data helpers; importing it also redraws time_to_prove.png)

CFG = ["sonnet-med", "opus-low", "opus-med", "opus-high"]


def llm_before_close(r):
    """LLM seconds spent before the certified bracket first closed (sum of request -> reply intervals)."""
    close, llm, last = P.cert_seconds(r), 0.0, None
    for line in (ROOT / "traces" / r["trace"]).open():
        e = json.loads(line)
        if e["t"] > close:
            break
        if e.get("ev") == "llm_request":
            last = e["t"]
        elif e.get("ev") == "llm_reply" and last is not None:
            llm, last = llm + e["t"] - last, None
    return llm


per = {}   # spec -> {cfg: (receivers, total, llm)}
for r in P.runs:
    for c in CFG:
        if r["spec"] in P.SPECS and P.tagged(c + "-full")(r) and P.proven(r):
            per.setdefault(r["spec"], {})[c] = (r["receivers"], P.cert_seconds(r), llm_before_close(r))
band = sorted((v[CFG[0]][0], min(t for _, t, _ in v.values()), sorted(t for _, t, _ in v.values())[1:3],
               max(t for _, t, _ in v.values()), sorted(l for _, _, l in v.values())[1:3])
              for v in per.values() if len(v) == len(CFG))
X = [b[0] for b in band]
LO, HI = [b[1] for b in band], [b[3] for b in band]
MED = [sum(b[2]) / 2 for b in band]
LLM = [sum(b[4]) / 2 for b in band]

std_c, kus_c = P.baseline_pts(ROOT / "results" / "baselines_ctr.jsonl")
_s, _k = P.baseline_pts(ROOT / "results" / "baselines_ctr_xl.jsonl")
kus = sorted(dict((x, (x, y)) for x, y in kus_c + _k).values())
std = sorted(std_c + _s)

kd = dict(kus)
i24, i26 = X.index(10626), X.index(14950)
b26 = per["K(26,4)"]
th = sorted(l for _, _, l in b26.values())
pi = sorted(t - l for _, t, l in b26.values())


def draw(out, figsize, dpi, fs, agent, agent_l, kneser, stdc, legend_fs, ann1, ann2):
    TXT, SUB, GRID = "#191919", "#555555", "#e3e3e3"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": fs, "axes.edgecolor": GRID,
                         "xtick.color": SUB, "ytick.color": SUB})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.fill_between(X, LO, HI, color=agent_l, lw=0, zorder=1)
    ax.plot(X, MED, color=agent, lw=2.4, marker="o", ms=4.5, mec="white", mew=0.7, zorder=3,
            label=f"LLM agent, total (4 Claude models: median, range shaded): {len(X)}/{len(P.SPECS)} proven")
    ax.plot(X, LLM, color=agent, lw=1.5, ls=":", zorder=3, label="   of which LLM thinking (median)")
    kx, ky = zip(*kus)
    ax.plot(kx, ky, color=kneser, lw=2.4, marker="D", ms=4.8, mec="white", mew=0.7, zorder=4,
            label=f"Kneser agent, no LLM: {len(kus)}/{len(P.SPECS)} proven")
    if std:
        sx, sy = zip(*std)
        ax.scatter(sx, sy, marker="s", s=40, color=stdc, edgecolor="white", lw=0.8, zorder=5,
                   label=f"standard methods (SAT, clique cover): {len(std)}/{len(P.SPECS)} proven in 60 s")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(170, 19000)
    ax.set_ylim(0.01, 1500)
    ax.grid(True, which="major", color=GRID, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("receivers (vertices of the graph)", color=SUB)
    ax.set_ylabel("seconds to prove β (log scale)", color=SUB)
    ax.annotate(f"10,626 receivers\nagent {LO[i24]:.0f}–{HI[i24]:.0f} s vs {kd[10626]:.0f} s", xy=(10626, MED[i24]),
                xytext=ann1, fontsize=fs, va="top", color=TXT, ha="left",
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8))
    ax.annotate(f"14,950 receivers: agent {LO[i26]:.0f}–{HI[i26]:.0f} s vs {kd[14950]:.0f} s\n"
                f"(LLM thinking {th[0]:.0f}–{th[-1]:.0f} s + pipeline {pi[0]:.0f}–{pi[-1]:.0f} s)",
                xy=(14950, HI[i26]), xytext=ann2, fontsize=fs, va="top", color=TXT, ha="left",
                arrowprops=dict(arrowstyle="-", color=SUB, lw=0.8))
    ax.legend(loc="lower right", frameon=False, fontsize=legend_fs, labelcolor=TXT)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / out, facecolor="white")
    plt.close(fig)
    print("wrote", HERE / out)


# wide (16:9 slides) and 4:3 Beamer (default beamer blue for the agent, as in the deck's colour theme)
draw("time_to_prove_deck.png", (9.2, 5.6), 220, 12, "#4b2fc0", "#d9d2fa", "#f47230", "#257de0", 10,
     (4300, 5), (1500, 1100))
draw("time_to_prove_beamer.png", (6.4, 4.1), 300, 9, "#3333b3", "#d4d4f2", "#e0631a", "#8a8a8a", 7.5,
     (3800, 5), (900, 1100))
for x, lo, m, hi, l in zip(X, LO, MED, HI, LLM):
    print(f"{x:>6}  agent {lo:6.1f} {m:6.1f} {hi:6.1f}  llm {l:5.1f}  kneser {kd.get(x, float('nan')):6.2f}")


def draw_models(out):
    """4:3 per-model chart for the Beamer appendix: each Claude configuration, Qwen (GPU server), Kneser agent."""
    plt.rcParams.update({"font.size": 9})
    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=300)
    names = {"sonnet-med": "Sonnet 5.5 medium", "opus-low": "Opus 5.5 low", "opus-med": "Opus 5.5 medium",
             "opus-high": "Opus 5.5 high"}
    col = {"sonnet-med": "#ee7fac", "opus-low": "#b3a5f2", "opus-med": "#8a70ea", "opus-high": "#3333b3"}
    for c in CFG:
        pts = P.agent_pts(P.tagged(c + "-full"))
        x, y = zip(*pts)
        ax.plot(x, y, color=col[c], lw=1.6, marker="o", ms=3, label=f"agent + {names[c]}: {len(pts)}/25")
    q = P.agent_pts(P.server_qwen)
    x, y = zip(*q)
    ax.plot(x, y, color="#f2a600", lw=1.4, ls="--", marker="^", ms=3,
            label=f"agent + Qwen2.5-Coder-32B (GPU server): {len(q)}/25")
    kx, ky = zip(*kus)
    ax.plot(kx, ky, color="#e0631a", lw=1.8, marker="D", ms=3, label=f"Kneser agent, no LLM: {len(kus)}/25")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(170, 19000)
    ax.grid(True, which="major", color="#e3e3e3")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("receivers", color="#555555")
    ax.set_ylabel("seconds to prove β (log), LLM time included", color="#555555")
    ax.legend(loc="lower right", frameon=False, fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(HERE / out, facecolor="white")
    plt.close(fig)
    print("wrote", HERE / out)


draw_models("time_to_prove_models_beamer.png")
