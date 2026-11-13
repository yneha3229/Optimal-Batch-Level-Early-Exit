"""Figures for the revised progress report (docs/IE624_progress_report_v2.pdf).

Reads only result files: results/graph_*_gpu.csv, results/main_*_real_*_gpu*.csv (server, RTX A6000) and
results/local_* (this machine, GTX 1650; experiments/local_hw.py). Writes figures/report2_*.png.

Usage: python3 analysis/report_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
# fixed categorical order: colour follows the method everywhere
COL = {"theory+plan": "#2a78d6", "theory": "#2a78d6", "unified": "#eb6834", "threshold": "#1baf7a",
       "majority": "#eda100", "random_collective": "#e87ba4", "threshold+plan": "#6250d6"}
NAME = {"theory+plan": "Cost rule + exit plan (ours)", "theory": "Cost rule, all exits (ours)",
        "unified": "Unified exit", "threshold": "Per-sample threshold", "majority": "Majority vote",
        "random_collective": "Random search (150 rules)", "threshold+plan": "Threshold + exit plan"}
MARK = {"theory+plan": "o", "theory": "o", "unified": "s", "threshold": "v", "majority": "^",
        "random_collective": "D", "threshold+plan": "P"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 220, "savefig.bbox": "tight"})


def line(ax, x, y, fam, **kw):
    ls = "--" if fam == "theory" else "-"
    ax.plot(x, y, ls, color=COL[fam], marker=MARK[fam], ms=5, lw=2 if fam.startswith("theory") else 1.4,
            mec=SURF, mew=0.8, label=NAME[fam], **kw)


def fig_cost() -> None:
    """Measured per-layer cost on the GTX 1650 at three sequence lengths, with the affine fit."""
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), sharey=True)
    for ax, T in zip(axs, (16, 32, 128)):
        d = json.loads((ROOT / f"results/local_cost_cuda_T{T}.json").read_text())
        B = np.array([p["B"] for p in d["points"]])
        y = np.array([p["full_ms"] for p in d["points"]]) / 12
        n = np.geomspace(1, 128, 100)
        ax.plot(n, d["a"] + d["b"] * n, color=INK2, lw=1, ls="--", label="affine fit a + b·n")
        ax.plot(B, y, "o", color="#2a78d6", ms=5, mec=SURF, mew=0.8, label="measured")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(B)
        ax.set_xticklabels(B)
        ax.set_title(f"sequence length {T}\naffine error ≤ {100 * d['affine_max_rel_err']:.0f}%, "
                     f"ρ(8) = {max(d['rho8'], 0):.2f}", fontsize=7.5, loc="left")
        ax.set_xlabel("active inputs n")
    axs[0].set_ylabel("time per layer (ms)")
    axs[2].legend(loc="lower right", fontsize=7.5)
    fig.savefig(ROOT / "figures/report2_cost_gtx1650.png")
    plt.close(fig)


def fig_server() -> None:
    """Server (RTX A6000) real wall-clock speedups, eager and CUDA graphs, MNLI and QQP."""
    fams = ["theory", "unified", "majority", "threshold"]
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.3), sharex=True, sharey="row")
    for i, task in enumerate(("mnli", "qqp")):
        eager = pd.concat([pd.read_csv(ROOT / f"results/main_{task}_real_{task}_gpu_B8.csv"),
                           pd.read_csv(ROOT / f"results/main_{task}_real_{task}_gpu_B16-128.csv")])
        graph = pd.read_csv(ROOT / f"results/graph_{task}_gpu.csv")
        for j, (d, mode) in enumerate(((eager, "eager PyTorch"), (graph, "CUDA graphs"))):
            ax = axs[i, j]
            ax.axhline(1, color=INK2, lw=0.8)
            for f in fams[::-1]:
                g = d[d.family == f].drop_duplicates("B").sort_values("B")
                line(ax, np.arange(len(g)), g.real_speedup, f)
            ax.set_xticks(range(4))
            ax.set_xticklabels(["B=8", "B=16", "B=64", "B=128"])
            ax.set_title(f"{task.upper()}, {mode}", fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel("speedup vs full model")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h[::-1], [x.replace(", all exits", "") for x in l[::-1]], loc="upper center", ncol=4, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.03))
    fig.savefig(ROOT / "figures/report2_server_speedups.png")
    plt.close(fig)


def local_table() -> pd.DataFrame:
    fs = sorted(ROOT.glob("results/local_time_*_cuda_T*.csv"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


def fig_local() -> None:
    """GTX 1650 real wall-clock speedups of certified policies. Rows: task; columns: sequence length."""
    d = local_table()
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    fams = ["threshold", "unified", "majority", "random_collective", "theory", "theory+plan"]
    Ts = sorted(d["T"].unique())
    tasks = [t for t in ("qqp", "mnli") if t in set(d.task)]
    fig, axs = plt.subplots(len(tasks), len(Ts), figsize=(7.2, 2.2 * len(tasks) + 0.3), sharex=True, sharey="row",
                            squeeze=False)
    for i, task in enumerate(tasks):
        for j, T in enumerate(Ts):
            ax = axs[i, j]
            ax.axhline(1, color=INK2, lw=0.8)
            g0 = d[(d.task == task) & (d["T"] == T)]
            Bs = sorted(g0.B.unique())
            for f in fams:
                g = g0[g0.family == f].set_index("B").reindex(Bs)
                x = np.arange(len(Bs))
                ax.vlines(x, g.real_lo, g.real_hi, color=COL[f], lw=1, alpha=0.6)
                line(ax, x, g.real_speedup, f)
            ax.set_xticks(range(len(Bs)))
            ax.set_xticklabels([f"B={b}" for b in Bs])
            ax.set_title(f"{task.upper()}, sequence length {T}", fontsize=8.5, loc="left")
            if j == 0:
                ax.set_ylabel("speedup vs full model")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.0 + 0.09 / len(tasks) * 2))
    fig.savefig(ROOT / "figures/report2_local_speedups.png")
    plt.close(fig)


def fig_resplit() -> None:
    """Share of random re-splits whose test-split accuracy drop exceeds the 1-point budget."""
    fs = sorted(ROOT.glob("results/local_resplit_*_cuda_T16.csv"))
    if not fs:
        return
    fig, axs = plt.subplots(1, len(fs), figsize=(3.6 * len(fs), 2.2), sharey=True, squeeze=False)
    for ax, f in zip(axs[0], fs):
        d = pd.read_csv(f)
        task = f.name.split("_")[2]
        g = d.groupby(["B", "family"]).agg(naive=("naive_drop", lambda s: (s > 0.01).mean()),
                                           cert=("cert_drop", lambda s: (s > 0.01).mean())).reset_index()
        lab = [f"{NAME[r.family].split(' (')[0].split(',')[0]}\nB={r.B}" for r in g.itertuples()]
        x = np.arange(len(g))
        ax.bar(x - 0.19, 100 * g.naive, 0.36, color="#eb6834", label="edge-of-budget selection (old protocol)")
        ax.bar(x + 0.19, 100 * g.cert, 0.36, color="#2a78d6", label="certified selection")
        ax.axhline(10, color=INK2, lw=0.8, ls="--")
        ax.text(len(g) - 0.5, 11, "δ = 10%", ha="right", va="bottom", fontsize=7, color=INK2)
        for xi, (a, b) in enumerate(zip(g.naive, g.cert)):
            ax.text(xi - 0.19, 100 * a + 1, f"{100 * a:.0f}", ha="center", fontsize=6.5, color=INK2)
            ax.text(xi + 0.19, 100 * b + 1, f"{100 * b:.0f}", ha="center", fontsize=6.5, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=6.3)
        ax.set_title(f"{task.upper()}", fontsize=8.5, loc="left")
        ax.grid(axis="x", visible=False)
    axs[0, 0].set_ylabel("re-splits over budget (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, 1.12))
    fig.savefig(ROOT / "figures/report2_resplit.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_cost()
    fig_server()
    if list(ROOT.glob("results/local_time_*_cuda_T*.csv")):
        fig_local()
    fig_resplit()
    print("figures written")
