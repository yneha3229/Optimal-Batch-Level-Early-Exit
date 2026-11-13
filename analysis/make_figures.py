"""Exhibits 1-8 and the main results table, from the CSVs in results/. Each exhibit is skipped if its inputs are
missing. Usage: ~/venv/bin/python analysis/make_figures.py --task mnli
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

# Reference categorical palette (dataviz skill), fixed order; colour follows the method, never its rank.
COLORS = {"threshold": "#2a78d6", "threshold_nocompact": "#eb6834", "majority": "#1baf7a", "unified": "#eda100",
          "ceebert": "#e87ba4", "consensus": "#008300", "theory+rescue_conf": "#6250d6", "bandit": "#e34948",
          "theory": "#6250d6", "funsearch": "#e34948", "random": "#2a78d6", "bo": "#eda100"}
LABELS = {"threshold": "Per-sample threshold", "threshold_nocompact": "Wait-for-all (no compaction)",
          "majority": "Majority vote (DREX)", "unified": "Unified exit (SkipDecode-style)",
          "ceebert": "CeeBERT-style bandit", "consensus": "Consensus (DREX)", "theory": "Theory rule (ours)",
          "theory+rescue_conf": "Theory rule + rescue (ours)", "bandit": "Label-free bandit (ours)",
          "funsearch": "FunSearch best", "full": "Full model", "lanes": "Difficulty lanes",
          "random_collective": "Random search (150)", "theory_indiv_only": "Theory, individual exits only",
          "theory+rescue_learned": "Theory + learned rescue (ours)", "bandit+rescue_conf": "Bandit + rescue (ours)",
          "greedy": "Greedy (DREX)", "funsearch+rescue_conf": "FunSearch best + rescue",
          "theory_hist": "Theory, trajectory-aware (ours)", "theory_robust": "Robust theory, m=B/16 (ours)",
          "theory_robust8": "Robust theory, m=B/8 (ours)", "bandit_t3pct": "Label-free bandit, 3% target (ours)"}
HEADLINE = ["threshold", "threshold_nocompact", "majority", "unified", "ceebert", "theory", "bandit"]
OURS = {"theory", "theory+rescue_conf", "bandit"}
REG_TITLE = {"enc_eager_meas": "Encoder cost (measured, ρ₆₄≈0.3)", "dec_eager": "Decoder-like cost (ρ₆₄≈95)",
             "enc_graph": "Encoder, CUDA graphs"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False})


def reg_title(r: str) -> str:
    return REG_TITLE.get(r, r.replace("real_", "Real GPU runner cost, "))


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=160)
    plt.close(fig)
    print("wrote", FIG / name)


def ex1_speedup_vs_batch(task: str, sel: pd.DataFrame, gpu: dict[str, pd.DataFrame]) -> None:
    regs = list(sel.regime.unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 4.2), squeeze=False, sharey=True)
    for ax, reg in zip(axes[0], regs):
        s = sel[sel.regime == reg]
        for fam in HEADLINE:
            g = s[s.family == fam].sort_values("B")
            if g.empty:
                continue
            ax.plot(g.B, g.heldout_speedup, "-o", color=COLORS[fam], lw=2.4 if fam in OURS else 1.6,
                    ms=4.5, label=LABELS[fam], zorder=3 if fam in OURS else 2)
        if reg in gpu:
            gg = gpu[reg]
            for fam in HEADLINE:
                g = gg[gg.family == fam].sort_values("B")
                if len(g):
                    ax.plot(g.B, g.real_speedup, "s", mfc="none", mec=COLORS[fam], ms=8, mew=1.6)
        ax.axhline(1, color="#0b0b0b", lw=0.9)
        ax.set(xscale="log", xticks=sorted(s.B.unique()), xticklabels=sorted(s.B.unique()), title=reg_title(reg),
               xlabel="Batch size")
    axes[0][0].set_ylabel("Speedup vs full model\n(held-out, accuracy within 1 pt)")
    axes[0][-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    if gpu:
        fig.text(0.01, 0.01, "lines: simulator   hollow squares: real GPU wall-clock", fontsize=8, color="#52514e")
    save(fig, f"ex1_speedup_vs_batch_{task}.png")


def ex2_cost(task: str) -> None:
    rc = ROOT / f"results/runner_cost_{task}.json"
    if not rc.exists():
        return
    d = json.loads(rc.read_text())
    pts = pd.DataFrame(d["points"])
    if "layers28" in pts:              # decoder: one decode step through all 28 layers
        y, nl, what = pts.layers28, 28, "ms per decode step (28 layers)"
    else:
        y, nl, what = pts.full_ms_per_batch, 12, "ms per batch (12 layers)"
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    n = np.linspace(1, pts.B.max(), 200)
    ax.plot(pts.B, y, "o", color=COLORS["threshold"], ms=7, label="Measured, full model")
    ax.plot(n, nl * (d["a"] + d["b"] * n), "-", color="#52514e", lw=1.6,
            label=f"{nl}·(a + b·n), a={d['a']:.3f} ms, b={d['b']:.4f} ms")
    ax.set(xlabel="Batch size n", ylabel=what, ylim=(0, None), title=f"Cost model, {task} on RTX A6000")
    ax.legend(fontsize=8)
    save(fig, f"ex2_cost_model_{task}.png")


def ex3_flops_gap(task: str, sel: pd.DataFrame, gpu: dict[str, pd.DataFrame]) -> None:
    reg = next((r for r in gpu), None) or "enc_eager_meas"
    s = sel[(sel.regime == reg) & (sel.B == 64) & sel.family.isin(HEADLINE)].set_index("family")
    fams = [f for f in HEADLINE if f in s.index]
    if not fams:
        return
    flops = 1 / s.loc[fams, "heldout_flops_frac"]
    real = (gpu[reg][gpu[reg].B == 64].set_index("family").reindex(fams).real_speedup if reg in gpu
            else s.loc[fams, "heldout_speedup"])
    y = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    ax.barh(y + 0.19, flops, 0.36, color="#c3c2b7", label="FLOPs speedup (1 / FLOPs fraction)")
    ax.barh(y - 0.19, real, 0.36, color=[COLORS[f] for f in fams],
            label="Wall-clock speedup" + (" (real GPU)" if reg in gpu else " (simulated)"))
    ax.axvline(1, color="#0b0b0b", lw=0.9)
    ax.set(yticks=y, yticklabels=[LABELS[f] for f in fams], xlabel="Speedup at B = 64",
           title=f"FLOPs saved are not time saved ({reg_title(reg)})")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    save(fig, f"ex3_flops_vs_wallclock_{task}.png")


def ex4_pareto(task: str, allr: pd.DataFrame) -> None:
    h = allr[(allr.split == "heldout") & (allr.B == 64)]
    regs = list(h.regime.unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 4.0), squeeze=False)
    for ax, reg in zip(axes[0], regs):
        g = h[h.regime == reg]
        full = g[g.family == "full"].iloc[0]
        for fam in ["threshold", "majority", "consensus", "ceebert", "theory", "theory+rescue_conf"]:
            gf = g[g.family == fam].sort_values("rel_latency")
            if gf.empty:
                continue
            # Pareto front only
            gf = gf[gf.accuracy >= gf.accuracy[::-1].cummax()[::-1] - 1e-12]
            ax.plot(gf.rel_latency, 100 * gf.accuracy, "-", color=COLORS[fam], lw=2.2 if fam in OURS else 1.4,
                    label=LABELS[fam])
        ax.plot(1, 100 * full.accuracy, "*", color="#0b0b0b", ms=11, label="Full model")
        ax.axhline(100 * (full.accuracy - 0.01), color="#52514e", lw=0.8, ls=":")
        ax.set(xlim=(0.3, 1.1), ylim=(100 * full.accuracy - 6, 100 * full.accuracy + 1), title=reg_title(reg),
               xlabel="Latency relative to full model", ylabel="Held-out accuracy (%)")
    axes[0][0].legend(fontsize=8, loc="lower right")
    save(fig, f"ex4_pareto_B64_{task}.png")


def ex5_bandit(task: str) -> None:
    for f in sorted((ROOT / "results").glob(f"bandit_{task}_*.csv")):
        tag = f.stem.replace("bandit_", "")
        df = pd.read_csv(f)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
        ax = axes[0]
        for m, col in [("bandit", COLORS["bandit"]), ("bandit_probe0.3", "#e87ba4"), ("bandit_probe3", "#eda100"),
                       ("ceebert", COLORS["threshold"])]:
            reg = df.groupby("seed").apply(lambda g: (g.oracle_fid_fid_score - g[f"{m}_fid_score"]).cumsum().to_numpy())
            R = np.stack(reg.to_list())
            t = np.arange(1, R.shape[1] + 1)
            ax.plot(t, R.mean(0), color=col, lw=2.2 if m == "bandit" else 1.4,
                    label={"bandit": "Bandit (probe c=1)", "bandit_probe0.3": "Bandit (c=0.3)",
                           "bandit_probe3": "Bandit (c=3)", "ceebert": "CeeBERT-style"}[m])
        T = len(t)
        ref = R.mean(0)[-1] if False else None
        ax.set(xlabel="Batches processed", ylabel="Cumulative regret (fidelity objective)",
               title="Regret vs oracle with known tables")
        ax.legend(fontsize=8)
        thr = ROOT / f"results/bandit_threshold_{tag}.csv"
        if thr.exists():
            th = pd.read_csv(thr)
            th = th[th.t == th.t.max()]
            ax = axes[1]
            for e, col in zip(sorted(th.exit.unique())[1:4], ["#2a78d6", "#1baf7a", "#6250d6"]):
                g = th[th.exit == e].groupby("n_active").threshold.mean()
                ax.plot(g.index, g.values, "-o", color=col, ms=4.5, lw=1.8, label=f"exit at layer {th[th.exit == e].layer.iloc[0]}")
            ax.set(xscale="log", xlabel="Inputs still active (n)", ylabel="Learned stop threshold (confidence)",
                   title="Learned threshold drops as the batch drains")
            ax.legend(fontsize=8)
        save(fig, f"ex5_bandit_{tag}.png")


def ex6_attack(task: str, att: pd.DataFrame) -> None:
    a = att[att.B == 64]
    regs = list(a.regime.unique())
    if not regs:
        return
    fig, axes = plt.subplots(1, len(regs), figsize=(5.0 * len(regs), 4.0), squeeze=False, sharey=True)
    for ax, reg in zip(axes[0], regs):
        for fam in ["threshold", "majority", "consensus", "unified", "theory", "bandit"]:
            g = a[(a.regime == reg) & (a.family == fam)].sort_values("k")
            if len(g):
                ax.plot(g.k, g.victim_rel_full, "-o", color=COLORS[fam], lw=2.2 if fam in OURS else 1.4, ms=5,
                        label=LABELS[fam])
        ax.axhline(1, color="#0b0b0b", lw=0.9)
        ax.set(xticks=[0, 1, 2, 4], xlabel="Slowdown inputs per batch of 64", title=reg_title(reg))
    axes[0][0].set_ylabel("Victims' latency / full-model latency")
    axes[0][-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    kind = a.attack.iloc[0]
    fig.text(0.01, 0.01, f"attack inputs: {kind}", fontsize=8, color="#52514e")
    save(fig, f"ex6_attack_{task}.png")


def ex7_funsearch(task: str) -> None:
    base = ROOT / f"results/evolve_baselines_{task}_val.csv"
    evo = ROOT / f"results/evolve_curve_{task}.csv"
    if not base.exists() and not evo.exists():
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    if base.exists():
        b = pd.read_csv(base)
        for m in ["random", "bo"]:
            g = b[b.method == m].groupby("i").best_so_far.mean()
            ax.plot(g.index, g.values, color=COLORS[m], lw=1.6, label={"random": "Random search", "bo": "Bayesian opt."}[m])
    if evo.exists():
        e = pd.read_csv(evo)
        ax.plot(e.i, e.best_so_far, color=COLORS["funsearch"], lw=2.4, label="FunSearch (OpenEvolve)")
    if (ROOT / f"results/evolve_reference_{task}.json").exists():
        ref = json.loads((ROOT / f"results/evolve_reference_{task}.json").read_text())
        ax.axhline(ref["theory"], color=COLORS["theory"], lw=1.2, ls="--", label="Theory rule (best λ)")
    ax.set(xlabel="Programs tried", ylabel="Best score so far (evolve split)", title="Search at equal budget")
    ax.legend(fontsize=8)
    save(fig, f"ex7_funsearch_{task}.png")


def ex8_rescue(task: str, sel: pd.DataFrame) -> None:
    s = sel[sel.B == 64]
    fams = ["theory", "theory+rescue_conf", "theory+rescue_learned", "majority", "threshold", "full"]
    regs = list(s.regime.unique())
    fig, axes = plt.subplots(1, len(regs), figsize=(4.6 * len(regs), 3.6), squeeze=False, sharey=True)
    for ax, reg in zip(axes[0], regs):
        g = s[(s.regime == reg)].set_index("family").reindex(fams).dropna(subset=["hard10_acc"])
        cols = ["#c3c2b7" if f == "full" else COLORS.get(f, "#6250d6") for f in g.index]
        ax.bar(range(len(g)), 100 * g.hard10_acc, color=cols, width=0.62)
        ax.set(xticks=range(len(g)), title=reg_title(reg))
        ax.set_xticklabels([LABELS[f].replace(" (ours)", "*") for f in g.index], rotation=35, ha="right", fontsize=8)
    axes[0][0].set_ylabel("Accuracy on hardest 10% (%)")
    save(fig, f"ex8_rescue_hardest10_{task}.png")


def main_table(task: str, sel: pd.DataFrame, gpu: dict[str, pd.DataFrame]) -> None:
    rows = []
    order = ["full", "threshold", "threshold_nocompact", "majority", "consensus", "unified", "ceebert", "lanes",
             "random_collective", "theory", "theory+rescue_conf", "theory+rescue_learned", "bandit",
             "bandit+rescue_conf", "funsearch", "funsearch+rescue_conf"]
    lines = [f"# Main results, {task}, B = 64 (held-out split)\n"]
    for reg in sel.regime.unique():
        s = sel[(sel.regime == reg) & (sel.B == 64)].set_index("family")
        g = gpu[reg][gpu[reg].B == 64].set_index("family") if reg in gpu else None
        lines.append(f"\n## {reg_title(reg)}\n")
        hdr = "| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |"
        lines += [hdr, "|" + "---|" * 7]
        for f in order:
            if f not in s.index:
                continue
            r = s.loc[f]
            real = f"{g.loc[f, 'real_speedup']:.2f}×" if g is not None and f in g.index else "—"
            vs = f"{r.victim_slowdown_k1:.2f}×" if pd.notna(r.get("victim_slowdown_k1", np.nan)) else "—"
            h10 = f"{100 * r.hard10_acc:.1f}" if pd.notna(r.get("hard10_acc", np.nan)) else "—"
            lines.append(f"| {LABELS.get(f, f)} | {100 * r.heldout_acc:.2f} | {100 * r.heldout_acc_drop:.2f} | "
                         f"{r.heldout_speedup:.2f}× | {real} | {h10} | {vs} |")
    out = ROOT / f"results/main_table_{task}.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


def ex6b_robust(task: str) -> None:
    """Victims' latency vs number of slowdown inputs: plain vs certified-robust theory rule and the baselines."""
    for f in sorted((ROOT / "results").glob(f"robust_{task}_*_B64.csv")):
        if "violations" in f.name:
            continue
        df = pd.read_csv(f)
        vf = ROOT / "results" / f.name.replace("robust_", "robust_violations_")
        viol = pd.read_csv(vf) if vf.exists() else None
        ks = [int(c[1:]) for c in df.columns if c.startswith("k") and c[1:].isdigit()]
        style = {"theory m=0": ("#6250d6", "--", 1.8, "Theory rule, plain (m=0)"),
                 f"theory m={df.m.max():.0f}": ("#6250d6", "-", 2.6, f"Robust theory rule (m={df.m.max():.0f}, ours)"),
                 "majority": (COLORS["majority"], "-", 1.4, LABELS["majority"]),
                 "consensus": (COLORS["consensus"], "-", 1.4, LABELS["consensus"]),
                 "threshold": (COLORS["threshold"], "-", 1.4, LABELS["threshold"])}
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for _, r in df.iterrows():
            if r.policy not in style:
                continue
            col, ls, lw, lab = style[r.policy]
            ax.plot(ks, [r[f"k{k}"] for k in ks], ls, color=col, lw=lw, marker="o", ms=4.5,
                    label=f"{lab}  (clean {r.clean_speedup:.2f}×)")
        ax.axhline(1, color="#0b0b0b", lw=0.9)
        ax.set(xticks=ks, xlabel="Slowdown inputs in a batch of 64", ylabel="Victims' latency / full-model latency",
               title=f"{task}: certified robustness, {f.stem.split('_', 2)[2].rsplit('_B', 1)[0]}")
        ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        if viol is not None:
            fig.text(0.01, 0.01, f"pathwise bound depth(C∪K; m) ≤ depth(C; m−k): {int(viol.violations.sum())} violations "
                     f"in {int(viol.batches.sum())} batches", fontsize=8, color="#52514e")
        save(fig, f"ex6b_{f.stem}.png")


def matched_table(task: str, allr: pd.DataFrame) -> None:
    """Fair family comparison: best held-out speedup among each family's candidates whose *held-out* accuracy is
    within X of the full model (B = 64, where every candidate was also run on held-out). This removes the
    evolve -> held-out selection noise that makes some tuned policies land 1-2 points below the full model."""
    h = allr[(allr.split == "heldout") & (allr.B == 64)]
    lines = [f"# Matched-accuracy comparison, {task}, B = 64 (held-out Pareto fronts)\n",
             "Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.\n"]
    for reg, g in h.groupby("regime"):
        full = g[g.family == "full"].accuracy.iloc[0]
        rows = []
        for fam, gf in g.groupby("family"):
            r = {"family": fam}
            for tol in (0.005, 0.01, 0.02):
                ok = gf[gf.accuracy >= full - tol]
                r[tol] = ok.speedup.max() if len(ok) else np.nan
            rows.append(r)
        t = pd.DataFrame(rows).sort_values(0.01, ascending=False)
        lines += [f"\n## {reg_title(reg)} (full model {100 * full:.2f}%)\n",
                  "| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |", "|---|---|---|---|"]
        for _, r in t.iterrows():
            f = lambda v: "—" if pd.isna(v) else f"{v:.2f}×"
            lines.append(f"| {LABELS.get(r.family, r.family)} | {f(r[0.005])} | {f(r[0.01])} | {f(r[0.02])} |")
    out = ROOT / f"results/matched_table_{task}.md"
    out.write_text("\n".join(lines) + "\n")
    print("wrote", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    args = ap.parse_args()
    t = args.task
    sel_p = ROOT / f"results/main_{t}_selected.csv"
    gpu = {p.stem.replace(f"main_{t}_", "").replace("_gpu", ""): pd.read_csv(p)
           for p in (ROOT / "results").glob(f"main_{t}_*_gpu.csv")}
    if sel_p.exists():
        sel = pd.read_csv(sel_p)
        ex1_speedup_vs_batch(t, sel, gpu)
        ex3_flops_gap(t, sel, gpu)
        ex4_pareto(t, pd.read_csv(ROOT / f"results/main_{t}_all.csv"))
        matched_table(t, pd.read_csv(ROOT / f"results/main_{t}_all.csv"))
        ex6_attack(t, pd.read_csv(ROOT / f"results/main_{t}_attack.csv"))
        ex8_rescue(t, sel)
        main_table(t, sel, gpu)
    ex2_cost(t)
    ex5_bandit(t)
    ex6b_robust(t)
    ex7_funsearch(t)


if __name__ == "__main__":
    main()
