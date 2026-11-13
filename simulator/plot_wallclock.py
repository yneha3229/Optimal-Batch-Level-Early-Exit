"""Figure: real GPU speedup vs batch size per policy, and simulator-predicted vs real speedup.
Usage: ~/venv/bin/python simulator/plot_wallclock.py enc_eager_meas
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
regime = sys.argv[1] if len(sys.argv) > 1 else "enc_eager_meas"
df = pd.read_csv(ROOT / f"results/p6_wallclock_{regime}.csv")
df = df[~df.family.isin(["full", "heads_only"])]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
for fam, g in df.groupby("family"):
    g = g.sort_values("B")
    ax1.plot(g.B, g.real_speedup, "o-" if fam.startswith("theory") else ".--", lw=2.5 if fam == "theory" else 1, label=fam)
    ax2.scatter(g.sim_speedup, g.real_speedup, label=fam, s=25)
ax1.axhline(1, color="k", lw=0.8)
ax1.set(xscale="log", xticks=[16, 64, 128], xticklabels=[16, 64, 128], xlabel="batch size",
        ylabel="real wall-clock speedup vs full model", title=f"RTX A6000, RoBERTa-base MNLI, held-out ({regime})")
ax1.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax1.legend(fontsize=7)
ax1.grid(alpha=0.3)
lo, hi = 0.6, 1.25
ax2.plot([lo, hi], [lo, hi], "k:", lw=0.8)
ax2.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="simulator speedup", ylabel="real speedup", title="simulator vs GPU")
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ROOT / f"figures/p6_wallclock_{regime}.png", dpi=150)
print("saved")
