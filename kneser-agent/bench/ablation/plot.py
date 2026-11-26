"""Model ablation chart from scores.jsonl -> model_ablation.png (three small multiples, one series each)."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = [("Qwen3-4B", "Qwen3\n4B"), ("Qwen2.5-32B", "Qwen 32B\nGPU\npre-fix"), ("Qwen2.5-32B-cpu", "Qwen 32B\nCPU Q4\nfixed"), ("haiku-med", "Haiku\n5.5\nmedium"),
         ("sonnet-med", "Sonnet\n5.5\nmedium"), ("opus-low", "Opus\n5.5\nlow"), ("opus-med", "Opus\n5.5\nmedium"),
         ("opus-high", "Opus\n5.5\nhigh")]
rows = [json.loads(l) for l in (HERE / "scores.jsonl").read_text().splitlines()]
by = defaultdict(list)
for r in rows:
    by[r["config"]].append(r)
cfgs = [(c, lab) for c, lab in ORDER if c in by]
n_graphs = max(len(v) for v in by.values())
agg = {
    "names": [sum(r["first_reply_names_nk"] for r in by[c]) for c, _ in cfgs],
    "false": [sum(r["false_objections"] or 0 for r in by[c]) for c, _ in cfgs],
    "calls": [sum(r["llm_calls"] for r in by[c]) for c, _ in cfgs],
}
brackets = {r["graph"]: set() for r in rows}
for r in rows:
    brackets[r["graph"]].add(tuple(r["bracket"]))
same = all(len(v) == 1 for v in brackets.values())
panels = [
    (f"Named the right K(n,k) in the first reply\n(of {n_graphs} graphs; higher is better)", agg["names"], n_graphs),
    (f"False Critic objections, labelled by hand\n(total over {n_graphs} graphs; lower is better)", agg["false"],
     max(agg["false"] + [1])),
    (f"LLM calls needed\n(total over {n_graphs} graphs; lower is better)", agg["calls"], max(agg["calls"])),
]
BLUE, TXT, SUB, GRID = "#257de0", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": SUB, "ytick.color": SUB})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=200)
for ax, (title, vals, top) in zip(axes, panels):
    bars = ax.bar(range(len(cfgs)), vals, color=BLUE, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, str(v), ha="center", va="bottom", color=TXT, fontsize=9)
    ax.set_xticks(range(len(cfgs)), [lab for _, lab in cfgs], fontsize=8)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(title, color=TXT, fontsize=10.5, loc="left")
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
fig.suptitle("Agent reasoning quality by model (blind runs, same prompts and graphs)", x=0.01, ha="left",
             color=TXT, fontsize=13, fontweight="bold")
note = ("Final certified bracket identical for every model: " if same else "Brackets differ between models: ") + \
    ", ".join(f"{g} {sorted(v)[0] if same else sorted(v)}" for g, v in brackets.items()) + ". One run per cell. Claude rows and Qwen CPU (llama.cpp, GGUF Q4_K_M): fixed code (d22d46c); Qwen GPU (vLLM, AWQ): before that fix."
fig.text(0.01, 0.005, note, color=SUB, fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 0.93))
fig.savefig(HERE / "model_ablation.png", facecolor="white")
print("wrote", HERE / "model_ablation.png")
