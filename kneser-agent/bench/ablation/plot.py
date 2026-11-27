"""Model ablation chart from scores.jsonl -> model_ablation.png (three small multiples, one series each).

Configurations have different numbers of runs (the Qwen GPU pilot has 3; the others up to 12), so every panel shows a
per-run rate and every bar label carries its run count.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ORDER = [("Qwen3-4B-cpu", "Qwen3 4B\nCPU Q4"), ("Qwen2.5-32B", "Qwen 32B\nGPU, pre-fix"),
         ("Qwen2.5-32B-cpu", "Qwen 32B\nCPU Q4"), ("haiku-med", "Haiku 5.5\nmedium"),
         ("sonnet-med", "Sonnet 5.5\nmedium"), ("opus-low", "Opus 5.5\nlow"), ("opus-med", "Opus 5.5\nmedium"),
         ("opus-high", "Opus 5.5\nhigh")]
rows = [json.loads(line) for line in (HERE / "scores.jsonl").read_text().splitlines() if line.strip()]
by = defaultdict(list)
for r in rows:
    by[r["config"]].append(r)
cfgs = [(c, lab + f"\n(n={len(by[c])})") for c, lab in ORDER if c in by]
agg = {
    "names": [100 * sum(r["first_reply_names_nk"] for r in by[c]) / len(by[c]) for c, _ in cfgs],
    "false": [sum(r["false_objections"] or 0 for r in by[c]) / len(by[c]) for c, _ in cfgs],
    "calls": [sum(r["llm_calls"] for r in by[c]) / len(by[c]) for c, _ in cfgs],
}
brackets = defaultdict(set)
for r in rows:
    brackets[r["graph"]].add(tuple(r["bracket"]))
same = all(len(v) == 1 for v in brackets.values())
panels = [
    ("Right family / (n, k) in the first Solver reply\n(% of runs; higher is better)", agg["names"], 100, "{:.0f}%"),
    ("False Critic objections per run\n(labelled by hand; lower is better)", agg["false"], max(agg["false"] + [1]),
     "{:.2f}"),
    ("LLM calls per run\n(lower is better)", agg["calls"], max(agg["calls"]), "{:.1f}"),
]
BLUE, TXT, SUB, GRID = "#257de0", "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "xtick.color": SUB, "ytick.color": SUB})
fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), dpi=200)
for ax, (title, vals, top, fmt) in zip(axes, panels):
    bars = ax.bar(range(len(cfgs)), vals, color=BLUE, width=0.6, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center", va="bottom", color=TXT,
                fontsize=8.5)
    ax.set_xticks(range(len(cfgs)), [lab for _, lab in cfgs], fontsize=7.5)
    ax.set_ylim(0, top * 1.18)
    ax.set_title(title, color=TXT, fontsize=10.5, loc="left")
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.suptitle("Agent reasoning quality by model (blind runs, same prompts and graphs)", x=0.01, ha="left",
             color=TXT, fontsize=13, fontweight="bold")
ng = len(brackets)
diff = sum(len(v) > 1 for v in brackets.values())
note = ((f"Final certified bracket identical for every model on all {ng} graph instances. " if same else
         f"Same verdict for every model on all {ng} graph instances; the bracket differs on {diff} (the random control: "
         "lower bound 336 for Claude, 331-333 for Qwen, from fewer independent-set restarts). ")
        + "Graphs: K(16,4), K(16,4) minus 1% of edges, K(14,4) (rule exception), K(13,4), K(15,5), random 6-regular "
        "(1000 vertices); 2 relabelling seeds each, one run per cell.\nQwen 32B GPU (vLLM, AWQ): 3-graph pilot on the "
        "code before 9081f70; every other bar: fixed code. Qwen CPU bars: llama.cpp, GGUF Q4_K_M; Qwen3 4B with a 1,500-token reply limit.")
fig.text(0.01, 0.005, note, color=SUB, fontsize=7.5)
fig.tight_layout(rect=(0, 0.07, 1, 0.93))
fig.savefig(HERE / "model_ablation.png", facecolor="white")
print("wrote", HERE / "model_ablation.png")
