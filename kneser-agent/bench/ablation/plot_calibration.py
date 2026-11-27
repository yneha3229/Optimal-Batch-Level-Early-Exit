"""How confidently is each model wrong? -> calibration.png (4:3 slide size).

Left: the confidence the Solver states in its first family guess, averaged over the exact Kneser graphs and over the
edge-deleted ("perturbed") graphs of the 12-cell grid, with every wrong first guess marked. Confidences on the
non-Kneser control graphs are left out: models read the scale differently there (Haiku and Sonnet give 0.1 = "no
family", Opus gives 0.8 = "sure there is none").
Right: Critic objections per run, split into true and false by the hand labels in label_detail.json.
A first guess is right if it names the right kind: Kneser for exact graphs, perturbed for edge-deleted graphs, no
Kneser family for the random regular controls.
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import score as S

HERE = Path(__file__).resolve().parent
LABELS = json.loads((HERE / "label_detail.json").read_text())
CF = [("haiku-med", "Haiku"), ("sonnet-med", "Sonnet"), ("opus-low", "Opus low"), ("opus-med", "Opus medium"),
      ("opus-high", "Opus high"), ("Qwen2.5-32B-cpu", "Qwen2.5-Coder 32B"), ("Qwen3-4B-cpu", "Qwen3 4B")]


def right(fh, kind):
    name = str(fh.get("name", "")).lower()
    if kind == "control":
        return "kneser" not in name
    if kind == "perturbed":
        return "perturb" in name
    return "kneser" in name and "perturb" not in name


rows = {}
for c, _ in CF:
    out = []
    for g, (_, _, kind) in S.TRUTH.items():
        p = S.TR / S.CONFIGS[c].format(g=g)
        if not p.exists():
            continue
        ev = [json.loads(l) for l in open(p)]
        sol = [e["reply"] for e in ev if e["ev"] == "llm_reply" and e.get("agent") == "solver"]
        objs = LABELS.get(f"{g}|{c}", [])
        fh = (sol[0].get("family_hypothesis") or {}) if sol else None
        out.append({"kind": kind, "conf": fh.get("confidence") if fh else None,
                    "right": right(fh, kind) if fh else None,
                    "obj": len(objs), "false": sum(o["false"] for o in objs)})
    rows[c] = out

BLUE, BLUE_L, RED, GREY = "#3333b3", "#b8b8e6", "#c0392b", "#555555"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#cccccc",
                     "xtick.color": GREY, "ytick.color": GREY})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.3), dpi=300, sharey=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
ys = list(range(len(CF)))[::-1]
for y, (c, lab) in zip(ys, CF):
    r = rows[c]
    ex = [x["conf"] for x in r if x["kind"] in ("rule applies", "rule exception") and x["conf"] is not None]
    pe = [x["conf"] for x in r if x["kind"] == "perturbed" and x["conf"] is not None]
    a1.plot([st.mean(pe), st.mean(ex)], [y, y], color="#dddddd", lw=3, zorder=1)
    a1.scatter(st.mean(ex), y, s=46, color=BLUE, zorder=3, label="exact Kneser graphs (mean)" if y == ys[0] else None)
    a1.scatter(st.mean(pe), y, s=46, facecolor="white", edgecolor=BLUE, lw=1.5, zorder=3,
               label="edge-deleted graphs (mean)" if y == ys[0] else None)
    wrong = [x["conf"] for x in r if x["right"] is False]
    if wrong:
        a1.scatter(wrong, [y + 0.28] * len(wrong), marker="x", s=34, color=RED, lw=1.6, zorder=4,
                   label="wrong first guess" if c.startswith("Qwen2") else None)
    t, f = sum(x["obj"] - x["false"] for x in r) / len(r), sum(x["false"] for x in r) / len(r)
    a2.barh(y, t, color=BLUE_L, height=0.6, label="true" if y == ys[0] else None)
    a2.barh(y, f, left=t, color=RED, height=0.6, label="false" if y == ys[0] else None)
    a2.text(t + f + 0.25, y, f"{f:.1f} false" if f else "0 false", va="center", fontsize=8,
            color=RED if f else GREY)
a1.set_yticks(ys, [lab for _, lab in CF])
a1.set_xlim(-0.03, 1.03)
a1.set_xlabel("confidence stated in the first guess", color=GREY)
a1.set_title("How sure is the Solver?", fontsize=10, loc="left")
a1.legend(loc="upper left", fontsize=7, frameon=False, ncol=3, bbox_to_anchor=(-0.45, -0.2), handletextpad=0.2, columnspacing=1.0)
a2.set_xlim(0, 20)
a2.set_xlabel("Critic objections per run", color=GREY)
a2.set_title("Is the Critic right?", fontsize=10, loc="left")
a2.legend(loc="upper right", fontsize=7, frameon=False, title="objections", title_fontsize=7)
for a in (a1, a2):
    a.grid(True, axis="x", color="#eeeeee")
    a.spines[["top", "right"]].set_visible(False)
    a.axhline(1.5, color="#cccccc", lw=0.8, ls="--")
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(HERE / "calibration.png", facecolor="white")
print("wrote", HERE / "calibration.png")
