"""Build docs/IE624_progress_report_v2.{html,pdf} from result files (WeasyPrint; no LaTeX on the local machine).

Usage: python3 analysis/report_figures.py && python3 analysis/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["threshold", "threshold+plan", "unified", "majority", "random_collective", "theory", "theory+plan"]
NAME = {"threshold": "Per-sample threshold", "threshold+plan": "Threshold + exit plan", "unified": "Unified exit",
        "majority": "Majority vote", "random_collective": "Random search (150 rules)",
        "theory": "Cost rule, all exits (ours)", "theory+plan": "Cost rule + exit plan (ours)"}


def local() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("results/local_time_*_cuda_T*.csv"))], ignore_index=True)
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    return d


def cell_table(d: pd.DataFrame, task: str) -> str:
    """Measured speedup per (sequence length, batch size) for one task; best in bold; * = held-out drop > 1 pt."""
    g = d[d.task == task]
    cols = sorted({(t, b) for t, b in zip(g["T"], g.B)})
    h = "<tr><th rowspan=2>Policy</th>" + "".join(
        f"<th colspan={sum(1 for c in cols if c[0] == T)}>sequence length {T}</th>" for T in sorted({c[0] for c in cols})) + "</tr>"
    h += "<tr>" + "".join(f"<th>B={b}</th>" for _, b in cols) + "</tr>"
    best = {c: g[(g["T"] == c[0]) & (g.B == c[1]) & g.family.isin(FAMS)].real_speedup.max() for c in cols}
    body = ""
    for f in ["heads_only"] + FAMS:
        row = f"<td>{NAME.get(f, 'All heads, no exits')}</td>"
        for c in cols:
            r = g[(g["T"] == c[0]) & (g.B == c[1]) & (g.family == f)]
            if r.empty:
                row += "<td>–</td>"
                continue
            r = r.iloc[0]
            s = f"{r.real_speedup:.2f}"
            if f in FAMS and r.real_speedup >= best[c] - 1e-9:
                s = f"<b>{s}</b>"
            if r.ho_drop_pts > 1.0:
                s += "*"
            row += f"<td>{s}</td>"
        body += f"<tr class='{'ours' if 'ours' in NAME.get(f, '') else ''}'>{row}</tr>"
    return f"<table class='num'>{h}{body}</table>"


def regret_table(d: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Per family over all measured cells: mean and worst speedup relative to the best policy of the cell."""
    g = d[d.family.isin(FAMS)].copy()
    g["best"] = g.groupby(["task", "T", "B"]).real_speedup.transform("max")
    g["rel"] = g.real_speedup / g.best
    g["err"] = (g.model_speedup - g.real_speedup).abs() / g.real_speedup
    g["aerr"] = (g.affine_speedup - g.real_speedup).abs() / g.real_speedup
    s = g.groupby("family").agg(mean=("rel", "mean"), worst=("rel", "min"), wins=("rel", lambda x: int((x > 0.97).sum())),
                                n=("rel", "size"), viol=("ho_drop_pts", lambda x: int((x > 1.0).sum())),
                                maxdrop=("ho_drop_pts", "max")).reindex(FAMS)
    rows = "".join(
        f"<tr class='{'ours' if 'ours' in NAME[f] else ''}'><td>{NAME[f]}</td><td>{r['mean']:.3f}</td><td>{r['worst']:.3f}</td>"
        f"<td>{int(r['wins'])} / {int(r['n'])}</td><td>{int(r['viol'])} / {int(r['n'])}</td><td>{r['maxdrop']:.2f}</td></tr>"
        for f, r in s.iterrows())
    t = ("<table class='num'><tr><th>Policy</th><th>Mean speed relative to the cell's best</th><th>Worst cell</th>"
         "<th>Cells within 3% of the best</th><th>Cells over the 1-pt budget (held-out)</th><th>Largest held-out drop (pts)</th></tr>"
         + rows + "</table>")
    return t, g


def resplit_table() -> str:
    out = ""
    for f in sorted(ROOT.glob("results/local_resplit_*_cuda_T16.csv")):
        d = pd.read_csv(f)
        task = f.name.split("_")[2].upper()
        for (B, fam), g in d.groupby(["B", "family"]):
            out += (f"<tr><td>{task}</td><td>{B}</td><td>{NAME[fam].replace(' (ours)', '')}</td>"
                    f"<td>{100 * (g.naive_drop > 0.01).mean():.0f}%</td><td>{100 * (g.cert_drop > 0.01).mean():.0f}%</td>"
                    f"<td>{100 * g.naive_drop.mean():.2f}</td><td>{100 * g.cert_drop.mean():.2f}</td>"
                    f"<td>{g.naive_speedup.mean():.2f}×</td><td>{g.cert_speedup.mean():.2f}×</td><td>{len(g)}</td></tr>")
    if not out:
        return ""
    return ("<table class='num'><tr><th>Task</th><th>B</th><th>Policy</th><th>Over budget, old protocol</th>"
            "<th>Over budget, certified</th><th>Mean drop, old (pts)</th><th>Mean drop, certified (pts)</th>"
            "<th>Speedup, old</th><th>Speedup, certified</th><th>Re-splits</th></tr>" + out + "</table>")


def cost_rows() -> str:
    out = ""
    for T in (16, 32, 128):
        d = json.loads((ROOT / f"results/local_cost_cuda_T{T}.json").read_text())
        y = {p["B"]: p["full_ms"] / 12 for p in d["points"]}
        out += (f"<tr><td>{T}</td><td>{y[1]:.2f}</td><td>{y[8]:.2f}</td><td>{y[64]:.2f}</td><td>{d['head']:.2f}</td>"
                f"<td>{max(d['rho8'], 0):.2f}</td><td>{max(d['rho64'], 0):.3f}</td><td>{100 * d['affine_max_rel_err']:.0f}%</td></tr>")
    return out


CSS = """
@page { size: A4; margin: 21mm 20mm 20mm 20mm; @bottom-center { content: counter(page); font: 9pt 'C059', serif; } }
body { font: 9.6pt/1.36 'C059', 'Century Schoolbook L', 'DejaVu Serif', serif; color: #111; text-align: justify; hyphens: auto; }
h1 { font-size: 19pt; text-align: center; margin: 0 0 2pt; }
.sub { text-align: center; font-size: 12.5pt; margin: 0; }
.sub2 { text-align: center; font-size: 10pt; margin: 3pt 0 12pt; }
.auth { text-align: center; font-size: 10.5pt; margin: 0; }
.inst { text-align: center; font-size: 9pt; margin: 2pt 0 10pt; }
.abs { margin: 0 10mm 8pt; font-size: 9pt; }
.abs b.t { display: block; text-align: center; margin-bottom: 3pt; }
h2 { font-size: 12.5pt; margin: 13pt 0 4pt; page-break-after: avoid; }
h3 { font-size: 10.3pt; margin: 9pt 0 3pt; page-break-after: avoid; }
p { margin: 0 0 5pt; }
ul { margin: 2pt 0 5pt; padding-left: 14pt; } li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin: 5pt auto 3pt; font-size: 8.3pt; text-align: left; page-break-inside: avoid; }
th { border-top: 0.9pt solid #111; border-bottom: 0.5pt solid #111; padding: 2.5pt 5pt; font-weight: 600; vertical-align: bottom; }
td { padding: 2pt 5pt; vertical-align: top; }
table tr:last-child td { border-bottom: 0.9pt solid #111; }
table.num td:not(:first-child), table.num th:not(:first-child) { text-align: center; }
tr.ours td { background: #eef4fc; }
.cap { font-size: 8.3pt; margin: 2pt 6mm 9pt; text-align: justify; }
.fig { text-align: center; margin: 7pt 0 0; page-break-inside: avoid; } .fig img { max-width: 100%; }
.eq { text-align: center; margin: 5pt 0; font-size: 10pt; }
.fb { margin-left: 12pt; } .fb i.l { font-style: italic; }
.tag { font-size: 7.6pt; font-weight: 600; padding: 0 3pt; border: 0.5pt solid #555; border-radius: 2pt; white-space: nowrap; }
.refs p { font-size: 8.3pt; margin: 0 0 2pt; padding-left: 16pt; text-indent: -16pt; text-align: left; }
code { font: 8.2pt 'DejaVu Sans Mono', monospace; }
i.v { font-style: italic; }
"""


def build() -> None:
    d = local()
    reg_html, g = regret_table(d)
    ours = g[g.family == "theory+plan"]
    plain = g[g.family == "theory"]
    n_cells = len(ours)
    tab_err = g.err.median() * 100
    aff_err = g.aerr.median() * 100
    tab_err_max, aff_err_max = g.err.max() * 100, g.aerr.max() * 100
    best_other = g[~g.family.isin(["theory", "theory+plan"])].groupby("family").rel.min().max()
    rs = resplit_table()
    fig = lambda name, cap: (f"<div class='fig'><img src='../figures/{name}'></div><p class='cap'>{cap}</p>"
                             if (ROOT / 'figures' / name).exists() else "")
    v = lambda x: f"<i class='v'>{x}</i>"
    html = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>IE624 progress report, revision 2</title></head><body>
<h1>When Should a Batch Stop?</h1>
<p class='sub'>Optimal Batch-Level Early Exit, Robust to Slowdown Attacks</p>
<p class='sub2'>IE624 Course Project: Progress Report and Plan (revision 2)</p>
<p class='auth'>Divyanshu Gupta (23B0306) &nbsp;&nbsp; Kushagra Gupta (23B3956) &nbsp;&nbsp; Neha Yadav (23B1807)</p>
<p class='inst'>Course instructor: Prof. Manjesh K. Hanawal, IIT Bombay<br>13 November 2026</p>

<div class='abs'><b class='t'>Abstract</b>
Early-exit networks answer easy inputs at intermediate layers, but under batched GPU inference one hard input keeps
the whole batch running, so most of the promised speedup disappears. We measure the per-layer cost {v('c')}({v('n')})
of running {v('n')} active inputs, derive a batch-stopping rule from it, learn the rule online without labels with a
partial-feedback bandit, and give it a certified bound on the damage done by slowdown inputs. This revision corrects
three claims of the first report and adds three results. Corrections: the rule is not the fastest in every setting,
the simulator matches the GPU only at large batches, and the earlier headline speedups were not accuracy-matched.
New: (i) on a second GPU the cost is a floor followed by a linear part, not {v('a')} + {v('b n')}; (ii) letting the
rule choose which exits to evaluate narrows its small-batch gap; (iii) parameters are now chosen with a
statistical test on the accuracy drop. In {n_cells} measured settings on a GTX 1650 (two text tasks, three
sequence lengths, three batch sizes, real wall-clock), the rule with an exit plan is at most
{100 * (1 - ours.rel.min()):.1f}% behind the fastest method, while every baseline is at least
{100 * (1 - best_other):.0f}% behind in some setting. On QQP it stays within the one-point budget everywhere; on
MNLI most methods, ours included, exceed it on the fixed held-out half, although on 30 random re-splits the certified rule stays within it in 93–100% of cases. The report states what
holds, what does not, and the work planned until 26 November.</div>

<h2>1. Problem statement</h2>
<p>Early-exit networks attach small classifiers (exits) after some layers, so an easy input can stop after a few
layers instead of all of them [1, 2]. On a GPU, inputs are served in batches and a batch runs layer by layer for all
its members together. A single hard input, a <i>straggler</i>, keeps the batch running to the last layer, so early
exits save compute (FLOPs) but little wall-clock time. DREX, GEEP and Miao et al. report the same effect for batched
BERT and LLM inference [3, 5, 4].</p>
<p><b>Cost model.</b> One layer run on the {v('n')} inputs still active costs {v('c')}({v('n')}). With exits after layers
{v('l')}<sub>1</sub> &lt; … &lt; {v('l')}<sub>E</sub> = {v('L')}, a set {v('S')} of exits that are actually evaluated, and a cost
{v('h')} per evaluated exit, the latency of a batch is</p>
<p class='eq'>{v('T')}<sub>batch</sub> = Σ<sub>e ∈ S</sub> ({v('l')}<sub>e</sub> − {v('l')}<sub>e−1</sub>) · {v('c')}({v('n')}<sub>e</sub>) + {v('h')} · |{v('S')}|. &nbsp;&nbsp;&nbsp; (1)</p>
<p>The first report assumed {v('c')}({v('n')}) = {v('a')} + {v('b n')} and described the regime by ρ = {v('a')}/({v('b n')}).
When ρ is small (compute-bound), removing an exited input saves real time, so per-input exits work. When ρ is large
(fixed-cost-bound), removing one input saves almost nothing and only stopping the whole batch saves time. That
picture stands, but Section 4.3 shows the affine form is a poor fit on a second GPU, so the method now takes the
measured {v('c')}({v('n')}) directly.</p>
<p><b>Gap.</b> Existing batch-level rules are heuristics: Consensus waits until every input is confident, and
Majority and Greedy stop when half or one of them is [3, 4]. None is derived from the cost. Apparate chooses which
exits to keep by their saving minus their overhead, but all inputs still run to the end of the model and accuracy
is monitored without a guarantee [6]. Risk-controlled early exit gives a guarantee, but for per-input exits with no
batch coupling [29].</p>
<p><b>Security coupling.</b> A slowdown attacker crafts inputs that never become confident [8, 9, 10]. In a
synchronous batch, such an input delays every co-batched user, not only itself.</p>
<p><b>Research questions.</b> Given the measured {v('c')}({v('n')}): (1) when should a batch stop, and which exits are
worth evaluating; (2) can the rule's parameters be set with a guarantee on accuracy, and learned online without
labels; (3) can the damage from slowdown inputs be bounded without detecting them?</p>

<h2>2. Approach</h2>
<p><b>(a) Batch-stopping rule for a measured cost.</b> Let {v('g')}<sub>j</sub>({v('e')}→{v('e′')}) be the expected accuracy gain
of carrying input {v('j')} from exit {v('e')} to a later exit {v('e′')}, estimated from a calibration table of
P(correct at {v('e′')} | confidence at {v('e')}). With {v('F')} the full model's latency and λ the price of latency,
input {v('j')} leaves alone if</p>
<p class='eq'>{v('g')}<sub>j</sub>({v('e')}→{v('E')}) / {v('B')} &lt; λ · [{v('c')}({v('n')}) − {v('c')}({v('n')}−1)] · ({v('L')} − {v('l')}<sub>e</sub>) / {v('F')}, &nbsp;&nbsp;&nbsp; (2)</p>
<p>and the remaining inputs {v('R')} stop together if no later evaluated exit is worth reaching:</p>
<p class='eq'>∀ {v('e′')} ∈ {v('S')}, {v('e′')} &gt; {v('e')}: &nbsp; (1/{v('B')}) Σ<sub>j ∈ R</sub> {v('g')}<sub>j</sub>({v('e')}→{v('e′')}) &lt; λ · [({v('l')}<sub>e′</sub> − {v('l')}<sub>e</sub>) · {v('c')}(|{v('R')}|) + ({v('e′')} − {v('e')}) · {v('h')}] / {v('F')}. &nbsp;&nbsp;&nbsp; (3)</p>
<p>An individual exit saves only the marginal cost {v('c')}({v('n')}) − {v('c')}({v('n')}−1); a collective stop saves all of
{v('c')}(|{v('R')}|). For the affine cost these are {v('b')} and {v('a')} + {v('b')}|{v('R')}|, which recovers the rule of
the first report and its threshold ratio of about 1/(1 + ρ). For a cost with a floor the marginal cost is zero below
the knee, so the rule switches from per-input exits to whole-batch stopping <i>inside a batch</i> as it drains.</p>
<p><b>(b) Exit plan (new).</b> The first rule evaluated every exit and paid {v('h')} for each. The plan {v('S')} is now a
choice: any subset of the intermediate exits, with the rule (2)–(3) run on that subset. With one exit kept the plan
behaves like a unified exit; with all exits kept it is the earlier rule. The plan and λ are chosen together by
replaying calibration traces under the measured {v('c')}({v('n')}) and {v('h')}; this needs no GPU time.</p>
<p><b>(c) Certified selection (new).</b> The first report chose the fastest setting whose accuracy on the tuning
split was within one point of the full model, and it then lost 0.8–2.4 points on held-out data. We now split the
tuning data in two. On the first part each method's candidates are reduced to a path ordered from safest to
fastest. On the second part the path is tested in order: for candidate θ, the drop in accuracy against the full
model is computed per batch on {v('K')} disjoint batches, {v('Z')}<sub>1</sub>, …, {v('Z')}<sub>K</sub>, and θ is certified
if a one-sided {v('t')}-test rejects "mean drop &gt; α" at level δ. Testing stops at the first candidate that fails
(fixed-sequence testing, as in Learn-then-Test [30]); the fastest certified candidate is used. We use α = 1 point
and δ = 10%. Batches, not inputs, are the unit, because a batch-level rule couples the inputs of a batch. The
{v('t')}-test is valid for large {v('K')}; a finite-sample bound is planned (Section 5.2). The certificate holds for the
batch size and cost it was computed for, and recomputing it for a new deployment is a replay on the CPU.</p>
<p><b>(d) Label-free, partial-feedback bandit.</b> Unchanged. The bandit replaces "correct" with "agrees with the full
model", which upper-bounds the accuracy loss. Feedback is censored: once a batch stops, its deeper exits are never
seen. Probe batches run to full depth with probability {v('c t')}<sup>−1/3</sup>, the partial-monitoring rate for a
revealing action [12]. Prior bandits for early exit tune per-input thresholds [13, 14].</p>
<p><b>(e) Certified robustness.</b> Unchanged. The robust rule leaves the {v('m')} largest gains out of the collective
sum in (3). For {v('k')} ≤ {v('m')} slowdown inputs {v('K')} among clean inputs {v('C')},
TS<sub>m</sub>({v('C')}) ≤ TS<sub>m</sub>({v('C')} ∪ {v('K')}) ≤ TS<sub>m−k</sub>({v('C')}), hence
depth({v('C')} ∪ {v('K')}; {v('m')}) ≤ depth({v('C')}; {v('m')} − {v('k')}) for every batch.</p>

<h2>3. Experimental setup</h2>
<table><tr><th>Model</th><th>Task</th><th>Exits after layer</th><th>Full-model acc.</th><th>Used in</th></tr>
<tr><td>RoBERTa-base [21]</td><td>MNLI [22]</td><td>2, 4, …, 12</td><td>87.8%</td><td>server and local</td></tr>
<tr><td>RoBERTa-base</td><td>QQP [23]</td><td>2, 4, …, 12</td><td>91.1%</td><td>server and local</td></tr>
<tr><td>ViT-B/16 [24]</td><td>CIFAR-100 [25]</td><td>2, 4, …, 12</td><td>93.3%</td><td>server</td></tr>
<tr><td>Qwen2.5-1.5B, Llama-3.1-8B (frozen) [26, 27]</td><td>WikiText-103 [28]</td><td>6 exits</td><td>49.9%, 53.6%</td><td>server (negative result)</td></tr></table>
<p class='cap'><b>Table 1:</b> Models and tasks. Encoder backbones were fine-tuned jointly with their six exits.</p>
<ul>
<li><b>Server.</b> One RTX A6000, fp16. Two execution modes: eager PyTorch and CUDA graphs (a compiled stack). All
server numbers come from trained checkpoints running on real inputs.</li>
<li><b>Local (new).</b> One GTX 1650 (4 GB), fp32 (fp16 is about four times slower on this card). The trained checkpoints
are not on this machine. The timed network has the trained model's architecture and tensor shapes with random
weights; every layer, exit head, host synchronisation, batch compaction and policy call is executed and timed, and
the exit decisions and predictions are those recorded for the trained model on the same inputs. Accuracy is
therefore exact for the trained model and time is exact for this hardware. Sequence lengths 16 and 32 use decisions
recorded at length 128; they show how the same policies behave when per-input compute is smaller, and they are not a
claim about truncated inputs.</li>
<li><b>Protocol.</b> Each validation set is split in half. Parameters are chosen on the tuning half by certified
selection (Section 2c; 40% for tables and candidate paths, 60% for the test) and reported on the held-out half.
Local timing runs 1,920 held-out inputs per pass at lengths 16 and 32 (5 rounds) and 768 at length 128 (3 rounds),
with policies interleaved within a round; we report the median over rounds of the paired speedup and its range.</li>
<li><b>Baselines.</b> Per-sample threshold (DeeBERT-style), unified exit (SkipDecode-style [20], one head evaluated),
Majority (as in DREX), a 150-trial random search over a 3-parameter collective rule, and the per-sample threshold
given the same exit-plan search as our rule (an Apparate-style selection of exits [6]).</li>
</ul>

<h2>4. Current status</h2>
<h3>4.1 Corrections to the first report</h3>
<table><tr><th>First report said</th><th>What the evidence shows</th><th>Source</th></tr>
<tr><td>The rule matches or beats the best heuristic in every cell</td><td>With CUDA graphs it is never the single fastest rule: unified exit wins at B ≤ 16 on text, a random search ties or wins at B ≥ 64. At B = 8 on MNLI it was the slowest rule (0.86×)</td><td><code>results/graph_*_gpu.csv</code></td></tr>
<tr><td>1.41×, 2.10×, 1.46× at B = 64 within one accuracy point</td><td>Those were simulated and selected on the held-out split. The real-GPU runs lost 0.8–2.4 points on held-out data</td><td><code>docs/RESULTS.md</code></td></tr>
<tr><td>The simulator tracks the GPU within about 5%</td><td>Only at B ≥ 64 in eager mode; 15–32% off at B ≤ 16, and it overestimates in graph mode</td><td><code>docs/RESULTS.md</code></td></tr>
<tr><td>Every collective rule beats per-sample exits at B ≤ 16</td><td>True on MNLI and QQP, not on CIFAR-100</td><td><code>results/main_cifar100_*</code></td></tr>
<tr><td>Robustness costs no clean speedup</td><td>True at high ρ. At low ρ on CIFAR-100 the robust rule reaches 1.12× against 1.46×</td><td><code>results/robust_*</code></td></tr>
<tr><td>Tuned parameters carry over between deployments (assumed)</td><td>Tested and false for every method: in simulation the rule broke the one-point budget in 29–31% of transfers</td><td><code>results/transfer_summary.csv</code></td></tr></table>
<p class='cap'><b>Table 2:</b> Claims of the first report that were narrowed or withdrawn.</p>

<h3>4.2 Claim 1: no rule wins everywhere, and the winner flips with the deployment (server)</h3>
{fig('report2_server_speedups.png', '<b>Figure 1:</b> Real wall-clock speedup against the full model on an RTX A6000, held-out split. Left: eager PyTorch. Right: CUDA graphs. These runs used the old selection protocol and are not accuracy-matched (on MNLI the cost rule lost 1.8–2.5 points, unified exit 0.8–0.9); Section 5.1 reruns them with certified selection.')}
<p>Figure 1 shows the server measurements. Three statements hold in both execution modes. Wait-for-all early exit,
which keeps exited inputs in the batch, is slower than no early exit (0.78–0.97×). Per-sample exits fall below 1× at
small batches on text (0.77–0.90× eager, 0.89× on MNLI with CUDA graphs). And the ordering of rules changes between
B = 8 and B = 128. Compiling cuts the fixed per-layer cost about six-fold, and the cause of the small-batch loss
changes with it: under CUDA graphs it is the exit heads, which alone cost 19% of the runtime at B = 8.</p>

<h3>4.3 New: the cost is a floor followed by a linear part (GTX 1650)</h3>
{fig('report2_cost_gtx1650.png', '<b>Figure 2:</b> Measured time of one RoBERTa-base layer on a GTX 1650 (fp32) against the number of active inputs, at three sequence lengths, with the affine fit of the first report. Both axes are logarithmic.')}
<table class='num'><tr><th>Sequence length</th><th>c(1) ms</th><th>c(8) ms</th><th>c(64) ms</th><th>h ms</th><th>ρ at n=8</th><th>ρ at n=64</th><th>Affine fit, largest error</th></tr>{cost_rows()}</table>
<p class='cap'><b>Table 3:</b> Per-layer cost and exit-head cost on the GTX 1650 (<code>results/local_cost_cuda_T*.json</code>).</p>
<p>A layer costs about 1.8 ms however few inputs are active, and then grows in proportion to the number of tokens in
the batch. The knee is near 128 tokens: one input at length 128, four at length 32, eight at length 16. The affine
model is off by up to 41% at short sequences. One GPU therefore covers both regimes: at length 128 it is
compute-bound at every batch size, and at length 16 with B = 8 a per-input exit saves nothing. Policies below are
chosen with the measured table. The gain in predicting <i>speedups</i> is small, because a speedup is a ratio and
errors partly cancel: over the {len(g)} policy runs timed in Section 4.4 the table's median error is {tab_err:.1f}%
(largest {tab_err_max:.0f}%), against {aff_err:.1f}% (largest {aff_err_max:.0f}%) for the affine fit applied to the
same decisions. The floor matters for the rule's decisions more than for the prediction.</p>

<h3>4.4 New: exit plan and certified selection, real wall-clock (GTX 1650)</h3>
{fig('report2_local_speedups.png', '<b>Figure 3:</b> Real wall-clock speedup on the GTX 1650 of the certified policy of each method, held-out inputs. Markers are the median over rounds; vertical bars are the range. Dashed blue: the cost rule with all exits; solid blue: with an exit plan.')}
{cell_table(d, 'qqp')}
<p class='cap'><b>Table 4:</b> QQP, GTX 1650, measured speedup of each method's certified policy. Bold: fastest in the column. * held-out accuracy drop above one point.</p>
{cell_table(d, 'mnli')}
<p class='cap'><b>Table 5:</b> MNLI, same layout.</p>
{reg_html}
<p class='cap'><b>Table 6:</b> Summary over the {n_cells} measured settings (2 tasks × 3 sequence lengths × 3 batch sizes). Speed is relative to the fastest method of each setting.</p>
RESULTS_TEXT
<h3>4.5 New: does the certificate hold?</h3>
RESPLIT_TEXT
<h3>4.6 Claim 2: the label-free bandit learns the rule (unchanged, server)</h3>
<p>Against an oracle that knows the true tables, cumulative regret over 3,000 batches of 64 grows with log–log slope
0.49–0.65 on MNLI and QQP, while a CeeBERT-style per-input bandit's regret grows linearly. On the GPU the bandit uses
no labels, costs 0.2–0.3 accuracy points, and reaches 1.26–1.77× at B = 64–128. Two cautions that the first report
did not state: these are single runs fitted on a few points, and a slope below 2/3 does not "confirm" the
{v('T')}<sup>2/3</sup> rate, which is a worst-case bound. Section 5.4 replaces them with ten seeds.</p>
<h3>4.7 Claim 3: certified, detection-free robustness (unchanged, server)</h3>
<p>The bound depth({v('C')} ∪ {v('K')}; {v('m')}) ≤ depth({v('C')}; {v('m')} − {v('k')}) held in all 10,500 batches tested
across five tasks, including real DeepSloth inputs on the ViT; the property test passes on the local machine as
well (19 of 19 unit tests). Consensus loses its whole speedup to a single attacker. Victims' latency under our rule
still rises with {v('k')}, and the robust rule has a clean-speed cost in the low-ρ regime (Table 2).</p>
<h3>4.8 What did not work</h3>
<ul>
<li><b>A faster rule at a fixed setting.</b> Random search, program search (FunSearch-style, 300 programs),
trajectory-aware calibration and a hindsight-best batch depth all tie the rule. A hidden-state probe adds at most
0.026 AUROC over confidence. The limit is the exit signal, not the stopping rule.</li>
<li><b>Decoders.</b> The cost model holds on Qwen2.5-1.5B and Llama-3.1-8B, but exits are too weak: frozen heads and
public LayerSkip checkpoints are both about 5 points short at 75% depth, and no rule beats the full model within
one point.</li>
<li><b>Transfer of tuned parameters.</b> See Table 2. This is why selection is now redone per deployment with a
certificate, which costs a CPU replay.</li>
<li><b>Straggler rescue</b> brought no measurable gain.</li>
</ul>

<h2>5. Planned work until 26 November</h2>
<p>Each item lists what we will do, what we expect, and the fallback.</p>
<h3>5.1 Accuracy-matched server tables with the exit plan (16–18 Nov)</h3>
<p>Rerun the 24 server settings (3 tasks × 4 batch sizes × eager and CUDA graphs) with the exit plan, the measured
cost table and certified selection, on trained checkpoints. This is the main evidence still missing.</p>
<p class='fb'><i class='l'>Expected:</i> the rule with a plan within 8% of the fastest method in every setting, as on the
GTX 1650, with no setting over the accuracy budget beyond the 10% allowed.<br>
<i class='l'>Fallback:</i> if it is more than 10% behind at B ≤ 16 under CUDA graphs, report the plan as a partial fix and
state the settings where a unified exit should be used.</p>
<h3>5.2 Tighten the certificate (14–16 Nov)</h3>
<p>Replace the {v('t')}-test with a finite-sample bound for bounded batch losses (Hoeffding–Bentkus), add the
label-free version that controls disagreement with the full model, and repeat the re-split check on CIFAR-100.</p>
<p class='fb'><i class='l'>Expected:</i> the same violation rates with a speed cost of a few percent.<br>
<i class='l'>Fallback:</i> keep the {v('t')}-test and state it as an asymptotic guarantee with the measured violation rate.</p>
<h3>5.3 Fluctuating load (19–21 Nov)</h3>
<p>Replay an arrival trace in which the batch size moves between 1 and 128. Compare the rule, re-certified for each
batch size from one set of calibration traces, against heuristics tuned at a single batch size and against
heuristics re-tuned per batch size.</p>
<p class='fb'><i class='l'>Expected:</i> within 5% of per-batch-size re-tuned heuristics and ahead of heuristics tuned once.<br>
<i class='l'>Fallback:</i> if tuned-once heuristics are as good, drop the load claim and keep the per-setting results.</p>
<h3>5.4 Bandit with error bars (22–23 Nov)</h3>
<p>Ten seeds with confidence bands, a probe-rate ablation ({v('t')}<sup>−1/2</sup>, {v('t')}<sup>−1/3</sup>, constant), and a
warm start from a small labelled set.</p>
<p class='fb'><i class='l'>Expected:</i> slopes between 0.5 and 0.7 where the oracle stops batches.<br>
<i class='l'>Fallback:</i> report the measured slopes without a rate claim.</p>
<h3>5.5 Robustness in the compute-bound regime (22–23 Nov)</h3>
<p>Run the attack experiments where majority vote loses its clean speedup, and test an attacker that uses
{v('k')} &gt; {v('m')} inputs.</p>
<p class='fb'><i class='l'>Expected:</i> the bound holds; the clean-speed cost at low ρ is quantified per task.<br>
<i class='l'>Fallback:</i> claim only the certificate.</p>
<h3>5.6 Trained checkpoints on the local GPU (when checkpoints are copied)</h3>
<p>Repeat Section 4.4 with the trained weights and real tokenised inputs, with batches padded to their own longest
input, so that the short-sequence settings are measured on real short inputs.</p>
<p class='fb'><i class='l'>Expected:</i> the same ordering; timing does not depend on weight values.<br>
<i class='l'>Fallback:</i> keep the replayed-decision results, labelled as such.</p>
<h3>5.7 LLM decoders, second attempt (future work; scripts ready, not run)</h3>
<p>The first attempt failed for reasons we can now name: the backbone was frozen while its exits were trained (the
same fault the encoders had before joint training), LayerSkip checkpoints are trained for drafting and not for
returning the early answer, the test was run only at batch sizes 16 and 64, and the budget was one point of
next-token accuracy on free text. The second attempt trains LoRA adapters and exit heads together with
self-distillation from the final layer on Qwen2.5-1.5B (<code>decoder/train_exit_lora.py</code>), keeps a frozen
baseline at the same budget, and sweeps batch sizes from 2 to 64. The runbook with commands and pass and kill
conditions for each step is <code>docs/PLAN_LLM.md</code>; the driver is <code>scripts/run_llm_v2.sh</code>. Nothing in this
track has been run, including the functional test of the new training script.</p>
<p class='fb'><i class='l'>Expected:</i> the 75%-depth exit within about 2 points of the full model (5.6 with a frozen
backbone), and a speedup at small batch sizes. This is a forecast.<br>
<i class='l'>Fallback:</i> if the exit stays more than 4 points short, the LLM track ends as a negative result with its
cause.</p>
<h3>5.8 Theory and write-up (24–26 Nov)</h3>
<p>Write the optimality statement for a general non-decreasing {v('c')}({v('n')}) under the stated assumptions, the
regime switch at the knee of a floor-shaped cost, and the regret sketch.</p>
<p class='fb'><i class='l'>Fallback:</i> clearly labelled proof sketches.</p>
<h3>5.9 Timeline</h3>
<table><tr><th>Dates</th><th>Work</th></tr>
<tr><td>13–15 Nov</td><td>Local timing (done, Section 4.4); re-split check (Section 4.5); finite-sample certificate; label-free certificate</td></tr>
<tr><td>16–18 Nov</td><td>Server rerun of all 24 settings with the exit plan and certified selection</td></tr>
<tr><td>19–21 Nov</td><td>Fluctuating-load experiment</td></tr>
<tr><td>22–23 Nov</td><td>Bandit with ten seeds; robustness at low ρ</td></tr>
<tr><td>24–26 Nov</td><td>Theory write-up, final figures, presentation</td></tr>
<tr><td>When a server GPU is free</td><td>LLM second attempt, steps 0–2 of <code>docs/PLAN_LLM.md</code> (about 4 GPU-hours, estimate)</td></tr></table>
<h3>5.10 Claims and their status</h3>
<table><tr><th>Claim</th><th>Status</th><th>Evidence still required</th></tr>
<tr><td>Which rule is fastest flips with batch size, execution mode and sequence length</td><td><span class='tag'>HOLDS</span> on two GPUs</td><td>Accuracy-matched server rerun (5.1)</td></tr>
<tr><td>The per-layer cost is a floor plus a linear part, not affine</td><td><span class='tag'>HOLDS</span> on the GTX 1650</td><td>The same profile on the A6000; whether it changes realised speed, not only the fit</td></tr>
<tr><td>The rule is never far from the fastest method (within 8% in 18 settings; baselines 15–43%)</td><td><span class='tag'>HOLDS</span> locally, replayed decisions</td><td>Trained checkpoints on the server (5.1), CIFAR-100</td></tr>
<tr><td>The exit plan fixes the small-batch weakness</td><td><span class='tag'>PARTLY</span> small effect locally</td><td>Server run under CUDA graphs, where heads cost 19% (5.1)</td></tr>
<tr><td>Certified selection keeps the accuracy budget</td><td>CERT_STATUS</td><td>Finite-sample bound (5.2)</td></tr>
<tr><td>The label-free bandit learns the rule</td><td><span class='tag'>PARTLY</span> single seeds</td><td>Ten seeds (5.4)</td></tr>
<tr><td>Certified robustness</td><td><span class='tag'>HOLDS</span> theorem, 0 violations</td><td>Low-ρ cost, adaptive attacker (5.5)</td></tr>
<tr><td>Benefit on LLM decoders</td><td><span class='tag'>FAILED</span> first attempt</td><td>Second attempt with jointly trained exits (5.7), not started</td></tr></table>

<h2>6. Expected outcome</h2>
<p>By 26 November we expect an accuracy-matched demonstration on two GPUs that each existing rule is far from the
best in some deployment while the cost-derived rule with an exit plan is not; a selection procedure with a stated
guarantee on the accuracy drop; a label-free bandit with error bars; and the certified robustness result. We do not
expect a rule that is fastest everywhere. A positive result on LLM decoders depends on the second attempt (5.7) and is not assumed. For a conference submission the
evidence would still need more models, trained checkpoints timed on both GPUs, and the load experiment; with those,
the work fits a systems-leaning venue.</p>

<h2>References</h2><div class='refs'>
<p>[1] J. Xin, R. Tang, J. Lee, Y. Yu, J. Lin. DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. ACL 2020.</p>
<p>[2] W. Zhou, C. Xu, T. Ge, J. McAuley, K. Xu, F. Wei. BERT Loses Patience: Fast and Robust Inference with Early Exit. NeurIPS 2020.</p>
<p>[3] X. Liu et al. Dynamic Rebatching for Efficient Early-Exit Inference with DREX. arXiv:2512.15705, 2025.</p>
<p>[4] R. Miao, Y. Yan, X. Yao, T. Yang. An Efficient Inference Framework for Early-exit Large Language Models. arXiv:2407.20272, 2024.</p>
<p>[5] L. Li et al. Accelerating BERT Inference with GPU-Efficient Exit Prediction. Frontiers of Computer Science 18(3), 2024.</p>
<p>[6] Y. Dai, R. Pan, A. Iyer, K. Li, R. Netravali. Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving. SOSP 2024.</p>
<p>[7] A. Iyer et al. Improving DNN Inference Throughput Using Practical, Per-Input Compute Adaptation (E3). SOSP 2024.</p>
<p>[8] S. Hong, Y. Kaya, I.-V. Modoranu, T. Dumitraş. A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference. ICLR 2021.</p>
<p>[9] Z. Coalson, G. Ritter, R. Bobba, S. Hong. BERT Lost Patience Won't Be Robust to Adversarial Slowdown. NeurIPS 2023.</p>
<p>[10] Y. Chen et al. Dynamic Transformers Provide a False Sense of Efficiency. ACL 2023.</p>
<p>[11] Y. S. Chow, H. Robbins, D. Siegmund. Great Expectations: The Theory of Optimal Stopping. Houghton Mifflin, 1971.</p>
<p>[12] N. Cesa-Bianchi, G. Lugosi, G. Stoltz. Minimizing Regret with Label Efficient Prediction. IEEE Trans. Information Theory 51(6), 2005.</p>
<p>[13] D. J. Bajpai, M. K. Hanawal. CeeBERT: Cross-Domain Inference in Early Exit BERT. Findings of ACL 2024.</p>
<p>[14] R. G. Pacheco et al. UCBEE: A Multi-Armed Bandit Approach for Early-Exit in Neural Networks. IEEE Trans. Network and Service Management, 2024.</p>
<p>[18] M. Elhoushi et al. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024.</p>
<p>[20] L. Del Corro et al. SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference. arXiv:2307.02628, 2023.</p>
<p>[21] Y. Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692, 2019.</p>
<p>[22] A. Williams, N. Nangia, S. Bowman. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. NAACL 2018.</p>
<p>[23] A. Wang et al. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR 2019.</p>
<p>[24] A. Dosovitskiy et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.</p>
<p>[25] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Tech. report, University of Toronto, 2009.</p>
<p>[26] Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.</p>
<p>[27] Llama Team, Meta AI. The Llama 3 Herd of Models. arXiv:2407.21783, 2024.</p>
<p>[28] S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer Sentinel Mixture Models. ICLR 2017.</p>
<p>[29] M. Jazbec, A. Timans, et al. Fast yet Safe: Early-Exiting with Risk Control. NeurIPS 2024.</p>
<p>[30] A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. arXiv:2110.01052, 2021.</p>
<p>Reference numbers follow the first report; [15]–[17] and [19] (program search, CALM) are not cited in this revision.</p>
</div></body></html>"""
    extra = json.loads((ROOT / "docs/report_v2_text.json").read_text())
    html = html.replace("RESULTS_TEXT", extra["results"]).replace("RESPLIT_TEXT", (rs + extra["resplit"]) if rs else extra["resplit_pending"])
    html = html.replace("CERT_STATUS", extra["cert_status"] if rs else "<span class='tag'>PENDING</span>")
    out = ROOT / "docs/IE624_progress_report_v2.html"
    out.write_text(html)
    from weasyprint import HTML
    HTML(str(out)).write_pdf(str(out.with_suffix(".pdf")))
    print("wrote", out.with_suffix(".pdf"))
    print(g.groupby("family").agg(mean=("rel", "mean"), worst=("rel", "min")).round(3))


if __name__ == "__main__":
    build()
