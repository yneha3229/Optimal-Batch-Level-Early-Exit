# Plan v2: two tracks built on what held up (written 2026-11-12)

> **Parked.** The team chose the direction in `docs/DIRECTION.md` instead (2026-11-13). Nothing here was started.

Alternative to the paper track in `docs/PLAN_PAPER.md`, whose steps 1–3 all failed.

The two tracks share one write-up: Track A says what label-free learning of an exit rule can and cannot achieve;
Track B says what ceiling any exit rule faces on real hardware.

- **Track A (theory, core):** label-free early exit as a feedback-graph bandit, with a lower bound.
- **Track B (empirical, supporting):** a reality check of early exit under batching and compilation, with a ceiling
  analysis.

---

## Track A: what does it cost to learn exits without labels?

### Claim to prove
Without labels and without a dominance assumption, learning an exit rule online has minimax regret Θ(T^(2/3)). The
probe-based bandit in `bandit/bandit.py` matches that rate. Under an extra assumption (calibrated confidence, or
strong dominance), the rate improves.

### Why this is plausibly new
- UEE-UCB (Hari Narayan, Hanawal, Bhardwaj, arXiv 2209.09480) gets O(log n) regret without labels, but only under
  the Strong Dominance assumption, which the authors say Elastic BERT only "nearly" satisfies.
- Our setting drops that assumption. Then an exited input is never checked: only a full-depth pass reveals whether an
  earlier exit agreed with the full model. Each exit arm sees its own cost but not its own error; the full-model arm
  sees everything. In the feedback-graph classification of Alon, Cesa-Bianchi, Dekel and Koren (arXiv 1502.07617)
  that is a weakly observable graph, whose minimax regret is Θ(T^(2/3)), against Θ(T^(1/2)) for strongly observable
  graphs.
- **Not yet checked:** whether Bajpai and Hanawal's later papers (CeeBERT, UAT, DAdEE) or the unsupervised sequential
  selection line already state this. Step A0 settles it.

### Steps

| Step | Work | Pass condition | Kill criterion | Time (estimate) |
|---|---|---|---|---|
| A0. Literature check | Read UEE-UCB, CeeBERT, UAT, DAdEE, the unsupervised sequential selection papers (Hanawal, Szepesvári, Saligrama), Alon et al. | No paper states a T^(2/3) lower bound for label-free exit selection without dominance | Already stated: drop the lower bound, keep A2–A4 as an empirical study | 1 day |
| A1. Problem statement | Write the model in `theory/THEORY.md`: arms, costs, fidelity loss, what each arm observes; show it is a weakly observable graph | A formal statement the team agrees on | n/a | 1 day |
| A2. Lower bound | Two-exit hard instance: exit 1's disagreement rate is μ ± ε; only the full pass reveals it, and the full pass is suboptimal by a constant when exit 1 is good. Standard change-of-measure argument | Ω(T^(2/3)) for the stochastic structured instance, not only by citing the adversarial result | Proof does not go through in 4 days: state it as a reduction to Alon et al. with the conditions made explicit, and say so | 3–4 days |
| A3. Upper bound | Regret proof for the existing algorithm (probe rate c·t^(−1/3), empirical-Bernstein bonus) | Õ(T^(2/3)) with explicit constants | Fall back to an explore-then-commit variant with the textbook proof | 2–3 days |
| A4. When √T is possible | Under strong dominance or calibrated confidence the graph becomes strongly observable; state the rate and test whether the assumption holds on our traces | Assumption measured on MNLI, QQP, CIFAR-100 with a number, plus a run showing the faster rate where it holds | Assumption fails everywhere: report that as the finding | 2 days |
| A5. Experiments | Regret curves with 10 seeds and confidence bands; fitted slopes; comparison with UEE-UCB and CeeBERT-style baselines; probe-rate ablation (t^(−1/2), t^(−1/3), constant) | Slope within 0.60–0.72 where the oracle is non-trivial, and the t^(−1/3) rate is the best of the three | Slopes do not separate: report as measured | 2 days, CPU |

### What exists already
- `bandit/bandit.py`, `bandit/run_bandit.py`: the algorithm and the regret harness.
- Measured slopes 0.49–0.65 on MNLI and QQP (3,000 batches of 64); CeeBERT-style baseline has linear regret.
- Caveat to fix: single seeds, and slopes fitted on four points. A5 replaces these.

### Risks
- The batch setting adds contexts (the confidences) and a continuous threshold. Keep the theorem to finitely many
  exits and a finite threshold grid; state the batch case as the experimental setting.
- Slopes of 0.49 are below 2/3. That is allowed (a lower bound is worst-case), but the write-up must not claim the
  experiments "confirm" T^(2/3).

---

## Track B: does early exit survive batching and compilation?

### Claim to support
Savings reported in FLOPs or at batch size 1 shrink or vanish under batched, compiled GPU inference, and the limit is
the exit signal, not the stopping rule.

### Why this is plausibly new
- Systems papers (Apparate; E3, citation to verify) note the batching problem and engineer around it. A search found
  no systematic evaluation of the gap, and none with a ceiling analysis.
- The new parts are the hindsight bound and the probe result, not the observation that batching hurts.

### Evidence already in hand

| Finding | Where |
|---|---|
| Wait-for-all gives 0.78–0.97× on real GPU | `results/main_*_gpu.csv` |
| Per-sample exits 0.77–0.90× at B ≤ 16 (eager); 0.89× at B = 8 on MNLI (compiled), caused by exit-head overhead | `results/graph_*_gpu.csv` |
| Compiling cuts the fixed per-layer cost about 6×; no rule is consistently fastest | `results/runner_cost_graph_*.json` |
| A hindsight-best batch depth does not beat the rule; label hindsight is 2× higher | `analysis/headroom.py` |
| Hidden-state probes add at most +0.026 AUROC over confidence | `results/probe_*.csv` |
| Random search, FunSearch and trajectory calibration tie the derived rule | `docs/RESULTS.md` |
| Frozen heads and LayerSkip exits are about 5 points short at 75% depth | `results/layerskip_*_layer_accuracy.csv` |

### Steps

| Step | Work | Pass condition | Time (estimate) | GPU |
|---|---|---|---|---|
| B0. Literature check | Apparate, E3, SelfXit, the early-exit survey (arXiv 2106.05022), "efficiency misnomer" | Confirm no prior systematic study with a ceiling bound; fix the E3 citation | 1 day | none |
| B1. Fix the evaluation | Three-way split (fit / select / report); 5 seeds of batch draws; error bars on every table | Every headline number has an interval, and selection never touches the report split | 2 days | none |
| B2. Matched accuracy on real GPU | Select at a stricter margin so held-out drops stay within 1 point; rerun eager and compiled timings | All timed policies within 1 point, or reported on a speed–accuracy curve | 1 day | GPU 5, idle |
| B3. Make the simulator honest in compiled mode | Model head cost as a function of n and the bucket padding; refit | Simulated vs real within 10% at every batch size, else report real GPU only | 1–2 days | GPU 5 |
| B4. Ceiling analysis as a result | Write the decomposition (given the stop depth, per-input exits decouple) and report each policy as a fraction of the hindsight bounds, with seeds | A table of "fraction of headroom captured" for 3 tasks × 2 execution modes | 1 day | none |
| B5. One more backbone | A second encoder (for example DeBERTa-v3-base or BERT-large) through the same pipeline | Same qualitative picture, or a documented exception | 1–2 days | GPU 5 |

### Risks
- It reads as a list of negatives. The ceiling analysis (B4) is what makes it a finding, so it gets written first.
- Two encoders and one ViT are a narrow base for a broad title. Scope the claim to "encoder classification on one
  GPU family" unless B5 and a second GPU are added.
- Venue level: TMLR, EMNLP Findings or a workshop. Not a main-conference result on its own.

---

## Timeline to the presentation (Nov 26)

| Dates | Track A | Track B |
|---|---|---|
| Nov 13 | A0 literature check | B0 literature check |
| Nov 14–15 | A1 problem statement | B1 splits, seeds, error bars |
| Nov 16–19 | A2 lower bound, A5 seeded regret runs | B2 matched-accuracy GPU reruns, B4 ceiling table |
| Nov 20–22 | A3 upper bound, A4 assumption test | B3 simulator fix (or drop to real GPU only) |
| Nov 23–24 | Write-up | B5 second backbone if time allows |
| Nov 25 | Slides and rehearsal | |

Suggested split for three people: one on A1–A3 (proofs), one on A4–A5 and B1 (experiments, CPU), one on B2–B5
(GPU work).

## Decision points
- **A0 finds the lower bound already published:** Track A becomes a seeded empirical comparison against UEE-UCB and
  CeeBERT; Track B becomes the core.
- **A2 stalls:** present the reduction to the feedback-graph result with the conditions stated, and mark the direct
  proof as open.
- **B2 changes the ordering of rules:** report the new ordering; the claim "no rule is consistently fastest" already
  allows it.

## After Nov 26
A paper needs, beyond the above: the direct lower-bound proof, a second GPU family, and a decoder section limited to
the measured negative. Realistic effort is 4–6 weeks more.

## Sources checked on 2026-11-12
- UEE-UCB: https://arxiv.org/abs/2209.09480
- Feedback graphs, strongly vs weakly observable: https://arxiv.org/abs/1502.07617
- Early-exit survey: https://arxiv.org/abs/2106.05022
