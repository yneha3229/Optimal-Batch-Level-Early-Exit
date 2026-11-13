# Plan: from course project to paper (written 2026-11-12)

> **Stopped.** Steps 1–3 failed their kill criteria (section 7). The current direction is `docs/DIRECTION.md`.

Status: **steps 1–3 run on 2026-11-12. All three hit their kill criteria (section 7).** Earlier results are in
`docs/RESULTS.md`.

## 1. Why the plan changes

A better stopping policy on the encoders is not the route to a paper. Four results say the rule is already at the
ceiling of what confidence scores allow:

- A random search over a 3-parameter rule ties or beats the theory rule.
- FunSearch ties it (1.406 vs 1.411) and overfits.
- Trajectory-aware calibration is within ±0.07× everywhere.
- A hindsight-best batch depth does not beat it (table below).

Headroom check, 2026-11-12 (held-out split, B = 64, simulator, real runner cost, speedup within 1 point of the full
model). Script: `analysis/headroom.py`.

| | MNLI | QQP | CIFAR-100 |
|---|---|---|---|
| Theory rule | 1.38× | 2.05× | 1.40× |
| Hindsight-best single depth per batch | 1.39× | 2.04× | 1.23× |
| Theory rule on difficulty-sorted batches | 1.39–1.40× | 2.25–2.27× | 1.41–1.42× |
| Hindsight with labels (knows which exit is right) | 2.89× | 4.00× | 2.73× |

Caveats: the batch draw differs from the report's, so the theory row is 0.03–0.06× below the report's
1.41 / 2.10 / 1.46. The hindsight rows are optimistic bounds selected on the held-out split. Sorted batches ignore
re-batching cost.

What this means:

- **The remaining gap is in the signal, not the rule.** The 2× headroom comes from knowing which early exits are
  correct.
- **Exit outputs hold nothing beyond confidence.** A gradient-boosted predictor on all exit probabilities gives the
  same AUROC as plain confidence (MNLI layer 8: 0.824 vs 0.827). Hidden states are untested (step 1).
- **Sorting by difficulty helps only QQP** (about +10%). Re-batching across exits is prior work (E3, SOSP 2024; verify
  the citation before using it).

## 2. Thesis of the paper

**Batched LLM decoding is where batch-level exit is unavoidable.**

- Measured on Qwen2.5-1.5B: a layer costs `a = 1.07 ms` with `b ≈ 0`, so one sequence exiting saves nothing.
- Sequences cannot be re-batched inside a token step, so the system fix used for encoders does not apply.
- Prior work there is SkipDecode's fixed, position-based schedule. An adaptive per-step batch depth is open.
- The cost model, the label-free bandit and the trimmed robustness bound carry over unchanged.
- The encoder results become the supporting section (regime switch on real hardware).

## 3. Steps, with kill criteria

Time figures are estimates, not measurements.

| Step | Question it answers | Pass condition | Kill criterion | Time | GPU |
|---|---|---|---|---|---|
| 1. Hidden-state correctness probe (encoders) | Is the label headroom reachable with a better signal? | Held-out AUROC at least 0.03 above confidence at two or more exits, and a matched-accuracy speedup gain | AUROC gain under 0.03 | about 2 h | one 3080 Ti |
| 2. CUDA-graph timing (encoders) | Does the regime switch survive a compiled stack? | Per-sample exits below 1× at some batch size, or the theory rule ahead of threshold at B = 64 | Per-sample exits never below 1× and the rule behind threshold on both text tasks | about 2 h | GPU 5, idle |
| 3. Exit-trained decoder | Is any decoder exit usable? | An exit at 75% depth or less within 1 point of the full model's next-token accuracy | No such exit | 3 h with a LayerSkip checkpoint; 8–12 h with LoRA plus exit loss on Qwen2.5-1.5B | GPU 5 |
| 4. Adaptive per-step batch depth vs SkipDecode | The headline result | Faster than SkipDecode at matched generation quality, on real GPU | No gain | 2–3 days | GPU 5 |
| 5. Paper-grade evidence | Does it hold up? | Seeds with error bars, three-way split, second model, serving trace | n/a | 2–3 weeks | any |

### Step 1 detail
- Re-extract traces with the pooled hidden state at each exit (`traces/train_exits.py` checkpoints, no retraining).
- Fit a small probe on the evolve split to predict "this exit is correct"; compare AUROC with confidence on held-out.
- If it passes, plug the probe into the theory rule in place of the calibration table and rerun the matched table.

### Step 2 detail
- Capture each layer segment as a CUDA graph (or `torch.compile`) per batch size, refit `a`, `b`, `head`.
- Rerun the real-GPU table for MNLI and QQP at B = 8, 16, 64, 128.
- Earlier simulation predicts ρ = 0.27 at B = 8 and 0.03 at B = 64, with per-sample exits never below 1×. The run
  confirms or refutes that. Timing must run on an idle GPU.

### Step 3 detail
- Route A: LayerSkip checkpoints (gated on Hugging Face, need access approval). Start with
  `facebook/layerskip-llama3.2-1B`, then `facebook/layerskip-llama3-8B`.
- Route B: LoRA plus an early-exit loss with layer dropout on Qwen2.5-1.5B, using `decoder/train_decoder_exits.py`
  as the base.
- Reuse `decoder/decode_runner.py` for traces and wall-clock. Baseline for comparison: frozen exits were 3.7 points
  (Llama layer 28) and 4.3 points (Qwen layer 24) below the full model.
- Disk: 91 GB free on `/home`. One 8B checkpoint fits; several do not.

### Step 4 detail
- Real generation, not teacher forcing. Quality by task metric (ROUGE or exact match) rather than next-token accuracy.
- Baselines: full model, SkipDecode schedule, majority vote, per-sequence exit.

## 4. Decision points

- **Step 3 fails:** stop the paper track. Present the course project with the encoder results and the negatives.
- **Step 2 fails:** the encoder section becomes "regime switch holds in eager PyTorch only"; the decoder thesis is
  unaffected because `b ≈ 0` there is not an eager artifact (to be rechecked in step 3).
- **Step 1 passes:** it becomes a second contribution (a better exit signal), for encoders and decoders.

## 5. Timeline

- **By Nov 26 (course presentation):** steps 1–3, plus the claim fixes from the validation report (three-way split,
  error bars, reworded abstract).
- **After Nov 26:** steps 4–5. Realistic paper effort is 2–3 months.

## 6. Open questions for the team

1. Is LLM decoding acceptable as the centre of the paper, or must the course deliverable stay on encoders?
2. Has LayerSkip access been granted? If not, step 3 uses route B.
3. Target venue and deadline.

## 7. Results of steps 1–3 (2026-11-12)

### Step 1: hidden-state probe. **Fails.**
Best probe AUROC minus confidence AUROC, held-out, per exit (`results/probe_<task>.csv`,
`analysis/probe_correctness.py`, hidden states from `traces/extract_hidden.py`):

| Exit | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| MNLI | +0.026 | +0.003 | −0.008 | −0.019 | 0.000 |
| QQP | +0.004 | +0.001 | −0.005 | −0.004 | −0.002 |
| CIFAR-100 | +0.010 | +0.002 | +0.007 | −0.003 | −0.014 |

The hidden state carries no usable correctness signal beyond confidence. The 2× label headroom is not reachable this
way.

### Step 2: CUDA-graph timing. **Mostly fails.**
`runner/graph_runner.py`; costs in `results/runner_cost_graph_<task>.json`, timings in `results/graph_<task>_gpu.csv`.
Compiling cuts the fixed per-layer cost from 0.34–0.52 ms to 0.07–0.08 ms: ρ is 0.22–0.40 at B = 8 and 0.03–0.05 at
B = 64.

Real speedup, theory rule / per-sample threshold / best other rule:

| | B = 8 | B = 16 | B = 64 | B = 128 |
|---|---|---|---|---|
| MNLI | 0.86 / 0.89 / 1.08 unified | 1.06 / 0.99 / 1.23 unified | 1.41 / 1.34 / 1.50 random | 1.61 / 1.50 / 1.61 random |
| QQP | 1.23 / 1.07 / 1.58 unified | 1.59 / 1.37 / 1.74 unified | 2.16 / 1.92 / 2.31 random | 2.31 / 1.91 / 2.27 random |
| CIFAR-100 | 0.95 / 1.03 / 1.05 consensus | 1.20 / 1.26 / 1.26 threshold | 1.39 / 1.41 / 1.42 random | 1.40 / 1.40 / 1.41 random |

- Per-sample exits still fall below 1× at B = 8 on MNLI (0.89×), but the cause is exit-head overhead (heads alone cost
  9–19% at B = 8), not the per-layer fixed cost.
- The theory rule is never the fastest rule. Unified exit wins at B ≤ 16 on text, random search ties or wins at
  B ≥ 64.
- MNLI accuracy drops are not matched (theory 1.8–2.5 pts vs unified 0.8–0.9 pts).
- The simulator overestimates in graph mode (MNLI theory B = 64: 1.76× simulated, 1.41× real).

### Step 3: LayerSkip exits. **Fails.**
Next-token accuracy on WikiText-103, points below the full model (`results/layerskip_<tag>_layer_accuracy.csv`,
`decoder/layerskip_exits.py`):

| Depth | 50% | 75% | 87.5% | 94% |
|---|---|---|---|---|
| LayerSkip Llama-3.2-1B (16 layers) | 12.3 | 5.2 | 2.0 | 0.8 |
| LayerSkip Llama-3-8B (32 layers) | 10.6 | 4.9 | 2.5 | 1.1 |
| Frozen Llama-3.1-8B + our heads (earlier) | — | 5.6 (layer 24) | 3.7 (layer 28) | — |

Simulated batch-stopping on the 8B LayerSkip traces (B = 64, Llama-8B runner cost, exits at 12…32): best speedup
within 1 point is 0.99× (majority vote); within 5 points 1.20×. LayerSkip exits are trained for lossless
self-speculative decoding, not for lossy exits, and they are about as weak as our frozen heads.

### Decision
Per section 4, the paper track as planned stops here. Present the course project with the encoder results and these
three negatives.
