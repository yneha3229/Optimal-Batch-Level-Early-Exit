# LLM track, second attempt: diagnosis and runbook (written 2026-11-13)

Status: **GPU steps not started.** The CPU analysis of section 5 was run on 2026-11-15; steps 0–4 have not been
executed (section 6). The plan was checked against the code and the result files on 2026-11-15; the gates in
steps 0–3 were recalibrated then, and each change is marked "(2026-11-15)". This file is the runbook: an engineer or an agent with shell access to the A6000 server
can execute it top to bottom. Every step has a command, an output to check, a pass condition and a kill condition.

Related: `docs/PLAN_PAPER.md` (first attempt, stopped), `docs/RESULTS.md` (frozen-decoder numbers),
`docs/DIRECTION.md` (encoder direction).

## 1. What went wrong the first time

| # | Problem | Evidence | Fix in this plan |
|---|---|---|---|
| 1 | **The backbone was never trained for its exits.** `decoder/train_decoder_exits.py` freezes the model and trains a `d×d` adapter per exit for 1,500 steps of 8 windows. | Layer-28 exit on Llama-3.1-8B is 3.7 points below the full model and agrees with it on 74.5% of tokens. The encoders had the same symptom on a frozen backbone (MNLI layer-8 exit 70.6%) and joint training fixed it (87.4%). | Step 1: LoRA on every layer + exit heads + self-distillation, `decoder/train_exit_lora.py`. |
| 2 | **LayerSkip checkpoints were the wrong tool.** They are trained so that an early exit is a good *draft* that the remaining layers verify, not so that the early exit can be returned as the answer. | 4.9–5.2 points short at 75% depth, about as weak as our frozen heads (`results/layerskip_*_layer_accuracy.csv`). | Not used. |
| 3 | **Batch size 64 was the worst place to look.** With `b ≈ 0` only a whole-batch stop saves time, and a whole-batch stop can only use what the batch has in common. The mean confidence of 64 random tokens is nearly the same in every batch, so an adaptive stop has nothing to adapt to and collapses to a fixed exit. | Measured 2026-11-15 on the frozen traces, section 5: the gain over a fixed exit is about 10% at B = 1, 1% at B = 8 and 0 at B = 64. | Step 2 covers B ∈ {1, …, 64} in depth; step 3 times B ∈ {2, 4, 8, 16, 64} (B = 1 is not timed: about 3 GPU hours for one batch size). The claim is made where adaptivity can exist. |
| 4 | **The budget was one point of next-token accuracy on WikiText.** Free text has many acceptable next tokens; top-1 accuracy of the full model is only 50–54%, so most "errors" of an exit are disagreements between two plausible tokens. | Full-model accuracy 49.9% (Qwen) and 53.6% (Llama). | Report agreement with the full model and budgets of 1 and 3 points; step 5 (after the gate) moves to a generation task with a task metric. |
| 5 | **Selection at the edge of the budget.** Same flaw as on the encoders: parameters tuned to exactly one point on the tuning split. | Encoder runs lost 0.8–2.4 points on held-out data. | Certified selection (`experiments/local_hw.py`, section 2c of the report) once step 2 passes. |

Problems 1 and 3 are the ones that decide whether there is a result. Problem 1 is about the model; problem 3 is about
where the method can help at all.

### Points raised in review of the progress report (2026-11-13)

1. **KV cache.** A token that exits at layer `l` leaves layers `l+1..L` without keys and values, and later tokens
   attend to them. `decoder/decode_runner.py` already handles this with CALM-style state propagation
   (`propagate_from`: the exit hidden state is projected into the K/V of every skipped layer; about 2% of a layer
   per skipped layer in the first attempt). It is an approximation. Two consequences for this plan:
   - `decoder/train_exit_lora.py` trains and scores exits with teacher forcing on full-depth states, so its
     accuracy table does **not** include the error from propagated K/V. The number that includes it is the
     runner's, in step 3. Compare the two; a gap of more than 1 point means propagation, not the exit, is the limit.
   - State this in any write-up before being asked.
2. **Metric.** One point of next-token accuracy on free text is a weak target; agreement with the full model's
   greedy output (`fidelity` in the CSVs) is the better one. What the code does (2026-11-15): step 1 gates on both.
   Step 2 runs with both budgets (`--metric fidelity`). Step 3 still **selects** on accuracy, because
   `experiments/run_main.py` budgets on accuracy only; its fidelity is read from the runner CSV and must be quoted
   next to every speedup. A policy inside a 1-point accuracy budget can still change 5–10% of the tokens.
3. **Alternative direction: exits as drafts (self-speculative decoding).** If exits stay too weak to be returned
   as answers, use them as drafts that the remaining layers verify, as LayerSkip does. The output then equals the
   full model's greedy output, so a weak exit costs speed and not accuracy. Batched speculation has the same
   straggler problem (sequences in a batch accept different numbers of draft tokens), and the draft depth and
   length can be chosen per batch from the measured `c(n)`. This is the fallback of step 1 (see there). Nothing
   is written for it yet.
4. **Priority.** This track must not take time from the accuracy-matched server rerun (report section 5.1).
   Run it only on a GPU that the rerun is not using.

## 2. What we expect, stated before running

- **Step 1 (exit quality).** Baseline on the model step 1 trains (frozen Qwen2.5-1.5B,
  `results/train_wikitext_exit_accuracy.csv`): layer 20 is 10.1 points below the full model (fidelity 0.58), layer 24
  is 4.3 below (fidelity 0.70). Reference for what exit training reaches: the LayerSkip checkpoints we measured are
  4.9–5.2 below at 75% depth and 2.0–2.5 below at 87.5% (fidelity 0.82–0.83). Forecast: joint training brings layer
  24 to about 2.5 below and layer 20 to about 6 below. This is a forecast, not a result. (2026-11-15: the earlier
  text said "to within about 2 points, from 5.6"; 5.6 was a Llama figure, not the Qwen baseline.)
- **Step 3 (adaptivity).** The adaptive rule should beat a fixed exit at B ≤ 8 and tie it at B = 64. If it also ties
  at B ≤ 8, batch-level adaptivity has no value on decoders and the LLM section becomes a negative result with an
  explanation.
- **Wall-clock.** With `b ≈ 0` latency is proportional to depth, so depth saved is time saved, minus overheads.
  Measured on Qwen (`results/runner_cost_wikitext.json`): a layer costs 1.065 ms, flat from B = 1 to 128; one exit
  check (confidence head + host sync) costs 0.405 ms, 38% of a layer; KV propagation costs 0.02 ms per skipped layer.
  **The exit checks are the overhead that matters** (2026-11-15): checking all five exits costs 2.0 ms per token,
  6.6% of the full model's 30.5 ms, and the first attempt measured 8–9% (`heads_only` rows). Consequences:
  - A rule that checks every exit needs a depth speedup above about 1.08× before it gains any wall-clock. Step 2's
    1.10× is therefore roughly break-even, and step 3's 1.05× in wall-clock needs about 1.15× in depth.
  - Stopping at layer 24 after five checks pays only if about half of the tokens stop there; with one check (that
    exit only) it pays from about 10%. Which exits to check is the exit plan of `docs/DIRECTION.md` section 3; it is
    not implemented for the decoder runner.
  - On Llama-3.1-8B `b` is not zero above B = 16 (per-layer time 0.79 ms at B ≤ 16, 0.99 ms at B = 64). The runner
    does not shrink the batch, so this does not change the argument, but "`b ≈ 0`" is a Qwen statement.

## 3. Environment

- Server: RTX A6000, repository at `~/ie624`, Python at `~/venv/bin/python` (torch 2.14, transformers 5.17).
- Use one idle A6000 (indices 2–5 on this server; 0–1 are 3080 Ti):
  `export GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n>"`. Check it is idle with `nvidia-smi`
  first; timing steps need it idle. GPU 5 is the one `docs/DIRECTION.md` gives to the accuracy-matched rerun
  (16–18 Nov); do not use it while that runs.
- `Qwen/Qwen2.5-1.5B` is already downloaded (it was used by the first attempt). No gated model is needed.
- Disk: the merged backbone is about 3 GB in bf16 (two are written: `smoke_merged`, `qlora_merged`). `/home` had
  66 GB free (99% full) on 2026-11-15; check `df -h ~` first. Checkpoints, data and `*_merged/` directories are
  git-ignored; do not commit them.
- All steps are wrapped in `scripts/run_llm_v2.sh`, which writes one log per step to `logs/llm2_<step>.log` and a
  status line to `logs/llm2_status.log`. It runs **one stage per call** and stops, so that each gate table is read
  before the next stage: `GPU_ID=<n> nohup scripts/run_llm_v2.sh train &` (steps 0–1), then `adapt` (step 2),
  `time` (step 3), `post` (step 4). It refuses to start without `GPU_ID` and fails a stage if the runner's sanity
  line is below 0.97.

## 4. Steps

### Step 0. Functional test (5 minutes)

```
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag smoke --steps 20 --bs 2 --train-lines 2000 --eval-windows 16
$GPU ~/venv/bin/python decoder/decode_runner.py --task wikitext_smoke --max-windows 8 --smoke
git checkout traces/wikitext_smoke_val.parquet results/train_wikitext_smoke_exit_accuracy.csv
```

Pass: the first command prints an accuracy table and writes `checkpoints/smoke_merged/`; the second prints
`sanity: own decode loop vs HF forward, final-exit token agreement` at 0.97 or higher and `smoke decode ok`
(this exercises collective stops, per-sample exits and KV propagation). If the agreement is below 0.95, the merged
backbone and the runner disagree: stop and fix before spending GPU hours.
- The bar is 0.97, not 0.98 (2026-11-15): the first attempt's working runner scored 0.9756 on Qwen and 0.9873 on
  Llama over 1,024 tokens (`logs/final_dec_calib.log`, `logs/final_llama_calib.log`).
- `--smoke` with a tagged task no longer appends `_smoke` to file names (fixed 2026-11-15 in `decode_runner.py`).
- The third line restores two tracked files of the first attempt that the smoke run overwrites.

### Step 1. Train exits with the backbone, and the frozen baseline at the same budget (about 3 hours, estimate)

```
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag qlora   --rank 16 --steps 6000 > logs/llm2_train_qlora.log 2>&1
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag qfrozen --rank 0  --steps 6000 > logs/llm2_train_qfrozen.log 2>&1
```

Read `results/train_wikitext_qlora_exit_accuracy.csv` (rows with `split == all`) and the last three lines of the log.

| Check | Pass | Kill |
|---|---|---|
| Fine-tuned full model vs original model accuracy | within 0.3 points | more than 1 point lower: the anchor failed; rerun with `--lr 1e-4` |
| Layer-24 exit (86% depth), points below the full model | 2.5 or less | more than 3.5: under 1 point better than frozen (4.3); go to the fallback below |
| Layer-24 agreement with the full model (`fidelity`) | 0.80 or higher | below 0.75 (frozen: 0.70) |
| Layer-20 exit (71% depth), points below the full model | 6 or less | more than 8.5 (frozen: 10.1) |
| `qlora` vs `qfrozen` at layer 20 | at least 1.5 points closer to the full model | no difference |

The bars were recalibrated on 2026-11-15 to the frozen-Qwen baseline and the LayerSkip measurements (section 2).
The earlier bars (layer 20 within 2.5, layer 24 within 1.5 and fidelity 0.85) were stricter than anything measured in
this repository, including the LayerSkip checkpoints. `qfrozen` trains the heads for 6,000 steps instead of 1,500, so
its numbers can differ from the frozen baseline quoted here; the comparison of record is `qlora` against `qfrozen`.

Between pass and kill on any row: record "partial" in section 7 and go on to step 2, which decides. A partial result
does not justify step 3 by itself.

Fallback if a kill row triggers: one rerun with `--rank 64 --steps 12000 --positions 128`. If that also fails, stop
the early-answer track and record the numbers in section 7; the diagnosis in section 1 row 1 is then wrong or
incomplete. The next thing to try is then step 6 (exits as drafts), not more training.

### Step 2. Is there anything for a batch-level rule to adapt to? (CPU, 2 minutes)

```
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext_qfrozen wikitext wikitext_llama --budgets 0.01 0.03
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext_qfrozen wikitext wikitext_llama --metric fidelity --budgets 0.05 0.10
```

Outputs: `results/adaptivity_vs_batch.csv` and `results/adaptivity_vs_batch_fidelity.csv`. **Both commands overwrite
the section 5 outputs**; the frozen tasks are in the task list so that the baseline rows stay in the file. Latency is
proportional to depth here, so `static_speedup` and `adaptive_speedup` are speedups before overheads; see section 2
for what the exit checks take away. The rows below are judged on the accuracy file; quote the fidelity file next to it.

Reference, frozen Qwen (`wikitext`), budget 1 point, B = 8: `adaptive_speedup` 1.047, gain over static 1.4%
(section 5). These rows fail the table below, as they should.

| Check | Pass | Kill |
|---|---|---|
| `wikitext_qlora`, budget 1 point, B = 8: `adaptive_speedup` | 1.10 or more | below 1.03 at both budgets and every B ≤ 8 |
| Same row: `gain_over_static_pct` | 3 or more at some B ≤ 8 | zero or negative at every B: adaptivity adds nothing; report a fixed exit |
| `adaptive_drop_pts` on held-out | at most budget + 0.5 | larger: tuning noise; rerun with `--cap 40000` |

### Step 3. Real wall-clock on the A6000 (about 2 hours, estimate; GPU must be idle)

```
$GPU ~/venv/bin/python decoder/decode_runner.py --task wikitext_qlora --calibrate
~/venv/bin/python experiments/run_main.py --task wikitext_qlora --regimes real_wikitext_qlora --batch-sizes 2 4 8 16 64 --workers 32
for B in 2 4 8 16 64; do
  $GPU ~/venv/bin/python decoder/decode_runner.py --task wikitext_qlora --batch-sizes $B --repeats 3 --out-suffix _B$B
done
```

Outputs: `results/runner_cost_wikitext_qlora.json` (expect `b` near 0 and `rho64` above 5, as before),
`results/main_wikitext_qlora_selected.csv`, `results/main_wikitext_qlora_real_wikitext_qlora_gpu_B*.csv`.

| Check | Pass | Kill |
|---|---|---|
| Runner sanity line (own loop vs HF forward) | 0.97 or higher (first attempt: 0.9756) | below 0.95: stop, the runner and the merged model disagree |
| Timing is clean: a policy with `layers_frac` = 1.0 against `heads_only`, same B | within 3% | further apart: the GPU was not idle or the rows come from different runs; rerun that B |
| Any policy with `acc_drop_pts` ≤ 1.0 and `real_speedup` > 1.05 at some B | yes | none at any B: report the depth result of step 2 only, as an upper bound that overheads erase |
| Theory rule vs unified exit at B ≤ 8, both within the budget | theory at least as fast | unified faster everywhere: the LLM result is "a fixed exit is enough", stated as such |
| Per-sample threshold | below 1.0 (it saves nothing when `b ≈ 0`) | n/a (this is the regime check) |

Why the timing row exists (2026-11-15): in the first attempt's Qwen file at B = 16, `full` took 50.1 s and
`heads_only` 54.4 s, while policies that never exited took 72 s. At B = 64 and on Llama they agree. The Qwen B = 16
speedups of the first attempt should not be quoted.

The selection inside `run_main.py` is the old edge-of-budget protocol. Expect held-out drops of up to about 2 points
at a nominal 1; that is known (section 1 row 5). If step 3 passes its second row, redo the selection with the
certified procedure before quoting numbers.

### Step 4. Robustness and the bandit on the new traces (CPU, 1 hour)

```
~/venv/bin/python experiments/robust_bound.py --task wikitext_qlora --regime real_wikitext_qlora
~/venv/bin/python bandit/run_bandit.py --task wikitext_qlora --regime real_wikitext_qlora --B 8 --T 3000
```

Pass: zero violations of the pathwise bound; the bandit's oracle stops batches (non-zero regret curve). On the frozen
decoders the oracle never stopped, so the bandit result there was empty.

### Step 5. Only if steps 1–3 pass: a generation task (after 26 November)

Teacher-forced next-token accuracy is a proxy. A paper needs free-running generation scored by a task metric
(summarisation with ROUGE-L, or extractive QA with exact match), with the same batch sizes and baselines plus
SkipDecode's position schedule. This needs a generation loop in `decoder/decode_runner.py` (feed back the produced
token instead of the true one) and is not scripted yet.

### Step 6. Fallback direction: exits as drafts (not scripted)

Only if step 1 hits its kill row twice. Needs a draft-then-verify loop in `decoder/decode_runner.py`: draft `d`
tokens at exit layer `l` for the whole batch, verify them with the remaining layers in one pass, keep the accepted
prefix of each sequence. Choices to make per batch from `c(n)`: the draft layer `l` and the draft length `d`.

| Check | Pass | Kill |
|---|---|---|
| Output equals the full model's greedy output | 100% of tokens (by construction; this is the correctness test) | any mismatch: bug |
| Mean accepted draft tokens per round at the layer-20 exit | 2 or more | below 1.2: drafts are too weak to pay for verification |
| Wall-clock against the full model at B ≤ 8 | above 1.1× | at or below 1.0× at every B |
| Per-batch choice of `(l, d)` against the best fixed `(l, d)` | faster at some B | no gain: report the fixed choice |

## 5. Evidence for problem 3, from traces already in the repository

**Measured 2026-11-15** (CPU, under one minute per command). Outputs: `results/adaptivity_vs_batch.csv`,
`results/adaptivity_vs_batch_fidelity.csv`, logs `logs/adaptivity_vs_batch*.log`.

```
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks mnli qqp wikitext wikitext_llama wikitext_ls8b
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks wikitext wikitext_llama wikitext_ls8b --metric fidelity --budgets 0.05 0.10
```

Depth speedup (before overheads), budget of 1 point of accuracy, held-out split:

| Trace | Fixed exit | Adaptive, B = 1 | B = 2 | B = 4 | B = 8 | B = 64 |
|---|---|---|---|---|---|---|
| `wikitext` (frozen Qwen) | 1.033 | 1.138 | 1.091 | 1.060 | 1.047 | 1.034 |
| `wikitext_llama` (frozen Llama) | 1.035 | 1.139 | 1.086 | 1.077 | 1.044 | 1.039 |
| `wikitext_ls8b` (LayerSkip 8B) | 1.052 | 1.298 | 1.188 | 1.118 | 1.082 | 1.067 |

- **Row 3 of section 1 holds in direction.** The spread of the batch-mean confidence falls as 1/sqrt(B) (Qwen:
  0.218 at B = 1, 0.077 at B = 8, 0.027 at B = 64), and the adaptive stop ties a fixed exit at B = 64 on both frozen
  decoders.
- **The size is small.** Gain over the fixed exit on frozen Qwen: 10% at B = 1, 5.6% at B = 2, 2.6% at B = 4, 1.4% at
  B = 8. At 3 points the pattern is the same (16%, 7.9%, 4.1%, 1.7%). With a budget on fidelity (5% of tokens may
  differ) it is 11%, 4.8%, 2.3%, 1.0%.
- **Better exits widen it.** LayerSkip 8B: 23% at B = 1, 6.2% at B = 4, 2.8% at B = 8. This is the case for step 1.
- **None of this clears the exit-check overhead** of section 2 (about 8%) at B ≥ 4 on the frozen traces.
- **Held-out drops overshoot at small B**: 1.74 points at B = 1 and 1.28 at B = 2 on Qwen for a 1-point budget. The
  B = 1 and B = 2 speedups above are therefore optimistic.
- Encoders, same script: at 1 point the adaptive stop is no better than a fixed exit on QQP at B ≥ 2 (0 to −5%) and
  mixed on MNLI (+8% at B = 2, −1% at B = 8).

Correction (2026-11-15): the script's fixed-exit baseline was 1.000× whenever no intermediate exit met the budget; it
now mixes the last exit with the full model, as its docstring says. The earlier baseline overstated the gain at
B = 8 on these traces by 3 to 11 points.

## 6. Functional test of the scripts

`analysis/adaptivity_vs_batch.py` has been run (section 5). `decoder/train_exit_lora.py` has **not** been run on a
GPU. Checked on 2026-11-15 on a CPU with a 2-layer random model: the module names it uses exist for Qwen2.5 in
transformers 5.17, and merging LoRA reproduces the unmerged model's logits to 7e-6. Not checked: the training loop,
memory at `--bs 8`, the trace writing. Step 0 is therefore a real test, not a formality: expect to fix small errors there (tensor dtypes under autocast,
the `model.layers` / `model.norm` attribute names for the chosen backbone, memory at `--bs 8`).

## 7. Results

Run on 2026-11-15 and 2026-11-16 (classifier timing added 2026-11-23). Steps 3–4 were not run (step 3 was held on the layer-24 fidelity result).

### Step 0 (functional test): pass
Runner against trace, final exit: 0.9883 (bar 0.97). `logs/llm2_smoke_decode.log`.

### Step 1 (exits trained with the backbone, Qwen2.5-1.5B, WikiText-103 validation and test windows, positions 128–255): partial
Original model 49.94% next-token accuracy.

| Run | Fine-tuned full model | Layer 20, points below full | Layer 24, points below full | Layer 20 fidelity | Layer 24 fidelity |
|---|---|---|---|---|---|
| Frozen, heads only (`qfrozen`) | 49.94% | 9.50 | 4.01 | 0.591 | 0.705 |
| LoRA rank 16, 6,000 steps (`qlora`) | 49.31% | 5.73 | 2.43 | 0.657 | 0.740 |
| LoRA rank 64, 12,000 steps, 128 positions (`qlora64`, the fallback) | 49.18% | 3.86 | 1.36 | 0.690 | 0.762 |

- Against section 4: layer-24 and layer-20 accuracy bars pass; LoRA is closer than frozen by more than 1.5 points (pass);
  layer-24 fidelity is 0.740 (below the 0.75 kill bar) at rank 16 and 0.762 (between kill and pass) at rank 64.
- The fine-tuned full model is 0.6–0.8 points below the original, so "points below full" flatters the exits by that much.
- Files: `results/train_wikitext_{qfrozen,qlora,qlora64}_exit_accuracy.csv`, `logs/llm2_train_*.log`.

### Step 2 (anything for a batch-level rule to adapt to?), `qlora`, 1-point accuracy budget: partial
Depth speedup, fixed exit 1.064; adaptive 1.192 (B=1), 1.134 (2), 1.100 (4), 1.090 (8, bar 1.10), 1.076 (16), 1.065 (64).
The gain over a fixed exit falls from 12.1% at B=1 to 0.2% at B=64. Held-out drops are over budget at B=1 and 2
(1.81, 1.55 points). `results/adaptivity_vs_batch.csv`. Not rerun on `qlora64`.

### Longer prompts (not in the original plan)
`experiments/context_scaling.py`, `experiments/context_policies.py`; `results/context_*`.
- Exit quality barely improves with context: layer-24 agreement 0.737 (positions 128–256) to 0.765 (2,048–4,096).
- Eager timing of a decode step is launch-bound: about 0.95 ms per layer on an RTX 3080 Ti whatever the batch size and
  context, against about 0.2 ms with CUDA graphs. "b ≈ 0" on Qwen is largely an eager-mode artifact.
- With CUDA graphs the per-sequence share of a layer's time at B=64 grows from 0.27 (192 tokens of context) to 0.72
  (3,072), so the cost regime depends on prompt length.
- Simulated with these costs, no rule gives a meaningful speedup within one point at any context length (0.85–1.07×).

### Verdict on decoding
Joint training roughly halves the exit gap, but free-text next-token prediction leaves too little agreement for
any stopping rule to use. Decoding stays a measured negative with three causes: weak exits on a high-entropy
task, a launch-bound decode step, and batch-mean confidence concentrating as 1/√B.

### New direction started 2026-11-16: the LLM as a batched classifier
`decoder/train_cls_exit_lora.py` (LoRA + one linear head per exit on the last prompt token, prefill only, no KV
cache). Traces use the encoder schema; `experiments/local_hw.py` now takes any number of exits and has an
`LLMDevice` (trained weights, real prompts). Run it with `--name <machine>` on the server: without it the GTX 1650
result files are overwritten. Pipeline: `scripts/run_llm_cls_hw.sh`.

Qwen2.5-1.5B, LoRA rank 16, 8,000 steps of 16 (128,000 examples), full validation sets:

| Task | Layer 8 | Layer 12 | Layer 16 | Layer 20 | Layer 24 | Layer 28 (full) | Layer-16 agreement with full |
|---|---|---|---|---|---|---|---|
| QQP | 85.3% | 88.6% | 89.9% | 90.1% | 90.1% | 90.1% | 0.983 |
| MNLI | 75.7% | 82.6% | 88.4% | 89.3% | 89.3% | 89.4% | 0.958 |

Cost on the A6000 (bf16, prompts padded to 128 tokens, eager): a floor of about 23 ms per pass up to B=4, then
about 4.2 ms per input (ρ = 0.48 at B=8, 0.06 at B=64); exit head 0.36 ms. `results/local_cost_a6000_T128_qwen.json`.

Speed at a certified one-point target, held-out, from the cost table (not timed), `results/local_front_*_qwen_a6000_T128.csv`:
- QQP: cost rule 2.25× (B=8), 2.62× (16), 2.95× (64), 3.07× (128), held-out drop 0.54–0.64 points; best baseline
  within budget 1.74× (B=8), 2.50× (16), 2.50× (64), 2.68× (128).
- MNLI: cost rule 1.12–1.18×, with an exit plan 1.37–1.47×; unified exit 1.40× at a 0.02-point drop; the per-sample
  threshold reaches 1.36–1.94× but is over budget on held-out (1.2 points).

Timed on the A6000, QQP, 2,048 held-out inputs per pass, 5 rounds, median paired speedup
(`results/local_time_qqp_qwen_a6000_T128.csv`). Decisions are replayed from the recorded traces on the trained
weights and real prompts; the device's own predictions agree with the traces on 99.2–100% of inputs at the
intermediate exits (97.3% at the last exit in this run, caused by a missing final norm in the device, fixed after it).

| B | Per-sample threshold | Unified exit | Random search | Cost rule | Cost rule + plan | Rule's held-out drop (pts) |
|---|---|---|---|---|---|---|
| 8 | 1.52× | 1.73× | 2.02× (drop 1.15, over budget) | 2.15× | 2.18× | 0.64 |
| 16 | 1.97× | 1.74× | 2.41× | 2.52× | 2.52× | 0.60 |
| 64 | 2.55× | 1.75× | 2.56× | 2.96× | 2.96× | 0.55 |
| 128 | 2.58× | 1.75× | 2.60× | 3.02× | 3.01× | 0.54 |

The cost table predicted these within 4% (largest gap at B=8).

### Timed results, all four classifier settings (added 2026-11-23)

Same protocol (A6000, eager, bf16, prompts padded to 128 tokens, 2,048 held-out inputs per pass, 5 rounds, median
paired speedup, certified selection at δ = 10%, one-point budget). Files: `results/local_time_{qqp,mnli}_{llama,qwen}_a6000_T128.csv`,
logs `logs/cls_time_{qqp,mnli}_{llama,qwen}_a6000.log`. "Best baseline" = fastest of per-sample threshold, threshold +
plan, unified exit, majority vote and random search whose held-out drop is at most one point.

Llama-3.1-8B (LoRA + exit heads at layers 8–32; full model 90.9% on both tasks, layer 16 within 0.1 point of full on both, `results/train_{qqp,mnli}_llama_exit_accuracy.csv`). Cost is linear: 31 ms at B=1, 2,384 ms at B=128,
ρ = 0.09 at B=8 (`results/local_cost_a6000_T128_llama.json`).

| Task | B | Cost rule (best of all exits / plan) | Rule's drop (pts) | Best baseline | Its drop | Rule ahead by |
|---|---|---|---|---|---|---|
| QQP | 8 | 3.48× | 0.42 | 3.13× random search | 0.53 | 11% |
| QQP | 16 | 3.52× | 0.42 | 3.11× random search | 0.53 | 13% |
| QQP | 64 | 3.66× | 0.44 | 3.50× random search | 0.29 | 5% |
| QQP | 128 | 3.62× | 0.42 | 3.43× random search | 0.23 | 5% |
| MNLI | 8 | 2.97× | 0.73 | 2.56× threshold + plan | 0.77 | 16% |
| MNLI | 16 | 3.05× | 0.74 | 2.66× random search | 0.76 | 14% |
| MNLI | 64 | 3.12× | 0.70 | 2.66× random search | 0.53 | 17% |
| MNLI | 128 | 3.14× | 0.72 | 2.66× random search | 0.45 | 18% |

QQP B=128 is in the timing log and the CSV, so the B=128 rerun listed in `docs/NEXT_STEPS.md` task 5 is not needed.
On Llama MNLI the unified exit is over budget (1.13–1.17 points) and majority vote is over budget at B ≥ 64 (1.34).

Qwen2.5-1.5B MNLI (QQP is in the table above):

| B | Cost rule, all exits | Cost rule + plan | Plan's drop (pts) | Best baseline | Its drop | Plan vs best baseline |
|---|---|---|---|---|---|---|
| 8 | 1.10× | 1.38× | 0.20 | 1.48× random search | 1.00 | −7% |
| 16 | 1.16× | 1.41× | 0.20 | 1.40× unified exit | 0.02 | +1% |
| 64 | 1.18× | 1.44× | 0.21 | 1.54× majority vote | 0.97 | −6% |
| 128 | 1.19× | 1.48× | 0.21 | 1.40× unified exit | 0.02 | +5% |

- The rule with all exits is the slowest certified policy on Qwen MNLI: the three exits below layer 16 are weak
  (67–83% against 89.4% full) and each costs a head evaluation (0.36 ms) at a floor-dominated cost (ρ = 0.48 at B=8).
  The plan drops them and the rule then sits at a 0.2-point drop, far inside the budget, so the certificate is
  conservative here.
- The per-sample threshold reaches 1.34–1.94× and the threshold + plan 1.70–1.73×, both over budget (1.21–1.46 points).
- Majority vote and random search have no certified policy at some batch sizes (missing rows in the CSV).
- Cost-table predictions are within 2% of the timed speedups on Llama and on Qwen MNLI, and within 5% on Qwen QQP.
- The device's own predictions agree with the recorded traces on 97.3–100% of inputs at every exit
  (`results/local_check_*_a6000_T128.json`; the 97.3% is Qwen QQP's last exit, from the missing final norm noted above).

Verdict on the classifier direction: the cost rule is the fastest certified policy in 14 of the 16 (model, task,
batch size) cells, by 1–18%. The exceptions are Qwen MNLI at B=8 and B=64, where it is 6–7% behind even with the
exit plan (and without the plan it is the slowest certified policy at every batch size). This is the strongest evidence for the "near the best everywhere" claim, and Qwen MNLI is the case to show
as the exception.

## 8. Files

| File | Role |
|---|---|
| `decoder/train_exit_lora.py` | Step 1. LoRA + exit heads + self-distillation; `--rank 0` is the frozen baseline |
| `decoder/train_decoder_exits.py` | First attempt (frozen). Unchanged; `ExitHeads` is imported from it |
| `decoder/decode_runner.py` | Steps 0 and 3. Reads the merged backbone through the checkpoint's `model` field; `--smoke` file naming fixed for tagged tasks |
| `analysis/adaptivity_vs_batch.py` | Step 2 |
| `experiments/run_main.py` | Step 3 selection. Reads `traces/*_val.meta.json` to learn new decoder tasks |
| `scripts/run_llm_v2.sh` | Steps 0–4, one stage per call (`train`, `adapt`, `time`, `post`), with logs |
