# LLM track, second attempt: diagnosis and runbook (written 2026-11-13)

Status: **not started.** Scripts are written and syntax-checked only; nothing in this track has been executed
(section 6). This file is the runbook: an engineer or an agent with shell access to the A6000 server
can execute it top to bottom. Every step has a command, an output to check, a pass condition and a kill condition.

Related: `docs/PLAN_PAPER.md` (first attempt, stopped), `docs/RESULTS.md` (frozen-decoder numbers),
`docs/DIRECTION.md` (encoder direction).

## 1. What went wrong the first time

| # | Problem | Evidence | Fix in this plan |
|---|---|---|---|
| 1 | **The backbone was never trained for its exits.** `decoder/train_decoder_exits.py` freezes the model and trains a `d×d` adapter per exit for 1,500 steps of 8 windows. | Layer-28 exit on Llama-3.1-8B is 3.7 points below the full model and agrees with it on 74.5% of tokens. The encoders had the same symptom on a frozen backbone (MNLI layer-8 exit 70.6%) and joint training fixed it (87.4%). | Step 1: LoRA on every layer + exit heads + self-distillation, `decoder/train_exit_lora.py`. |
| 2 | **LayerSkip checkpoints were the wrong tool.** They are trained so that an early exit is a good *draft* that the remaining layers verify, not so that the early exit can be returned as the answer. | 4.9–5.2 points short at 75% depth, about as weak as our frozen heads (`results/layerskip_*_layer_accuracy.csv`). | Not used. |
| 3 | **Batch size 64 was the worst place to look.** With `b ≈ 0` only a whole-batch stop saves time, and a whole-batch stop can only use what the batch has in common. The mean confidence of 64 random tokens is nearly the same in every batch, so an adaptive stop has nothing to adapt to and collapses to a fixed exit. | Not measured yet; see section 5. | Step 3 sweeps B ∈ {1, 2, 4, 8, 16, 64}; the claim is made where adaptivity can exist. |
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
2. **Metric.** One point of next-token accuracy on free text is a weak target. The primary metric from step 1 on
   is agreement with the full model's greedy output (`fidelity` in the CSV); accuracy against the reference token
   is reported next to it.
3. **Alternative direction: exits as drafts (self-speculative decoding).** If exits stay too weak to be returned
   as answers, use them as drafts that the remaining layers verify, as LayerSkip does. The output then equals the
   full model's greedy output, so a weak exit costs speed and not accuracy. Batched speculation has the same
   straggler problem (sequences in a batch accept different numbers of draft tokens), and the draft depth and
   length can be chosen per batch from the measured `c(n)`. This is the fallback of step 1 (see there). Nothing
   is written for it yet.
4. **Priority.** This track must not take time from the accuracy-matched server rerun (report section 5.1).
   Run it only on a GPU that the rerun is not using.

## 2. What we expect, stated before running

- **Step 1 (exit quality).** Joint training should bring the 75%-depth exit to within about 2 points of the full
  model, from 5.6. Basis: the encoder result and the LayerSkip/EE-LLM literature. This is a forecast, not a result.
- **Step 3 (adaptivity).** The adaptive rule should beat a fixed exit at B ≤ 8 and tie it at B = 64. If it also ties
  at B ≤ 8, batch-level adaptivity has no value on decoders and the LLM section becomes a negative result with an
  explanation.
- **Wall-clock.** With `b ≈ 0` latency is proportional to depth, so any depth saved is time saved, minus the exit-head
  and KV-propagation overheads already measured (0.02 ms per skipped layer).

## 3. Environment

- Server: RTX A6000, repository at `~/ie624`, Python at `~/venv/bin/python` (torch 2.14, transformers 5.17).
- Use one idle GPU: `export GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5"`. Check it is idle with
  `nvidia-smi` first; timing steps need it idle.
- `Qwen/Qwen2.5-1.5B` is already downloaded (it was used by the first attempt). No gated model is needed.
- Disk: the merged backbone is about 3 GB in bf16. Checkpoints, data and `*_merged/` directories are git-ignored;
  do not commit them.
- All steps are wrapped in `scripts/run_llm_v2.sh`, which writes one log per step to `logs/llm2_<step>.log` and a
  status line to `logs/llm2_status.log`. Run it with `nohup scripts/run_llm_v2.sh &`, or run the steps by hand.

## 4. Steps

### Step 0. Functional test (5 minutes)

```
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag smoke --steps 20 --bs 2 --train-lines 2000 --eval-windows 16
$GPU ~/venv/bin/python decoder/decode_runner.py --task wikitext_smoke --max-windows 8 --smoke
```

Pass: the first command prints an accuracy table and writes `checkpoints/smoke_merged/`; the second prints
`sanity: own decode loop vs HF forward, final-exit token agreement` at 0.98 or higher and `smoke decode ok`.
If the agreement is low, the merged backbone and the runner disagree: stop and fix before spending GPU hours.
Note: `decode_runner.py --smoke` appends `_smoke` to file names; if it looks for `smoke_exits_smoke.pt`, run it
without `--smoke` (`--task wikitext_smoke --calibrate`) as the functional check instead.

### Step 1. Train exits with the backbone, and the frozen baseline at the same budget (about 3 hours, estimate)

```
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag qlora   --rank 16 --steps 6000 > logs/llm2_train_qlora.log 2>&1
$GPU ~/venv/bin/python decoder/train_exit_lora.py --tag qfrozen --rank 0  --steps 6000 > logs/llm2_train_qfrozen.log 2>&1
```

Read `results/train_wikitext_qlora_exit_accuracy.csv` (rows with `split == all`) and the last three lines of the log.

| Check | Pass | Kill |
|---|---|---|
| Fine-tuned full model vs original model accuracy | within 0.3 points | more than 1 point lower: the anchor failed; rerun with `--lr 1e-4` |
| Layer-20 exit (71% depth), points below the full model | 2.5 or less | more than 4: no better than frozen; go to the fallback below |
| Layer-24 exit (86% depth), points below the full model | 1.5 or less | more than 3 |
| Layer-24 agreement with the full model (`fidelity`) | 0.85 or higher | below 0.80 |
| `qlora` vs `qfrozen` at layer 20 | at least 1.5 points closer to the full model | no difference |

Fallback if the kill row triggers: one rerun with `--rank 64 --steps 12000 --positions 128`. If that also fails, stop
the early-answer track and record the numbers in section 7; the diagnosis in section 1 row 1 is then wrong or
incomplete. The next thing to try is then step 6 (exits as drafts), not more training.

### Step 2. Is there anything for a batch-level rule to adapt to? (CPU, 2 minutes)

```
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext_qfrozen wikitext_llama --budgets 0.01 0.03
```

Output: `results/adaptivity_vs_batch.csv`. Latency is proportional to depth here, so `static_speedup` and
`adaptive_speedup` are speedups before overheads.

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
| Runner sanity line (own loop vs HF forward) | 0.98 or higher | lower: stop, the runner and the merged model disagree |
| Any policy with `acc_drop_pts` ≤ 1.0 and `real_speedup` > 1.05 at some B | yes | none at any B: report the depth result of step 2 only, as an upper bound that overheads erase |
| Theory rule vs unified exit at B ≤ 8, both within the budget | theory at least as fast | unified faster everywhere: the LLM result is "a fixed exit is enough", stated as such |
| Per-sample threshold | below 1.0 (it saves nothing when `b ≈ 0`) | n/a (this is the regime check) |

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

**Not measured yet.** `analysis/adaptivity_vs_batch.py` runs on the existing traces (`wikitext_llama`,
`wikitext_ls8b`, `mnli`, `qqp`) in a few minutes on a CPU, but it was not run on 2026-11-13: the team deferred the
whole LLM track. Until it is run, row 3 of section 1 is an argument (the standard deviation of a mean of B
confidences falls as 1/sqrt(B)), not a result. Run it first; it needs no GPU:

```
~/venv/bin/python analysis/adaptivity_vs_batch.py --tasks mnli qqp wikitext_llama wikitext_ls8b
```

## 6. Functional test of the scripts

**Not run.** `decoder/train_exit_lora.py` and `analysis/adaptivity_vs_batch.py` have only been syntax-checked.
Step 0 is therefore a real test, not a formality: expect to fix small errors there (tensor dtypes under autocast,
the `model.layers` / `model.norm` attribute names for the chosen backbone, memory at `--bs 8`).

## 7. Results

Not run yet. Record each step's table here with the date, and state pass or kill against section 4.

## 8. Files

| File | Role |
|---|---|
| `decoder/train_exit_lora.py` | Step 1. LoRA + exit heads + self-distillation; `--rank 0` is the frozen baseline |
| `decoder/train_decoder_exits.py` | First attempt (frozen). Unchanged; `ExitHeads` is imported from it |
| `decoder/decode_runner.py` | Steps 0 and 3. Unchanged; reads the merged backbone through the checkpoint's `model` field |
| `analysis/adaptivity_vs_batch.py` | Step 2 |
| `experiments/run_main.py` | Step 3 selection. Reads `traces/*_val.meta.json` to learn new decoder tasks |
| `scripts/run_llm_v2.sh` | Steps 0–4 in order, with logs |
