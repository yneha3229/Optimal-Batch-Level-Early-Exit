# Direction: one cost-aware exit plan for every deployment (2026-11-13)

This is the current direction of the project. It keeps the original idea (batch-level early exit derived from the
cost model `c(n) = a + b·n`) and changes the claim.

Status: **documented; step 1 (transfer test) running.** Results will be added in section 7.

## 1. What changed and why

**Old claim:** the theory rule is the fastest batch-level exit rule.

**Why it cannot stand.** Every attempt to raise the speed ceiling at a fixed setting failed:

| Attempt | Outcome | Where |
|---|---|---|
| Random search over a 3-parameter rule | Ties or beats the theory rule | `docs/RESULTS.md` |
| FunSearch (OpenEvolve) | Ties (1.406 vs 1.411) and overfits | `docs/RESULTS.md` |
| Trajectory-aware calibration | Within ±0.07× | `docs/RESULTS.md` |
| Hindsight-best batch depth | Does not beat the rule | `analysis/headroom.py` |
| Hidden-state correctness probe | At most +0.026 AUROC over confidence | `results/probe_*.csv` |
| Exit-trained decoders (LayerSkip 1B, 8B) | About 5 points short at 75% depth | `results/layerskip_*_layer_accuracy.csv` |

Confidence is all the information a policy has, and the rule already uses it fully. Peak speed is not where the
contribution is.

**New claim:** no existing rule is good across deployments, and which one wins flips with batch size, execution
mode and task. A rule that takes the measured cost `(a, b, head, n)` as input can be tuned once and stay near the
best method everywhere, with no per-deployment re-tuning.

## 2. The evidence that motivates it

Real-GPU speedup with CUDA graphs (RTX A6000, fp16; `results/graph_<task>_gpu.csv`):

| | MNLI B=8 | MNLI B=128 | QQP B=8 | QQP B=128 | CIFAR-100 B=64 |
|---|---|---|---|---|---|
| Unified exit (SkipDecode-style) | **1.08×** | 1.45× | **1.58×** | 1.93× | 0.98× |
| Per-sample threshold | 0.89× | 1.50× | 1.07× | 1.91× | **1.41×** |
| Majority vote (DREX) | 1.05× | 1.45× | 1.33× | 1.98× | 1.04× |
| Theory rule (ours) | 0.86× | **1.61×** | 1.23× | **2.31×** | 1.39× |

- Unified exit wins at small batches on text and gives no speedup on CIFAR-100.
- Per-sample exits are near the best on CIFAR-100 and slower than no early exit on MNLI at B = 8.
- Majority vote is never the best in these cells.
- The theory rule is the best or close at large batches and the worst rule at B = 8 on MNLI.

Accuracy caveat: these runs are not accuracy-matched on MNLI (theory drops 1.8–2.5 points, unified 0.8–0.9). Step 2
reruns them matched.

## 3. What is missing in the rule

The rule decides who leaves and when the batch stops. It never decides **which exits to evaluate**, so it pays for
every exit head.

Example, MNLI at B = 8 with CUDA graphs: evaluating the five intermediate heads costs 19% of the runtime
(`heads_only` = 0.81×). Unified exit evaluates one head and reaches 1.08×. The theory rule evaluates all five and
reaches 0.86×.

**Extension: the exit plan.** From the same cost model, choose the subset of exits to evaluate.

- An exit is worth evaluating only if its expected saving exceeds its head cost:
  `E[layers saved after e] · (a + b·n) > head(n)`, with the expectation taken from the calibration tables.
- With one exit kept, the plan behaves like unified exit. With all exits kept and cheap heads, it behaves like the
  current rule. The cost model picks between them.
- The knob stays λ. The plan is recomputed from `(a, b, head, n)`, which take about one minute to measure on a new
  machine (`runner/gpu_runner.py --calibrate`, `runner/graph_runner.py`).

Prediction, not yet a result: the plan reaches at least unified exit's speed at B ≤ 16 and keeps the rule's speed at
B ≥ 64.

## 4. Second half: real load

A server does not run at one batch size. With dynamic batching the batch size moves between 1 and 128 with traffic.
A heuristic tuned at B = 64 is mis-tuned when load changes. A rule that takes `n` as input is not.

Experiment: replay an arrival trace with fluctuating batch sizes; compare the plan tuned once against heuristics
tuned at one batch size and against heuristics re-tuned per batch size (the upper reference).

## 5. Tests, in order, with kill criteria

| Step | Question | Pass | Kill | Cost (estimate) |
|---|---|---|---|---|
| 1. Transfer test | Do tuned policies transfer across settings? | The theory rule's worst-case loss against the per-setting best is at least 5 points smaller than every heuristic's and random search's | Heuristics and random search transfer as well as the rule | CPU, about 2 h |
| 2. Exit plan | Does choosing exits fix small batches? | Within 5% of the best method in every real-GPU cell (3 tasks × 4 batch sizes × 2 execution modes), accuracy matched within 1 point | Still more than 10% behind at B ≤ 16 | 1–2 days, GPU 5 |
| 3. Load trace | Does it hold under fluctuating batch size? | Tuned-once plan within 5% of per-batch-size re-tuned heuristics, and ahead of tuned-once heuristics | No gap between tuned-once methods | 2 days, GPU 5 |

### Step 1 design
- Settings: 3 tasks × batch sizes {8, 16, 64, 128} × cost regimes {eager runner, CUDA-graph runner}. Transfer is
  within a task (a policy cannot move between tasks).
- For each method and each source setting: select parameters on the evolve split at the source (within 1 point), then
  evaluate the same parameters at every target setting on the held-out split.
- Metric per target: speedup relative to the best method tuned at that target, counted only if accuracy stays within
  1 point; a violation counts as 1.0× (fall back to the full model).
- Report: mean and worst-case relative speedup over targets, per method.
- Simulator only. The simulator is within 5% of the GPU at B ≥ 64 in eager mode and worse elsewhere, so step 1 is a
  screen; step 2 is the real-GPU evidence.

## 6. How the existing pieces fit

| Piece | Role in the new story |
|---|---|
| Cost model and theory rule (`theory/THEORY.md`, `simulator/policies.py`) | The core. Extended with the exit plan |
| Label-free bandit (`bandit/`) | How λ is tuned in deployment without labels |
| Trimmed robust rule (`theory/THEORY.md` §6) | A certified option when collective stops are used; unchanged |
| CUDA-graph runner (`runner/graph_runner.py`) | Second execution mode for every real-GPU table |
| Headroom and probe analyses (`analysis/`) | The argument for why peak speed is capped |
| Decoders, rescue, FunSearch | Negative results, one section |

## 7. Results

(to be filled as steps complete)

## 8. Honest scope

- This is a robustness-across-deployments result, not a peak-speed result. Expected gain is "never far from the best
  method", not "faster than every method".
- Evidence base is two text tasks on RoBERTa-base and one vision task on ViT-B/16, on one GPU model.
- Venue level if steps 1–3 pass: systems-leaning (MLSys, EMNLP). No result here is a guaranteed paper.

## 9. Parked alternatives
- `docs/PLAN_V2.md`: bandit lower bound (Track A) and reality-check paper (Track B). Not started.
- `docs/PLAN_PAPER.md`: decoder-centred paper track. Stopped after steps 1–3 failed.
