# Direction: one cost-aware exit plan for every deployment (2026-11-13)

This is the current direction of the project. It keeps the original idea (batch-level early exit derived from the
cost model `c(n) = a + b·n`) and changes the claim.

Status: **step 1 (transfer test) done: fails its pass criterion (section 7).** Steps 2–3 not started.

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

### Step 1: transfer test (2026-11-13). **Fails the pass criterion.**

`experiments/transfer_test.py`, simulator only. Outputs: `results/transfer_summary.csv`, `results/transfer_pairs.csv`,
`results/transfer_<task>_all.csv`, log `logs/transfer_test.log`. Setup: 3 tasks × 8 settings (B ∈ {8, 16, 64, 128} ×
{eager, CUDA-graph} cost), 6 methods; parameters selected on evolve and scored on held-out (3 seeds).

Relative score = speedup ÷ best in-domain speedup at the target. A held-out accuracy drop above 1 point scores 1.0×.
Selection margin 0.5 pt, all tasks pooled:

| Method | Mean, other batch size | Mean, other mode | Worst, other batch size | Worst, other mode | Violation rate (transfer) |
|---|---|---|---|---|---|
| Theory rule | **0.93** | **0.93** | 0.45 | 0.39 | 29–31% |
| Random search (3 params) | 0.89 | 0.89 | 0.48 | 0.48 | 27–28% |
| Per-sample threshold | 0.88 | 0.88 | **0.64** | **0.64** | **0%** |
| Majority vote | 0.78 | 0.77 | 0.41 | 0.41 | 11–13% |
| Unified exit | 0.76 | 0.76 | 0.41 | 0.39 | 17% |
| Consensus | 0.67 | 0.69 | 0.38 | 0.38 | 33–42% |

- **Against the criterion:** the theory rule's worst-case loss is 55–61%, the threshold's is 36%. The rule is
  worse in the worst case, not 5 points better. Kill condition met.
- **Mean:** the theory rule has the best mean transfer score (0.93), 4–5 points above threshold and random search.
  This is below the 5-point bar and does not count as a pass.
- **Cause of the worst cases:** accuracy, not speed. The rule's worst pairs are QQP real B=64 → graph B=64 / real
  B=128 (drops of 1.01–1.04 pts, just over the 1-pt budget, while speedup is 2.35–2.61×) and MNLI B=16/64 → B=128
  (drop 1.71 pts). The threshold rule never violated the budget in transfer.
- **Margin 1.0 pt** (also in the summary): violation rates of 42–88% for every method, so that margin is too loose
  for the 1-pt budget; worst cases collapse to about 0.37 for all methods.
- Per task (transfer pairs, mean / worst): theory 1.00 / 0.96 on CIFAR-100, 0.81 / 0.62 on MNLI, 0.98 / 0.39 on QQP.

Status after step 1: per section 5, the transfer claim as stated does not hold in simulation. Steps 2–3 have not
been started.

## 8. Honest scope

- This is a robustness-across-deployments result, not a peak-speed result. Expected gain is "never far from the best
  method", not "faster than every method".
- Evidence base is two text tasks on RoBERTa-base and one vision task on ViT-B/16, on one GPU model.
- Venue level if steps 1–3 pass: systems-leaning (MLSys, EMNLP). No result here is a guaranteed paper.

## 9. Parked alternatives
- `docs/PLAN_V2.md`: bandit lower bound (Track A) and reality-check paper (Track B). Not started.
- `docs/PLAN_PAPER.md`: decoder-centred paper track. Stopped after steps 1–3 failed.
