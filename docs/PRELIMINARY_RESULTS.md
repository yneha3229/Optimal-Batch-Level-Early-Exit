# Preliminary results (2026-10-28)

Setting: RoBERTa-base, MNLI-matched validation. The backbone is `textattack/roberta-base-MNLI`, frozen. Exit heads sit at layers 2/4/6/8/10/12 and were trained DeeBERT-style (stage 2) on 100k MNLI-train samples. Policies were tuned on the *evolve* half (4,907 samples) and are reported on the *held-out* half (4,908). "Matched accuracy" means within 1 point (absolute) of the full model, checked on the evolve split. Real runs: RTX A6000 (GPU 3), fp16, eager PyTorch, inputs padded to 128.

## 1. Exit heads (`results/p2_exit_accuracy.csv`)

| exit layer | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|
| accuracy | 42.5 | 49.2 | 57.3 | 70.6 | 87.5 | **87.9** |

The early exits are weak because the backbone was fine-tuned only for its last layer. Layer 10 is the only exit close to full accuracy, so **no policy can reach much more than 12/10 = 1.2× at matched accuracy.** This ceiling binds every result below.

## 2. Simulation, all measured cost regimes (`results/p6_sim_selected.csv`, `figures/p6_sim_speedup_vs_batch.png`)

Held-out speedup at matched accuracy:

| regime (measured a, b) | B | per-sample threshold | threshold, no compaction | Consensus | Majority | random-search collective | theory (indiv. only) | **theory** |
|---|---|---|---|---|---|---|---|---|
| encoder, eager (ρ₆₄ = 0.28) | 16 | 1.00 | 0.93 | 1.10 | 1.11 | 1.12 | 1.05 | **1.12** |
| | 64 | 1.11 | 0.97 | 1.12 | 1.17 | 1.18 | 1.13 | **1.16** |
| | 128 | 1.16 | 0.98 | 1.12 | 1.18 | 1.21 | 1.19 | **1.19** |
| encoder, CUDA graph (ρ₆₄ = 0.03) | 64 | 1.20 | 0.99 | 1.16 | 1.20 | 1.22 | 1.22 | **1.22** |
| encoder, RTX 3080 Ti | 64 | 1.13 | 0.97 | 1.13 | 1.17 | 1.19 | 1.15 | **1.18** |
| **decoder-like cost (Qwen2.5-1.5B decode, ρ₆₄ ≈ 95)** | 16 | **0.95** | 0.95 | 1.11 | 1.13 | 1.12 | **0.96** | **1.12** |
| | 64 | **0.94** | 0.94 | 1.09 | 1.13 | 1.12 | **0.94** | **1.12** |
| | 128 | **0.94** | 0.94 | 1.07 | 1.12 | 1.13 | **0.94** | **1.12** |

In the decoder row, the MNLI traces are replayed under the measured Qwen decode cost profile. It is a what-if on the cost model, not a decoder experiment.

What this shows:
- **H1 (FLOPs saved ≠ time saved) holds, in two different forms:**
  - *Implementation straggler:* when exited samples stay in the batch (no compaction, as in most research code), per-sample exits are **slower than the full model** in every regime (0.93–0.99×).
  - *Cost-model straggler:* when the fixed cost dominates (decoder-like ρ), even *compacted* per-sample exits are slower (0.94×), because removing samples saves nothing.
- **The regime switch appears, in simulation.** With small ρ (encoder), the theory rule restricted to individual exits matches the full theory rule (1.22 vs 1.22 under CUDA graphs). With large ρ (decoder), individual exits alone lose (0.94×), and only collective termination helps (1.12×). This is the qualitative prediction of the target theorem.
- **H2 holds only partially.** Collective termination recovers the speedup at under 1 point of accuracy loss, but the gain over a well-tuned heuristic is small. The theory rule comes within about 0.02× of the best baseline and never clearly beats it. Random search over a hand-parameterized collective rule (150 programs) is the strongest baseline. This is the bar that evolution must beat.

## 3. Real GPU wall-clock (`results/p6_wallclock_enc_eager_meas.csv`, `figures/p6_wallclock_enc_eager_meas.png`)

Same policies, tuned in simulation, run for real on the held-out split:

| B | threshold | threshold no-compact | Consensus | Majority | random collective | theory indiv-only | **theory** | acc. drop (theory) |
|---|---|---|---|---|---|---|---|---|
| 16 | 0.66 | 0.65 | 0.76 | 0.76 | 0.75 | 0.69 | **0.73** | 0.57 pt |
| 64 | 0.97 | 0.84 | 0.99 | 1.01 | 1.03 | 0.97 | **1.00** | 0.41 pt |
| 128 | 1.08 | 0.91 | 1.04 | 1.09 | 1.11 | 1.08 | **1.09** | 0.58 pt |

- **The headline speedup is not there yet.** Real wall-clock speedup at matched accuracy is at most **1.11× (B=128)** and about 1.0× at B=64. Below B=64, early exit makes the system *slower*.
- **Why: exit overhead.** Each exit point costs **0.50–0.67 ms** in practice: the head plus a GPU→CPU sync plus a host-side decision plus compaction. The head alone was profiled at 0.15 ms. That is about 1.3× a whole layer's fixed cost, paid 5 times per batch. Evaluating every head without ever exiting (`heads_only`) already costs 12–33% of runtime.
- **The ranking holds on real hardware.** Collective rules beat per-sample threshold at B=16 (0.73–0.76 vs 0.66) and at B=64. No-compaction is always worst.
- **The simulator ranks correctly but is optimistic:** about 3–5% too high at B=128 and 20–30% at B=16. The remaining gap is eager-mode launch overlap, which makes the real full model faster at small batches than the summed per-layer costs predict. P1 measured this as an encoder/(12×layer) ratio of 0.72. The simulator should be calibrated against end-to-end full-model time, not isolated layer time.

## 4. Honest status of H1–H4

| | Status |
|---|---|
| H1: FLOPs ≠ time; speedup → ≤1× as B grows | **Supported, with a twist.** Real speedup falls *as B shrinks* in eager mode, because the fixed per-exit overhead dominates small batches. Without compaction, it stays below 1× at every B. |
| H2: collective termination recovers speedup at <1% loss | **Partly.** It beats per-sample exits on the GPU, but the absolute gain is small (≤1.11×), capped by weak exits and overhead. |
| H3: evolution rediscovers a ρ-dependent rule | **Not run yet.** The equal-budget random-search baseline exists and is strong. |
| H4: learned threshold falls as the batch drains | **Not run yet** (bandit). |

## 5. Cheapest next steps that would change these numbers

1. **Train exits jointly** by fine-tuning the backbone with all 6 exit losses (about 1–2 h on GPU 3). This raises early-exit accuracy and the 1.2× ceiling, and it is the biggest lever.
2. **Cut exit overhead:**
   - decide on the GPU and sync only one stop flag;
   - drop the useless exits at layers 2 and 4;
   - wrap layer segments in CUDA graphs.
   This could move B=64 from about 1.0× to about 1.1× or more without any policy change.
3. **Calibrate the simulator** on end-to-end time: fit a, b and the exit overhead from real runs.
4. **Run a real decoder early-exit experiment** (Qwen2.5-1.5B with exit heads), the regime where the theory predicts collective termination matters most. So far it exists only as a simulation what-if.
5. Then **OpenEvolve (P4)** against the simulator, and the **bandit (P5)**.

## Reproduce

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python traces/train_exit_heads.py configs/p2_mnli.yaml   # ~4 min
~/venv/bin/python simulator/run_sim.py                                                                                 # ~4 min, 48 CPU workers
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python simulator/wallclock.py --regime enc_eager_meas --batch-sizes 16 64 128
~/venv/bin/python simulator/plot_wallclock.py enc_eager_meas
```
Decoder profile: `profiling/profile_decoder.py` (ran on GPU 5). Second-GPU profile: `configs/p1_profile_3080ti.yaml` (ran on GPU 0).
