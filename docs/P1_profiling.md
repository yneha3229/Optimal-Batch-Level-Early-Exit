# P1: Hardware profiling (measure a and b)

Date: 2026-10-28. GPU: RTX A6000 (nvidia-smi index 3). Software: torch 2.14.0+cu130, transformers 5.17, SDPA attention.
Reproduce with: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python profiling/profile_layer.py configs/p1_profile.yaml`
Outputs: `results/p1_timing_A6000.csv`, `results/p1_fit_A6000.json`, `figures/p1_cost_fit_A6000.png`

Setup: one RoBERTa-base encoder layer, random weights (they don't affect timing), CUDA events, 10 warm-up runs and 50 timed repeats, median reported. n = 1…128, seq_len ∈ {64, 128, 256}, fp32/fp16. Two modes:
- **eager**: plain PyTorch, the way most early-exit research code runs.
- **cuda_graph**: the layer is captured once and replayed. This removes Python and kernel-launch overhead and approximates an optimized server (TensorRT, CUDA graphs in vLLM and similar).

## Answer

**Gate 1 (`a ≫ b·n`) fails for RoBERTa-base at the headline batch of 64.** In every setting we measured, ρ = a/(b·n) < 1 at n = 64. Almost all of the fixed cost `a` in eager mode is Python and kernel-launch overhead, not weight loading. CUDA graphs cut it by about 7×.

| Setting (fp16) | a (ms) | b (ms/sample) | n* = a/b (ρ = 1) | ρ at n = 64 | R² |
|---|---|---|---|---|---|
| eager, L=64 | 0.443 | 0.0095 | 47 | 0.73 | 0.961 |
| eager, L=128 (MNLI plan) | 0.382 | 0.0216 | 18 | 0.28 | 0.990 |
| eager, L=256 | 0.330 | 0.0459 | 7 | 0.11 | 0.998 |
| cuda_graph, L=64 | 0.042 | 0.0129 | 3.3 | 0.05 | 0.997 |
| cuda_graph, L=128 | 0.053 | 0.0245 | 2.1 | 0.03 | 0.999 |
| cuda_graph, L=256 | 0.080 | 0.0480 | 1.7 | 0.03 | 0.999 |

In fp32, n* ≤ 6 everywhere: the layer is compute-bound almost immediately. The full table is in `results/p1_fit_A6000.json`.

The **memory/overhead-bound range** (ρ > 1) is n < n*: n ≲ 18 for the planned eager fp16 L=128 setting, and n ≲ 2 with CUDA graphs.

## Details

- **Shape of c(n).** In eager fp16 the curve is flat at about 0.55 ms up to a knee (n ≈ 12 at L=128), then linear. It is closer to `max(a, b·n)` than to `a + b·n`. A hinge fit `a + b·max(0, n−k)` raises R² from 0.990 to 0.997 at L=128 and from 0.961 to 0.989 at L=64. Under CUDA graphs the plain linear model fits well (R² ≥ 0.997). **Implication for P3:** the theory should allow a roofline-style cost, or at least state that a + b·n is a linearization.
- **Why a is overhead.** One layer holds about 7M fp16 parameters, roughly 14 MB. At the A6000's ~768 GB/s, loading them takes about 0.02 ms, yet eager mode measures a ≈ 0.4–0.5 ms. The gap is Python and kernel launches. [VERIFY: the 768 GB/s spec figure]
- **Additivity check (simulator assumption).** We compared the full 12-layer encoder with 12 × one layer:
  - CUDA graphs: the ratio is 0.92–1.10, so summing per-layer costs holds within ±10%.
  - Eager: the ratio drops to 0.72 at small n. Launch overhead overlaps with GPU work, so the simulator would *overestimate* eager latency at small batches by up to about 28%.
- **Exit head** (dense, tanh, out_proj, softmax-max on CLS):
  - Eager: about 0.13–0.16 ms per call, almost flat in n. That is about 9% of a layer at n = 64 and about 28% at n = 1, so it is not negligible.
  - CUDA graphs: about 0.02 ms, roughly 1% of a layer at n = 64.

## What this means for the thesis

1. When `b·n` dominates and exited samples are *removed* from the batch, latency ≈ b·Σ n_l. Per-sample exits then already turn FLOPs saved into time saved, and the straggler costs only its own share. **The straggler effect in this regime comes from implementation, not from the cost:** it appears only when exited samples stay in the batch (masking, or "wait for all"). H1 has to separate these two cases.
2. **Large ρ should occur in decoder token-by-token generation.** Each sample then contributes one token per step while the layer weights (Qwen2.5-1.5B: ~50M params per layer, ~100 MB fp16) still have to be loaded. That setting is weight-bandwidth-bound, so `a ≫ b·n` should hold up to large n. [VERIFY by measuring]
3. This makes **the regime switch itself the story**: the theory predicts per-sample exits for encoder classification (small ρ) and collective termination for decoder decoding (large ρ). Measuring both regimes tests the theorem directly, rather than relying on one setting where it happens to hold.

## Next measurements (cheapest first)

1. **Qwen2.5-1.5B decode layer:** n sequences × 1 new token, with KV cache at context 256/1024, in eager and CUDA-graph modes. This decides whether the large-ρ regime exists on this GPU, and takes about 30 minutes.
2. **Short sequences** (SST-2, padded L=32/64) in eager mode, the setting where encoder ρ is largest.
3. **A second GPU** (RTX 3080 Ti, indices 0–2) for the "different a/b per GPU" ablation. **Needs your OK, since we agreed to use GPU 3.**
