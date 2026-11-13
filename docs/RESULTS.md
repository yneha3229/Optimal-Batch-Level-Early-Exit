# Results: When Should a Batch Stop? (2026-11-06)

> **Corrections, 2026-11-13.** A re-check of every number (validation report, Nov 12) and new runs (CUDA graphs,
> headroom bound; `docs/PLAN_PAPER.md` §7) narrowed several statements in this file. The text below is corrected in
> place. The current direction is in `docs/DIRECTION.md`.
> - The theory rule is **not** the fastest rule everywhere. A tuned 3-parameter random search beats it on MNLI by
>   0.07–0.09× and ties it elsewhere; unified exit beats it at B ≤ 16.
> - "Every collective rule is faster at B ≤ 16" holds on MNLI and QQP, not on CIFAR-100.
> - The simulator matches the real GPU within 5% only at B ≥ 64; at B ≤ 16 it is 15–32% off.
> - The robust rule has a clean cost at low ρ on CIFAR-100 (1.12× vs 1.46×).
> - The matched-accuracy numbers are simulated and selected on the held-out split, so they are optimistic.
> - With CUDA graphs the fixed per-layer cost drops about 6× and the ordering of rules changes again.

**Bottom line.** The three claims hold on encoder models in a narrower form than first written.

- **Claim 1 (cost-derived rule):** a rule derived from `c(n) = a + b·n` is competitive in both cost regimes with one
  knob. It is not the fastest rule in every setting.
- **Claim 2 (bandit):** a label-free bandit learns that rule with sublinear regret.
- **Claim 3 (robustness):** a trimmed variant has a certified, detection-free bound against slowdown inputs. It has
  no clean cost at high ρ and a clear clean cost at low ρ on CIFAR-100.

Two things do not hold: early exit on frozen-backbone *decoders* (Qwen2.5-1.5B, Llama-3.1-8B), and straggler
rescue. Both are reported below.

Setup:
- **Encoders:** RoBERTa-base (MNLI, QQP) and ViT-B/16 (CIFAR-100), each with 6 jointly trained exits.
- **Decoders:** Qwen2.5-1.5B and Llama-3.1-8B, frozen, with trained exit heads, doing teacher-forced batched
  decoding on WikiText-103.
- **Hardware:** RTX A6000, fp16/bf16, eager PyTorch.
- **Protocol:** policies are tuned on the evolve split with accuracy within 1 point of the full model, and
  reported on the held-out split.
- "Real GPU" means wall-clock of the runner, which executes the simulator's own `decide()` code.

## Claim 1: one rule, derived from a + b·n, that adapts to both regimes

**Real-GPU wall-clock speedup.** The measured runner cost gives ρ = a/(b·n) of about 1.3–2.6 at B = 8–16 and
0.15–0.4 at B = 64–128 (`results/runner_cost_*.json`, `results/main_<task>_real_<task>_gpu.csv`).

| Policy | MNLI B=8 | MNLI B=16 | MNLI B=64 | MNLI B=128 | QQP B=8 | QQP B=16 | QQP B=64 | QQP B=128 |
|---|---|---|---|---|---|---|---|---|
| Per-sample threshold (DeeBERT-style) | 0.90× | 0.77× | 1.36× | 1.56× | 0.87× | 0.78× | 1.63× | 1.93× |
| Wait-for-all (most research code) | 0.89× | 0.78× | 0.94× | 0.97× | 0.87× | 0.78× | 0.94× | 0.97× |
| Majority vote (DREX) | 1.12× | 1.13× | 1.43× | 1.45× | 1.24× | 1.15× | 1.60× | 1.96× |
| Unified exit (SkipDecode-style) | **1.22×** | **1.16×** | 1.39× | 1.45× | **1.52×** | **1.51×** | 1.81× | 1.91× |
| **Theory rule (ours)** | 1.11× | 1.06× | **1.63×** | **1.75×** | 1.37× | 1.26× | **2.16×** | **2.43×** |

What the real-GPU table shows:
- **The regime switch is real, on hardware.**
  - At small batches (large ρ), per-sample exits are *slower than no early exit* (0.77–0.90×), while the
    collective rules in the table are faster. This holds on MNLI and QQP, not on CIFAR-100.
  - At large batches (small ρ), per-sample exits pay off, and majority vote falls behind the theory rule.
- **The theory rule is the fastest rule in this table at B = 64 and 128.** The comparison is not accuracy-matched
  (see the accuracy caveat), and a tuned random search is not in the table.
- **At B = 8–16 a static truncation wins on MNLI and QQP.** Unified exit evaluates fewer exit heads, and each head
  costs 0.3–0.36 ms (19% of runtime at small B, from the `heads_only` row). The theory rule pays for every head.
  Choosing *which* exits to evaluate from the same cost model is the obvious next step.
- **CIFAR-100 (ρ₆₄ = 0.15) is all low-ρ.** At B = 64 the rule (1.47×) ≈ per-sample threshold (1.49×), and majority
  vote is 1.05×, as the theory predicts.
- **Accuracy caveat.** The real-GPU runs use the policies selected in simulation. Their held-out accuracy drops
  are 0.8–2.4 points (λ chosen at the edge of the constraint on a 4,900-input split). For a fair comparison, use
  the matched-accuracy table below.

**Matched accuracy (simulated; selected on the held-out split, so optimistic).** Best held-out speedup among each method's candidates within 1 point of the
full model's held-out accuracy, simulator at B = 64 (`results/matched_table_*.md`):

| Cost regime | MNLI | QQP | CIFAR-100 |
|---|---|---|---|
| Real runner cost: theory / best heuristic | **1.41×** / 1.37× majority | **2.10×** / 1.66× majority | **1.46×** / 1.45× threshold |
| Measured encoder cost: theory / best heuristic | **1.36×** / 1.33× majority | **2.03×** / 1.59× majority | **1.26×** / 1.25× threshold |
| Decoder-like cost (ρ₆₄ ≈ 95): theory / best heuristic | 1.36× / **1.39×** majority | **1.80×** / 1.66× majority | **1.02×** / 1.02× majority |
| Per-sample threshold in the decoder-like regime | 0.94× | 0.94× | 0.94× |

What the matched table shows:
- **Each heuristic collapses in one regime.** Majority vote does at low ρ (CIFAR 0.98–1.05×). Per-sample exits do
  at high ρ (0.94× on every task).
- **The rule matches or beats the best hand-written heuristic,** within 0.03×.
- **It is not unbeatable.** A random search over a 3-parameter hand-written collective rule (150 labelled
  evaluations) beats the theory rule on MNLI by 0.07–0.09× and ties it elsewhere. The rule's advantage is one knob
  (λ) and a derivation, not a higher ceiling.

## Claim 2: the label-free bandit learns the rule online

- **Labels and probes.** It uses no labels. Its target is agreement with the full model's answer, which
  upper-bounds the accuracy loss. Occasional full-depth probe batches reveal whole traces, with probe rate ∝ t^(−1/3).
- **Regret is sublinear** against an oracle that knows the true tables. Cumulative regret over 3,000 batches of 64
  has log-log slopes of:
  - 0.65 (MNLI, decoder-like cost)
  - 0.49 (MNLI, encoder cost)
  - 0.52 (QQP, decoder-like cost)

  The partial-monitoring sketch predicts T^(2/3).
  On CIFAR and the decoders the oracle itself never stops, and neither does the bandit (regret ≈ 0).
- **Learned stopping threshold, MNLI.** It falls from about 0.85 confidence at n = 64 active inputs to about 0.1 at
  n ≤ 4. That is the "threshold drops as the batch drains" prediction (`figures/ex5_bandit_*`).
- **CeeBERT-style per-sample bandit:** its regret is linear (`figures/ex5_bandit_mnli_dec_eager_B64.png`).
- **Real GPU.** The bandit costs only 0.18–0.29 points of accuracy and reaches 1.26–1.77× at B = 64–128 on
  MNLI and QQP.
- **The price of the label-free guarantee is conservatism.** Its latency is 0.73–0.83× of the full model's,
  against 0.62–0.74× for the oracle.

## Claim 3: certified, detection-free robustness to slowdown inputs

**Theorem.** The robust rule trims the `m` largest gains from the collective test. For every batch, and every
`k ≤ m` slowdown inputs, the attacked batch ends no later than the same clean inputs would under the rule trimmed by
`m − k`. The proof is in `theory/THEORY.md` §6.

**Verified.** 0 violations in 2,100 batches on each of five tasks, including real DeepSloth inputs on the ViT
(`results/robust_violations_*.csv`, `tests/test_core.py::test_robust_rule_pathwise_depth_bound`).

| CIFAR-100, real DeepSloth inputs, decoder-like cost, B = 64 | Clean speedup | k=0 | k=1 | k=4 | k=8 |
|---|---|---|---|---|---|
| Theory rule, plain | 1.05× | 0.94 | 0.96 | 1.05 | 1.07 |
| **Robust theory rule, m = 8** | **1.07×** | 0.92 | 0.94 | **1.01** | 1.07 |
| Consensus (Miao et al. / DREX) | 1.04× | 0.95 | **1.07** | 1.07 | 1.07 |
| Majority vote (DREX) | 1.05× | 0.95 | 0.96 | 0.99 | 1.04 |

The k columns are victims' latency as a fraction of the full model's.

- **No clean cost at high ρ.** At matched accuracy the robust rule is as fast or slightly faster with decoder-like
  cost: QQP 1.94× vs 1.99× within 2 points, MNLI 1.39× vs 1.36× within 1 point, CIFAR 1.07× vs 1.05×.
- **A clean cost at low ρ.** With the real runner cost on CIFAR-100 the robust rule (m = B/8) reaches 1.12× against
  1.46× for the plain rule.
- **Consensus loses its whole speedup to one attacker.**
- **Majority vote** is robust by construction up to B/2 attackers. It degrades faster than the robust rule on QQP
  (0.60 vs 0.53 at k = 8).
- **Per-sample exits** have nothing to lose. Natural stragglers already push almost every batch to full depth.
- **Limit: victims' latency still rises with k.** Flat latency is impossible without detection, because ignoring
  the attackers' gains means ignoring the hardest clean inputs too.

## Supporting results

- **FunSearch / OpenEvolve:** local qwen2.5-coder:7b, 300 programs.
  - **Evolve-split score:** the best program scores 1.406, against 1.411 for the theory rule, 1.40–1.42 for random
    search and BO at the same budget, and 1.03 for the seed.
  - **What it found:** per-input exits plus whole-batch stops, the same two-part structure as the theory rule. It
    always stops at layer 8 and cuts the batch when fewer than a quarter of the inputs remain.
  - **Its ρ test is spurious:** it compares a boolean against 1.2.
  - **It overfits:** held-out score 0.86 vs 0.91 for the theory rule.
  - **Search mechanics:** with Ollama's default 4k context, 82% of diff edits failed. A 16k context with full
    rewrites failed only 6%.
- **Straggler rescue (combining exits for cut-off inputs):** negative. It never raised hardest-10% accuracy, and
  sometimes cost speedup.
- **Trajectory-aware calibration** (confidence plus its change since the last exit): within ±0.07× of the plain rule
  at matched accuracy everywhere. It does not help.

## What did not work: frozen-backbone decoders

The cost model is confirmed on both decoders:
- Qwen2.5-1.5B decode steps cost a = 1.07 ms per layer with b ≈ 0, so ρ is effectively infinite.
- Llama-3.1-8B costs a = 0.78 ms, b = 0.0024 ms per sequence, so ρ₆₄ = 5.2.
- Fused KV propagation for skipped layers costs 0.02 ms per layer (2% of a layer).

But the exits are too weak for any batch-level stop within 1 point:
- Llama's layer-28 exit is 3.7 points below the full model in next-token accuracy (49.9% vs 53.6%), and agrees with
  it on only 74.5% of tokens. Qwen's layer 24 is 4.3 points below.
- No policy beats the full model within 1 point (real GPU: 0.68–0.95×).
- With 5 points of slack, majority vote reaches 1.14–1.17×.

LLM early exit needs a backbone trained for it (LayerSkip, EE-LLM). That is out of scope for 3 weeks. Present it in
one line: "cost model confirmed; frozen exits too weak".

## How to present it

1. **Lead with the real-GPU regime switch** on MNLI/QQP. Per-sample exits run below 1× at B ≤ 16 and pay off at
   B ≥ 64. Collective rules show the reverse ordering.
2. **The theory rule is competitive in both regimes with one knob.** Show the matched table, and be upfront that a
   tuned 3-parameter random search ties or beats it. The stronger framing is in `docs/DIRECTION.md`.
3. **The bandit:** label-free, with sublinear regret matching T^(2/3), and the threshold-vs-n plot.
4. **Robustness:** the theorem, 0 violations, and Consensus breaking at k = 1.
5. **One slide of negatives:** decoders, rescue, FunSearch overfitting, and unified exit winning at B ≤ 16 because
   of head overhead (the next step).

## Reproduce

All results came from `scripts/run_final.sh`, `scripts/run_parallel.sh` and `scripts/run_lane4.sh`, which run
GPU 3/4/5 in parallel. Logs are in `logs/final_*.log`, status in `logs/final_status.log`. Runners were calibrated on
GPU 3 (A6000); some timing ran on GPUs 4/5, which are identical A6000s (85–86 dense fp16 TFLOPS each in a check).
Tests: `~/venv/bin/python -m pytest tests simulator/test_simulator.py -q`.
