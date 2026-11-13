# Feasibility check of the plan "When Should a Batch Stop?" (2026-11-06)

> **Superseded by `docs/RESULTS.md`** (final results with real-GPU numbers, the robustness theorem and the decoder experiments). This file is the first feasibility check.

**Verdict: the project can be finished on time. Every module in the plan is written and has been run end to end.
Three of the plan's claims do not hold as written, so the pitch has to change before the presentation.** Gaps
that remain are listed in section 4.

GPU cost so far: 18 min (MNLI) + 23 min (QQP) + 12 min (CIFAR-100) of joint training, plus 2 min of DeepSloth
attacks, all on one A6000. That is far below the plan's 20–25 GPU-hour budget.

## 1. Claim by claim (simulation on held-out data, B = 64; real-GPU numbers in section 3)

The "matched" numbers compare methods at equal held-out accuracy, within 1 point of the full model
(`results/matched_table_<task>.md`), so selection noise cannot favour any method.

| Claim in the plan | Status | Evidence |
|---|---|---|
| H1: existing methods save FLOPs but barely any wall-clock time at B = 64 | **Only partly true.** | See H1 breakdown below. |
| H2: the theory rule stays within 1% and recovers the speedup | **Holds, but the gain over the best baseline is task-dependent.** | Speedup at matched accuracy: MNLI 1.36×, QQP 2.03× (encoder cost) / 1.80× (decoder-like), CIFAR-100 1.26×. It beats majority vote clearly on QQP (1.80× vs 1.66×; 2.03× vs 1.59×). It ties majority vote on MNLI with decoder-like cost (1.36× vs 1.39×). A random search over a 3-parameter rule matches it on MNLI and CIFAR. |
| Regime switch in ρ | **Holds (simulation).** | With encoder cost (ρ₆₄ ≈ 0.3), the rule restricted to individual exits equals the full rule. With decoder-like cost (ρ₆₄ ≈ 95), per-sample exits fall below 1×, while the collective rule gives 1.36–1.80×. |
| Gate 1: `a ≫ b·n` | **Fails for the encoders** (ρ₆₄ ≈ 0.3, `docs/P1_profiling.md`). | It holds only for decoder-like decoding cost. That regime is a cost what-if: classification traces replayed under a measured Qwen2.5-1.5B decode cost. |
| H3: the bandit learns the rule online without labels; its threshold drops as the batch drains | **Holds.** | Regret is sublinear (log-log slope 0.65 with decoder cost, 0.49 with encoder cost, against T^(2/3) in theory). The learned stop threshold falls from about 0.85 at n = 64 to about 0.1 at n ≤ 4 (`figures/ex5_bandit_*`). The catch: it is conservative, with 1.09–1.47× at B = 64 versus 1.36–2.03× for the label-tuned rule. |
| H4: with 1–4 slow inputs, others' latency rises under standard early exit and stays flat under our rule | **Not supported.** | See H4 breakdown below. |
| H5a: FunSearch rediscovers the rule | **Pending** (the run is in progress). | After 85 programs its best score was 1.16, against 1.40–1.42 for random search and BO at 300 programs each. |
| H5b: rescue recovers accuracy on cut-off inputs | **Not supported.** | Hardest-10% accuracy never improves, and rescue sometimes costs speedup (QQP decoder: 1.48× vs 1.99×). Early exits are weaker than the stop exit, so pooling dilutes the prediction. |

**H1 breakdown.**
- **True for wait-for-all:** exited inputs stay in the batch, which is what most research code does. It gives
  0.90× despite saving 40–60% of FLOPs.
- **True for per-sample exits under decoder-like cost:** 0.94×.
- **False for compacted per-sample exits under encoder cost:** 1.25–1.50× at B = 64.

**H4 breakdown.**
- **Per-sample exits have nothing left to lose.** At B = 64, natural stragglers already push almost every batch
  to full depth. With real DeepSloth inputs on ViT, victim latency moves only from 0.81× to 0.83× of the full
  model with 4 attackers.
- **Our rule is not flat either.** Under decoder-like cost, its victim latency rises from 0.93× to 1.05× as
  attackers go from 0 to 4.
- **What does hold:** one attacker defeats Consensus on QQP (1.17× victim slowdown), and our rule's damage grows
  only linearly in k.

## 2. Suggested reframing (keeps all three novel pieces, all of them now backed by data)

- **Title claim:** "Per-sample early exit stops paying once the fixed per-layer cost dominates. We derive the
  batch-stopping rule from the measured cost and show it switches regimes at ρ = a/(b·n)."
- **Headline exhibit:** speedup vs batch size in both cost regimes (`figures/ex1_*`), plus the matched-accuracy
  table. QQP is the strongest task to lead with.
- **Bandit:** present it as "label-free with a guarantee". The 1% disagreement target bounds the accuracy drop,
  regret is sublinear, and the learned threshold drops as the batch drains. Say plainly that it is conservative.
- **Attacks:** turn H4 into a finding. "At B = 64, synchronous per-sample exits are already maximally slowed by
  natural stragglers, so slowdown attacks matter for collective rules. Consensus breaks with one attacker; our
  rule degrades linearly."
- **Rescue:** report it as a negative result in one line, or drop it.

## 3. Real GPU (RTX A6000, fp16, eager PyTorch)

Real-GPU wall-clock numbers are written by the pipeline to `results/main_<task>_real_<task>_gpu.csv`.

The runner now runs the simulator's exact `decide()` code with one host sync per exit point. An early functional
run at B = 64 on MNLI tracked the simulator within about 5%:

| Policy | Real speedup | Simulated |
|---|---|---|
| Majority vote | 1.42× | 1.36× |
| Theory rule, individual exits only | 1.62× | 1.60× |
| Wait-for-all | 0.92× | 0.90× |

That run shared the GPU with training, so it is indicative only. The preliminary results had about 1.0×
real speedup at B = 64; the gain comes from joint exit training plus the lower exit overhead.

## 4. What was fixed along the way (worth knowing before you extend it)

- **Frozen exit heads were too weak.** Joint training lifted MNLI exit accuracy: layer 6 from 57% to 82%, layer 8
  from 71% to 87%. The final layer dropped 0.1 points.
- **Label order.** The `textattack/roberta-base-MNLI` classifier outputs GLUE labels in reverse order
  (permutation 2,1,0). The remap lives in `configs/train_mnli.yaml`.
- **Calibration noise.** Negative estimated gains in the theory rule made inputs exit even at λ = 0. Calibration
  is now shrunk and isotonic, and gains are clipped at 0.
- **Bandit optimism.** The bonus was variance-free and uncapped, so a 5-sample bin blocked every stop. It is now an
  empirical-Bernstein bonus, capped at 1, on the log of the number of probes.
- **FunSearch context.** At Ollama's default 4k context, 82% of diff edits failed. The run now uses a 16k context
  and full rewrites.

## 5. Reproduce

```
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python traces/train_exits.py configs/train_mnli.yaml   # also qqp, cifar100
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python attacks/deepsloth_vit.py --n 500
~/venv/bin/python experiments/run_main.py --task mnli          # all methods x regimes x batch sizes, + attack test
~/venv/bin/python bandit/run_bandit.py --task mnli --regime dec_eager --B 64
~/venv/bin/python evolve/baselines.py --budget 300; ~/venv/bin/openevolve-run evolve/initial_program.py evolve/evaluator.py --config evolve/config.yaml --output evolve/runs/mnli
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python runner/gpu_runner.py --task mnli --calibrate
~/venv/bin/python analysis/make_figures.py --task mnli
~/venv/bin/python -m pytest tests -q
```
Everything after training is chained in `scripts/run_pipeline.sh`.
