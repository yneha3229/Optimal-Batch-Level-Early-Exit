# Reproducing "When Should a Batch Stop?" (IE624 progress report, 13 Nov; Section 5 of 23 Nov)

This file maps every table, figure and numeric claim of `docs/IE624_progress_report_v2.{pdf,html}` (built by
`analysis/make_report.py`) to the command, inputs, result file, log and hardware that produced it. It also records
the final status of the planned items (§5.1–5.8) and of the claims table (§5.10).

**Rule used throughout.** Every number points to a file in this repository. A number that needs a GPU to regenerate
is labelled **GPU required, logs provided**: its result file and raw log are committed, and the CPU checks below
recompute everything derived from it.

Branch: `feature/early-exit-repro`, from `feature/kneser-minrank-theorem` (08b7ad4). That branch did not have some
early-exit evidence; commit 09d7d37 adds it unchanged (see §6). Merged into `main` on 26 Nov, where the report is
regenerated with the calibration-cache fix (§6) and includes §5.8 and Table 12b.

Quick start:

```bash
~/venv/bin/python analysis/check_report_numbers.py   # every report number against results/ (2 min, CPU)
scripts/reproduce_cpu.sh                              # regenerate the CPU results into results_repro/ and diff them (hours, CPU)
scripts/reproduce_gpu.sh                              # print the GPU steps (plan only; RUN=1 to execute one)
```

Check outputs (committed): `results_repro/REPORT_NUMBERS.md` and `results_repro/report_numbers.csv` (one row per
number), `results_repro/DIFF.md` and `results_repro/diff_summary.csv` (one row per regenerated file),
`results_repro/timing.tsv` (runtime of each CPU step), `results_repro/gpu_spot/` (the GPU spot check, §4).

---

## 1. Inventory

Hardware labels:
- **A6000 eager**: RTX A6000 on the server, eager PyTorch.
- **A6000 graphs**: the same GPU under CUDA graphs.
- **GTX 1650**: the laptop GPU, 4 GB, fp32.
- **CPU replay**: recorded traces replayed under a measured cost table, with no GPU.

"Check" counts come from `analysis/check_report_numbers.py`. A MATCH means the number was recomputed from the file
and, rounded to the printed precision, equals the reported value. "Repro" is the result of
`scripts/reproduce_cpu.sh` for files a CPU can regenerate (§3).

| item | claim | command | inputs | output file | log | hardware | GPU required? | runtime | status |
|---|---|---|---|---|---|---|---|---|---|
| Table 1 | Full-model accuracy: MNLI 87.8, QQP 91.1, CIFAR-100 93.3; WikiText 49.9 (Qwen2.5-1.5B), 53.6 (Llama-3.1-8B) | `scripts/train_text.sh`, `scripts/train_vision.sh` (`traces/train_exits.py configs/train_<task>.yaml`); `decoder/train_exit_lora.py --tag qfrozen --rank 0`; `decoder/train_decoder_exits.py --model meta-llama/Llama-3.1-8B-Instruct --tag llama` | HF datasets `nyu-mll/glue` (mnli, qqp), torchvision CIFAR-100, `Salesforce/wikitext` (wikitext-103-raw-v1) | `results/train_{mnli,qqp,cifar100}_exit_accuracy.csv`, `train_wikitext_{qfrozen,llama}_exit_accuracy.csv` | `logs/train_{mnli,qqp,cifar100}.log`, `logs/llm2_train_qfrozen.log`, `logs/final_llama_train.log` | A6000 | GPU required, logs provided | not logged (est. 1–3 h per model) | Check 5/5 MATCH |
| Table 2 | Corrections to the first report (6 rows) | as Figure 1 (rows 1–3); `experiments/robust_bound.py` (row 5); `experiments/transfer_test.py` (row 6) | server traces, `results/runner_cost_*.json` | `results/graph_*_gpu.csv`, `main_*_real_*_gpu*.csv`, `matched_table_*.md`, `robust_cifar100_real_cifar100_B64_lowrho.csv`, `transfer_summary.csv` | `logs/final_*.log`, `logs/step2_graph_*.log`, `logs/robust_lowrho_*.log`, `logs/transfer_test.log` | A6000 eager / graphs; simulator (row 6) | GPU required for rows 1–3, logs provided; rows 5–6 CPU | transfer test 8 min (CPU, 6 workers) | Check 12/15 MATCH. MISMATCH: "15–32%" simulator error (definition not stated; per policy 1–47%); "1.12×" robust rule at low ρ (the committed file is a 22 Nov rerun, 1.24×; the 13 Nov file is not in the repository) |
| Figure 1, §4.2 | Server speedups, eager and CUDA graphs; wait-for-all 0.78–0.97×; per-sample 0.77–0.90× at B ≤ 16; graphs cut the fixed cost about 6×; heads cost 19% at B = 8 | `runner/gpu_runner.py --task <t> --calibrate`, `--regime real_<t>` (`scripts/run_parallel.sh`, `run_gpu_redo.sh`); `runner/graph_runner.py --task <t>`; plot `analysis/report_figures.py` | trained checkpoints (`checkpoints/<task>.pt`), traces | `results/main_{mnli,qqp}_real_*_gpu_B8.csv`, `_B16-128.csv`, `graph_{mnli,qqp}_gpu.csv`, `runner_cost[_graph]_*.json`; `figures/report2_server_speedups.png` | `logs/final_gpu8_*.log`, `logs/final_gpu_*.log`, `logs/step2_graph_*.log` | A6000 eager and graphs | GPU required, logs provided | est. 3–4 h in total | Check 12/13 MATCH. MISMATCH: heads' share under graphs is 19% on MNLI but 12% on QQP |
| Figure 2, Table 3, §4.3 | Per-layer cost on the GTX 1650: a floor, then linear; affine fit off by up to 41% | `experiments/local_hw.py profile --seq-len {16,32,128}` | random-weight RoBERTa-base (timing only) | `results/local_cost_cuda_T{16,32,128}.json`; `figures/report2_cost_gtx1650.png` | `logs/local_profile_cuda_T{16,32,128}.log` | GTX 1650 | Measured on local GTX 1650, raw logs: `logs/local_profile_cuda_T*.log` | est. 5 min per length | Check 22/22 MATCH |
| Figure 3, Tables 4–6, §4.4 | Timed speedup of each method's certified policy; the rule is at most 7.5% behind the fastest, but not at equal accuracy | `scripts/run_local.sh` (`local_hw.py select` on CPU, then `local_hw.py time`) | `local_cost_cuda_T*.json`, `traces/{qqp,mnli}_val.parquet` | `results/local_select_*_cuda_T*.json` (CPU), `results/local_time_{qqp,mnli}_cuda_T{16,32,128}.csv` (GPU); `figures/report2_local_speedups.png` | `logs/local_select_*_T*.log`, `logs/local_time_*_T*.log`, `logs/local_status.log` | GTX 1650 (timing); CPU (selection) | Timing measured on local GTX 1650, raw logs: `logs/local_time_*_T*.log`; selection CPU | timing 63 min; selection ~6 min (CPU) | Check 213/213 MATCH (Tables 4–6 and the §4.4 text); repro: see §3 |
| Figure 4, Tables 7–9, §4.5 | At matched held-out drop ≤ 1 point the rule is 2–17% ahead on QQP; on MNLI within 2% in 4 of 6 settings and 6.5–8.5% behind in 2; the loss lands on hard inputs | `scripts/run_front.sh` (`local_hw.py front`) | `local_cost_cuda_T{16,128}.json`, traces | `results/local_front_{qqp,mnli}_cuda_T{16,128}.csv`; `figures/report2_front.png` | `logs/local_front_*_T*.log` | CPU replay (GTX 1650 cost table) | No | ~5 min | Check 203/203 MATCH; repro: see §3 |
| Table 10, §4.6 | Certified selection keeps the budget on 30 re-splits (cost rule 0–3% over budget against 23–47% for the old protocol; before the 26 Nov cache fix: 0–7% against 13–40%) | `local_hw.py resplit --seq-len 16 --batch-sizes 8 64 --seeds 30` | `local_cost_cuda_T16.json`, traces | `results/local_resplit_{qqp,mnli}_cuda_T16.csv`; `figures/report2_resplit.png` | `logs/local_resplit_*_T16.log` | CPU replay | No | re-split 8 min for both tasks (CPU, 6 workers) | Check 103/103 MATCH; repro: cost-rule rows differed (cache bug, §6), fixed and regenerated on 26 Nov; all other rows MATCH (§3) |
| §4.7 | Bandit, single runs: regret slope 0.49–0.65; 0.2–0.3 points; 1.26–1.77× at B = 64–128 | `bandit/run_bandit.py --task <t> --regime <r> --B 64 --T 3000` (`scripts/run_final.sh`); GPU rows from `runner/gpu_runner.py` | `results/main_<task>_selected.csv`, `runner_cost_*.json` | `results/bandit_{mnli,qqp}_{dec_eager,enc_eager_meas}_B64.csv`, `main_*_real_*_gpu*.csv` (bandit rows) | `logs/final_bandit_*.log` | CPU (regret); A6000 eager (speedups) | Regret: no. Speedups: GPU required, logs provided | ~minutes | Check 4/6 MATCH. MISMATCH: slopes. The committed files hold a later 3-seed run; a refit gives 0.56–0.63; the 6 Nov single-run fit is not in the repository. Superseded by §5.4 |
| §4.8 | Robustness bound held in all 10,500 batches (5 tasks); 19/19 unit tests; consensus loses its speedup to one attacker | `experiments/robust_bound.py --task <t> --regime dec_eager` (`run_final.sh`); `pytest tests simulator/test_simulator.py` | traces, `attacks/deepsloth_vit.py` output (`traces/cifar100_attack.parquet`) | `results/robust_violations_*_B64.csv`, `robust_*_dec_eager_B64.csv` | `logs/final_robust_*.log`, `logs/final_dec_robust.log` | CPU | No (the DeepSloth inputs needed a GPU once; committed) | ~minutes | Check 4/6 MATCH (tests re-run: 19 passed). MISMATCH: consensus collapses at k = 1 only on CIFAR-100; on MNLI/QQP its latency goes 0.70→0.74 and 0.72→0.85 |
| §4.9 | Program search, 300 programs; probe adds ≤ 0.026 AUROC; exits ~5 points short at 75% depth | `evolve/` (OpenEvolve, `logs/evolve_mnli.log`); `analysis/probe_correctness.py`; `decoder/layerskip_exits.py`; `decoder/train_decoder_exits.py` | traces; HF `facebook/layerskip-llama3.2-1B`, `facebook/layerskip-llama3-8B` | `results/probe_*.csv`, `layerskip_ls{1b,8b}_layer_accuracy.csv`, `train_wikitext_llama_exit_accuracy.csv` | `logs/evolve_mnli.log`, `logs/step1_probe_*.log`, `logs/step3_ls*_exits.log` | local Ollama (evolve), CPU (probe), A6000 (LayerSkip) | LayerSkip and decoder exits: GPU required, logs provided | — | Check 4/5 MATCH. MISMATCH: frozen Llama heads are 8.2 points short at 75% depth (layer 24 of 32); LayerSkip 4.9/5.2 MATCH |
| Tables 11–12, §5.1 | A6000, RoBERTa, eager, certified: QQP T128 rule + plan fastest within budget at every B (1.95–2.46×, 8–14% ahead); MNLI over budget | `scripts/run_server_a6000.sh` (`local_hw.py profile`, `select` and `front` on CPU, `time` on GPU) | `traces/{qqp,mnli}_val.parquet` | `results/local_cost_a6000_T{128,32}.json`, `local_select_*_a6000_T*.json` (CPU), `local_time_{qqp,mnli}_a6000_T{128,32}.csv` (GPU) | `logs/server_{profile,select,front,time}_*_a6000_T*.log`, `logs/server_a6000_status.log` | A6000 eager; CPU (selection) | Timing: GPU required, logs provided | 28 min (22 Nov) | Check 112/112 MATCH; repro (selection): see §3. **Labelling:** see §6 |
| Table 13, §5.2 | t-test within δ on re-splits (1.5/5.2/8.5%; before the cache fix 1.3/5.3/8.5%); HB never certifies; WSR only on QQP (49%); label-free costs speed; 10% data costs the rule more than random search (−28% against −20% on QQP) | `scripts/run_cert_check.sh`; `scripts/run_mnli_delta.sh` (δ = 5%, `experiments/delta_check.py`) | `local_cost_a6000_T128.json`, `runner_cost_cifar100.json`, traces | `results/cert_check_{qqp,mnli,cifar100,mnli_d05}[_summary].csv`, `delta_check_mnli.csv` | `logs/cert_check_*.log`, `logs/delta_check_mnli.log` | CPU replay | No | 1 h 47 min with 14 workers (22 Nov) | Check 47/47 MATCH (+ §5.2 text in Part C); repro (3 h 5 min, 6 workers): cost-rule rows do not reproduce (cache bug, §6); all other rows MATCH (§3) |
| Table 14, §5.3 | Fluctuating load: passes on QQP (2.32× vs 2.11×), CIFAR-100 only with the plan, fails on MNLI (budget) | `scripts/run_load_trace.sh`, `run_mnli_delta.sh` | as Table 13 | `results/load_trace_{qqp,mnli,mnli_d05,cifar100}{,_levels,_selected,_batches}.csv` | `logs/load_trace_*.log` | CPU replay | No | 7 min per task | Check 32/32 MATCH; repro: see §3 |
| §5.4 | Bandit, 10 seeds: slopes 0.55–0.77 in 5 settings, 4 in 0.5–0.7; t^−1/3 best in 3 of 5; warm start −2 to −12% | `scripts/run_bandit_seeds.sh` | `results/main_<task>_selected.csv`, `runner_cost_*.json` | `results/bandit_slopes_*_B64_10seeds.csv` (raw 25 MB traces not committed) | `logs/bandit_10seeds_*.log` | CPU | No | 22 min (10 seeds, 6 settings) (CPU) | Part C MATCH; repro: see §3 |
| §5.5 | Bound held in 25,200 batches; majority and consensus collapse; threshold also robust; trimming buys little | `scripts/run_robust_cert.sh` (and `run_robust_lowrho.sh`) | as §5.4 | `results/robust_*_B64_cert.csv`, `robust_violations_*_B64_cert.csv` | `logs/robust_cert_*.log` | CPU | No | 7 min (12 settings) (CPU) | Part C MATCH; repro: see §3 |
| §5.6 | Trained checkpoints on the local GPU | — | — | none | none | GTX 1650 | yes | — | NOT RUN (no file) |
| Tables 15–16, §5.7 | LLM classifiers: rule fastest within budget on Llama (QQP 5–13%, MNLI 14–18% ahead) and Qwen QQP; Qwen MNLI mixed. Decoding: adaptive gain falls from 12% (B = 1) to 0.2% (B = 64) | `decoder/train_cls_exit_lora.py`; `scripts/run_llm_cls_hw.sh`, `run_llama_cls_queue.sh`; `analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext wikitext_llama` | `Qwen/Qwen2.5-1.5B`, `meta-llama/Llama-3.1-8B-Instruct`, `nyu-mll/glue`, `Salesforce/wikitext` | `results/local_time_{qqp,mnli}_{llama,qwen}_a6000_T128.csv`, `local_check_*_a6000_T128.json`, `local_cost_a6000_T128_{llama,qwen}.json`, `local_select/front_*_{llama,qwen}_a6000_T128.*` (CPU), `adaptivity_vs_batch.csv` (CPU), `train_wikitext_{qlora,qfrozen}_exit_accuracy.csv` | `logs/cls_*`, `logs/llm_cls_status.log`, `logs/llm2_*.log`, `logs/adaptivity_vs_batch.log` | A6000 (bf16, trained weights, real prompts); CPU (selection, adaptivity) | Timing and training: GPU required, logs provided | Llama 60–63 min per task, Qwen 17–21 min per task (timing) | Check 112/112 MATCH (+ Part C); §5.7 plan text 1/2 (the "5.6" of the Expected paragraph does not match; the result paragraph prints 4.0); repro (selection, adaptivity): see §3 |
| §5.8 (added in 09d7d37, not in this report version) | Optimality of the lookahead rule: proved in the monotone case for the collective stop; rule (2) never late, loss bounded; corrected rule (2) gives no gain at matched accuracy | `theory/verify_58.py`; `experiments/theory58_front.py --task <t> --name {cuda,a6000} --seq-len T --plans` | cost tables, traces | `results/theory58_checks.json`, `results/theory58_front_*.csv`; `theory/THEORY_58.md` | `logs/theory58_*.log` | CPU | No | checks 10 s; real-trace fronts 20 min (CPU) | repro: see §3 |
| Abstract, §4.3 numbers, §5 result paragraphs | Generated by `make_report.py` from the files above | `analysis/make_report.py` | `results/*` | `docs/IE624_progress_report_v2.{html,pdf}` | — | CPU | No | 30 s | Part C: report re-rendered from the committed results, all **1,904** numbers in its text identical |

## 2. Planned work (§5.1–5.8) and claims (§5.10): final status

### 2.1 Planned items

| item | status | headline number | file |
|---|---|---|---|
| 5.1 Accuracy-matched server tables (24 settings: 3 tasks × 4 B × eager/graphs) | **PARTIAL**: 9 of 24 settings. Text eager: 8/8, plus 8 extra at length 32. ViT eager: 1/4 (B = 8; GPU 5 failed during B = 16 on 25 Nov). CUDA graphs: 0/12 | QQP, length 128: rule + plan fastest within budget at every B, 1.95–2.46× at 0.60–0.66 points, 8–14% ahead. MNLI: rule over budget (1.23–1.28 points). ViT, B = 8: rule + plan 1.44× vs threshold + plan 1.48× (both within budget), 2.8% behind | `results/local_time_{qqp,mnli}_a6000_T{128,32}.csv`, `results/local_time_cifar100_a6000_T197.csv`, `logs/server_time_cifar100_a6000_T197.log`, `logs/chain51_status.log` |
| 5.2 Tighten the certificate | **DONE**; outcome = fallback (asymptotic t-test) | t-test mean violation 1.3% QQP, 5.3% MNLI, 8.5% CIFAR-100; HB certifies nothing; WSR 49% of QQP cases (1.50× vs 2.31×); 10% data: rule 2.30→1.83×, random search 2.13→1.71× | `results/cert_check_*_summary.csv`, `delta_check_mnli.csv` |
| 5.3 Fluctuating load | **DONE** | QQP pass (2.32× vs 2.11×); CIFAR-100 pass with the plan only (1.39× vs 1.38×); MNLI fail (drop 1.48 / 1.23 points) | `results/load_trace_*_levels.csv` |
| 5.4 Bandit with error bars | **DONE** | slopes 0.55–0.77 in 5 settings (±0.01–0.03), 4 of 5 in 0.5–0.7; warm start −2 to −12% regret | `results/bandit_slopes_*_B64_10seeds.csv` (committed in 09d7d37) |
| 5.5 Robustness at low ρ, k > m | **DONE** | 0 violations in 25,200 batches; majority 1.01–1.11× and consensus 0.97–1.10× full-model latency under attack; trimming 8 on CIFAR-100: 0.91–1.02× vs 1.15–1.57× clean | `results/robust_*_B64_cert.csv`, `robust_violations_*_B64_cert.csv` |
| 5.6 Trained checkpoints on the local GPU | **NOT RUN** | — | no file |
| 5.7 LLMs | **DONE** (decoding negative; classification positive except Qwen MNLI) | Llama: QQP 3.48–3.66× vs 3.11–3.50×; MNLI 2.97–3.14× vs 2.56–2.66×. Qwen MNLI with the plan: +1%/+5% at B = 16/128, −7%/−6% at B = 8/64. Decoding: adaptive gain 12% at B = 1 → 0.2% at B = 64 | `results/local_time_*_{llama,qwen}_a6000_T128.csv`, `adaptivity_vs_batch.csv` |
| 5.8 Theory | **PARTIAL** (proofs for the monotone case and the two affine limits; open for 0 < ρ < ∞ and above the knee) | rule (3) optimal in 210/210 monotone instances, suboptimal in 17/190 non-monotone ones; rule (2) suboptimal in 165/301 monotone single-input instances, never late, bound holds 31/31; real traces, 40 settings: corrected rule −1.4% at matched drop (13 wins, 21 losses) | `theory/THEORY_58.md`, `results/theory58_checks.json`, `results/theory58_front_*.csv` |

### 2.2 Claims table (§5.10 of the report, 23 Nov) with final status

No status tag changes. One claim is added, two are narrowed by files, and one label is corrected (§6).

| claim | status (report, 23 Nov) | final status | change | evidence |
|---|---|---|---|---|
| Which rule is fastest flips with batch size, execution mode and sequence length | HOLDS on two GPUs | HOLDS on two GPUs | none. The certified CUDA-graph rerun is still NOT RUN | Figure 1 (old protocol, both modes), Tables 4, 11–12 |
| The per-layer cost is a floor plus a linear part | HOLDS (GTX 1650; Qwen on the A6000 up to B = 4) | HOLDS | none | `results/local_cost_cuda_T*.json`, `local_cost_a6000_T128_qwen.json` |
| Re-certified per setting, the rule is near the best certified method everywhere | HOLDS on QQP and Llama MNLI; exceptions RoBERTa MNLI, Qwen MNLI | HOLDS with the same exceptions | evidence added: ViT, A6000, B = 8: rule + plan 2.8% behind threshold + plan (the only ViT batch size timed) | `results/local_time_cifar100_a6000_T197.csv` |
| Stopping a batch spreads the accuracy loss evenly | FALSE | FALSE | none | Table 9 |
| The exit plan fixes the small-batch weakness | PARTLY | PARTLY | none. Under CUDA graphs still not measured; ViT B = 8: plan 1.44× vs 1.40× without | Tables 11, 14, 16; `local_time_cifar100_a6000_T197.csv` |
| Certified selection keeps the accuracy budget | WEAKENED (asymptotic) | WEAKENED | none | Table 13 |
| The MNLI held-out overshoot is an unlucky split | PARTLY consistent | PARTLY consistent | none | `delta_check_mnli.csv`, `cert_check_mnli*_summary.csv` |
| Under fluctuating load, one tuning is enough | HOLDS QQP; CIFAR-100 with the plan; FAILS MNLI | same | none | Table 14 |
| The label-free bandit learns the rule | HOLDS (ten seeds) | HOLDS | none. Its summary files were not committed; now they are (09d7d37) | `results/bandit_slopes_*_10seeds.csv` |
| The rule is optimal | (row absent in this report version) | **PARTLY** | added from §5.8 | `theory/THEORY_58.md`, `results/theory58_checks.json` (reproduced, §3) |
| Bounded damage from slowdown inputs | HOLDS: 0 violations; majority and consensus collapse | HOLDS | narrowed: consensus loses its speedup at k = 1 only on CIFAR-100 (DeepSloth); on MNLI/QQP its latency goes 0.70→0.74 and 0.72→0.85 at k = 1, and it collapses at k = 8 (§5.5). Table 2's low-ρ cost of robustness is 1.24× vs 1.46× in the committed file (report: 1.12×) | `results/robust_{mnli,qqp,cifar100}_dec_eager_B64.csv`, `robust_cifar100_real_cifar100_B64_lowrho.csv` |
| Benefit on LLMs | FAILED decoding; HOLDS classification (Llama, Qwen QQP); mixed Qwen MNLI | same | none | Tables 15–16 |

## 3. CPU reproduction (`scripts/reproduce_cpu.sh`)

This run (25 Nov) compares against the files committed before the cache fix. After the fix (26 Nov) the cost-rule
rows of Tables 10 and 13 were regenerated (`scripts/rerun_cache_fix.sh`) and are deterministic; the other 92 files
are unchanged by the fix.

Run on 25 Nov 2026 on the A6000 server, CPU only (`CUDA_VISIBLE_DEVICES=""`), 6 worker processes, while other jobs
kept the load average at 40–70. All **72 steps** finished OK (`results_repro/timing.tsv`). The steps add up to 5 h 16 min
(the run was resumed once, so wall-clock time was longer). The three largest groups were the certificate checks
(3 h 5 min), the bandit (22 min) and theory58 (20 min). The per-file diff is `results_repro/DIFF.md`
(`results_repro/diff_summary.csv`), using the tolerances in the script header: MATCH = every number within
atol 1e-9 + rtol 1e-6.

| group | files | MATCH | MISMATCH | inputs (GPU-measured, committed) |
|---|---|---|---|---|
| theory (§5.8 checks) | 1 | 1 | 0 | none |
| gtx (Tables 7–10, Figure 4) | 12 | 10 | 2 (re-splits) | `local_cost_cuda_T*.json` |
| a6000 (Tables 11–12, 15–16 selection, ViT) | 18 | 18 | 0 | `local_cost_a6000_*.json` |
| delta (§5.2 δ text) | 1 | 1 | 0 | `local_cost_a6000_T128.json` |
| load (Table 14) | 16 | 16 | 0 | as above, `runner_cost_cifar100.json` |
| robust (§5.5) | 24 | 24 | 0 | `runner_cost_*.json`, server traces |
| bandit (§5.4, 10 seeds) | 6 | 6 | 0 | as robust |
| theory58 (§5.8 real traces) | 10 | 10 | 0 | GTX 1650 and A6000 cost tables |
| transfer (Table 2 row 6, §5.7) | 6 | 6 | 0 | traces |
| cert (Table 13) | 8 | 0 | 8 | `local_cost_a6000_T128.json`, `runner_cost_cifar100.json` |
| **total** | **102** | **92** | **10** | |

No file is CLOSE or MISSING. All 92 MATCH files agree to the last digit printed.

- **The 10 MISMATCH files differ only in the cost-rule rows** (`theory`, `theory+plan`). In the four certificate files
  that is 558, 337, 460 and 268 `theory` rows plus 223, 162, 153 and 110 `theory+plan` rows of 5,760 (1,920 for δ = 5%).
  In the re-splits it is 54 and 50 `theory` rows. Every row of every other family (threshold, majority, unified,
  random search) matches exactly. The cause is the calibration-cache bug in §6: the cost rule's result depends on
  how jobs are scheduled on the workers. Neither the committed run (12–14 workers) nor this one (6 workers) is
  the correct value. A run with the fix gives the same result with 3 and 6 workers (`results_repro/cache_fix/`).
- **Report.** `make_report.py` run on the regenerated files (`results_repro/docs/IE624_progress_report_v2.html`)
  prints the same 1,904 numbers in the same places. They differ in 32 table cells, all in cost-rule rows: 20 in
  Table 10 and 12 in Table 13. The §4.6 and §5.2 text that quotes those cells changes with them. Details are in §6.
- **Figures 1–4.** The PNG bytes differ from the committed ones because the committed PNGs were rendered elsewhere
  (sizes differ by 2–6 pixels). `report_figures.py` run on the committed results on this machine gives
  pixel-identical Figures 1–4 to the reproduction. `report2_resplit.png` differs because of Table 10, but the
  report does not use it.

## 4. GPU steps (`scripts/reproduce_gpu.sh`) and what cannot be reproduced without a GPU

`scripts/reproduce_gpu.sh` prints the GPU steps in order, each with its GPU, time and expected output (g1–g10). With
`RUN=1 GPU_ID=<id> STEPS=g5` it runs a scripted step in a shadow copy (`results_repro/gpu/`). It was **not** run in
full.

These cannot be reproduced without a GPU. All are labelled **GPU required, logs provided**:
- **Checkpoints and traces** (g1, g2, g8–g10): RoBERTa (MNLI, QQP), ViT-B/16, the LoRA classifiers and the decoder
  exits. The traces (`traces/*.parquet`) are committed, so every CPU step runs without the checkpoints.
- **All wall-clock speedups**: Figure 1 and §4.2 (A6000, both modes); Tables 4–6 and Figure 3 (GTX 1650, laptop);
  Tables 11–12, 15–16 and the ViT row (A6000 eager); the cost tables `local_cost_*.json` and `runner_cost_*.json`
  that the CPU replays take as input.
- **Never run** (no file exists): the certified CUDA-graph tables (5.1, g6) and ViT at B = 16, 64, 128 (g7).
  `experiments/check_graph_device.py` passes on GPU 4 after the aliasing fix (`logs/check_graph_device_gpu4_fix.log`).
  It failed on GPU 5 before the fix (`logs/check_graph_device_gpu5.log`). Item 5.6 (trained weights on the GTX 1650)
  also has no file.

**GPU spot check (step 6, 25 Nov 08:56 UTC).** One Table 11 cell (QQP, length 128, B = 8, 2,048 inputs, 5 rounds)
on the idle A6000 at PCI 01:00.0, written to `results_repro/gpu_spot/`.
- Held-out drops: identical to the committed file for all 9 policies.
- Speedups: lower, and the rounds are very spread (e.g. cost rule + plan 1.62×, rounds 1.44–4.15×, against 1.96×,
  rounds 1.94–1.97×, on 22 Nov).
- Cause: the CPU was oversubscribed (load average 54–71 on 64 cores, from the minrank SAT sweeps and
  `reproduce_cpu.sh`). This timing path synchronises with the host after every exit.
- So this spot check is **not a valid replication**. A timing check needs a quiet CPU.

## 5. Environment

- **Server**: Linux 6.8.0-151-generic; 2× Intel Xeon Silver 4216 (64 threads); 4× RTX A6000 (48 GB) and
  2× RTX 3080 Ti; NVIDIA driver 580.159.04, CUDA 13.0 (driver).
- **Python**: 3.12.3, venv `~/venv`. Key packages: torch 2.14.0+cu130, transformers, datasets, pandas 3.0.6,
  numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, weasyprint 70.0. Exact versions: `requirements-lock.txt`
  (`pip freeze` of the venv used for every run in this file).
- **Laptop** (Tables 3–5): GTX 1650 4 GB, fp32, 4 CPU threads (`logs/local_profile_cuda_T*.log`). Its package
  versions were not recorded.
- **Datasets** (Hugging Face): `nyu-mll/glue` (mnli: validation_matched; qqp: validation); `Salesforce/wikitext`
  (wikitext-103-raw-v1); CIFAR-100 from `torchvision.datasets.CIFAR100`. DeepSloth inputs:
  `attacks/deepsloth_vit.py --n 500` → `traces/cifar100_attack.parquet`.
- **Checkpoints** (Hugging Face IDs):
  - RoBERTa: `textattack/roberta-base-MNLI` (MNLI), `roberta-base` (QQP).
  - ViT: `google/vit-base-patch16-224-in21k`.
  - LLMs: `Qwen/Qwen2.5-1.5B`, `meta-llama/Llama-3.1-8B-Instruct`.
  - LayerSkip: `facebook/layerskip-llama3.2-1B`, `facebook/layerskip-llama3-8B`.
- **Fine-tuned checkpoints**: not in the repository (`checkpoints/` is ignored). On the server they are in
  `~/ie624/checkpoints/`. To retrain them:

  | checkpoint | command | expected full-model accuracy (Table 1) |
  |---|---|---|
  | `mnli.pt` | `scripts/train_text.sh` (`traces/train_exits.py configs/train_mnli.yaml`) | 87.8% (`logs/train_mnli.log`: 0.8777) |
  | `qqp.pt` | same, `configs/train_qqp.yaml` | 91.1% (0.9106) |
  | `cifar100.pt` | `scripts/train_vision.sh` (`configs/train_cifar100.yaml`) | 93.3% (0.9331) |
  | `{mnli,qqp}_qwen_cls.pt` | `decoder/train_cls_exit_lora.py --task <t> --model Qwen/Qwen2.5-1.5B --tag qwen` | see `results/train_<t>_qwen_exit_accuracy.csv` |
  | `{mnli,qqp}_llama_cls.pt` | `decoder/train_cls_exit_lora.py --task <t> --model meta-llama/Llama-3.1-8B-Instruct --tag llama --exits 8 12 16 20 24 28 32 --steps 8000 --bs 16` | see `results/train_<t>_llama_exit_accuracy.csv` |
  | `qfrozen_exits.pt`, `qlora_exits.pt` | `decoder/train_exit_lora.py --tag qfrozen --rank 0 --steps 6000` / `--tag qlora --rank 16 --steps 6000` | WikiText 49.9% full model (Qwen) |

- **Seeds**:
  - Training: `seed: 0` in every `configs/train_*.yaml`. The evolve/held-out split of each trace is
    `default_rng(seed)` in `traces/train_exits.py`.
  - Selection: disjoint batches from `default_rng(seed)`, `seed + 1`, `seed + 2` (`experiments/local_hw.py`).
    Random-weight timing network: `torch.manual_seed(0)`.
  - Re-splits: seeds 0–29. Bandit: seeds 0–9 (`--seeds 0 … 9`). Load traces: 3 arrival seeds.
  - `reproduce_cpu.sh` sets `PYTHONHASHSEED=0` and `CUDA_VISIBLE_DEVICES=""`.

## 6. Labelling issues found, and what this branch adds

- **Tables 11–12 are not "trained checkpoint, real inputs".** The captions say "trained RoBERTa-base checkpoint,
  real inputs", and Section 3 says "server, fp16". But `experiments/local_hw.py` `Device` (13 Nov and 22 Nov
  versions; `git show 8ef0592:experiments/local_hw.py`, lines 313–316) times a randomly initialised fp32
  RoBERTa-base on random token ids. It then replays the trained model's recorded decisions. This is the same method
  as the GTX 1650 tables. Accuracy is exact and timing does not depend on weight values, but the captions should
  say "replayed decisions, random-weight network, fp32". The ViT row uses the same path. Tables 15–16 do use
  trained weights and real prompts (`LLMDevice`, bf16).
- **Missing evidence.** `feature/kneser-minrank-theorem` lacked the 5.8 theory, the ViT timing path, the
  GraphDevice fix and the 25 Nov ViT outputs. These are on `claude/keen-dijkstra-fpe319` or were uncommitted.
  Commit 09d7d37 brings them in unchanged. On `main` (26 Nov) the report is built with the 24 Nov
  `analysis/make_report.py`, which adds the 5.8 paragraph and the conditional Table 12b (ViT, B = 8 only; its caption
  and the 5.1 text now say the run stopped during B = 16 and that the CUDA-graph run was not done).
- **Ignored summary files.** `.gitignore` excluded `results/bandit_slopes_*_10seeds.csv`, the files behind §5.4.
  Its pattern for the 25 MB raw traces matched them too. Fixed in 09d7d37.
- **A cache bug makes the cost rule's re-split results depend on scheduling.** `experiments/local_hw.py:210`
  caches the calibration table of the cost rule under `("cal", id(cal_tr), plan)`. `resplit`
  (`_resplit_job`) and `experiments/cert_check.py` (line 156) build new `fit`/`cert` traces in every job. When a
  job's traces are garbage-collected, a later job in the same worker can get the same `id()` and reuse a
  calibration table fitted on another re-split. Only the families `theory` and `theory+plan` use the cache, so
  only their rows can change, and the result depends on the worker count and job order. `select`, `front`,
  `time`, `load_trace`, `delta_check` and `theory58_front` keep one `fit` object alive for the whole run, so they
  are not affected (they reproduce exactly, §3). **Fixed on 26 Nov** (`experiments/local_hw.py` keeps a weak reference to the traces and refits when the object
  differs; 3- and 6-worker runs now give identical files). Rerun with `scripts/rerun_cache_fix.sh` (16 workers; logs
  `logs/cache_fix_*.log`), then the report was regenerated. Before the fix, this reproduction (6 workers) and the
  committed files (12–14 workers) disagreed in 32 report cells, all in cost-rule rows (§3). Changes in the report,
  committed values → fixed values. **No claim changes status.**
  - **Table 10 (§4.6), cost rule, certified share over budget (B = 8 / 64):** QQP 0% / 3% → 0% / 0%; MNLI 7% / 0% →
    3% / 3%. Old protocol: QQP 13% / 13% → 33% / 23%; MNLI 40% / 37% → 40% / 47%. Certified speedup: QQP 1.80× /
    2.54× → 1.79× / 2.52×; MNLI 1.33× / 1.59× → 1.33× / 1.55×. The §4.6 text now reads "0–3% against 23–47%", mean
    drop 0.34–0.64 points, speed cost about 6% on QQP and 9–14% on MNLI (was 2–4% and 8–12%), old protocol 33–47% of
    MNLI re-splits (was 33–43%).
  - **Table 13 (§5.2):** mean share over budget QQP 1.3% → 1.5%, MNLI 5.3% → 5.2%, CIFAR-100 8.5% (unchanged). Cost
    rule + plan: QQP 2.31× (1.7%) unchanged; MNLI 1.52× (6.7%) → 1.50× (7.5%); CIFAR-100 1.26× (11.7%) → 1.27× (12.5%),
    still slightly above δ. WSR 1.50× → 1.51×. 10% calibration data, QQP: 2.30× → 1.83× becomes 2.32× → 1.66×; the
    report now says random search loses less (−20% against −28%) instead of "about as much"; the conclusion (the
    data-efficiency hypothesis is not supported) is unchanged.
  - **§5.2 MNLI δ text:** violation rate at δ = 10% 7.5% → 5.8% (with the plan 6.7% → 7.5%); lowering δ to 5% costs
    about 2% speed (1.50× → 1.47×) instead of 3% (1.51× → 1.46×).
  - `results_repro/cache_fix/` keeps the scratch runs that showed the effect before the fix.
- **Ten numbers do not match their files** (`results_repro/REPORT_NUMBERS.md`). None of them is in a table, and
  none changes a claim's status:
  - Table 2: the "15–32%" simulator error (its definition is not stated) and the "1.12×" (the file now says 1.24×).
  - §4.2: heads' share under graphs is 19% on MNLI only (QQP 12%).
  - §4.7: the single-run slopes 0.49–0.65, superseded by §5.4.
  - §4.8: consensus collapses at k = 1 only on CIFAR-100.
  - §4.9: frozen Llama heads are 8.2 points short at 75% depth, not about 5.
  - §5.7 "Expected": 5.6 where the result paragraph of the same report prints 4.0.
