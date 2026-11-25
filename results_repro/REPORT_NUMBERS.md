# Report numbers recomputed from the result files

`analysis/check_report_numbers.py` on `docs/IE624_progress_report_v2.html`. Totals: **887 MATCH, 10 MISMATCH, 0 NO FILE** (897 numbers).

| item | MATCH | MISMATCH | NO FILE |
|---|---|---|---|
| Part C | 1 | 0 | 0 |
| Table 1 | 5 | 0 | 0 |
| Table 10 | 84 | 0 | 0 |
| Table 11 | 56 | 0 | 0 |
| Table 12 | 56 | 0 | 0 |
| Table 13 | 47 | 0 | 0 |
| Table 14 | 32 | 0 | 0 |
| Table 15 | 56 | 0 | 0 |
| Table 16 | 56 | 0 | 0 |
| Table 2 | 12 | 3 | 0 |
| Table 3 | 21 | 0 | 0 |
| Table 4 | 72 | 0 | 0 |
| Table 5 | 72 | 0 | 0 |
| Table 6 | 35 | 0 | 0 |
| Table 7 | 42 | 0 | 0 |
| Table 8 | 42 | 0 | 0 |
| Table 9 | 84 | 0 | 0 |
| §4.2 / Figure 1 | 12 | 1 | 0 |
| §4.3 text | 1 | 0 | 0 |
| §4.4 text | 34 | 0 | 0 |
| §4.4-4.6 text | 3 | 0 | 0 |
| §4.5 text | 35 | 0 | 0 |
| §4.6 text | 16 | 0 | 0 |
| §4.7 bandit (single runs) | 4 | 2 | 0 |
| §4.8 robustness | 4 | 2 | 0 |
| §4.9 what did not work | 4 | 1 | 0 |
| §5.7 plan text | 1 | 1 | 0 |

## Every number that is not MATCH

| item | where | reported | recomputed | status | file | note |
|---|---|---|---|---|---|---|
| §4.2 / Figure 1 | exit heads' share of runtime at B = 8 under CUDA graphs, QQP (%) | 19 | 12.0044 | MISMATCH | `results/main_{mnli,qqp}_real_*_gpu_B8.csv, _B16-128.csv (eager), results/graph_{mnli,qqp}_gpu.csv (CUDA graphs); A6000` | eager B = 8: 18.8% on both tasks |
| Table 2 | simulator error at B <= 16, eager (%): rules of Figure 1 (low end) | 15 | 1.0000 | MISMATCH | `results/main_{mnli,qqp}_real_*_gpu_B8.csv, _B16-128.csv (eager), results/graph_{mnli,qqp}_gpu.csv (CUDA graphs); A6000` | definition not stated in the report; per policy and task the error is 1.5–47% (cost rule 23–47%) |
| Table 2 | simulator error at B <= 16, eager (%): rules of Figure 1 (high end) | 32 | 47.0000 | MISMATCH | `results/main_{mnli,qqp}_real_*_gpu_B8.csv, _B16-128.csv (eager), results/graph_{mnli,qqp}_gpu.csv (CUDA graphs); A6000` | definition not stated in the report; per policy and task the error is 1.5–47% (cost rule 23–47%) |
| Table 2 | low rho, CIFAR-100: robust rule (m = 8) | 1.12 | 1.2393 | MISMATCH | `results/robust_cifar100_real_cifar100_B64_lowrho.csv` | this file is the 22 Nov rerun (--tag _lowrho); the 13 Nov run that printed 1.12x is not in results/ |
| §4.7 bandit (single runs) | log-log regret slope, MNLI and QQP (seed 0, current fit) (low end) | 0.49 | 0.5644 | MISMATCH | `results/bandit_{mnli,qqp}_{dec_eager,enc_eager_meas}_B64.csv` | refit with run_bandit.py's method (seed 0, t >= 100; t >= 1 gives 0.62-0.69); the committed files hold a later 3-seed run with ablation variants, and the 6 Nov single-run fit that printed 0.49/0.52/0.65 is not in results/ or logs/ |
| §4.7 bandit (single runs) | log-log regret slope, MNLI and QQP (seed 0, current fit) (high end) | 0.65 | 0.6318 | MISMATCH | `results/bandit_{mnli,qqp}_{dec_eager,enc_eager_meas}_B64.csv` | refit with run_bandit.py's method (seed 0, t >= 100; t >= 1 gives 0.62-0.69); the committed files hold a later 3-seed run with ablation variants, and the 6 Nov single-run fit that printed 0.49/0.52/0.65 is not in results/ or logs/ |
| §4.8 robustness | consensus loses its whole speedup to one attacker, mnli (victims' latency >= full model at k = 1) | k=1 | False | MISMATCH | `results/robust_mnli_dec_eager_B64.csv` | consensus latency k=0 0.70, k=1 0.74 |
| §4.8 robustness | consensus loses its whole speedup to one attacker, qqp (victims' latency >= full model at k = 1) | k=1 | False | MISMATCH | `results/robust_qqp_dec_eager_B64.csv` | consensus latency k=0 0.72, k=1 0.85 |
| §5.7 plan text | frozen Qwen2.5-1.5B, gap of the layer-24 exit (pts), 'Expected' paragraph | 5.6 | 4.0146 | MISMATCH | `results/train_wikitext_qfrozen_exit_accuracy.csv` | the 5.7 result paragraph of the same report prints 4.0 from this file; 5.6 was written on 13 Nov |
| §4.9 what did not work | frozen Llama-3.1-8B heads, gap at 75% depth (layer 24 of 32, pts) | 5 | 8.1767 | MISMATCH | `results/train_wikitext_llama_exit_accuracy.csv` | layer 28 (88%): 3.7; frozen Qwen layer 20 (71%): 9.5, layer 24 (86%): 4.0 |
