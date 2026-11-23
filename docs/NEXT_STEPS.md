# Next steps: instructions for the `next-steps` agent (written 23 Nov 2026, 04:15 UTC)

You are picking up work on the branch `claude/keen-dijkstra-fpe319` of this repository, on the shared GPU server.
Read this whole file first, then work through the tasks in order. Report back to the user with short status
messages. The user is coding the CUDA-graph device in parallel on another branch (spec:
`docs/PROMPT_CUDA_GRAPH_DEVICE.md`), so **do not edit the `Device` / `LLMDevice` classes, `make_device`, `tag` or
the CLI in `experiments/local_hw.py`**.

## 1. The two projects

1. **Early exit (due 26 Nov).** Batch-level early exit with a cost-aware stopping rule derived from theory. The
   report is `analysis/make_report.py`; its section 5 lists the planned items 5.1–5.8, and 5.10 is the claims
   table. The direction and pass/kill criteria are in `docs/DIRECTION.md`; the LLM-classifier plan is
   `docs/PLAN_LLM.md`.
2. **Minrank over F_2 (new, from a teammate).** The plan is `docs/MINRANK_PLAN.md`, the code `minrank/` (see
   `minrank/README.md`), and the scripts `scripts/minrank/00..09`. Option A is OpenEvolve with a local LLM; Option B
   is exact values and separations; Option D is an early-exit cascade for candidate evaluation.

## 2. Machine rules (important)

- **GPUs (`nvidia-smi`):**
  - GPUs 2 and 3 are used by another user (prakhar: vLLM servers). **Never touch their processes.**
  - GPU 4 runs our minrank vLLM (`~/venv-vllm`, port 8000, Qwen2.5-Coder-32B-Instruct-AWQ, max_model_len 32768)
    until the evolve run finishes.
  - GPUs 0 and 1 hold small idle allocations by someone else; ask the user before using them.
  - Use `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<id>`.
- **CPU:** pin everything with `taskset`.

  | Cores | Use |
  |---|---|
  | 0-3,32-35 | GPU timing only; keep them quiet |
  | 4-11,36-43 | early-exit CPU jobs |
  | 12-15,44-47 | the evolve run |
  | 16-31,48-63 | free for CPU jobs (the minrank CPU chain is done) |
- **Killing processes:** kill **by PID only**, never with `pkill -f` / `pgrep -f` patterns. They match your own
  shell's command line and have killed it twice. Check a process's parent and age before killing it.
- **Disk:** `/home` has ~11 GB free. Never write new checkpoints, and don't keep large intermediate files. Ask the
  user before deleting anything in `checkpoints/`.
- **Environments:**
  - `~/venv/bin/python` for everything, except vLLM, which uses `~/venv-vllm`.
  - Run with `HF_HUB_OFFLINE=1`; the datasets are cached.
- **Waiting on jobs:** use background commands, and don't chain sleeps.

## 3. Where things stand (all numbers on held-out data, budget = 1 point accuracy drop)

**Done:**

| Item | Result | Files |
|---|---|---|
| 5.1 eager, RoBERTa | QQP: the rule with plan is fastest at every B (1.96–2.46×, 0.6 pt). MNLI: random search is faster at B ≥ 16, and the rule is over budget (1.25 pt). T=32, B ≤ 16: the plan is 3–6% behind unified exit | `logs/server_{front,time}_*_a6000_T{128,32}.log` |
| LLM classifiers (timed) | Llama-3.1-8B QQP: rule 3.48/3.52/3.66× at B=8/16/64 vs best baseline 3.13/3.11/3.50× (0.42 pt). Llama MNLI: 2.97–3.14× vs 2.66× (0.72 pt). Qwen2.5-1.5B QQP: 2.15/2.52/2.96× vs 2.00/2.41/2.56×. Qwen2.5-1.5B MNLI: the rule loses (1.10–1.18×; with plan 1.38–1.44×) vs 1.40–1.54× for the best in-budget baselines | `logs/cls_time_{qqp,mnli}_{llama,qwen}_a6000.log` |
| 5.2 certificates | t-test violation rate ≤ δ=10% on all tasks (1.3% QQP, 5.3% MNLI, 8.5% CIFAR-100). Hoeffding–Bentkus certifies nothing. WSR certifies only on QQP (1.46× vs 2.30×). Label-free: 0 violations, slower. 10% of the calibration data: QQP 2.30→1.83× | `results/cert_check_*_summary.csv`, `logs/cert_check_*.log` |
| 5.3 fluctuating load | QQP passes (2.32× vs 2.11×, ratio 1.10). CIFAR-100 passes only with the plan (1.39 vs 1.38×). MNLI fails on the budget (1.48 pt) | `results/load_trace_*_levels.csv` |
| 5.4 bandit | regret slopes 0.55–0.78; the t^-1/3 probe rate is best in 4 of 5 settings; the warm start helps a little | `results/bandit_slopes_*.csv`, `logs/bandit_10seeds_*.log` |
| 5.5 robustness, accuracy-matched (`robust_bound.py --tune cert`) | 0 pathwise-bound violations in all 12 settings. Majority and consensus reach ≥ 1.0× full latency at k=16–32. The cost rule degrades smoothly (QQP k=32: 0.60–0.90). The per-sample threshold is also robust. Trimming (m>0) helps little on text and costs a lot on CIFAR (m=8 → about 1.0×). MNLI is still 1.4–1.6 pt over budget for the cost rule | `logs/robust_cert_*.log`, `results/robust_*_cert.csv` |
| MNLI δ check (`experiments/delta_check.py`) | On the fixed split every family overshoots: theory 1.25–1.28 pt at δ=0.10, about the same at 0.05, and 0.47–0.80 pt at δ=0.02 (except B=128) at about 10% less speed. The split is random (`traces/train_exit_heads.py`), so this is an unlucky split, not a shift | `results/delta_check_mnli.csv` |
| Minrank CPU chain | Baselines (11,180 runs): SAT is best on planted graphs (within 1.4% of best at 60 s; local search 14–16% worse); some method beats the clique cover on 24/51 planted graphs and 0/121 others. Exact: 29/53 proven, 24 timed out at 600 s (n=50 planted_d: lb 8 vs best 15–16). Separations: 0 in 520 small graphs | `minrank/results/*.csv`, `logs/minrank_*.log` |

**Running when this was written:**

- **Item 2 chain** (`logs/delta_check_status.log`, cores 4-11,36-43):
  - `load_trace.py --task mnli --delta 0.05 --tag _d05`, then `cert_check.py --task mnli --methods t --delta 0.05 --tag _d05`.
  - It ends with a `DONE` line; expected around 05:00–05:30 UTC.
- **Minrank evolve** (GPU 4, `logs/minrank_evolve_evo1.log`):
  - Resumed from checkpoint 50 with a smaller prompt (`minrank/evolve/config.yaml`: 2 top programs, 1 diverse
    program, artifacts capped at 4 KB, `max_tokens` 8192), after the 32k context overflowed.
  - At iteration 230, about 1.8 min per iteration; ends at 300, around 06:15 UTC.
  - **The best score is still the seed's (−0.0075).**

## 4. Tasks, in order

1. **Item 2 results.** When `logs/delta_check_status.log` shows `DONE`, summarize:
   - MNLI under fluctuating load at δ=0.05: does the cost rule get within budget?
   - Re-split violation rate and speed at δ=0.05 vs 0.10.
2. **Minrank gate 1** (after evolve reaches iteration 300; confirm the process exited):
   - Read `scripts/minrank/07_eval_heldout.sh`, `08_cascade_replay.sh` and `09_aggregate.sh` for their env vars, then
     run them in order, pinned to cores 16-31,48-63, with `PY=~/venv/bin/python RUN=evo1`.
   - Write `docs/MINRANK_GATE1.md`. Cover:
     - the best evolved program vs the seed vs the baselines, on held-out;
     - the Option D precondition: evaluation time vs LLM time per iteration (from the evolve log). It was about
       30 s of evaluation vs about 3 min of generation, so the cascade has little to save;
     - a recommendation. The current one: stop blind evolution. Re-target to the hard graphs where SAT times out
       (n=50 planted_d), with fitness relative to SAT at the same budget, or evolve the SAT encoding / a hybrid.
   - **Do not start a new evolve run or keep vLLM running without the user's approval.** Tell the user GPU 4 can be
     freed for the CUDA-graph timing.
3. **Minrank exact reruns (CPU, can start right away):**
   - Rerun the 24 `call_timeout` graphs of `minrank/results/exact.csv` with a longer limit. Read
     `scripts/minrank/04_exact.sh` and `python -m minrank.exact --help` for the flags; try 3600 s per call.
   - Use 16 workers on cores 16-31,48-63, and write to a separate output so `exact.csv` is not clobbered.
   - Report how many become proven.
4. **Docs and report.** Rewrite the claims to match the evidence, without overclaiming:
   - Update `docs/PLAN_LLM.md` §7 and `docs/DIRECTION.md` §7 with the results in section 3.
   - Update `analysis/make_report.py`: section 5 results for 5.1 (eager), 5.2, 5.3, 5.4 and 5.5; the LLM
     classifiers; and the 5.10 claims table. Then regenerate the report and check the figures render.
   - Claims as they now stand:

     | Claim | Status |
     |---|---|
     | Near-best everywhere without re-tuning | **Holds**, strongest on the LLM classifiers. Show RoBERTa MNLI and Qwen-1.5B MNLI as the exceptions |
     | Certified | **Weaken** to "an asymptotic certificate, checked over 30 re-splits". Finite-sample bounds are too conservative at these sample sizes; label-free costs speed |
     | Fluctuating load | **Holds** on QQP; CIFAR-100 only with the plan; MNLI fails on budget |
     | Bandit | **Holds** |
     | Robustness | The bound **holds**, and collective majority/consensus collapse. But the per-sample threshold is also robust, and trimming buys little; say so |
     | MNLI overshoot | Present as split luck (all families overshoot; the re-split rate is within δ). Optionally show δ=0.02 |
5. **Llama QQP at B=128:** the timing log has B=8/16/64 only. When a free GPU is available (not 2 or 3; ask the
   user), run the `time` step for B=128, following `scripts/run_llm_cls_hw.sh`.
6. **ViT / CIFAR-100 device (5.1):** only after the user's CUDA-graph branch is merged, because it touches the same
   code. Ask the user before starting.

## 5. Committing

- Commit under the configured git identity (the user's). Use small, focused commits, each ending with the
  attribution line the session's instructions give.
- First `git fetch` and `git merge --ff-only origin/claude/keen-dijkstra-fpe319` (or rebase if needed); then push
  to `origin claude/keen-dijkstra-fpe319`.
- Never commit:
  - `checkpoints/`;
  - the raw per-step bandit CSVs (`results/bandit_*_10seeds.csv`, gitignored, about 25 MB each);
  - anything over about 20 MB;
  - `.env` / tokens.
- Don't push to the user's CUDA-graph branch.
