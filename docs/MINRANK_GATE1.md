# Minrank gate 1: first OpenEvolve run (`evo1`), 23 Nov 2026

**Headline.** After 300 iterations no evolved program beat the seed. The best program of the run *is* the seed
(training fitness −0.0075; the best LLM-written candidate reached −0.0090). On held-out graphs both the seed and the best LLM-written program are behind every classical baseline, including SAT. About two thirds of
the LLM's candidates crashed inside the evolved functions. The seed's `try/except` caught every crash and returned the
starting clique cover, so the evaluator scored them as valid, cover-level programs. Evaluation takes about 4 s per
candidate against about 106 s of generation, so an early-exit cascade for evaluation (Option D) has almost nothing to
save. Recommendation: stop blind evolution of this setup. Re-target it to the graphs SAT cannot solve, with fitness
relative to SAT at equal time (the code is ready, not launched), and fix the silent-crash problem first.

## 1. The run

| | |
|---|---|
| Model | Qwen2.5-Coder-32B-Instruct-AWQ, vLLM on GPU 4 (port 8000), max_model_len 32768 |
| Config | `minrank/evolve/config.yaml`: full rewrites of the EVOLVE block, 3 islands, population 200, cascade (4 graphs, then all) |
| Training set | 52 graphs, split `train`, families planted, planted_d, gnp, n ≤ 50, 2 s per graph, 4 parallel graphs |
| Fitness | mean over graphs of (cover_rank − rank) / cover_rank (cover_rank = TabuCol, 10 s); −1 for crash, timeout or non-fitting answer |
| History | iterations 1–50, then the 32k context overflowed; resumed from checkpoint 50 with a smaller prompt (2 top + 1 diverse program, artifacts ≤ 4 KB, max_tokens 8192), ran to 300 (22:41 → 06:21 UTC) |
| Outputs | `minrank/evolve/runs/evo1/` (checkpoints 25–300, `best/`), traces `minrank/evolve/runs/evo1/traces/evals.jsonl`, log `logs/minrank_evolve_evo1.log` |

Numbers below are from the resumed run (iterations 51–300; `logs/minrank_evolve_evo1.log`) unless stated.

- **Best program = seed.** `best/best_program.py` has candidate id `a39003680687`, the same as
  `minrank/evolve/seed_program.py`; its `best_program_info.json` says generation 0, iteration 0, combined_score
  −0.00753. No evaluated candidate reached that score. The best of the 223 evaluations in the resumed run scored
  −0.0151. The best non-seed program in the final population (`06b3d73ed39e`, found at iteration 3 of the first
  part) scored −0.0090; it is saved as `minrank/evolve/runs/evo1/best_nonseed_program.py` and also evaluated on
  held-out data below.
- **27 iterations** produced only edits outside the EVOLVE-BLOCK and were rejected.
- **Silent crashes.** 63.7% of the 223 evaluations scored exactly −0.0754, i.e. the DSatur clique cover the skeleton
  starts from, in a few milliseconds per graph. In the final population (`checkpoint_300/programs`), 92 of 200
  programs have that score. With the skeleton's `except Exception: pass` disabled, **all 92 raise** on a G(30, 0.5)
  graph: `ValueError` 33 (mostly "truth value of an array is ambiguous", plus matmul shape mismatches), `NameError` 31
  (for example `rng` used where it is not in scope), `UnboundLocalError` 26, `UFuncTypeError` 2. The skeleton
  returns its fallback answer, the evaluator sees a fitting matrix, and the LLM is told "status ok, rank = cover"
  instead of the traceback. Most of the search was spent on programs that never ran their own code.

## 2. Held-out: best program (= seed) and best non-seed program vs the baselines

Source: `minrank/results/evolved.csv` (steps 07, 1,800 verified runs, 0 failures) against the val/test rows of
`minrank/results/baselines.csv`; seed `a39003680687`, best non-seed `06b3d73ed39e`. Mean rank / best rank found by any
run (lower is better):

| Group | Budget | SAT | factor_tabu | tabucol | seed | best non-seed |
|---|---|---|---|---|---|---|
| planted | 1 s | 1.157 | 1.151 | 1.173 | 1.227 | 1.234 |
| planted | 10 s | **1.033** | 1.147 | 1.168 | 1.189 | 1.195 |
| planted | 60 s | **1.014** | 1.139 | 1.165 | 1.168 | 1.171 |
| random | 60 s | 1.004 | 1.003 | **1.001** | 1.022 | 1.020 |
| structured | 60 s | 1.002 | 1.001 | **1.000** | 1.004 | 1.003 |

Paired on the same (graph, budget, seed), 300 pairs per budget, the best non-seed program against SAT at 60 s is better
on 4, tied on 210 and worse on 86 (mean +0.48 rank); against factor_tabu at 60 s: 9 / 232 / 59 (+0.28). The seed is
within noise of the non-seed program. Neither is competitive with the classical baselines on held-out graphs.

## 3. Option D precondition: evaluation time vs LLM time

| Per iteration (resumed run) | Median | Mean | 90th pct | Max |
|---|---|---|---|---|
| LLM generation (request → response), 239 calls | 106 s | 108 s | | |
| Evaluation of one candidate (OpenEvolve "Evaluated program … in"), 223 | 3.7 s | 8.1 s | 21.1 s | 26.5 s |
| Whole iteration ("completed in") | 217 s | 222 s | | |

The evaluation is about 2% of an iteration at the median and at most 12%. It is short because most candidates
crashed and returned immediately; the seed, which uses its whole budget, takes 26 s. Even a cascade that skipped
evaluation entirely would shorten the run by a few percent. The G2b bar (≥ 2× less evaluation compute at equal
final fitness) could be met on paper, but it would not matter for wall time while generation costs 25–50× more.

**Cascade replay (step 08, `minrank/results/cascade/summary.csv`, 33 generation batches of the evo1 traces).**
Every policy kept the final incumbent (the seed) and lost no winner. Evaluation compute: successive halving −52%,
Hyperband −39%, the cost rule −68% (3.1× less evaluation, at every threshold ≥ 0.003), the certified cost rule and the
Hoeffding race 0%. This looks good on paper, but (i) only one batch contained a winner, so "no lost winner" is weak
evidence, and (ii) evaluation is about 2% of wall time, so a 68% saving shortens the run by about 1–2%. Option D's
precondition (evaluation dominates cost) fails for this setup.

## 4. Exact values (Option B), longer rerun

**Longer rerun (`scripts/minrank/04c_exact_rerun.sh`, 3,600 s per SAT call, same encoding).** 16 of the 24 timed-out
graphs have finished: all 16 timed out again; one improved its upper bound (planted n = 50 s2: 9 → 8). A longer limit
does not help with the current encoding.

**Update (23 Nov, after this gate): independent-set symmetry breaking settles most of them.** If I is an independent
set with |I| ≤ r, then A[I, I] = Id for every fitting A, and the GL(r) symmetry lets us fix U_I = V_I = [Id | 0] (for
directed graphs with an acyclic induced I, only U_I). This is sound by a short argument and was checked against every
known value (29 proven benchmark graphs and 511 small graphs, 1,080 SAT/UNSAT checks, 0 mismatches). With it:
Kneser K(7,2) = 7, K(7,3) = 18, **K(8,3) = 22** (clique cover 28, α = 21; rank 21 UNSAT in 822 s), Mycielski M5 = 12,
M6 = 24, cycle power C(31,3) = 8, and planted n = 50 graphs = 8 are proven, most in under a second; directed planted
n = 50 upper bounds improve (e.g. 15 → 14, 14 → 13). K(8,3) is an explicit 56-vertex graph whose F_2 minrank is six
below its clique cover. Cross-checks (second solver, other independent set) and the move into `minrank/exact/`
with tests are in progress; until then these values are preliminary.

## 5. Recommendation

1. **Stop blind evolution of this setup.** 300 iterations (about 9.5 hours of one A6000 running the 32B model, 20:49 → 06:21 UTC) did not beat the
   seed on its own training set, so the held-out comparison is a comparison of the seed with the baselines.
2. **Fix the evaluator feedback before any new run.** Crashes inside evolved functions must reach the LLM. Two
   options, neither applied yet: (a) a seed whose skeleton re-raises, or records the exception and returns it in the
   artifacts, so that a crash scores −1 with its traceback; (b) keep the fallback but report "evolved code raised X at
   line Y" in the per-graph artifact. Option (a) changes fitness values, so it belongs in a new run, not in `evo1`.
3. **Re-target to the hard graphs, fitness relative to SAT at equal time.** Implemented as an opt-in evaluator mode
   (`MINRANK_EVO_TARGET=sat`, `minrank/evolve/config_hard.yaml`, `scripts/minrank/06b_evolve_hard.sh`; see
   `minrank/README.md`). The default evaluator is unchanged. A CPU-only dry run (`PREPARE_ONLY=1`, on a scratch copy,
   10 s per graph, 4 workers) selects 15 hard training graphs: planted_d n = 50 (s0–s3), planted n = 50 (s1–s3),
   planted_d and planted n = 100 (s0–s3 each). The seed then scores −0.038 against SAT: it beats SAT on 1 graph,
   ties on 7, and is behind on 7 (for example 26–27 against 24–25 on the directed n = 100 graphs). Not launched:
   needs your approval and a GPU.
4. **Alternative to evolving the local search (now the main lead, see Section 4):** improve or evolve the SAT encoding or a SAT + local-search hybrid (symmetry
   breaking, column order, warm starts). SAT is the strongest baseline on the planted graphs, and on n = 50
   planted_d it stalls at rank 14–16 against a planted rank of 8.
5. **GPU 4 can be freed.** The evolve run has exited. vLLM is still running (PID 3843328, port 8000) because stopping
   it was not part of this task; stop it with `kill "$(cat logs/minrank_vllm_8000.pid)"` (PID 3843328) when you want
   the GPU for the CUDA-graph timing.
