# Kneser agent benchmark

GPU server (64 cores; baselines pinned to cores 16-27, 3 graphs at a time; agent runs pinned to cores 28-31, 3 worker processes; other users' jobs were running on the machine). Standard methods: one thread each, 120 s each, in separate processes; 'time' = when their bracket first closed, counting them as if they ran in parallel. Kushagra's script: recognise (n, k), rule field, one rank, greedy star; falls back to the general minrank agent. Agent: budget 300 s; time = wall time of the run (graph loading and the one-time numba warm-up excluded); for LLM runs the compute time without LLM seconds is in brackets. Every agent bound is re-checked by the independent verifier. Bold = proven exact (lower = upper). '(rule)' = the family rule may be used (only an LLM hypothesis can invoke it, so without an LLM it makes no difference); '(generic only)' = --no-family-rule.

## Analysis (26 Nov 2026, GPU server)

Conditions: other users' jobs ran on the machine throughout. To save time, the agent runs overlapped with the
baselines (on separate cores), so the baseline timings were not measured on an otherwise idle machine. All benchmark Qwen
numbers below are **Qwen2.5-Coder-32B-AWQ with prompts v3, before the plan-results fix d22d46c** (results of checks
run by the Solver's own plan were not shown to the agents). The Qwen3-4B ablation could not run, and the Qwen 32B rerun on the fixed code ran
on CPU only for the three reference cases (see the model ablation below): from 18:59 UTC, `cuInit` failed on every GPU of the server (error 3, also on an idle
3080 Ti), which needs an admin reset; already-running processes of other users were not affected.

- **Proven exact, 20 benchmark graphs:** standard methods 2, Kushagra's script 14, agent without LLM 14, agent +
  Qwen 32B 14 (14 with generic rules only). All 14 are the regular Kneser graphs; the 3 exceptions and 3 controls
  end at gaps for every method; no method claimed a false value, and every agent bound passed the independent
  verifier.
- **Large graphs (4,845-8,855 receivers):** Kushagra's script 3/3, agent + Qwen 32B with the rule 3/3, agent without
  LLM 0/3, agent + Qwen with generic rules only 0/3, standard methods 0/3 (open; no lower bound on K(23,4)). The
  agent computes no spectrum above N = 4000, so its generic search has nothing to try there; the family rule,
  reachable only through an LLM hypothesis, closes these graphs.
- **Speed:** without an LLM the agent is slower than Kushagra's script on the shared exact graphs (median time ratio
  x0.61; it is slower on 8 of 14, by at most 0.6 s), and x135 faster than the standard methods on the 2 graphs
  they prove. On the large graphs the agent's compute time is 5-8x below Kushagra's script (2.3 / 4.5 / 8.0 s vs
  18.7 / 23.0 / 41.6 s), but with the LLM seconds its wall time is about 2x longer on two of them and equal on K(23,4).
- **LLM vs no LLM:** on the 20 benchmark graphs Qwen proved nothing more and its compute time was about the same
  (median x1.05). It adds a median 43 s of LLM time per run (max 56 s). Its value showed only on the large graphs
  (3 proofs that the no-LLM run did not get) and on one control (reg(1000,6): lower bound 332 vs 327, from the extra
  independent-set restarts it asked for).
- **Family guess vs truth.json:** the first Solver reply named the right family and (n, k) in 2/40 runs (with v3 it
  usually asks binomial_match before giving (n, k)); the accepted plan was right in 27/40. On 5 exact Kneser graphs
  (K(40,2), K(53,2), K(12,3), K(21,3), K(17,4), rule allowed) the final label was a wrong "perturbed Kneser",
  after false Critic objections; no verdict changed, because the verdict comes from the checks.
- **What the Critic caught:** 110 challenges in 40 runs. Useful: the false arithmetic of prompts v1 ("C(14,7) = 1001",
  "C(14,5) = 1820"; now also rejected by code), asking for binomial_match before trusting a guess, and making the
  Solver query the rule on K(14,4) (it names no field) before a generic search, as in the chat reference run. Not
  useful: repeated or wrong objections ("degree should be C(12,3) = 220" on K(16,4)).
- **Invalid replies:** 14 of 336 requests (9 were family_rule without arguments), 2 fallbacks to the default plan;
  no JSON syntax errors (vLLM JSON mode; also none in a run without it).
- **Generic rules only vs with the family rule:** identical on the 20 benchmark graphs (the generic spectral search
  closes every regular Kneser graph up to 3,003 receivers); 0 vs 3 on the large graphs.
- **Losses:** to Kushagra's script, on time, on most exact graphs (listed below); never on accuracy: on the 3
  exceptions both reach the same brackets ([84, 93], [286, 287], [495, 507]), on the 3 controls the agent's brackets
  are tighter (e.g. K(16,4)-del0.01: [455, 534] vs [0, 1818]). With prompts v1, Qwen lost to the no-LLM agent on
  K(14,4) ([286, 299] vs [286, 287]); the scheduler's safety net fixed this.
- **Model ablation (bench/ablation/, three reference cases, one run each):** every model ends with the same
  certified brackets ([455, 455], [455, 534], [286, 287]): the model changes the dialogue, not the result. The Claude
  Solvers (Haiku 5.5 medium, Sonnet 5.5 medium, Opus 5.5 low/medium/high) named the right (n, k) in the first reply on
  14 of 15 runs, Qwen 32B on 0 of 6; the Claude Critics made 0 false objections, Qwen's 5 (GPU, code before
  d22d46c) and 7 (CPU, fixed code); Qwen needed 27-28 LLM calls over the three graphs, the Claude configurations
  11-13. So the plan-results fix did not change Qwen's weak point: its Critic still objects to numbers that are right
  (e.g. "the common-neighbour counts [70, 126, 210, 330] do not match K(16,4)"). Qwen GPU = vLLM, AWQ, 26 Nov
  18:0x-18:4x; Qwen CPU = llama.cpp b12780, GGUF Q4_K_M, 12 cores, budget 7,200 s instead of 3,600 s (about 6.5 min
  per reply, 52-56 min per run), run because the GPUs stopped initialising CUDA at 18:59 UTC. False objections are
  labelled by hand (Qwen CPU rows labelled by Claude Code, same rule as the other rows).
- **32B vs 4B:** not run (GPU failure above; a 4B model on CPU was not attempted).

## Computed summary (benchmark graphs only; generated by bench/report.py)

- 20 graphs. Proven exact: standard methods 2, Kushagra's script 14, agent, no LLM 14, agent + Qwen 32B 14 (generic only 14)
- agent + Qwen 32B vs Kushagra's script, both exact (14 graphs): median time ratio x0.92 (min x0.05, max x2.11; > 1 = the agent is faster; agent time = compute time without LLM seconds)
- agent + Qwen 32B vs Kushagra's script, LLM seconds included, both exact (14 graphs): median time ratio x0.01 (min x0.00, max x0.15; > 1 = the agent is faster)
- agent + Qwen 32B vs standard methods, both exact (2 graphs): median time ratio x159.77 (min x130.91, max x188.63; > 1 = the agent is faster; agent time = compute time without LLM seconds)
- agent, no LLM vs Kushagra's script, both exact (14 graphs): median time ratio x0.61 (min x0.12, max x1.48; > 1 = the agent is faster)
- agent, no LLM vs standard methods, both exact (2 graphs): median time ratio x134.64 (min x108.69, max x160.58; > 1 = the agent is faster)
- agent + Qwen 32B: 40 runs (both rule settings); LLM seconds per run median 43.0 (max 56.3); LLM calls 296; invalid replies 14 of 336 requests, fallbacks to the default 2; Critic challenges 110; safety net ran in 13 runs
  - family guess vs bench/truth.json: first Solver reply right in 2/40, final plan right in 27/40 (wrong -> right during the dialogue: 25; right -> wrong: 0)
  - rule allowed vs no LLM: proven only with the LLM: none; only without: none; both proven (14): no-LLM wall / LLM-run compute time, median x1.05 (> 1 = the LLM run computes less)
  - generic only vs no LLM: proven only with the LLM: none; only without: none; both proven (14): no-LLM wall / LLM-run compute time, median x0.93 (> 1 = the LLM run computes less)

### Every loss of an agent run (rule allowed) to a baseline

- K(21,2) | agent, no LLM: slower than Kushagra's script (0.10 s vs 0.03 s)
- K(21,2) | agent + Qwen 32B: slower than Kushagra's script (0.14 s vs 0.03 s)
- K(29,2) | agent, no LLM: slower than Kushagra's script (0.23 s vs 0.07 s)
- K(29,2) | agent + Qwen 32B: slower than Kushagra's script (1.51 s vs 0.07 s)
- K(40,2) | agent + Qwen 32B: slower than Kushagra's script (0.28 s vs 0.24 s)
- K(53,2) | agent, no LLM: slower than Kushagra's script (1.27 s vs 0.68 s)
- K(53,2) | agent + Qwen 32B: slower than Kushagra's script (1.45 s vs 0.68 s)
- K(12,3) | agent, no LLM: slower than Kushagra's script (0.27 s vs 0.03 s)
- K(12,3) | agent + Qwen 32B: slower than Kushagra's script (0.15 s vs 0.03 s)
- K(13,3) | agent, no LLM: slower than Kushagra's script (0.16 s vs 0.04 s)
- K(13,3) | agent + Qwen 32B: slower than Kushagra's script (0.13 s vs 0.04 s)
- K(16,3) | agent, no LLM: slower than Kushagra's script (0.47 s vs 0.13 s)
- K(16,3) | agent + Qwen 32B: slower than Kushagra's script (0.23 s vs 0.13 s)
- K(21,3) | agent + Qwen 32B: slower than Kushagra's script (0.72 s vs 0.71 s)
- K(13,4) | agent, no LLM: slower than Kushagra's script (0.61 s vs 0.28 s)
- K(15,4) | agent, no LLM: slower than Kushagra's script (1.17 s vs 0.80 s)

## Per-graph results

| graph | receivers | standard methods | time | Kushagra's script | time | agent, no LLM | time | agent + Qwen 32B (rule) | time | agent + Qwen 32B (generic only) | time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K(21,2) | 210 | [20, 21] | open | **20** | 0.03 s | **20** | 0.10 s | **20** | 27.20 s (0.14) | **20** | 49.98 s (0.17) |
| K(29,2) | 406 | [28, 29] | open | **28** | 0.07 s | **28** | 0.23 s | **28** | 43.86 s (1.51) | **28** | 41.69 s (1.27) |
| K(40,2) | 780 | **39** | 36.13 s | **39** | 0.24 s | **39** | 0.23 s | **39** | 38.38 s (0.28) | **39** | 48.69 s (0.15) |
| K(53,2) | 1378 | [52, 59.98] | open | **52** | 0.68 s | **52** | 1.27 s | **52** | 55.38 s (1.45) | **52** | 42.11 s (1.04) |
| K(12,3) | 220 | **55** | 29.24 s | **55** | 0.03 s | **55** | 0.27 s | **55** | 43.74 s (0.15) | **55** | 41.77 s (0.14) |
| K(13,3) | 286 | [66, 71.5] | open | **66** | 0.04 s | **66** | 0.16 s | **66** | 42.39 s (0.13) | **66** | 35.81 s (0.10) |
| K(16,3) | 560 | [105, 160] | open | **105** | 0.13 s | **105** | 0.47 s | **105** | 28.98 s (0.23) | **105** | 41.96 s (0.36) |
| K(18,3) | 816 | [136, 411] | open | **136** | 0.30 s | **136** | 0.23 s | **136** | 44.33 s (0.27) | **136** | 40.15 s (0.45) |
| K(21,3) | 1330 | [190, 1152] | open | **190** | 0.71 s | **190** | 0.50 s | **190** | 40.44 s (0.72) | **190** | 37.45 s (0.64) |
| K(13,4) | 715 | [220, 604] | open | **220** | 0.28 s | **220** | 0.61 s | **220** | 30.15 s (0.25) | **220** | 48.80 s (0.38) |
| K(15,4) | 1365 | [364, 1341] | open | **364** | 0.80 s | **364** | 1.17 s | **364** | 29.54 s (0.52) | **364** | 50.27 s (1.95) |
| K(16,4) | 1820 | [455, 1800] | open | **455** | 1.90 s | **455** | 1.28 s | **455** | 52.97 s (0.90) | **455** | 47.33 s (1.31) |
| K(17,4) | 2380 | [560, 2368] | open | **560** | 2.32 s | **560** | 1.68 s | **560** | 40.86 s (1.90) | **560** | 49.09 s (1.99) |
| K(15,5) | 3003 | [1001, 3003] | open | **1001** | 4.38 s | **1001** | 3.62 s | **1001** | 28.52 s (2.88) | **1001** | 40.47 s (4.10) |
| K(10,4) (exception) | 210 | [84, 121] | open | [84, 93] | not exact | [84, 93] | 0.34 s | [84, 93] | 39.02 s (0.51) | [84, 93] | 48.69 s (0.52) |
| K(14,4) (exception) | 1001 | [286, 961] | open | [286, 287] | not exact | [286, 287] | 2.08 s | [286, 287] | 58.70 s (2.36) | [286, 287] | 28.68 s (2.37) |
| K(13,5) (exception) | 1287 | [378, 1287] | open | [495, 507] | not exact | [495, 507] | 3.15 s | [495, 507] | 58.76 s (3.44) | [495, 507] | 47.40 s (3.44) |
| K(16,4)-del0.01 (control) | 1820 | [455, 1800] | open | [0, 1818] | not exact | [455, 534] | 0.52 s | [455, 534] | 45.06 s (1.21) | [455, 534] | 51.84 s (0.66) |
| K(18,3)-del0.01 (control) | 816 | [136, 433] | open | [136, 732] | not exact | [136, 152] | 0.12 s | [136, 152] | 49.73 s (0.45) | [136, 152] | 50.00 s (0.44) |
| reg(1000,6) (control) | 1000 | [294, 994] | open | [293, 998] | not exact | [327, 537] | 0.16 s | [332, 537] | 53.88 s (0.59) | [332, 537] | 51.76 s (0.60) |
| K(20,4) (large) | 4845 | [969, 4841] | open | **969** | 18.66 s | [969, 1098] | 2.38 s | **969** | 40.27 s (2.25) | [969, 1098] | 45.05 s (0.35) |
| K(17,5) (large) | 6188 | [1491, 6188] | open | **1820** | 22.98 s | [1820, 2106] | 11.35 s | **1820** | 44.15 s (4.53) | [1820, 2106] | 97.28 s (53.14) |
| K(23,4) (large) | 8855 | [-, 8855] | open | **1540** | 41.62 s | [1540, 1789] | 15.73 s | **1540** | 40.85 s (8.04) | [1540, 1771] | 111.55 s (66.54) |

## Model ablation: blind agents on the three reference cases (bench/ablation/)

Each run gives the Solver, Critic and Verifier roles to one model answering blind (bench/ablation/README.md). Claude rows: fixed code (d22d46c). Qwen rows: GPU server, prompts v3, code before d22d46c. One run per cell; false objections labelled by hand (bench/ablation/false_objections.json).

![model ablation](bench/ablation/model_ablation.png)

| graph | model | verdict | bracket | LLM calls | invalid | fallbacks | (n, k) in first reply | Critic objections | false | safety net | codes tried |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K_16_4_s3 | Qwen2.5-32B | proven | [455, 455] | 8 | 0 | 0 | no | 3 | 2 | no | 5 |
| K_16_4_s3 | Qwen2.5-32B-cpu | proven | [455, 455] | 9 | 1 | 0 | no | 3 | 2 | no | 1 |
| K_16_4_s3 | haiku-med | proven | [455, 455] | 3 | 0 | 0 | yes | 0 | 0 | no | 1 |
| K_16_4_s3 | sonnet-med | proven | [455, 455] | 3 | 0 | 0 | yes | 0 | 0 | no | 1 |
| K_16_4_s3 | opus-low | proven | [455, 455] | 3 | 0 | 0 | yes | 0 | 0 | no | 1 |
| K_16_4_s3 | opus-med | proven | [455, 455] | 3 | 0 | 0 | yes | 0 | 0 | no | 1 |
| K_16_4_s3 | opus-high | proven | [455, 455] | 3 | 0 | 0 | yes | 0 | 0 | no | 1 |
| K_16_4-del0.01_s0 | Qwen2.5-32B | gap | [455, 534] | 9 | 0 | 0 | no | 5 | 3 | yes | 0 |
| K_16_4-del0.01_s0 | Qwen2.5-32B-cpu | gap | [455, 534] | 10 | 1 | 0 | no | 3 | 3 | yes | 0 |
| K_16_4-del0.01_s0 | haiku-med | gap | [455, 534] | 4 | 0 | 0 | no | 0 | 0 | yes | 1 |
| K_16_4-del0.01_s0 | sonnet-med | gap | [455, 534] | 6 | 0 | 0 | yes | 2 | 0 | yes | 1 |
| K_16_4-del0.01_s0 | opus-low | gap | [455, 534] | 4 | 0 | 0 | yes | 0 | 0 | yes | 1 |
| K_16_4-del0.01_s0 | opus-med | gap | [455, 534] | 4 | 0 | 0 | yes | 1 | 0 | yes | 1 |
| K_16_4-del0.01_s0 | opus-high | gap | [455, 534] | 4 | 0 | 0 | yes | 0 | 0 | yes | 3 |
| K_14_4_s0 | Qwen2.5-32B | gap | [286, 287] | 10 | 1 | 0 | no | 4 | 0 | yes | 32 |
| K_14_4_s0 | Qwen2.5-32B-cpu | gap | [286, 287] | 9 | 0 | 0 | no | 4 | 2 | yes | 31 |
| K_14_4_s0 | haiku-med | gap | [286, 287] | 4 | 0 | 0 | yes | 0 | 0 | yes | 31 |
| K_14_4_s0 | sonnet-med | gap | [286, 287] | 4 | 0 | 0 | yes | 0 | 0 | yes | 31 |
| K_14_4_s0 | opus-low | gap | [286, 287] | 4 | 0 | 0 | yes | 0 | 0 | yes | 31 |
| K_14_4_s0 | opus-med | gap | [286, 287] | 4 | 0 | 0 | yes | 0 | 0 | yes | 31 |
| K_14_4_s0 | opus-high | gap | [286, 287] | 4 | 0 | 0 | yes | 1 | 0 | yes | 31 |

## Every run with an LLM in the loop

llm = chat: an assistant answered as the agents through the file-based chat backend (reference runs); otherwise the named model on the server (vLLM, GPU 4). prompts: v1 = branch prompts; v2, v3 = after the fixes described in MERGE_NOTES.md and the commit messages.

| graph | llm | prompts | rule | verdict | bracket | wall | LLM s | compute | calls | invalid | Critic challenges | final hypothesis | truth |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| K_16_4_s3_chat | chat | - | yes | proven | [455, 455] | 55.4 s | 54.8 | 0.52 s | 4 | 0 | 1 | Kneser graph K(n,k) {'n': 16, 'k': 4} | kneser 16,4 |
| K_16_4-del0.01_s0_chat | chat | - | yes | gap | [455, 534] | 77.0 s | 61.6 | 15.39 s | 3 | 0 | 0 | Kneser K(16,4) with about 1% of its edges deleted (not an exact Kneser graph) {'n': 16, 'k': 4, 'deleted_fraction': 0.01} | kneser with deleted edges 16,4 |
| K_14_4_s0_chat | chat | - | yes | gap | [286, 287] | 92.9 s | 82.0 | 10.87 s | 5 | 0 | 1 | Kneser graph K(n,k) {'n': 14, 'k': 4} | kneser 14,4 |
| K_16_4_s3 | Qwen2.5-Coder-32B-Instruct-AWQ | v1 | yes | proven | [455, 455] | 68.8 s | 67.9 | 0.85 s | 7 | 0 | 3 | Kneser {'n': 16, 'k': 4} | kneser 16,4 |
| K_16_4-del0.01_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v1 | yes | gap | [455, 534] | 67.3 s | 66.6 | 0.65 s | 8 | 0 | 3 | Kneser {'n': 16, 'k': 4} | kneser with deleted edges 16,4 |
| K_14_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v1 | yes | gap | [286, 299] | 72.4 s | 71.6 | 0.87 s | 9 | 1 | 3 | Kneser {'n': 14, 'k': 7} | kneser 14,4 |
| K_16_4_s3_nojsonmode | Qwen2.5-Coder-32B-Instruct-AWQ | v1 | yes | proven | [455, 455] | 84.8 s | 84.0 | 0.89 s | 9 | 2 | 3 | Kneser {} | kneser 16,4 |
| K_16_4_s3 | Qwen2.5-Coder-32B-Instruct-AWQ | v2 | yes | proven | [455, 455] | 45.7 s | 44.8 | 0.93 s | 8 | 0 | 3 | Kneser {'n': 16, 'k': 4} | kneser 16,4 |
| K_14_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v2 | yes | gap | [286, 287] | 56.5 s | 53.3 | 3.26 s | 9 | 1 | 3 | perturbed Kneser {'n': 14, 'k': 4} | kneser 14,4 |
| K_16_4_s3 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [455, 455] | 37.3 s | 36.1 | 1.20 s | 7 | 0 | 3 | perturbed Kneser {} | kneser 16,4 |
| K_16_4-del0.01_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [455, 534] | 49.6 s | 48.2 | 1.36 s | 9 | 0 | 3 | none {} | kneser with deleted edges 16,4 |
| K_14_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [286, 287] | 28.4 s | 25.6 | 2.89 s | 5 | 1 | 3 | Kneser {'n': 14, 'k': 4} | kneser 14,4 |
| K_16_4-del0.01_s0_opus-low | chat | v3 | yes | gap | [455, 534] | 96.5 s | 13.6 | 82.91 s | 3 | 0 | 0 | perturbed Kneser {} | kneser with deleted edges 16,4 |
| K_14_4_s0_opus-low | chat | v3 | yes | gap | [286, 287] | 98.3 s | 41.3 | 56.95 s | 3 | 0 | 0 | Kneser {} | kneser 14,4 |
| K_16_4_s3_opus-low | chat | v3 | yes | proven | [455, 455] | 100.4 s | 8.9 | 91.50 s | 2 | 0 | 0 | Kneser {} | kneser 16,4 |
| K_16_4_s3_opus-med | chat | v3 | yes | proven | [455, 455] | 102.9 s | 12.4 | 90.45 s | 2 | 0 | 0 | Kneser {} | kneser 16,4 |
| K_16_4_s3_sonnet-med | chat | v3 | yes | proven | [455, 455] | 106.3 s | 13.9 | 92.37 s | 2 | 0 | 0 | Kneser {'n': 16, 'k': 4} | kneser 16,4 |
| K_16_4_s3_opus-high | chat | v3 | yes | proven | [455, 455] | 108.5 s | 16.4 | 92.05 s | 2 | 0 | 0 | Kneser K(n,k), likely K(16,4) {'n': 16, 'k': 4} | kneser 16,4 |
| K_14_4_s0_sonnet-med | chat | v3 | yes | gap | [286, 287] | 110.0 s | 52.3 | 57.67 s | 3 | 0 | 0 | Kneser K(14,4) {'n': 14, 'k': 4} | kneser 14,4 |
| K_16_4_s3_haiku-med | chat | v3 | yes | proven | [455, 455] | 110.9 s | 18.4 | 92.46 s | 2 | 0 | 0 | Kneser-type graph K(n,k) {} | kneser 16,4 |
| K_16_4-del0.01_s0_opus-med | chat | v3 | yes | gap | [455, 534] | 114.4 s | 19.6 | 94.76 s | 3 | 0 | 0 | perturbed Kneser (likely K(16,4) with a few edges deleted) {} | kneser with deleted edges 16,4 |
| K_14_4_s0_haiku-med | chat | v3 | yes | gap | [286, 287] | 115.8 s | 58.3 | 57.47 s | 3 | 0 | 0 | Kneser K(14,4) {'n': 14, 'k': 4} | kneser 14,4 |
| K_14_4_s0_opus-med | chat | v3 | yes | gap | [286, 287] | 116.6 s | 59.3 | 57.29 s | 3 | 0 | 0 | Kneser graph K(n,k) (likely K(14,4), to be confirmed by binomial_match) {} | kneser 14,4 |
| K_16_4-del0.01_s0_haiku-med | chat | v3 | yes | gap | [455, 534] | 120.0 s | 25.6 | 94.38 s | 3 | 0 | 0 | perturbed Kneser {} | kneser with deleted edges 16,4 |
| K_16_4-del0.01_s0_sonnet-med | chat | v3 | yes | gap | [455, 534] | 126.8 s | 31.5 | 95.31 s | 5 | 0 | 1 | perturbed Kneser K(16,4) {'n': 16, 'k': 4} | kneser with deleted edges 16,4 |
| K_16_4-del0.01_s0_opus-high | chat | v3 | yes | gap | [455, 534] | 130.0 s | 27.1 | 102.83 s | 3 | 0 | 0 | perturbed Kneser (possibly K(16,4) with some edges deleted) {'n': 16, 'k': 4} | kneser with deleted edges 16,4 |
| K_14_4_s0_opus-high | chat | v3 | yes | gap | [286, 287] | 131.7 s | 75.2 | 56.44 s | 3 | 0 | 0 | Kneser graph K(n,k) {} | kneser 14,4 |
| K_21_2_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [20, 20] | 27.2 s | 27.1 | 0.14 s | 6 | 0 | 2 | Kneser {'n': 21, 'k': 2} | kneser 21,2 |
| K_21_2_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [20, 20] | 50.0 s | 49.8 | 0.17 s | 8 | 0 | 3 | Kneser {'n': 21, 'k': 2} | kneser 21,2 |
| K_29_2_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [28, 28] | 43.9 s | 42.4 | 1.51 s | 7 | 0 | 3 | Kneser {'n': 29, 'k': 2} | kneser 29,2 |
| K_29_2_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [28, 28] | 41.7 s | 40.4 | 1.27 s | 7 | 0 | 3 | Kneser {'n': 29, 'k': 2} | kneser 29,2 |
| K_40_2_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [39, 39] | 38.4 s | 38.1 | 0.28 s | 7 | 0 | 3 | perturbed Kneser {} | kneser 40,2 |
| K_40_2_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [39, 39] | 48.7 s | 48.5 | 0.15 s | 7 | 0 | 3 | perturbed Kneser {'n': 40, 'k': 2} | kneser 40,2 |
| K_53_2_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [52, 52] | 55.4 s | 53.9 | 1.45 s | 8 | 0 | 3 | perturbed Kneser {} | kneser 53,2 |
| K_53_2_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [52, 52] | 42.1 s | 41.1 | 1.04 s | 7 | 0 | 2 | Kneser {} | kneser 53,2 |
| K_12_3_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [55, 55] | 43.7 s | 43.6 | 0.15 s | 7 | 0 | 3 | perturbed Kneser {'n': 12, 'k': 3} | kneser 12,3 |
| K_12_3_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [55, 55] | 41.8 s | 41.6 | 0.14 s | 7 | 0 | 3 | Kneser {'n': 12, 'k': 3} | kneser 12,3 |
| K_13_3_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [66, 66] | 42.4 s | 42.3 | 0.13 s | 7 | 0 | 3 | Kneser {'n': 13, 'k': 3} | kneser 13,3 |
| K_13_3_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [66, 66] | 35.8 s | 35.7 | 0.10 s | 7 | 0 | 3 | Kneser {'n': 13, 'k': 3} | kneser 13,3 |
| K_16_3_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [105, 105] | 29.0 s | 28.8 | 0.23 s | 6 | 0 | 2 | Kneser {'n': 16, 'k': 3} | kneser 16,3 |
| K_16_3_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [105, 105] | 42.0 s | 41.6 | 0.36 s | 7 | 0 | 3 | perturbed Kneser {} | kneser 16,3 |
| K_18_3_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [136, 136] | 44.3 s | 44.1 | 0.27 s | 7 | 0 | 3 | Kneser {'n': 18, 'k': 3} | kneser 18,3 |
| K_18_3_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [136, 136] | 40.2 s | 39.7 | 0.45 s | 7 | 0 | 3 | Kneser {'n': 18, 'k': 3} | kneser 18,3 |
| K_21_3_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [190, 190] | 40.4 s | 39.7 | 0.72 s | 7 | 0 | 3 | perturbed Kneser {'n': 21, 'k': 3} | kneser 21,3 |
| K_21_3_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [190, 190] | 37.4 s | 36.8 | 0.64 s | 7 | 0 | 3 | Kneser {'n': 21, 'k': 3} | kneser 21,3 |
| K_13_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [220, 220] | 30.2 s | 29.9 | 0.25 s | 6 | 0 | 2 | Kneser {'n': 13, 'k': 4} | kneser 13,4 |
| K_13_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [220, 220] | 48.8 s | 48.4 | 0.38 s | 7 | 0 | 3 | Kneser {'n': 13, 'k': 4} | kneser 13,4 |
| K_15_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [364, 364] | 29.5 s | 29.0 | 0.52 s | 6 | 0 | 2 | Kneser {'n': 15, 'k': 4} | kneser 15,4 |
| K_15_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [364, 364] | 50.3 s | 48.3 | 1.95 s | 7 | 0 | 3 | Kneser {'n': 15, 'k': 4} | kneser 15,4 |
| K_16_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [455, 455] | 53.0 s | 52.1 | 0.90 s | 7 | 0 | 3 | Kneser {'n': 16, 'k': 4} | kneser 16,4 |
| K_16_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [455, 455] | 47.3 s | 46.0 | 1.31 s | 9 | 2 | 3 | perturbed Kneser {} | kneser 16,4 |
| K_17_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [560, 560] | 40.9 s | 39.0 | 1.90 s | 7 | 0 | 3 | perturbed Kneser {} | kneser 17,4 |
| K_17_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [560, 560] | 49.1 s | 47.1 | 1.99 s | 8 | 2 | 3 | Kneser {'n': 17, 'k': 4} | kneser 17,4 |
| K_15_5_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [1001, 1001] | 28.5 s | 25.6 | 2.88 s | 6 | 0 | 2 | Kneser {'n': 15, 'k': 5} | kneser 15,5 |
| K_15_5_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | proven | [1001, 1001] | 40.5 s | 36.4 | 4.10 s | 7 | 0 | 3 | Kneser {'n': 15, 'k': 5} | kneser 15,5 |
| K_10_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [84, 93] | 39.0 s | 38.5 | 0.51 s | 7 | 0 | 2 | Kneser {} | kneser 10,4 |
| K_10_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [84, 93] | 48.7 s | 48.2 | 0.52 s | 8 | 0 | 3 | perturbed Kneser {} | kneser 10,4 |
| K_14_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [286, 287] | 58.7 s | 56.3 | 2.36 s | 9 | 1 | 3 | Kneser {'n': 14, 'k': 4} | kneser 14,4 |
| K_14_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [286, 287] | 28.7 s | 26.3 | 2.37 s | 5 | 0 | 1 | Kneser {'n': 14, 'k': 4} | kneser 14,4 |
| K_13_5_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [495, 507] | 58.8 s | 55.3 | 3.44 s | 9 | 1 | 3 | Kneser {'n': 13, 'k': 5} | kneser 13,5 |
| K_13_5_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [495, 507] | 47.4 s | 44.0 | 3.44 s | 8 | 0 | 3 | Kneser {'n': 13, 'k': 5} | kneser 13,5 |
| K_16_4-del0.01_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [455, 534] | 45.1 s | 43.8 | 1.21 s | 8 | 0 | 3 | none {} | kneser with deleted edges 16,4 |
| K_16_4-del0.01_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [455, 534] | 51.8 s | 51.2 | 0.66 s | 9 | 1 | 3 | perturbed Kneser {'n': 16, 'k': 4} | kneser with deleted edges 16,4 |
| K_18_3-del0.01_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [136, 152] | 49.7 s | 49.3 | 0.45 s | 8 | 0 | 3 | perturbed Kneser {'n': 18, 'k': 3} | kneser with deleted edges 18,3 |
| K_18_3-del0.01_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [136, 152] | 50.0 s | 49.6 | 0.44 s | 11 | 3 | 3 | none {} | kneser with deleted edges 18,3 |
| reg_1000_6_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | gap | [332, 537] | 53.9 s | 53.3 | 0.59 s | 9 | 1 | 3 | none {} | random regular , |
| reg_1000_6_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [332, 537] | 51.8 s | 51.2 | 0.60 s | 9 | 3 | 2 | none {} | random regular , |
| K_20_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [969, 969] | 40.3 s | 38.0 | 2.25 s | 7 | 0 | 3 | Kneser {'n': 20, 'k': 4} | kneser 20,4 |
| K_20_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [969, 1098] | 45.1 s | 44.7 | 0.35 s | 9 | 1 | 3 | Kneser {'n': 20, 'k': 4} | kneser 20,4 |
| K_17_5_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [1820, 1820] | 44.1 s | 39.6 | 4.53 s | 7 | 0 | 3 | Kneser {'n': 17, 'k': 5} | kneser 17,5 |
| K_17_5_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [1820, 2106] | 97.3 s | 44.1 | 53.14 s | 8 | 0 | 3 | Kneser {'n': 17, 'k': 5} | kneser 17,5 |
| K_23_4_s0 | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | yes | proven | [1540, 1540] | 40.9 s | 32.8 | 8.04 s | 7 | 0 | 3 | Kneser {'n': 23, 'k': 4} | kneser 23,4 |
| K_23_4_s0_nofam | Qwen2.5-Coder-32B-Instruct-AWQ | v3 | no | gap | [1540, 1771] | 111.5 s | 45.0 | 66.54 s | 9 | 1 | 3 | Kneser {'n': 23, 'k': 4} | kneser 23,4 |
| K_16_4_s3 | Qwen2.5-Coder-32B-Instruct-Q4_K_M-cpu | v4 | yes | proven | [455, 455] | 2813.8 s | 2802.3 | 11.51 s | 8 | 1 | 3 | Kneser {'n': 16, 'k': 4} | kneser 16,4 |
| K_14_4_s0 | Qwen2.5-Coder-32B-Instruct-Q4_K_M-cpu | v4 | yes | gap | [286, 287] | 2948.2 s | 2938.2 | 10.04 s | 8 | 0 | 3 | Kneser {} | kneser 14,4 |
| K_16_4-del0.01_s0 | Qwen2.5-Coder-32B-Instruct-Q4_K_M-cpu | v4 | yes | gap | [455, 534] | 3262.6 s | 3249.7 | 12.81 s | 9 | 1 | 3 | perturbed Kneser {} | kneser with deleted edges 16,4 |
