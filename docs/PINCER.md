# Pincer: an agentic pipeline that squeezes the broadcast rate and the minrank between certified bounds

Code: [`pincer/`](../pincer/) (built on the solvers in [`minrank/`](../minrank/)). Scripts: [`scripts/pincer/`](../scripts/pincer/).
Proposal it implements: *Pincer — An agent that squeezes the broadcast rate of an index-coding problem between two
certified bounds* (IE624, Nov 2026). Status as of 24 Nov 2026: everything below runs and is tested (66 tests); the
LLM parts are implemented and tested with a scripted backend, but **no LLM has been run in the loop yet** (this
container has no API key or GPU); see §7 for exactly what to run on the server.

## 0. Results in one table

Fresh split `test2` (26 random-family graphs, seeds never used for tuning), 30 s per graph, 4-core container,
every number from a re-verified certificate bundle (`pincer/results/summary_test2_b30.md`):

| method | mean minrank gap | minrank closed | both intervals closed | paired total gap vs classic (W/T/L) | time to reach classic's final quality (median) |
|---|---|---|---|---|---|
| classic (CP-SAT MAIS + TabuCol cover, no decomposition) | 8.46 | 2/26 | 2/26 | - | 15.5 s (its own run) |
| satdescend (CP-SAT MAIS + minrank B9 SAT descent) | 8.23 | 2/26 | 2/26 | 4/20/2 | - |
| Pincer, round-robin over the toolbox | 5.46 | 13/26 | 6/26 | 23/2/1 | 2.4 s (7x) |
| **Pincer, cost-aware index policy** | **5.35** | **13/26** | **6/26** | **22/4/0** | **1.6 s (11x)** |

On the first test split (35 graphs incl. 9 structured ones, `pincer/results/summary_test_b30.md`): minrank closed
on 22/35 (classic 2, satdescend 8), index beats classic on 31/35 and loses on 1, and matches classic's final
quality after a median 0.67 s instead of 15 s (23x). Examples: planted digraphs n = 50 (planted rank 8): classic
[8, 15], Pincer closes at 8 in 4-9 s; C5, C7, C9, complement of C7, Mycielski M4: Pincer certifies β exactly
(5/2, 7/2, 9/2, 7/3, 11/2), which MAIS + clique cover cannot.

Check against the one-line I + A(G) matrix (24 Nov; it is optimal on the extra_p = 0 planted graphs of `minrank/`): Pincer's planted graphs use extra_p = 0.05, where I + A has rank 36-43 against the planted 8, and I + A beats no method's upper bound on any of the 61 test / test2 graphs. So the numbers above stand. I + A is now part of `quick` and of both baselines anyway.

What does **not** work (yet): the index policy only ties round-robin on the *final* gap (5/16/5 on test2) — its
value is reaching good bounds sooner; the block-minrank LNS never improved a bound; FunSearch (non-LLM GP so far)
improves the greedy MAIS construction by 5 points on held-out graphs but the gain disappears once 1-2 s of
simulated annealing run after it (§4). Random digraphs with n >= 80 stay wide open for everyone (mean minrank gap
~10 for Pincer, e.g. n = 120, p = 0.3: [21, 44] for every method): neither MAIS nor the covers get close, and SAT does not scale there yet.


## 0b. Update, 24 Nov (third session)

- **DRAT-ready certificates.** The r = 9 UNSAT steps behind minrk₂ = 10 for all 8 undirected planted n = 100 graphs are
  exported as gzipped DIMACS (re-solved UNSAT here, 0.9-4.7 s each; header: graph digest, r, mode, fixed set):
  `pincer/certificates/unsat_cnf/` (8.9 MB). Check them on the server with
  `bash scripts/minrank/10_drat_check.sh pincer/certificates/unsat_cnf`. (Building CaDiCaL / drat-trim in this
  container was not permitted, so the DRAT run itself is still to do.)
- **Scheduling fix for slow one-shot tools.** The index policy priced the Shannon LP at its default 0.5 s slice, so
  on C(16,2) it ran the 60 s LP before a 0.04 s fractional cover and ran out of budget with β ∈ [16/3, 6]. One-shot
  tools are now charged their predicted run time (LP: about 1e-5·M² s for M subset orbits), skipped when they cannot
  finish, and the LP runs in a child process killed at the time left (certificate re-checked in the parent). C(16,2)
  and circ(16; 1,4) are now pinned by the index policy too (given ≥ 65 s; the LP alone needs ~61 s).
- **Negative: no cheap predictor of a good SAT fixed set.** 39 maximum independent sets on two planted n = 100 graphs,
  one call "minrk ≤ 10?" each (60 s limit): times 0.9 s to > 60 s; Spearman correlation with 7 set features
  (degrees, coverage, private neighbours) is weak and inconsistent between the graphs (|ρ| ≤ 0.47, the one p < 0.05
  among 14 tests is not robust). Relabelling test: the slowest set stays slow under 4 random vertex relabellings
  (3 × > 60 s, 32 s), the fastest regresses to 3.6-13 s. So some sets are reliably bad, the rest is solver luck;
  the restart portfolio (§0a) is the right response and a FunSearch target over set choice is not worth its cost.
  Data: `pincer/results/fixset_*.jsonl`, scripts `pincer/fixset_{study,analysis,relabel}.py`.

## 0a. Update, 24 Nov (second session)

Positive:
- **New exact minranks on the minrank benchmark.** `planted_n100_r10_e0.1_s2` and `_s6` (undirected, n = 100, listed as
  open in `minrank/results/exact_planted_e01.csv` after 300 s SAT calls) have **minrk₂ = 10**: Pincer proves it in
  34-61 s by asking SAT *from below* (r = 6, 7, 8, 9 UNSAT in < 1 s each, then r = 10 SAT) instead of descending
  from the clique cover (13). Cross-checked with a second solver (Glucose, different independent set): r = 9 UNSAT,
  r = 10 SAT. Bundles: `pincer/certificates/planted_n100_r10_e0.1_s{2,6}.bundle.json`. On all 8 undirected n = 100
  graphs Pincer's lower bound is 10 (driver: 8-9), and one direct SAT call at r = 10 then found rank-10 matrices for
  the remaining four (s1, s4, s5, s7; 1.6-539 s). So **all 8 undirected n = 100 planted e0.1 graphs have minrk₂ = 10**
  (6 of them were open). Every r = 9 UNSAT was re-run with Glucose and an independently found independent set;
  matrices in `pincer/certificates/planted_n100_r10_e0.1_s{1,4,5,7}.rank10.json`.
  Directed n = 100: both close all 8 (Pincer 7-23 s, driver 16-20 s). Table: §5.
- **Symmetry-reduced Shannon LP with exact certificates** (`lp.shannon_bound_sym`): one variable per Aut(G)-orbit of
  subsets; the checker re-verifies the generators and rebuilds the reduced LP. When rounded duals miss the optimum,
  the optimal HiGHS basis is solved exactly in rational arithmetic (`lp.exact_basis_duals`, python-flint). This
  extends exact β from n ≤ 12 to symmetric graphs with n ≤ 16. Structured split (23 graphs, 60 s,
  `pincer/results/summary_structured_b60.md`): **minrank pinned 23/23, β pinned 21/23** (classic 2/23, satdescend
  minrank 16/23, β 2/23). New exact values include β(C₁₃) = 13/2, β(C₁₅) = 15/2, β(C̄ₙ) = n/⌊n/2⌋ for n = 7…15,
  β(Paley 13) = 13/3, β(Petersen) = 5, β(C(13,2)) = 13/3, β(C(16,2)) = 16/3, β(circ(16; 1,4)) = 8.
- **Faster MAIS annealing.** Kept-neighbour bitmasks plus a 5-candidate tournament per move (found while building the
  FunSearch "sa" target, §4b): on the 16 held-out FunSearch test graphs at equal wall time, 0.5 s: 0.936 → 0.988 of
  the old 20 s reference; 2 s: 0.988 → 1.009 (22 wins, 26 ties, 0 losses vs plain SA). End to end the effect is
  small (test2 index: mean minrank gap 5.35 → 5.27, vs classic 24/2/0), because at 30 s the CP-SAT MAIS models
  supply most lower bounds there.

- **SAT restart portfolio.** At a fixed rank, SAT time depends heavily on which maximum independent set is fixed
  (planted n = 100, r = 10: 1.6 s to > 60 s across five sets). `sat_up` / `sat_down` now fix a different set of
  the same size on each retry: undirected planted n = 100 at 120 s, the agent alone closes 5/8 (was 3/8).
Negative:
- **Partial clique cover gives nothing on these graphs** (0/26 test2 graphs improve on the clique cover; no merge of
  two groups ever saves a transmission), so the planned GF(p) vector-linear upper bound (old next step 5) was not
  built. Random digraphs keep β gap = minrank gap.
- The index policy could lose a slow one-shot tool to the budget (C(16,2)); fixed in §0b.
- Mycielski M5 (n = 23) and Paley 17 stay open for β (LP too large / too little symmetry found).
- FunSearch on the SA candidate choice (§4b): GP beats its seed at a fixed move budget but not the library SA at
  equal wall time.
- Fixed on the way: fractional-cover certificates could carry 100-digit denominators (weights now on one grid).

---

## 1. What is computed

Side-information digraph `G` on `n` receivers: arc `i → j` iff receiver `i` knows `x_j`. For every graph Pincer
returns two certified intervals and a certificate bundle that a short checker re-verifies from the graph alone:

```
 MAIS(G)  ≤  Shannon LP  ≤  β(G)  ≤  fractional clique cover            (broadcast rate, rational)
 MAIS(G), ⌈Shannon⌉, UNSAT(r)+1  ≤  minrk₂(G)  ≤  rank of a fitting matrix  (scalar linear F₂ codes, integer)
 β(G) ≤ minrk₂(G) ≤ clique cover
```

| bound | certificate | checker (`pincer/certify.py`) | trusted? |
|---|---|---|---|
| L_β, L_mr: MAIS | vertex set | Kahn's algorithm on the induced subgraph | fully checked |
| L_β: Shannon (polymatroid) LP | rational Lagrange multipliers (+ automorphism generators for the reduced LP) | exact rational weak duality, LP rebuilt from G (generators checked to be automorphisms) | fully checked |
| L_mr: SAT UNSAT at rank r | (r, symmetry mode, fixed set) | re-run with `--recheck-unsat`; CNF exportable for DRAT (minrank/exact) | solver-trusted |
| U_mr, U_β: fitting matrix | n bit-rows | fit pattern + exact F₂ rank | fully checked |
| U_β: fractional clique cover | cliques + rational weights | every clique bidirected, every vertex covered ≥ 1, exactly | fully checked |

Additivity: all four quantities are sums over strongly connected components (SCCs). MAIS and the clique cover
trivially; minrk_q because a fitting matrix is block triangular in a topological order of the SCCs and
`rank [[A, X],[0, B]] ≥ rank A + rank B`; β by Tahmasbi–Shahrasbi–Gohari (capacity region = time sharing over
SCCs, [arXiv:1312.0132](https://arxiv.org/abs/1312.0132)). Tests check the first three by brute force and that the
Shannon LP of the whole graph equals the sum over SCCs.

## 2. The pipeline

```
graph G
  │  Stage 0  decompose (Tarjan SCC) ─ singletons closed at once (β = minrk = 1)
  ▼
per SCC: CompState {[mr_lo, mr_hi], [b_lo, b_hi], certificates}   ← only the checked offer_* methods can change it
  │  Stage 1  quick pass (ms): greedy-FVS acyclic set, DSatur clique cover, cycle-of-cliques cover
  ▼
  │  Stage 2  controller loop until all SCCs closed, budget spent, or no tool is worth its time:
  │      arms = {(SCC, tool) : tool useful for this SCC's state, not retired}
  │      policy picks (arm, time slice)  ── round-robin │ cost-aware index │ LLM (reads index estimates)
  │      tool runs within the slice, proposes certificates → CompState verifies and updates
  │      slice schedule: anytime tools double after a fruitless call; SAT calls double after a timeout
  ▼
  │  Stage 3  bundle = lifted block-diagonal matrix + union acyclic set + per-SCC LP/cover/UNSAT records
  ▼          verify_bundle(G, bundle) from scratch → reported numbers
[L_β, U_β], [L_mr, U_mr] + certificates
```

### Tools (`pincer/tools.py`)

| tool | moves | how | when it pays (measured, §5) |
|---|---|---|---|
| `quick` | both | greedy FVS + DSatur + cycle of cliques | always first |
| `mais_sa` | L | simulated annealing on topological orders (Galinier et al. 2013), bitmask conflicts, 5-candidate tournament | larger sparse SCCs |
| `mais_lazy` | L, MAIS bound | CP-SAT on y_v, cycle cuts added lazily, warm start | proves MAIS on sparse SCCs (n=80: bound 24 vs 41 for the ordering model) |
| `mais_order` | L, MAIS bound | CP-SAT ordering model (existing `minrank/exact/lower.py`) | dense SCCs |
| `cover_tabu` | U | TabuCol on the conflict graph | mid-size undirected |
| `partial_cover` | U | clique cover + disjoint cycles of cliques | rarely beyond `quick` |
| `sat_down` | U, closes mr | is minrk ≤ U−1? SAT with the independent-set symmetry fix (`indep_cols`) | n ≤ ~50 |
| `sat_up` | L_mr, closes mr | is minrk ≤ L? UNSAT raises L, SAT closes | when minrk = MAIS (planted digraph n=50: closes at 8 in 4 s; descending from the cover stalls at 13 after 60 s) |
| `shannon` | L_β | polymatroid LP: full (2ⁿ variables, n ≤ 12) or orbit-reduced under Aut(G) (n ≤ 16); exact basis duals as fallback | odd cycles and their complements, circulants, Paley, Mycielski: β pinned on 21/23 structured graphs |
| `frac_cover` | U_β | LP over cliques, column generation with CP-SAT max-weight-clique pricing | undirected / mutual-heavy SCCs |
| `block_lns` | U | merge blocks of a cover, re-solve exactly by SAT (optional, `--lns`) | **never** in our tests (negative) |
| `mais_evolved` | L | a FunSearch program (`--evolved PATH`) + SA from it | see §4 |

### Policies (`pincer/policies.py`)

All three share the slice schedule, so they differ only in the choice:
- **round-robin**: every useful arm in turn (the "every tool gets an equal share" baseline of the proposal).
- **index** (cost-aware): `index = P(success) × E[gap reduction | success] / slice`. `P(success)` is a Beta
  posterior per (tool, SCC size bucket, directed), shared across components and graphs, with an optimism bonus and
  a 0.7^k discount after k fruitless calls of the same arm. `E[gain | success]` is learned per bucket too and
  capped by what the current state allows. Priors come from round-robin runs on the train split
  (`pincer/priors.json`, `python -m pincer.bench --split train --methods roundrobin --calibrate`). With
  `min_rate > 0` it stops early once no arm is worth its time.
- **LLM**: sees the state, recent calls and the index policy's top-10 candidates with their estimates, replies
  `{"choice": k, "slice": s}` or `{"stop": true}`; any invalid reply or API error falls back to the index choice.
  LLM latency counts against the budget.

## 3. Which earlier ideas are reused (pre-project report and the minrank branch)

| earlier idea (where) | how it appears in Pincer |
|---|---|
| Cost-aware stopping rule "continue while expected gain > λ · cost" (early-exit theory, `theory/THEORY.md`) | the index policy is that rule applied to (SCC, tool) arms; `min_rate` = λ gives the early stop |
| "A batch waits for its hardest input" (straggler) | a graph's interval waits for its hardest SCC: close the cheap SCCs first, then concentrate budget; never let one SCC starve the others |
| Label-free bandit with censored feedback and forced exploration (`bandit/`) | a SAT timeout is a censored runtime observation: the slice doubles, the arm's success estimate drops; optimism bonus = forced exploration |
| One rule tuned once, tested for transfer (`docs/DIRECTION.md`) | priors are calibrated on the train split only and shared across sizes; evaluation on test and on a second, untouched split |
| OpenEvolve sandbox and static checks (`evolve/`, `minrank/evolve/`) | same pattern in `pincer/funsearch/` (isolated `python -I`, import whitelist, memory and time limits) |
| Gate 1: two thirds of candidates crashed silently behind a fallback (`docs/MINRANK_GATE1.md`) | no fallback in the skeleton; a crash scores as a failure and its traceback goes into the next prompt of that island |
| Gate 1: evolving a local search SAT already beats is wasted effort | the FunSearch target is chosen where the measured headroom is (§4), and the evolved program is offered to the controller as one more tool, not as the whole solver |
| Gate 1: evaluation ≈ 2% of an iteration, so the evaluation cascade (Option D) cannot pay | here evaluation is 0.1 s against ~100 s of LLM time, so Option D is not used |
| Random search / BO tied FunSearch (`evolve/baselines.py`) | genetic programming baseline at the same number of evaluated programs (`--proposer gp`) |
| Independent-set symmetry breaking (`minrank/exact/symmetry.py`, WORK.md) | used by `sat_down` / `sat_up` with the best acyclic set found so far, so a better lower bound also speeds up SAT |

## 4. FunSearch component (`pincer/funsearch/`)

**Target and why.** The lower-bound side is where cheap heuristics matter: exact MAIS stops scaling at n ≈ 80
(table in §5), the classic greedy deletion rule (in-degree × out-degree) reaches only 0.85-0.88 of a 20 s SA
reference, and a bigger acyclic set also gives SAT a bigger symmetry-breaking set. Upper bounds were ruled out as
a target by gate 1 (SAT beats any evolved local search there). The evolved object is the FunSearch-style priority
`priority(v, ctx) -> float` inside a fixed greedy-FVS skeleton (`sandbox_runner.py`).

**Loop** (`loop.py`): islands database with score-signature clusters, Boltzmann cluster sampling with a decaying
temperature, short-program preference and periodic island resets (`database.py`, as in Romera-Paredes et al.);
best-shot prompt with two parents renamed `priority_v0`, `priority_v1` and the header of `priority_v2`; the last
error of the island (with the candidate's traceback lines) is appended to the next prompt.
**Evaluation** (`evaluate.py`): static check (only `math`, no I/O, no dunder, no writes to `ctx`), one isolated
`python -I` process per candidate with memory and time limits, every returned set re-checked for acyclicity in the
parent, score = mean |set| / reference over the split. 0.1-0.6 s per candidate.
**Selection**: top-8 by train score *plus the seed* are scored on val, the val winner (possibly the seed) on test.
**Proposers**: `anthropic[:model[:effort]]` (Claude through the `anthropic` SDK, default `claude-opus-5-5`),
`local:<url>|<model>` (the vLLM/Qwen server of `scripts/minrank/05_vllm_serve.sh`), `gp` (genetic programming over
the same feature set: the non-LLM baseline every LLM run must beat at equal evaluated programs).

**Results so far (GP only, 1,500 programs per run, `pincer/funsearch/runs/*/summary.json`):**

| run | train set | train seed → best | val seed → selected | test seed → selected | selected = seed? |
|---|---|---|---|---|---|
| gp_s0 | 14 graphs | 0.849 → 0.901 | 0.879 → 0.879 | 0.862 → 0.862 | yes (over-fitted) |
| gp_s1 | 14 graphs | 0.849 → 0.894 | 0.879 → 0.879 | 0.862 → 0.862 | yes (over-fitted) |
| gp42_s2 | 42 graphs | 0.874 → 0.899 | 0.879 → 0.890 | 0.862 → 0.873 | no |
| gp42_s3 | 42 graphs | 0.874 → 0.916 | 0.879 → 0.909 | **0.862 → 0.916** | no |

gp42_s3 wins on 12 of 16 test graphs (2 ties, 2 losses), including n = 200 and 400 (trained on n ≤ 100) and an
undirected family never trained on. With 14 training graphs GP over-fitted exactly as the early-exit FunSearch did;
a 3x larger training set fixed it (evaluation is cheap, so there is no reason to keep it small).

**Deployment check** (`deploy_check.py`, the way the controller would use it: greedy, then SA):

| SA seconds after the greedy start | seed + SA | evolved + SA | W/T/L (16 graphs × 3 SA seeds) |
|---|---|---|---|
| 0 | 0.862 | 0.916 | 36/6/6 |
| 0.5 | 0.952 | 0.957 | 17/20/11 |
| 2 | 0.989 | 0.983 | 5/28/15 |

So the evolved construction is better, but SA erases the head start within about a second. It is available as
the opt-in tool `mais_evolved` (`--evolved pincer/funsearch/runs/gp42_s3/best.py`), not a default. Lesson for the
next target (§7): score candidates **after** the downstream solver, at the deployment budget.

### 4b. Second target: the candidate choice inside simulated annealing (24 Nov)

Following the lesson above, the evolved object is now part of the downstream solver: `score(f)` ranks the 5
candidates SA samples per move (`sandbox_sa.py`; f = conflicts at the best slot, kept neighbours, degrees,
temperature, step). Fitness = set size after a fixed move budget of 40 n (load-independent) on harder graphs
(n = 120-250, splits `train_sa` / `val_sa`), seed = fewest conflicts first.

| | train | val | test (move budget) |
|---|---|---|---|
| seed (fewest conflicts) | 0.955 | 0.954 | 0.983 |
| GP, 500 programs (`runs/gpsa_s0`) | 0.978 | 0.985 | 0.998 |
| GP, 500 programs (`runs/gpsa_s1`) | 0.983 | 0.974 | 0.984 |

Deployment check at equal **wall time** on the 16 test graphs × 3 seeds (`deploy_check_sa.py`, and the formula
ported into `pincer.mais.sa_order(score=...)`):

| SA seconds | library default | gpsa_s0 in the library (W/T/L) | gpsa_s1 in the library (W/T/L) |
|---|---|---|---|
| 0.5 | 0.987-0.989 | 0.988 (11/26/11) | 0.987 (12/26/10) |
| 2 | 1.008 | 1.003 (4/30/14) | 1.001 (2/33/13) |

Verdict: the evolved score beats its seed at a fixed move budget, but not the library at equal time; the library
keeps the default. What this target did produce is the seed itself: building it showed that a 5-candidate
tournament plus kept-neighbour bitmasks beats plain SA clearly (§0a), and that is now the default `mais_sa`.
Two GP targets, two honest negatives for evolution against a tuned baseline; the LLM runs (§7.1) must clear the
same bar (deployment budget, equal wall time).


## 5. Experiments

All at 30 s per graph on a 4-core container (2-3 jobs in parallel), single runs (seed 0). Splits: random families
use seeds 0-1 for calibration (`train`), 5-6 (`test`, structured graphs included) and 7-8 (`test2`, untouched until
the final run). The index policy was corrected once after the first `test` pass (its gain model credited
`frac_cover` with half the β gap; it spent whole budgets there on n = 120); `test2` was run only after that fix.

- Families: random digraphs G(n, p) n 30-120, random undirected n 40/80, planted low-rank digraphs and undirected
  graphs (rank 8, n 50), clustered digraphs with 1-8 SCCs (local side information), and structured graphs (odd
  cycles, complement of C7, Mycielski M4, cycle power, circulant, Kneser K(7,2), Paley 13).
- Baselines: `classic` = the Pincer proposal's prototype pipeline (CP-SAT ordering MAIS for half the budget,
  TabuCol for the other half, whole graph); `satdescend` = the same lower bound plus minrank's B9 (TabuCol + SAT
  descent with the original encoding), the strongest single minrank solver of `minrank/`.
- Full tables: `pincer/results/summary_test_b30.md`, `pincer/results/summary_test2_b30.md`; raw rows (with every
  tool call) in `pincer/results/*.jsonl`.

Tool attribution on test2 (sum of gap reduction after the quick pass, index policy): `mais_order` 71,
`sat_up` 64, `mais_sa` 36, `sat_down` 17, `cover_tabu` 16, `mais_lazy` 8, `frac_cover` 6.2, `shannon` 0.5,
`partial_cover` 0. `sat_up` (SAT from below with the independent-set fix) is the single most useful minrank tool;
the Shannon LP matters only on the small structured graphs of the first test split.


**Planted extra_p = 0.1 benchmark of `minrank/` (n = 100, planted rank 10), Pincer index at 120 s per graph vs the
exact driver's run (`minrank/results/exact_planted_e01.csv`, 300 s per SAT call).** Raw rows:
`pincer/results/bench_planted_e01_n100_b120.jsonl`.

| graphs | exact driver | Pincer (120 s) |
|---|---|---|
| directed, 8 | all closed at 10, 16-20 s | all closed at 10, 7-23 s |
| undirected s0 | 10 (UNSAT 128 s) | 10 (closed at 45 s) |
| undirected s2, s6 | open [8, 13], [9, 13] | **10 (closed at 61 s, 34 s)** |
| undirected s3 | 10 (UNSAT 111 s) | [10, 12] (missed the rank-10 matrix) |
| undirected s1, s4, s5, s7 | open [8, 13] | open [10, 13]; then one direct SAT call "≤ 10?" (`pincer/probe_rank.py`, 600 s limit): **SAT in 539, 74, 14, 1.6 s → 10** |

Why it works: the driver starts at the clique cover and asks "≤ 12?", which times out; Pincer's `sat_up` asks
"≤ L?" with L the current lower bound, where the independent-set fix pins 80-90 % of the variables, so each UNSAT
step takes < 1 s and the first SAT step (r = 10) 3-5 s. Suggestion for the driver: ascend from α instead of probing
α once and then descending.

**Structured split (23 vertex-transitive / symmetric graphs, 60 s, `pincer/results/summary_structured_b60.md`):**

| method | minrank pinned | β pinned | mean β gap |
|---|---|---|---|
| classic | 2 | 2 | 1.22 |
| satdescend | 16 | 2 | 1.22 |
| Pincer round-robin | 23 | 21 | 0.14 |
| Pincer index | 23 | 20 | 0.17 |

Earlier measurements that shaped the toolbox (same container):

| question | measurement |
|---|---|
| exact MAIS models | 20 s each, n=80 p=0.3: lazy cycles bound 24-26 vs ordering 41-43; p=0.6: ordering proves 9, lazy bound 10; n=300 p=0.1: SA 3 s finds 68-70 vs ordering 56-58 |
| minrank SAT encodings | n=40 random digraph, 60 s: independent-set fix 21 → 18; original encoding (`sat_descend`) stays 21 |
| block-minrank LNS | 400 merge attempts on 6 graphs (n 40-150): 0 accepted |
| Shannon LP time | n=11: 2 s (interior point) vs 88 s (dual simplex); n=12: 9-17 s; orbit-reduced: C13 0.5 s, C15 9-13 s, circ(16;1,4) 32-51 s (dual simplex stalls > 300 s there, so it is only a time-limited fallback) |
| partial clique cover | greedy group merging on the 26 test2 graphs: 0 improvements over the clique cover |


## 6. Reproduce

```bash
bash scripts/pincer/00_test.sh                                   # install extras, 66 tests
PYTHONPATH=. python -m pincer.agent minrank/bench/graphs/cycle_5.json --budget 20 --out /tmp/c5_bundle.json
PYTHONPATH=. python -m pincer.certify minrank/bench/graphs/cycle_5.json /tmp/c5_bundle.json
SPLIT=train METHODS=roundrobin CALIBRATE=1 bash scripts/pincer/01_bench.sh      # priors for the index policy
SPLIT=test2 bash scripts/pincer/01_bench.sh                                     # the comparison in §5
PROPOSER=gp ITER=1500 bash scripts/pincer/02_funsearch.sh                       # non-LLM FunSearch baseline
```

## 7. Next steps (concrete, with pass criteria)

1. **LLM FunSearch on the server** (needs the A6000 or an API key):
   `PROPOSER="local:http://127.0.0.1:8000/v1|Qwen/Qwen2.5-Coder-32B-Instruct-AWQ" ITER=300 bash scripts/pincer/02_funsearch.sh`
   and `PROPOSER=anthropic ITER=300 ...`; compare with `PROPOSER=gp ITER=300` (same number of evaluated programs).
   Pass: val_selected ≥ gp's + 0.01 and test wins on ≥ 10/16 graphs. Cost: log `llm_usage` from summary.json.
2. ~~Re-target FunSearch at deployment conditions~~ done for SA (§4b, negative for GP). The fixed-set choice for
   SAT was studied instead of evolved (§0b): mostly solver luck, so no target there.
3. **LLM controller**: `bash scripts/pincer/03_llm_controller.sh` (test2; the budget includes LLM latency). Pass: paired total-gap wins > losses against `index`. If not, the LLM's place is offline (writing
   tools and priors), not in the loop.
4. ~~Bigger Shannon LPs~~ done for n ≤ 16 (§0a); M5 (n = 23) needs an LP over fewer inequalities (e.g. only
   inequalities on orbit representatives of small sets) — open.
5. ~~Partial clique cover over GF(p)~~ measured, no gain (§0a). Better β upper bounds for random digraphs need
   minrk over other fields (`minrank/exact/sat_fq.py`) or composite / vector codes — open.
6. **DRAT** for the exported CNFs: `bash scripts/minrank/10_drat_check.sh pincer/certificates/unsat_cnf` on the
   server (§0b).
7. **Ascending SAT search in the minrank driver** (done by hand here for the n = 100 planted graphs: all 8 settled);
   in Pincer, let `sat_up` keep a long slice when its question is r = L and the lower bound has just risen.
8. **Scale study**: 300 s budgets and n = 200-500 to find where the tool mix changes.

