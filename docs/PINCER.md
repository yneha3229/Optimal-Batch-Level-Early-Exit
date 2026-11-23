# Pincer: an agentic pipeline that squeezes the broadcast rate and the minrank between certified bounds

Code: [`pincer/`](../pincer/) (built on the solvers in [`minrank/`](../minrank/)). Scripts: [`scripts/pincer/`](../scripts/pincer/).
Proposal it implements: *Pincer — An agent that squeezes the broadcast rate of an index-coding problem between two
certified bounds* (IE624, Nov 2026). Status as of 23 Nov 2026: everything below runs and is tested (58 tests); the
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

What does **not** work (yet): the index policy only ties round-robin on the *final* gap (5/16/5 on test2) — its
value is reaching good bounds sooner; the block-minrank LNS never improved a bound; FunSearch (non-LLM GP so far)
improves the greedy MAIS construction by 5 points on held-out graphs but the gain disappears once 1-2 s of
simulated annealing run after it (§4). Random digraphs with n >= 80 stay wide open for everyone (mean minrank gap
~10 for Pincer, e.g. n = 120, p = 0.3: [21, 44] for every method): neither MAIS nor the covers get close, and SAT does not scale there yet.


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
| L_β: Shannon (polymatroid) LP | rational Lagrange multipliers | exact rational weak duality, rebuilt from G | fully checked |
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
| `mais_sa` | L | simulated annealing on topological orders (Galinier et al. 2013) | larger sparse SCCs |
| `mais_lazy` | L, MAIS bound | CP-SAT on y_v, cycle cuts added lazily, warm start | proves MAIS on sparse SCCs (n=80: bound 24 vs 41 for the ordering model) |
| `mais_order` | L, MAIS bound | CP-SAT ordering model (existing `minrank/exact/lower.py`) | dense SCCs |
| `cover_tabu` | U | TabuCol on the conflict graph | mid-size undirected |
| `partial_cover` | U | clique cover + disjoint cycles of cliques | rarely beyond `quick` |
| `sat_down` | U, closes mr | is minrk ≤ U−1? SAT with the independent-set symmetry fix (`indep_cols`) | n ≤ ~50 |
| `sat_up` | L_mr, closes mr | is minrk ≤ L? UNSAT raises L, SAT closes | when minrk = MAIS (planted digraph n=50: closes at 8 in 4 s; descending from the cover stalls at 13 after 60 s) |
| `shannon` | L_β | polymatroid LP, 2ⁿ variables, HiGHS interior point + crossover, n ≤ 12 | odd cycles, Mycielski, small SCCs with fractional β |
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

Earlier measurements that shaped the toolbox (same container):

| question | measurement |
|---|---|
| exact MAIS models | 20 s each, n=80 p=0.3: lazy cycles bound 24-26 vs ordering 41-43; p=0.6: ordering proves 9, lazy bound 10; n=300 p=0.1: SA 3 s finds 68-70 vs ordering 56-58 |
| minrank SAT encodings | n=40 random digraph, 60 s: independent-set fix 21 → 18; original encoding (`sat_descend`) stays 21 |
| block-minrank LNS | 400 merge attempts on 6 graphs (n 40-150): 0 accepted |
| Shannon LP time | n=11: 2 s (interior point) vs 88 s (dual simplex); n=12: 9-17 s |


## 6. Reproduce

```bash
bash scripts/pincer/00_test.sh                                   # install extras, 58 tests
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
2. **Re-target FunSearch at deployment conditions.** Score = acyclic set after greedy + 1 s SA (or the SA move
   selection itself), and a minrank target: the choice of the symmetry-breaking set / column order for `sat_up`,
   scored by SAT time to decide r = L on the hard planted/random graphs of `minrank/`. Pass: beats the default on
   val at the deployment budget.
3. **LLM controller**: `bash scripts/pincer/03_llm_controller.sh` (test2; the budget includes LLM latency). Pass: paired total-gap wins > losses against `index`. If not, the LLM's place is offline (writing
   tools and priors), not in the loop.
4. **Bigger Shannon LPs**: restrict to Aut(G)-invariant solutions (orbit variables) and expand the dual back for the
   certificate; target SCCs with n ≤ 20 (structured graphs). Pass: C11, C13, Mycielski M5 certified exactly.
5. **Upper bounds on β for digraphs**: vector-linear partial clique cover over GF(p) with an explicit code
   certificate (general decodability checker: rank of B restricted to unknown columns). On random digraphs the β
   gap equals the minrank gap today.
6. **DRAT** for the UNSAT records (tooling plan in WORK.md T4) so no bound is solver-trusted.
7. **Scale study**: 300 s budgets and n = 200-500 to find where the tool mix changes.

