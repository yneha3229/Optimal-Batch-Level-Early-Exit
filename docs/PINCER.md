# Pincer: an agentic pipeline that squeezes the broadcast rate and the minrank between certified bounds

Code: [`pincer/`](../pincer/) (built on the solvers in [`minrank/`](../minrank/)). Scripts: [`scripts/pincer/`](../scripts/pincer/).
Proposal it implements: *Pincer — An agent that squeezes the broadcast rate of an index-coding problem between two
certified bounds* (IE624, Nov 2026). Status as of 23 Nov 2026: everything below runs and is tested (58 tests); the
LLM parts are implemented and tested with a scripted backend, but **no LLM has been run in the loop yet** (this
container has no API key or GPU); see §8 for exactly what to run on the server.

RESULTS_PLACEHOLDER

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

| tool | moves | how | when it pays (measured, §6) |
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

FUNSEARCH_PLACEHOLDER

EXPERIMENTS_PLACEHOLDER

## 7. Reproduce

```bash
bash scripts/pincer/00_test.sh                                   # install extras, 58 tests
PYTHONPATH=. python -m pincer.agent minrank/bench/graphs/cycle_5.json --budget 20 --out /tmp/c5_bundle.json
PYTHONPATH=. python -m pincer.certify minrank/bench/graphs/cycle_5.json /tmp/c5_bundle.json
SPLIT=train METHODS=roundrobin CALIBRATE=1 bash scripts/pincer/01_bench.sh      # priors for the index policy
SPLIT=test2 bash scripts/pincer/01_bench.sh                                     # the comparison in §6
PROPOSER=gp ITER=1500 bash scripts/pincer/02_funsearch.sh                       # non-LLM FunSearch baseline
```

NEXT_PLACEHOLDER
