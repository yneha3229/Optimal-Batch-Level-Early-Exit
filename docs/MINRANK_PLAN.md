# LLM-Guided Search for the Minrank of Graphs, with Cost-Aware Early Exit for Candidate Evaluation

**Author:** Divyanshu Gupta
**Team:** Divyanshu Gupta, Kushagra Gupta, Neha Yadav (IE624, IIT Bombay, Prof. Manjesh K. Hanawal)
**Date:** 20 November 2026
**Status:** Proposal and implementation plan. Nothing in this document has been run. The literature notes come from
a web search on 20 November 2026; check each claim against the paper before quoting it.

---

## 0. Summary

- **The problem:** given a graph `G` with 20–200 vertices, find a matrix that fits `G` over `F_2` with rank as low as
  possible. That rank is `minrk_2(G)`, and it is the length of the best scalar linear index code for `G`. The problem
  is NP-hard, and it is NP-hard to approximate within any constant factor. Exact values are known only for graphs up
  to about 10 vertices.
- **What we do:** use LLM-guided evolutionary search (OpenEvolve, already in this repository under `evolve/`) to
  discover search heuristics for minrank. The heuristics must beat strong classical baselines at equal compute on
  held-out graphs.
- **Why it can work:** a candidate answer (a matrix) can be verified in milliseconds. Every result is therefore a
  certificate, and a wrong guess costs nothing. Lower bounds (maximum acyclic induced subgraph, independence number)
  are computed exactly by ILP, so every result also has a known gap to optimal.
- **Where the early-exit project fits:** most of the cost of LLM-evolved search is in *evaluating* candidate programs
  on many test graphs. Deciding when to stop evaluating a batch of candidates is the same problem as batch-level
  early exit. Our cost-derived rule, certified selection and bandit move over with little change.
- **Deliverables:**
  1. a certified minrank benchmark for `n = 20–200` (upper bound, lower bound, certificate per graph);
  2. LLM-evolved heuristics compared with all baselines at equal wall-clock;
  3. new exact values or explicit separations on structured graphs (Option B);
  4. a cost-aware early-exit evaluator that reaches the same search quality with less evaluation compute.

---

## 1. Background: index coding and minrank

### 1.1 Index coding
- A server holds messages `x_1, …, x_n`, each one bit (or one symbol of a field `F_q`).
- Receiver `i` wants `x_i` and already knows a subset `S_i` of the other messages (its *side information*).
- The server broadcasts as few symbols as possible so that every receiver can decode its message.
- **Side-information graph** `G`: a directed graph on `{1, …, n}` with an edge `i → j` iff `j ∈ S_i`. If side
  information is symmetric (`j ∈ S_i ⇔ i ∈ S_j`) the graph is undirected.

### 1.2 Minrank
- A matrix `A ∈ F_q^{n×n}` **fits** `G` if
  - `A_ii ≠ 0` for all `i` (over `F_2`: `A_ii = 1`), and
  - `A_ij = 0` whenever `i ≠ j` and `j ∉ S_i` (no edge `i → j`).
  - Entries on edges are free.
- `minrk_q(G) = min { rank_q(A) : A fits G }`.
- **Theorem (Bar-Yossef, Birk, Jayram, Kol 2006):** the shortest *scalar linear* index code over `F_2` has length
  exactly `minrk_2(G)`. Encoding: pick a basis of the row space of an optimal `A`; broadcast `⟨row, x⟩` for each basis
  row. Receiver `i` reconstructs `⟨A_i, x⟩ = x_i + Σ_{j ∈ S_i} A_ij x_j` and cancels the known terms.
- Linear codes are not always optimal: Lubetzky and Stav (2007) show non-linear codes can be much shorter. This
  project is about the linear quantity `minrk`.

### 1.3 Standard bounds (undirected `G`, `Ḡ` its complement)
- `α(G) ≤ minrk_q(G) ≤ χ(Ḡ)`:
  - **Lower:** an independent set `I` of `G` gives a submatrix of `A` on `I` that is the identity (no free
    entries), so its rank is `|I|`.
  - **Upper:** a clique cover of `G` (a colouring of `Ḡ`) with `k` cliques gives a rank-`k` fitting matrix: rows of a
    clique are all equal to the all-ones vector on that clique. Broadcast the XOR of each clique.
- **Directed:** `MAIS(G) ≤ minrk_q(G)`, where `MAIS` is the size of the largest induced acyclic subgraph. Upper
  bounds come from cycle covers, partial clique covers (Birk–Kol) and local chromatic number (Shanmugam–Dimakis–
  Langberg).
- Lovász `ϑ` and the fractional clique cover are other upper bounds on the broadcast rate. Haviv (2018) shows `ϑ`
  can be constant while `minrk` is polynomial, so `ϑ` is not a good proxy for minrank.

### 1.4 What is known (search of 20 November 2026)

| Fact | Source |
|---|---|
| Deciding `minrk ≤ 3` is NP-complete over any field | Peeters 1996 |
| NP-hard to approximate within any constant factor | Chawin–Haviv, STACS 2023 (arXiv 2301.00732) |
| `minrk(G(n,p)) = Θ(n / log n)` w.h.p.; clique cover matches up to constants | Golovnev–Regev–Weinstein 2017 (arXiv 1607.04842) |
| All index coding instances with ≤ 5 messages solved; linear optimal there | Arbabjolfaei et al.; Ong (arXiv 1401.7369, 1606.05982) |
| Binary minrank of all graphs up to 10 vertices, via SAT; 10 vertices took about 4.3 days | arXiv 1304.6832 |
| SDP algorithm for graphs with small minrank | Chlamtáč–Haviv (arXiv 1107.1958) |
| Rank-minimisation heuristics (alternating projections, over the reals), up to 13% better than colouring | Huang–El Rouayheb 2015 (arXiv 1504.06844) |
| Matrix-completion heuristic over finite fields | Esfahanizadeh–Lahouti–Hassibi 2014 (arXiv 1408.3046) |
| Minrank-critical edges reduce instance size | arXiv 1901.00164, 1901.04926 |
| No ML, RL or LLM work on minrank found | (gap) |
| LLM evolution (OpenEvolve) settled Zarankiewicz numbers, under $30 per instance | arXiv 2605.01120 |

**Consequence for our claims.** We cannot claim near-optimal answers in general (hardness of approximation). We claim
better certified upper bounds on specific instances and families, with a measured gap to a proven lower bound, at
equal compute. Do not claim "LLM solves minrank with high probability".

**Name clash.** "MinRank" in cryptography (multivariate MinRank, Gröbner bases, support minors) is a different
problem. Filter those papers out.

---

## 2. Problem statement

> **Input:** a graph `G` (undirected first, directed second) with `n ∈ [20, 200]` vertices and a compute budget `T`.
> **Output:** a matrix `A` over `F_2` that fits `G`, as low in rank as possible, plus a certified lower bound `L(G)`.
> **Score:** `rank(A)`; reported with the gap `rank(A) − L(G)` and the ratio `rank(A) / χ̂(Ḡ)` against the best
> clique cover found.

- Field: `F_2` first. `F_3`, `F_4` only if time allows (Option B separations).
- Every reported number is backed by a file holding `A` and a verification script anyone can run.

---

## 3. Options

### Option A (core): LLM-evolved minrank heuristics (FunSearch-style)

**Idea.** OpenEvolve evolves a program `solve(G, budget) → A`, or better, only the key decision function inside a
fixed search skeleton. The program is trained on one set of graphs and scored on held-out graphs. The result is a
heuristic that generalises, not a single construction.

**What the LLM evolves (in order of preference):**
1. **A priority / move-selection function** inside a fixed min-conflicts search (Section 4.3). Small search space;
   FunSearch had its successes this way.
2. **A constructive heuristic:** order in which vertices are added and how rows are merged (a generalised LDG /
   clique-cover greedy).
3. **The whole solver.** Only if 1 and 2 plateau.

**Fitness for a candidate program:**
- Run it on each training graph with a fixed time budget (for example 5 s per graph).
- Score per graph: `(χ̂(Ḡ) − rank(A)) / χ̂(Ḡ)`, the relative improvement over the best clique cover; 0 if the program
  crashes, times out or returns a matrix that does not fit.
- Fitness = mean over the training graphs. Report the held-out mean separately and never select on it.

**Benchmark families** (fixed instance lists with seeds, stored in the repository):
- `G(n, p)` undirected, `n ∈ {20, 30, 50, 100, 200}`, `p ∈ {0.3, 0.5, 0.7}`, 20 instances each.
- Directed `D(n, p)`, same sizes (second phase).
- Structured: Kneser `K(m, k)` (for example `K(7,2)` 21 vertices, `K(8,2)` 28, `K(8,3)` 56, `K(9,3)` 84), Paley
  graphs `P(q)` (`q = 29, 37, 41, 49, 53, 61, …`), circulants, powers of cycles and their complements, Mycielski
  graphs, generalised Kneser graphs.
- Instances from index coding papers, where available.
- Split: train / validation / held-out by instance, and also by family (train on random graphs, test on Kneser) to
  check generalisation.

**Kill criterion.** If the best evolved heuristic does not beat simulated annealing / tabu search (Section 4.3) at
equal wall-clock on the held-out graphs at two budgets or more, stop and report a negative result. Our own FunSearch
attempt in the early-exit project tied random search and overfitted; "Evolution or Illusion?" (arXiv 2609.19799)
warns of the same thing.

### Option B (stretch): new values and separations on structured graphs

1. **Exact minrank for specific graphs with 20–50 vertices.** Use Option A's best upper bound `r` as a warm start;
   prove `minrk ≥ r` by showing `minrk ≤ r − 1` is UNSAT (Section 4.4), with symmetry breaking from the graph's
   automorphism group. Targets: Kneser, Paley, circulants, vertex-transitive graphs, where the symmetry helps the
   solver.
2. **Separations with small explicit graphs:**
   - large gap between `minrk_2(G)` and the clique cover number `χ(Ḡ)` at small `n`;
   - graphs where `minrk_2 ≠ minrk_3` (field dependence);
   - graphs where `minrk` is far above `α` / `MAIS` (lower bounds are loose).
   An explicit new graph with a provable gap is a mathematical result, and it is checkable.
3. **Random graphs at practical sizes:** the constant in `Θ(n / log n)` at `n = 20–200` is open in practice. Exact
   values or tight bounds for `G(n, 1/2)` at `n = 20–40` give the first empirical curve.

### Option C (only if A shows structure): learned guidance inside a solver
A small GNN, or the LLM itself, proposes which entries of `A` to free or the order of elimination; local search
finishes. More ML engineering and more risk. Do it only if Option A finds heuristics that clearly exploit structure.

### Option D (the early-exit connection): cost-aware early exit for candidate evaluation
This is how the existing project's theory enters. Details in Section 5. It sits alongside Option A and needs Option
A's evaluation traces.

---

## 4. Technical details

### 4.1 Representation and verification
- Store graphs as adjacency bitsets (one Python `int` or `numpy.uint64` array per vertex). File format: edge list
  plus a JSON header (`n`, directed, family, seed).
- **Fit check:** `A_ii = 1` and `A_ij = 0` for every non-edge `i ≠ j`. `O(n^2)`.
- **Rank over `F_2`:** Gaussian elimination on bit-packed rows, `O(n^3 / 64)`. For `n = 200` this is well under a
  millisecond in numpy, so verification is never the bottleneck. Option: the `galois` package, but a bitset version
  is faster and has no dependency.
- **Certificate file:** `A` as bit rows plus the graph hash. `verify.py G.json A.txt` prints `fits: yes, rank: r`.
- Unit tests: random fitting matrices of known rank; clique-cover matrices; reproduce known minrank of all graphs on
  ≤ 7 vertices (enumerate with nauty `geng`) against the SAT solver of Section 4.4.

### 4.2 Lower bounds
- **Undirected:** `α(G)` by ILP or CP-SAT (OR-Tools). Exact up to `n ≈ 200` for medium density; for dense or large
  instances, report the best bound found within the time limit.
- **Directed:** `MAIS(G)` by ILP: binary `y_v` for vertices kept and cycle-elimination constraints added lazily (or
  an ordering formulation with integer positions).
- Optional, stronger: SAT proof that `minrk ≤ r − 1` is infeasible (Section 4.4); only feasible for small `n` or
  highly symmetric graphs.

### 4.3 The search space of a heuristic (the important insight)
`rank(A) ≤ r` iff `A = U Vᵀ` with `U, V ∈ F_2^{n×r}`. Write `u_i, v_i ∈ F_2^r` for the rows. Then `A` fits `G` iff
- `⟨u_i, v_i⟩ = 1` for every `i`, and
- `⟨u_i, v_j⟩ = 0` for every non-edge `i → j`.

So **`minrk_2(G) ≤ r` is a constraint satisfaction problem with `2nr` bits**. It looks much like graph colouring:
- Clique cover with `r` cliques is the special case `u_i = v_i = e_{colour(i)}`.
- Undirected graphs allow the symmetric restriction `u_i = v_i` (an orthogonal representation over `F_2`). It is
  smaller but not always optimal; try both.

Heuristics on this representation:
- **Min-conflicts / tabu search at fixed `r`** (as TabuCol does for colouring): cost = number of violated
  constraints; move = flip one bit of some `u_i` or `v_i`; incremental delta cost from the vertex's non-neighbours;
  tabu list; restarts. When the cost reaches 0, record `r`, then try `r − 1` starting from the solution with one
  coordinate removed.
- **Simulated annealing**, same moves.
- **Constructive:** start from the best clique cover (`r = χ̂`) and repeatedly try to remove a coordinate by
  repairing conflicts.
- The LLM evolves the move-selection / priority function in this skeleton (Option A, item 1).

### 4.4 Exact solving (Option B, and validation)
- Encode `∃ U, V: ⟨u_i, v_i⟩ = 1, ⟨u_i, v_j⟩ = 0 (non-edges)` as CNF with XOR constraints:
  - auxiliary `z_{ijk} = u_{ik} ∧ v_{jk}` (Tseitin, 3 clauses each);
  - XOR of `z_{ij·}` = target bit. Use CryptoMiniSat (native XOR clauses) or Kissat with XOR chains.
- **Symmetry breaking:** column permutations of `U, V` (lexicographic order on columns); `GL(r, F_2)` acting on the
  factorisation (fix `U` on a basis set of rows to a reduced form); graph automorphisms via nauty / `pynauty`.
- Variables: `2nr + (#non-edges + n)·r` auxiliaries. For `n = 50, r = 15` that is about 1,500 primary variables;
  dense graphs (few non-edges) are easier, as the 10-vertex enumeration paper found.
- Warm start from the best heuristic solution; ask the solver only for `r − 1`.

### 4.5 Baselines (all must be implemented and run at the same budgets)

| # | Baseline | Type | Notes |
|---|---|---|---|
| B1 | Greedy clique cover (DSatur colouring of `Ḡ`) | upper bound | the standard bound |
| B2 | TabuCol / strong colouring of `Ḡ` | upper bound | best clique cover we can afford; denominator of the score |
| B3 | LDG (Birk–Kol least-difference greedy) | upper bound | the classic index-coding heuristic |
| B4 | Partial clique cover / local chromatic number | upper bound | directed graphs |
| B5 | Alternating projections / matrix completion over `F_2` | upper bound | Huang–El Rouayheb idea moved to `F_2`; Esfahanizadeh et al. |
| B6 | Min-conflicts tabu on `U Vᵀ` (Section 4.3) | upper bound | **the strongest expected baseline** |
| B7 | Simulated annealing on `U Vᵀ` | upper bound | |
| B8 | Random search / random restarts | upper bound | sanity floor |
| B9 | SAT/ILP with a time limit (anytime) | upper and lower | CryptoMiniSat / CP-SAT |
| B10 | SDP (Chlamtáč–Haviv) | upper bound | optional; only for small-minrank instances |
| L1 | `α(G)` / `MAIS(G)` by ILP | lower bound | |
| E1 | OpenEvolve evolved heuristic | ours | Option A |

Protocol:
- Budgets per graph: 1 s, 10 s, 60 s, 600 s wall-clock on one CPU core (report all, as "Evolution or Illusion?"
  recommends).
- 5 seeds per (method, graph, budget). Report mean, best, and a 95% interval over instances.
- Anytime curves: best rank vs time.
- LLM cost reported separately (tokens, dollars, GPU-hours) and never hidden in "the heuristic is fast".

### 4.6 The OpenEvolve set-up
- Reuse `evolve/evaluator.py`, `evolve/sandbox_eval.py` and `evolve/config.yaml` as templates.
- **Seed program:** the B6 min-conflicts skeleton with a simple priority function.
- **Evaluator:** run the candidate on the training graphs under a time budget per graph, in a sandbox; verify every
  returned matrix; return fitness. Log per-graph scores for every candidate (needed for Option D).
- **Models:** an API model, or a local coder model served with vLLM on the A6000. Budget guide: the Zarankiewicz paper
  reports under $30 per instance. Plan for $50–200 for development runs (estimate).
- **Anti-overfitting:** held-out graphs never used in selection; a family-held-out test; compare against B6/B7 given
  the same total compute *including* the evolution run, and also the same compute at deployment.

---

## 5. Option D in detail: early exit for candidate evaluation

### 5.1 The mapping

| Batch-level early exit (current project) | Evaluating evolved candidates |
|---|---|
| A batch of `B` inputs | A generation of `B` candidate programs evaluated in parallel |
| Layers `1 … L` | Test graphs `1 … L`, evaluated in a fixed order (small and cheap first) |
| Confidence of input `j` at exit `e` | Candidate `j`'s running score after `e` graphs, relative to the current best |
| Calibration table `P(correct at e′ | conf at e)` | `P(candidate beats the best on all graphs | partial score after e graphs)`, estimated from earlier generations |
| Individual exit | Drop one candidate early |
| Collective stop | Stop the whole generation (no candidate left is likely to win) |
| Cost `c(n) = a + b·n` | Cost of one evaluation round over `n` live candidates on a worker pool: fixed set-up `a` (process start, graph load) plus `b` per candidate; measure it |
| Certified selection (Learn-then-Test) | Guarantee: P(drop a candidate that would have beaten the best) ≤ δ |
| Label-free bandit | Best-arm identification; successive halving and Hyperband are the baselines |

### 5.2 The claim
> A cost-derived stopping rule for candidate evaluation, with a statistical guarantee, reaches the same best minrank
> as full evaluation with several times less evaluation compute, and does better than OpenEvolve's fixed-threshold
> cascade and successive halving at equal compute.

OpenEvolve's built-in cascade uses fixed thresholds (for example "drop below 0.5 after the first stage", "drop below
0.8 × best"). One paper estimates about 8× savings from cascades (arXiv 2608.14555), but nothing derives the
thresholds from cost or gives a guarantee. That is the gap.

### 5.3 How to run it
1. Log full evaluation traces from Option A runs: score of every candidate on every graph, plus timing.
2. Offline replay simulator, modelled on `simulator/` in this repository: replay a policy on the traces and count
   evaluation compute and the final best candidate.
3. Policies: full evaluation; OpenEvolve cascade; successive halving; Hyperband; racing (Hoeffding races); our rule
   (`theory/THEORY.md` rule with the evaluation cost); our rule with certified thresholds (code in
   `experiments/local_hw.py`).
4. Online check: plug the best policy into the live evaluator for one full evolution run and compare the final
   minrank with a run on full evaluation at the same wall-clock.
5. Metrics: evaluation CPU-hours to reach a given fitness; final held-out fitness at equal compute; rate of "lost a
   winner" events (should be ≤ δ).

### 5.4 Precondition
This only pays if evaluation dominates cost. Measure it in the first Option A run: time in LLM calls versus time in
the evaluator. With dozens of graphs of `n ≥ 50` per candidate it should dominate. If it does not, Option D drops to a
short section.

---

## 6. Reading list

Read Tier 1 first (everyone). Tiers 2–4 by workstream (Section 8).

### Tier 1: foundations (everyone, week 1)
1. Bar-Yossef, Birk, Jayram, Kol. *Index Coding with Side Information.* FOCS 2006; IEEE Trans. Inf. Theory 2011.
   Minrank = optimal linear index code.
2. Arbabjolfaei, Kim. *Fundamentals of Index Coding.* Foundations and Trends in Communications and Information
   Theory, 2018. **The survey; read chapters on bounds and linear codes.**
3. Birk, Kol. *Informed-Source Coding-on-Demand (ISCOD).* INFOCOM 1998; IEEE Trans. Inf. Theory 2006. LDG heuristic.
4. Lubetzky, Stav. *Non-linear Index Coding Outperforming the Linear Optimum.* FOCS 2007. Where linear is not enough.
5. Romera-Paredes et al. *Mathematical Discoveries from Program Search with Large Language Models* (FunSearch).
   Nature, 2024.
6. Novikov et al. *AlphaEvolve: A Coding Agent for Scientific and Algorithmic Discovery.* arXiv 2506.13131.
7. Bhan, Nobili, Raghuraman, Langer. *New Bounds for Zarankiewicz Numbers via Reinforced LLM Evolutionary Search.*
   arXiv 2605.01120. **Closest template to Option A/B.**

### Tier 2: minrank algorithms, bounds, hardness (WS1, WS2)
8. Peeters. *Orthogonal Representations over Finite Fields and the Chromatic Number of Graphs.* Combinatorica 1996.
9. Chawin, Haviv. *Improved NP-Hardness of Approximation for Orthogonality Dimension and Minrank.* STACS 2023,
   arXiv 2301.00732.
10. Golovnev, Regev, Weinstein. *The Minrank of Random Graphs.* APPROX/RANDOM 2017, arXiv 1607.04842.
11. Huang, El Rouayheb. *Index Coding and Network Coding via Rank Minimization.* ITW 2015, arXiv 1504.06844.
12. Esfahanizadeh, Lahouti, Hassibi. *A Matrix Completion Approach to Linear Index Coding Problem.* arXiv 1408.3046.
13. Chlamtáč, Haviv. *Linear Index Coding via Semidefinite Programming.* CPC 2014, arXiv 1107.1958.
14. Shanmugam, Dimakis, Langberg. *Local Graph Coloring and Index Coding.* ISIT 2013.
15. *Polynomial Time Algorithm for Min-Ranks of Graphs with Simple Tree Structures.* arXiv 1304.6832. SAT encoding and
    the enumeration of all graphs up to 10 vertices.
16. Ong. *Linear Codes are Optimal for Index-Coding Instances with Five or Fewer Receivers.* arXiv 1401.7369; and
    arXiv 1606.05982 (finite alphabets).
17. Minrank-critical edges: arXiv 1901.00164 and arXiv 1901.04926.
18. Haviv. *On Minrank and the Lovász Theta Function.* APPROX/RANDOM 2018, arXiv 1802.03920.
19. Golovnev, Haviv. *The (Generalized) Orthogonality Dimension of (Generalized) Kneser Graphs.* CCC 2021,
    arXiv 2002.08580. Option B targets.
20. Alon, Hassidim, Lubetzky, Stav, Weinstein. *Broadcasting with Side Information.* FOCS 2008.
21. Galinier, Hertz. *A Survey of Local Search Methods for Graph Coloring* (TabuCol and relatives). C&OR 2006. The
    search template for B2/B6.

### Tier 3: LLM-guided search (WS3)
22. *Mathematical Exploration and Discovery at Scale.* arXiv 2511.02864. What AlphaEvolve does and does not do
    unaided.
23. *Evolution or Illusion? Rethinking Evaluation in LLM Evolutionary Search.* arXiv 2609.19799. **Evaluation
    protocol; read before running anything.**
24. *LEVI: Stronger Search Architectures Can Substitute for Larger LLMs in Evolutionary Search.* arXiv 2605.09764.
25. *LLM-Guided Search for Deletion-Correcting Codes.* arXiv 2504.00613. FunSearch with local batched inference.
26. *Reinforced Generation of Combinatorial Structures: Hardness of Approximation.* arXiv 2509.18057.
27. OpenEvolve repository and README: github.com/algorithmicsuperintelligence/openevolve.

### Tier 4: early stopping, bandits, certified selection (WS4)
28. *Discovering KV Cache Eviction Policies via LLM-Guided Program Evolution.* arXiv 2608.14555. Cascade evaluation
    and its savings.
29. Karnin, Koren, Somekh. *Almost Optimal Exploration in Multi-Armed Bandits.* ICML 2013 (successive halving).
30. Jamieson, Talwalkar. *Non-stochastic Best Arm Identification and Hyperparameter Optimization.* AISTATS 2016.
31. Li et al. *Hyperband.* JMLR 2018.
32. Maron, Moore. *Hoeffding Races.* NeurIPS 1994.
33. Angelopoulos, Bates, Candès, Jordan, Lei. *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk
    Control.* 2021.
34. Our own: `theory/THEORY.md`, `docs/DIRECTION.md`, `bandit/bandit.py`, `experiments/local_hw.py`.

### Tools
- Graphs: `networkx`, nauty `geng` / `pynauty` (enumeration, automorphisms).
- SAT: CryptoMiniSat (`pycryptosat`, native XOR), Kissat; ILP / CP: OR-Tools CP-SAT.
- `F_2` linear algebra: own bitset code; `galois` for checks.
- Evolution: OpenEvolve; vLLM for a local model.

---

## 7. Proposed code layout (new top-level directory `minrank/`)

| Path | Contents |
|---|---|
| `minrank/core/` | graph I/O, bitset `F_2` rank, fit check, certificate read/write, `verify.py` |
| `minrank/bench/` | instance generators (random, Kneser, Paley, circulant, …), fixed instance lists, splits |
| `minrank/baselines/` | B1–B10 |
| `minrank/exact/` | α / MAIS ILP, SAT encoding, symmetry breaking |
| `minrank/evolve/` | OpenEvolve config, seed program, evaluator with trace logging |
| `minrank/cascade/` | trace format, replay simulator, stopping policies (Option D) |
| `minrank/results/` | per-instance table: best upper bound, method, lower bound, certificate path |
| `minrank/certificates/` | matrices backing every reported upper bound |
| `minrank/tests/` | verifier and solver tests, known small values |

The early-exit code stays where it is; Option D imports from `simulator/`, `bandit/` and `experiments/local_hw.py`
where useful.

---

## 8. Plan of work, in parallel

### 8.1 Workstreams

| WS | Name | Suggested owner | Depends on | Produces |
|---|---|---|---|---|
| WS0 | Core and benchmark (shared) | everyone, days 1–4 | none | verifier, graph format, instance lists, test suite |
| WS1 | Heuristic baselines | Member 1 (Kushagra suggested) | WS0 | B1–B8 at all budgets; the baseline table |
| WS2 | Exact solving and lower bounds | Member 2 (Neha suggested) | WS0 | L1, B9, SAT encoding, validation on small graphs; Option B |
| WS3 | LLM evolution (Option A) | Member 3 (Divyanshu suggested; knows `evolve/`) | WS0; seed from WS1 (B6) | evolved heuristics, held-out results, evaluation traces |
| WS4 | Cost-aware evaluation (Option D) | Divyanshu with WS3; anyone free | traces from WS3 (synthetic traces until then) | replay simulator, policies, savings table |

Owners are suggestions; reassign freely. WS1 and WS2 are fully independent of each other. WS3 needs only the
verifier and a seed program to start (it can begin with a trivial seed and switch to B6 when ready). WS4 can build
its simulator on synthetic traces in week 2 and switch to real traces when WS3 produces them.

### 8.2 Timeline (weeks from start; adjust to the calendar)

| Week | WS0 | WS1 baselines | WS2 exact / lower | WS3 evolution | WS4 early exit |
|---|---|---|---|---|---|
| 1 | Verifier, bitset rank, graph format, generators, tests; **all read Tier 1** | read Tier 2 (11, 12, 14, 21) | read Tier 2 (8, 9, 10, 15, 19) | read Tier 3; set up OpenEvolve and the model | read Tier 4 |
| 2 | Freeze instance lists and splits | B1, B2, B3, B8 | α / MAIS ILP (L1); SAT encoding | evaluator with trace logging; smoke run with trivial seed | trace format; replay simulator on synthetic traces |
| 3 | — | B6, B7 (min-conflicts / SA on `UVᵀ`), B5 | validate SAT against all graphs ≤ 7 vertices; anytime B9 | first real run on small graphs (`n = 20–30`); measure LLM vs evaluator time | port the `THEORY.md` rule and certified selection to the simulator |
| **Gate G1 (end of week 3)** | | baseline table at 1 s / 10 s for `n ≤ 50` | lower bounds for all `n ≤ 100`; SAT validated | evolution runs end to end | simulator replays real traces |
| 4 | — | B4 (directed), B10 optional; run all budgets, all sizes | Option B: exact values for structured graphs `n = 20–40` | main runs: seed B6, train on random graphs | policies vs OpenEvolve cascade, successive halving, Hyperband (offline) |
| 5 | — | directed benchmark | Option B: separations search with WS3's best heuristic | held-out and family-held-out tests; equal-compute comparison against B6/B7 | online run of the best policy inside the live evaluator |
| **Gate G2 (end of week 5)** | | full baseline table | Option B first results | **kill or continue** (Section 3, Option A kill criterion) | **savings measured** (keep if ≥ 2× at equal final quality) |
| 6 | — | support: rerun baselines on any new instances | Option B: push largest exact values; certify the results table | second round: directed graphs, evolve whole solver if first round plateaued | sensitivity: budget, δ, generation size |
| 7 | all: results table, certificates, figures, write-up | | | | |

### 8.3 Gates and kill criteria

| Gate | Pass | Kill / change |
|---|---|---|
| G1 | Verifier passes tests; SAT reproduces known minrank of all graphs ≤ 7 vertices; evolution runs end to end; baseline table for `n ≤ 50` exists | Verifier or SAT disagree with known values: stop everything else and fix |
| G2a (Option A) | Evolved heuristic beats B6/B7 on held-out graphs at ≥ 2 budgets, at equal wall-clock | Ties or loses: report the certified benchmark and the negative result; move effort to Option B |
| G2b (Option D) | Same final best fitness with ≥ 2× less evaluation compute; "lost winner" rate ≤ δ | < 1.3× or the guarantee fails: one short section, no more effort |
| G2c (Option B) | At least one new exact value or one explicit separation at `n ≥ 20` | None: keep as a table of best-known bounds |

### 8.4 Synchronisation
- One results table (`minrank/results/`) with columns: graph, `n`, family, lower bound, best upper bound, method,
  budget, certificate path, verified (yes/no). Every new best goes in with its certificate, verified by `verify.py`
  in a test.
- Weekly 30-minute meeting at each gate; short status line per workstream in `logs/minrank_status.log`, as in the
  early-exit project.
- The verifier and instance lists are frozen at the end of week 2. Changes after that need everyone's agreement,
  otherwise results stop being comparable.

---

### 8.5 Compressed 7-day schedule (adopted 21 November 2026; full scope kept)

Everything in this document is in scope: Options A, B, C and D, all baselines B1–B10 and L1, directed graphs, the
reading list, and the early-exit integration (replay simulator, cost-derived rule, certified selection, bandit /
successive-halving baselines). The schedule below only orders the work. Claude writes code alongside the team.

| Day | Claude (code) | Kushagra: baselines | Neha: exact + Option B | Divyanshu: evolution + early exit |
|---|---|---|---|---|
| 1 | `minrank/` core, benchmark, B1/B2/B6/B7/B8, L1, SAT, tests | read Tier 1–2 | read Tier 1–2 | OpenEvolve + vLLM on the A6000 |
| 2 | B3 LDG, B5 alternating projections, B4, evolve evaluator with trace logging | full baseline sweep (server CPUs) | SAT vs brute force; L1 on all graphs | evolution smoke run with B6 seed |
| 3 | Option D replay simulator and policies (cost rule, certified, SH, Hyperband, cascade) | 60 s sweep; tables | Option B: exact values (Kneser, Paley, Mycielski) | first real evolution runs (train split) |
| **G1** | | | | |
| 4 | online cascade in the live evaluator; B10 SDP; Option C prototype | directed benchmark | Option B: separations, `minrk_2` vs `minrk_3` | held-out runs at equal compute |
| 5 | fixes, ablations | directed sweep | certify Option B results | cascade savings; Option C if A shows structure |
| **G2** | | | | |
| 6 | figures, results table | tables | tables | ablations |
| 7 | write-up (everyone) | | | |

## 9. Risks

| Risk | Effect | Mitigation |
|---|---|---|
| Clique cover is already asymptotically optimal on random graphs | Small gains on `G(n,p)` | Emphasise structured families; report gap to lower bound, not only the ratio |
| Evolved heuristic overfits training graphs | False positives | Held-out and family-held-out splits; equal-compute comparisons |
| Strong classical baselines (B6) leave little headroom | Option A negative | It is still a certified benchmark; Option B uses B6 as warm start anyway |
| SAT does not scale beyond `n ≈ 30–50` | Few exact values | Symmetric graphs only; report best bounds otherwise |
| Evaluation is not the cost bottleneck | Option D weak | Measure at week 3 before investing |
| `F_2` results do not move to larger fields | Narrower claim | State the field everywhere; `F_3` only for separations |
| LLM API cost | Budget overrun | Local model on the A6000 via vLLM; cascade evaluation (Option D) cuts evaluator cost |

---

## 10. What a finished project claims (if gates pass)
1. A certified benchmark of minrank upper and lower bounds for graphs with 20–200 vertices, with certificates.
2. LLM-evolved heuristics that beat the strongest classical baselines at equal compute on held-out graphs (or a clear
   negative result with the reason).
3. New exact minrank values or explicit separations on structured graphs.
4. A cost-derived, certified early-exit rule for evaluating candidates in LLM evolutionary search, with measured
   savings over fixed cascades and successive halving.
