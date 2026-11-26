# Gap agent: a learned agent that tightens β bounds on very large graphs

For index coding on a side-information digraph G (arc i → j: receiver i has message j):

```
largest acyclic set  ≤  β(G)  ≤  smallest clique cover
   (lower bound)                     (upper bound)
```

Both are NP-hard, and on graphs with thousands of receivers no exact solver gets anywhere in seconds. This agent
runs several **anytime strategies** on a shared search state and **learns when to switch** between them, so the gap
between the two bounds shrinks as fast as possible. Every bound it reports carries a certificate (the acyclic set,
checked by a topological sort; the cover, checked clique by clique).

## Strategies (`gap_agent/strategies.py`, compiled inner loops in `kernels.py`)

| side | strategy | what it does |
|---|---|---|
| lower | `lo_greedy` | fresh randomised greedy acyclic sets (restarts) |
| lower | `lo_anneal` | simulated annealing on the current set, high temperature (Galinier et al. insertion moves) |
| lower | `lo_descent` | the same moves at a low temperature |
| lower | `lo_kick` | drop a random fifth of the set, refill greedily |
| upper | `hi_dsatur` | randomised DSatur colourings of the conflict graph |
| upper | `hi_greedy` | Culberson's iterated greedy on the best cover (never worse) |
| upper | `hi_tabu` | TabuCol: try to fit everyone into one clique fewer |

A stopped strategy resumes where it left off (the tabu search keeps its colouring, conflict table and tabu list).

## The agent (`gap_agent/agent.py`, `gap_agent/learn.py`)

Every 0.25 s the agent reads the state (graph size and densities, both bounds, time left, and for each strategy:
time spent, time since it last helped, failures in a row, total gain, gain rate last time) and picks the strategy
for the next slice.

**Learned prior + online adaptation** (`LearnedBandit`, the agent we report). A model trained offline on 330 runs over
2000-8000-receiver graphs (gradient-boosted trees, 35 000 slices) predicts, from the current state, how fast each
strategy will close the gap in the next slice. That prediction counts as one pseudo-observation and is combined
with the rates the strategy actually showed in *this* run (last 8 uses), plus an exploration bonus; the agent runs
the strategy with the best estimate. So it starts with what it learned across graphs (e.g. tabu search pays on the
cover side, annealing early on the acyclic side) and corrects itself as soon as a strategy stalls on this graph.

Also tried, and kept for comparison: an **offline policy** learned by fitted Q-iteration (batch reinforcement
learning on the same runs). It does worse: with no online correction it over-commits to the cover side and
starves the acyclic set when a graph differs from the training graphs (e.g. 10 000 receivers).

Model selection (the prior strength, and FQI vs learned prior) was done on separate validation graphs
(`validate.py`), never on the test graphs.

## Results on unseen very large graphs (`RESULTS.md`)

27 test graphs with 3000, 6000 and 10 000 receivers (random digraphs, planted cliques, layered digraphs; training
never saw more than 8000), 30 s each, one core, same opening and seed for every controller. Gap integral = area under
gap(t)/gap(0) (lower = gap closed sooner).

| controller | mean gap integral | final gap (sum) | best on |
|---|---|---|---|
| **learned agent (learned prior + online)** | **0.724** | **17 245** | **14/27** |
| online bandit (UCB, adapts within the run, nothing learned) | 0.736 | 17 567 | 4/27 |
| fixed pair (annealing + tabu, alternating) | 0.741 | 17 710 | 1/27 |
| offline policy (fitted Q-iteration) | 0.767 | 19 400 | 6/27 |
| random switching | 0.770 | 18 291 | 0/27 |
| round robin | 0.771 | 18 560 | 2/27 |

Graph by graph, the learned agent beats the online bandit on 22/27 (gap integral -0.012 ± 0.003) and the fixed
pair on 23/27; it reaches the bandit's 30 s bounds in a median **20.5 s** and the fixed pair's in **18.7 s** (about
a third less time), and random switching's in 9.6 s.

What the numbers also say, plainly:
- The gains from smarter switching are real but moderate (a few percent of the gap integral, a third less time to
  the same bounds). Switching cannot close what the bounds cannot: on dense random graphs the largest acyclic set
  (~15) and the smallest clique cover (~500 at 5000 receivers) are far apart for structural reasons, so most of the
  remaining gap is the distance between the two bounds themselves, not search effort.
- On 500-2000 receivers (`results/medium/RESULTS.md`) every strategy reaches its plateau within seconds and the time
  split hardly matters (a sweep of the split gave curves flat to within seed noise), so no controller has much to
  gain there. Switching pays where the graph is too large for the strategies to converge within the budget.
- The standard exact solver gets nowhere at this size: CP-SAT on the textbook models (`bench.py cpsat`) did not
  finish *building* either model within the 30 s on any of the nine 3000-receiver test graphs, so it improved
  neither bound (gap integral 1.0 on all nine, vs 0.60 for the learned agent).

## v2: the parallel agent with structure analysis and lambda_2 (`gap_agent/parallel.py`, `RESULTS2.md`)

Three workers run at the same time; a coordinator keeps the best certified bound per strongly connected component
(SCC) from any of them and re-plans as results come in.

- **Search worker**: the learned agent above. When parts of the graph are finished it projects its solutions onto
  the rest and continues there only.
- **Structure worker** (`structure.py`, `exact.py`): splits the graph into SCCs (relabelling SCC by SCC makes the
  adjacency matrix block triangular, and all bounds, beta and lambda_2 add up over the blocks); shrinks each SCC
  for the acyclic-set bound with classic kernel rules (LOOP, IN0/OUT0, IN1/OUT1 contraction, CORE, PIE; brute-force
  checked on 600 random graphs to keep the exact optimum); then solves pieces exactly from the smallest up (CP-SAT
  on the kernel, CP-SAT on each mutual component for the cover, and the full lambda-agent, i.e. Shannon LP +
  vector-linear GF(2) codes, on SCCs of at most 10 receivers).
- **lambda_2 worker** (`lam.py`): GF(2) block codes better than the clique cover: SAT-found merges of a few cliques
  into one block of lower rank, cycles of cliques (k cliques in a ring, each knowing the next, cost k - 1), and
  directed cycles carved out of the cover with the leftover receivers re-covered exactly; it picks among the three
  moves by their success so far.
- **Re-planning**: a worker with nothing useful left (no structure to exploit, or no code savings for a while) is
  handed the open part of the graph and the best solutions so far, and becomes another search worker.

Note on "an isomorphic graph that is easier": relabelling alone changes no bound and no difficulty. What helps is
what the structure worker does: an equivalent *smaller* instance (kernel) and a *split* into independent pieces.

Results (30 unseen graphs, 3000-10000 receivers, 30 s; v1 = the single-core learned agent; portfolio = three v1
copies on three cores; mean gap integral, lower is better; "pinned" = beta = lambda_2 proven):

| family | v1, 1 core | v1 portfolio, 3 cores | **v2, 3 workers** |
|---|---|---|---|
| sparse side information | 0.102, pinned 0/6 | 0.090, pinned 1/6 | **0.085, pinned 2/6** |
| communities (clustered) | 0.641 | 0.622 | **0.619** (final gap 3742 vs 3842) |
| random dense | 0.879 | 0.876 | 0.876 |
| planted cliques | 0.804 | **0.799** | 0.801 |
| layered | 0.703 | **0.612** | 0.630 |

- On structured graphs v2 is best. On both 10 000-receiver sparse graphs it **proves beta = lambda_2 exactly**
  (7089 and 6909) in under 8 s: the kernel rules shrink the acyclic-set problem to nothing (5851 -> 0 and 7397 -> 0
  receivers) and the cover splits into small mutual components solved exactly. No single-search set-up proves it.
- On dense unstructured graphs there is nothing to find (one SCC, no rule fires, no code saving): v2 hands its
  helper cores to search after about 5 s and ends level with the portfolio, a little behind on layered and planted
  graphs because those cores start searching late.
- lambda_2 beyond the clique cover: code savings were found only on the clustered graphs (up to a few per graph);
  on dense random-like graphs local GF(2) codes do not beat the clique cover, as theory suggests (the GF(2) minrank
  of a random graph is of the same order as its clique cover number).

## Run it

```
python gap-agent/train.py           # ~60 min on 3 cores -> data/transitions.pkl, models/agent.pkl (FQI)
python gap-agent/validate.py        # model selection on validation graphs; fits models/rate.pkl
python gap-agent/bench.py run       # unseen test graphs, all controllers -> results/bench.jsonl
python gap-agent/bench.py report    # -> RESULTS.md
python gap-agent/bench2.py run       # v2 vs v1 vs portfolio -> results/bench2.jsonl
python gap-agent/bench2.py report    # -> RESULTS2.md
python -m pytest gap-agent/test_gap.py
```

Results: `RESULTS.md` (very large graphs) and `results/medium/RESULTS.md` (first attempt, 500-2000 receivers).
