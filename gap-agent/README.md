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
for the next slice. Its policy is a Q-function learned offline by **fitted Q-iteration** (batch reinforcement
learning, gradient-boosted trees) on real runs: round 0 uses random switching, later rounds the agent itself with
15 % random choices. The reward of a slice is the gap closed, weighted by the time still left, so the return of a
run is exactly the drop in the **gap integral** (area under gap(t)/gap(0) over the budget, lower is better).

## Run it

```
python gap-agent/train.py           # ~70 min on 3 cores -> models/agent.pkl
python gap-agent/bench.py run       # unseen test graphs, all controllers -> results/bench.jsonl
python gap-agent/bench.py report    # -> RESULTS.md
python -m pytest gap-agent/test_gap.py
```

Results: `RESULTS.md` (very large graphs) and `results/medium/RESULTS.md` (first attempt, 500-2000 receivers).
