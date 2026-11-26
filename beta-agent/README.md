# Pincer demo: estimate the broadcast rate of a network you draw

A self-contained demo of the project's core idea. Draw an index-coding network, and an agent squeezes the
broadcast rate β between two certified bounds:

```
lower bound L = largest acyclic induced subgraph (MAIS)   ≤   β   ≤   upper bound U = smallest clique cover
```

If L = U, β is known exactly. This folder does not use any other code in the repository.

## Run it

- **Locally:** open `index.html` in a browser (no server, no install). Everything runs offline; no language model.
- **Tests:** `node --test beta-agent/engine.test.js` (exact solvers checked against brute force on 300 random graphs,
  certificate checks, known cases).
- **Benchmark:** `node beta-agent/bench/suite.js` (writes the 46 graphs), `node beta-agent/bench/run.js` (greedy,
  plain exact search, Pincer), `python beta-agent/bench/cpsat.py` (Google OR-Tools CP-SAT), then
  `python beta-agent/bench/analyze.py` (writes `bench/RESULTS.md` and the table the page shows).

## Benchmark (46 graphs, 10-60 receivers, 30 s limit, one CPU core per method)

| method | both bounds optimal | missed transmissions | median time to answer | worst time to answer |
|---|---|---|---|---|
| Greedy heuristics | 28 / 35 of 46 | 40 | 2 ms | 20 ms |
| Plain exact search | 45 / 46 of 46 | 1 | 15 ms | 30.1 s |
| CP-SAT (Google OR-Tools) | 46 of 46 | 0 | 14 ms | 1.2 s |
| **Pincer agent** | **46 of 46** | **0** | **2 ms** | 1.1 s |

Pincer finds the optimum of both bounds on every graph and is the fastest to its answer (3-5x faster than CP-SAT on
graphs up to 40 receivers, on par above). Its weakness: on 50-60 receivers its in-browser exact search needs up to
30 s to *prove* the answer optimal, where CP-SAT needs at most 1.2 s. Full tables: `bench/RESULTS.md`.
Should a fine-tuned LLM do this instead? See `LLM_ANALYSIS.md` (short answer: no).

## Using the page

| Tool | What it does |
|---|---|
| Receiver (R) | click empty space to add receiver *i* (it wants message x*i*) |
| Knows → (K) | click A, then B: receiver A already has B's message (arc A → B) |
| Know each other ↔ (B) | click two receivers that have each other's messages |
| Move (M), Erase (E) | drag receivers; click a receiver or an arrow to remove it |
| Load example | five friends in a ring, a directed cycle, caching groups, a random network, … |
| Run agent | the agent works until L = U or both bounds are proven optimal |
| Compare methods | runs greedy, plain exact search and the agent on your graph and shows bounds and times side by side |
| Steps / Replay | every decision with its reason and the bounds after it; click a step or press Replay to watch the acyclic set and the cliques change on the graph |
| Graph as JSON | copy a drawn network, or paste one: `{"receivers": n, "knows": [[i, j], ...]}` (1-based) |

## What is inside

`engine.js` (plain JavaScript, also loads in Node):

- **Lower-bound tools:** `lower_greedy` (keep receivers that cannot lie on a cycle, delete the most connected one,
  repeat), `lower_local_search` (simulated annealing on a topological order), `lower_exact` (branch and bound;
  proves the acyclic set is maximum when it finishes in time).
- **Upper-bound tools:** `upper_greedy` (DSatur colouring of the receivers that do not know each other),
  `upper_local_search` (min-conflicts search for a cover with one group fewer), `upper_exact` (backtracking from
  the current U down to L; proves the cover is smallest when it finishes).
- **Decomposition:** both bounds add up over strongly connected components, so every tool solves the components
  separately.
- **Checks:** a tool only proposes a vertex set or a list of groups. The session checks it (no directed cycle /
  every group is a clique of two-way pairs, every receiver covered once) before it can change L or U.
- **The agent:** a `Session` holds the bounds and the step log; the planner decides each step and records why:
  greedy bounds first, then exact search with a short time slice that doubles only after it runs out, local search
  between attempts to improve the incumbent, and a stop as soon as L = U (which by itself proves both optimal).

`index.html`: the drawing board, the bound meter, the agent controls, the step log with replay, the two
certificates in words (the acyclic set, and the broadcast plan as one XOR per clique), the live comparison and the
benchmark table.

## Reading the result

- **β = k:** the bounds meet; k transmissions are necessary and sufficient.
- **L ≤ β ≤ U with both proven:** both bounds are as good as these two methods can be on this graph, so they
  cannot decide β here. Example: five friends in a ring, 2 ≤ β ≤ 3, while the true value is 2.5. Stronger tools
  (LP bounds, as in the research code under `pincer/`) are needed for such graphs.
- **L ≤ β ≤ U, not proven:** the time budget ran out; run again or simplify the graph.

Exact search is exponential: it finishes quickly up to about 25-30 receivers per component; larger drawings get
heuristic bounds.
