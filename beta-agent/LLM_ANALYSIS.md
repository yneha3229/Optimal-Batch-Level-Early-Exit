# Would a fine-tuned LLM (or a network of LLMs) beat the existing methods on this task?

**Short answer: no, not on time and not on accuracy.** We do not recommend spending GPU time on fine-tuning an LLM to
compute these bounds. The offline agent in this folder (a deterministic, cost-aware planner over certified solvers)
already gets every bound right on the benchmark and answers about 300 times faster than the fastest an LLM could
physically write its answer. Details and the evidence below.

## The task, stated precisely

For a side-information digraph, output (a) a large acyclic set of receivers (lower bound L) and (b) a partition of
the receivers into few mutual cliques (upper bound U), ideally optimal and with proof. Both are NP-hard. A useful
method must be **accurate** (close to the optimum), **fast**, and **trustworthy** (an answer that is wrong must never
be reported).

## Evidence 1: what the current methods already achieve (measured here)

46 graphs, 10-60 receivers, 30 s limit, one CPU core per method (`bench/RESULTS.md`):

| method | both bounds optimal | missed transmissions | median time to answer | worst time to answer |
|---|---|---|---|---|
| Greedy heuristics | 28 / 35 of 46 | 40 | 2 ms | 20 ms |
| Plain exact search | 45 / 46 of 46 | 1 | 15 ms | 30.1 s |
| CP-SAT (Google OR-Tools) | 46 of 46 | 0 | 14 ms | 1.2 s |
| **Pincer agent (offline)** | **46 of 46** | **0** | **2 ms** | 1.1 s (proofs on n = 50-60: up to 30 s) |

There is no accuracy left to gain on this range: the exact optimum is reached on every graph. The only room left is
speed of proofs on the largest graphs, and the known cure for that is a stronger exact solver (CP-SAT), not a
language model.

## Evidence 2: the speed floor of any LLM (computed from the benchmark graphs)

An LLM must read the graph (prompt tokens) and write the answer (output tokens) before anything can be checked.
On our suite the graph is 347 prompt tokens at the median (2,170 at most) and the answer about 78 tokens. With
typical single-request speeds, and assuming a single attempt with no reasoning text (the most favourable case):

| model and hardware (assumed speeds: prompt / output tokens per s) | median time per graph | median for n > 40 |
|---|---|---|
| 1.5B model on a laptop CPU (200 / 20) | 5.7 s | 13.6 s |
| 1.5B model on an A6000 GPU (5,000 / 150) | 0.61 s | 1.1 s |
| 8B model on an A6000 GPU (3,000 / 60) | 1.5 s | 2.6 s |

Pincer's median time to the optimal answer is 0.002 s. Even the fastest LLM setup is about 300 times slower per
graph before its answer is checked. A "network of LLMs" multiplies these times (each agent call is another
prompt and answer). Reasoning text, which the literature finds necessary for accuracy on such tasks, multiplies them
again.

## Evidence 3: accuracy of LLMs on NP-hard graph problems (published)

- **GraphArena** (ICLR 2025; 10 tasks, 10,000 problems, 10 LLMs): models do clearly worse on NP-complete tasks
  (including maximum independent set and maximum clique, the undirected cases of our two bounds) than on polynomial
  ones; on NP problems they mostly fall back to greedy-like answers and often produce *hallucinated* answers that
  look right but are infeasible. Chain-of-thought helps little, and for large NP instances "fine-tuning alone is
  insufficient". [arXiv 2407.00379](https://arxiv.org/abs/2407.00379)
- **GraphWiz** (KDD 2024): instruction-tuned and DPO-aligned 7B-13B open models (Llama 2, Mistral) on 72,785 training
  samples over nine graph tasks reach about 65 % average accuracy (GPT-4: 43.8 %). That is far below the 100 %
  that exact solvers reach on such sizes, and the NP-complete tasks are the hard part. [arXiv 2402.16029](https://arxiv.org/abs/2402.16029)
- **GraphThought / Llama-GT** (2025 preprint): an 8B Llama-3 model fine-tuned on reasoning traces for graph
  optimisation (including vertex cover and colouring) gets *close to* Gurobi on some tasks, by the authors' own
  account; it does not beat the solver. [arXiv 2502.11607](https://arxiv.org/abs/2502.11607)
- Related: [arXiv 2509.16865](https://arxiv.org/abs/2509.16865) trains LLMs as end-to-end combinatorial solvers
  with solver-generated labels (Gurobi as the ground truth). [arXiv 2510.08808](https://arxiv.org/abs/2510.08808)
  fine-tunes small LoRA models to *estimate* properties such as the chromatic number; generalisation beyond the
  training sizes is limited.

(These pages could not be opened from this environment; the statements above come from their abstracts and search
summaries, not from their full tables.)

## Evidence 4: our own LLM experiments in this project

- The first FunSearch run with a local **Qwen2.5-Coder-32B** (on the A6000, 300 iterations, about 9.5 GPU hours) did
  not produce a heuristic better than its hand-written seed; generation took about 106 s per program
  (`docs/MINRANK_GATE1.md`).
- Two later evolution targets (greedy acyclic set, simulated-annealing move choice) improved on their seeds only
  until compared with a tuned classical method at equal wall time, where the gain vanished (`docs/PINCER.md` §4).

## Verdict

| question | answer |
|---|---|
| Can a fine-tuned LLM compute L and U more accurately than the current methods? | No: they are already optimal on every benchmark graph; published LLM results stay below exact solvers on NP-hard graph tasks. |
| Faster? | No: about 0.6 s per graph at best on a GPU vs 0.002 s for the agent, before any checking. |
| Trustworthy without a checker? | No: LLMs produce infeasible "hallucinated" answers; any LLM output would still need our verifier and an exact proof. |
| Explainable? | The offline agent already explains every step (why it calls each tool, what it found, what certifies each bound). An LLM's explanation of its own answer is not a proof. |

## What we recommend instead

1. **Keep the agent offline and LLM-free** (as the demo now does): a cost-aware planner over certified tools. It is
   deterministic, explainable step by step, and measured above.
2. **Close the one weakness that remains** (proofs on n ≥ 50) with a stronger exact tool, for example CP-SAT as one of
   the agent's tools in an offline Python version. That would make the agent at least as good as CP-SAT on proofs
   while keeping its 2 ms answers.
3. If a learned component is wanted for the course, the defensible place is a **tiny learned policy** that predicts
   which tool and which time slice to use from graph features (size, density, components). It costs microseconds,
   can be trained on the benchmark logs without a GPU, and is explainable (feature importances). It can only
   save time where the planner wastes it, so expect small gains.

## If you still want to test an LLM first-hand

A fair pilot would take about a day with the A6000: generate 20,000 small graphs with optimal answers from our exact
solvers, fine-tune Qwen2.5-1.5B-Instruct (LoRA, then reinforcement learning with the verifier as the reward), and
score it on `bench/suite.json` for feasibility, optimality and time against the table above. Pass bar: at least as
many optimal answers as the agent at lower time. Based on everything above we expect it to fail that bar; this
environment cannot download models (huggingface.co is blocked), so the pilot would run on your server.
