# minrank: LLM-guided search for the minrank of graphs

Find matrices over F_2 that fit a side-information graph with rank as low as possible (= the shortest scalar linear
index code), with a certificate for every upper bound and a certified lower bound for every graph. Full plan,
background and reading list: [`docs/MINRANK_PLAN.md`](../docs/MINRANK_PLAN.md).

## Quick start

```bash
bash scripts/minrank/00_setup.sh            # pip install + unit tests (EXTRAS=1 also installs openevolve, cvxpy, vllm)
```

CPU part, on the server (everything resumable; keep `WORKERS` at or below the number of physical cores):

```bash
WORKERS=32 nohup bash scripts/minrank/run_cpu_all.sh > logs/minrank_cpu_all.log 2>&1 &
```

GPU part (Option A, needs an idle A6000):

```bash
GPU=5 PORT=8000 bash scripts/minrank/05_vllm_serve.sh        # serve the coder model (leave running)
PORT=8000 ITER=300 bash scripts/minrank/06_evolve.sh          # OpenEvolve run
bash scripts/minrank/07_eval_heldout.sh                       # best program on val/test at 1/10/60 s
bash scripts/minrank/08_cascade_replay.sh                     # Option D on the run's evaluation traces
bash scripts/minrank/09_aggregate.sh                          # tables + figures
```

## Scripts

| Script | What it does | Output |
|---|---|---|
| `00_setup.sh` | install requirements, run tests | |
| `01_bench.sh` | regenerate the frozen benchmark (already committed) | `minrank/bench/graphs/` |
| `02_baselines.sh` | sweep: graph x method x budget x seed | `minrank/results/baselines.csv`, `minrank/certificates/` |
| `03_lower_bounds.sh` | certified lower bounds (CP-SAT) | `minrank/results/lower_bounds.csv` |
| `04_exact.sh` | Option B: SAT descent to proven optima on structured graphs | `minrank/results/exact.csv` |
| `04c_exact_rerun.sh` | Option B, second pass: only the `call_timeout` graphs of `exact.csv`, 3600 s per SAT call | `minrank/results/exact_long.csv` |
| `04b_separations.sh` | Option B: search for explicit separations (gap to clique cover, F_2 vs F_3) | `minrank/results/separations.csv` |
| `05_vllm_serve.sh` | start the vLLM server for OpenEvolve | |
| `06_evolve.sh` | Option A: OpenEvolve run (writes evaluation traces) | `minrank/evolve/runs/` |
| `06b_evolve_hard.sh` | Option A, re-targeted: hard graphs, fitness relative to SAT at equal time (see below) | `minrank/evolve/runs/` |
| `07_eval_heldout.sh` | evaluate the best evolved program like a baseline | `minrank/results/evolved.csv` (method `evolved:<id>`) |
| `08_cascade_replay.sh` | Option D: replay early-exit policies on the traces | `minrank/results/cascade/` |
| `09_aggregate.sh` | summary tables, figures; fails if a certificate does not verify | `minrank/results/summary_*.csv`, `figures/` |
| `run_cpu_all.sh` | 00, 03, 02, 04, 09 in order | |

Every script takes `PY=/path/to/python` (default `python`); see the comments at the top of each for its variables.

## Layout

| Path | Contents |
|---|---|
| `core/` | `Graph` (bitmask side-information graphs), exact F_2 rank, fit verifier, certificate files |
| `bench/` | generators (G(n,p), Kneser, Paley, circulant, cycle power, Mycielski, orthogonality, planted) and the fixed instance list with splits |
| `baselines/` | B1 DSatur, B2 TabuCol, B8 random greedy (`clique_cover.py`); B6/B7 tabu/annealing on `A = U V^T` (`factor_search.py`); B5 alternating information-set decoding (`alt_f2.py`); B3 LDG, B4 partial clique cover, B10 SDP |
| `exact/` | L1 lower bound (`lower.py`); SAT for `minrk <= r` and anytime descent (`sat.py`); symmetry breaking, F_q encoding, Option B driver and separations |
| `evolve/` | Option A: seed program, sandboxed evaluator (writes traces), OpenEvolve config, held-out evaluation |
| `cascade/` | Option D: trace replay, policies (cost rule, certified, successive halving, Hyperband, Hoeffding race, OpenEvolve cascade), online gate |
| `analysis/` | aggregation into summary tables and figures |
| `run_baselines.py` | the sweep runner |
| `tests/` | unit tests (`python -m pytest minrank/tests -q`) |

## Re-targeted evolution: hard graphs, fitness relative to SAT

The first run (`06_evolve.sh`, `config.yaml`) scores a program against a clique cover on all training graphs with
n <= 50. SAT descent already beats the cover there in a few seconds, so that fitness rewards nothing SAT cannot do.
The re-targeted mode trains only on graphs that SAT does not solve within the budget, and scores against SAT:

- **Opt-in:** `MINRANK_EVO_TARGET=sat` in the evaluator's environment. Unset (or `cover`), the evaluator behaves
  exactly as before: same graphs, scores, metrics and trace records.
- **Graphs:** the training graphs of `MINRANK_EVO_FAMILIES` with `MINRANK_EVO_MIN_N <= n <= MINRANK_EVO_MAX_N`, kept if
  SAT at the evaluation budget neither proves optimality (UNSAT) nor reaches the certified lower bound
  (`MINRANK_EVO_HARD=auto`, the default). `MINRANK_EVO_HARD=all` keeps them all; a comma-separated list of graph names
  fixes the set by hand.
- **Score per graph:** `(sat_rank - rank) / sat_rank`, or -1 on a crash, timeout or non-fitting answer. `sat_rank` is
  what `sat_descend` (the `sat` baseline, seed `MINRANK_EVO_SAT_SEED`, default 0) reaches with the same budget
  `MINRANK_EVO_BUDGET`. It is computed once per (graph, budget, seed) and cached in `minrank/evolve/reference_sat.json`
  (`MINRANK_SAT_REFERENCE`; module `minrank/evolve/reference_sat.py`), like `reference.json` for the clique cover.
  Extra metrics are `n_tied_ref` and `n_below_cover`, and `n_improved` counts graphs below SAT. Trace records gain
  `ref_rank` and `target`.
- **Config:** `minrank/evolve/config_hard.yaml`: prompt describing the hard graphs and the SAT-relative score,
  cascade threshold -0.3, one evaluation at a time, 150 iterations.

Run it (the defaults are `planted_d,planted`, n 50-100, 10 s per graph, 4 workers on cores 12-15,44-47):

```bash
PY=~/venv/bin/python PREPARE_ONLY=1 bash scripts/minrank/06b_evolve_hard.sh   # references + seed check (CPU only)
PY=~/venv/bin/python RUN=evo_hard1 PORT=8000 bash scripts/minrank/06b_evolve_hard.sh      # needs 05_vllm_serve.sh
FAMILIES=planted_d MIN_N=50 MAX_N=50 bash scripts/minrank/06b_evolve_hard.sh              # n=50 planted_d only
MINRANK_EVO_TARGET=sat MINRANK_EVO_FAMILIES=planted_d,planted MINRANK_EVO_MIN_N=50 MINRANK_EVO_MAX_N=100 \
  MINRANK_EVO_BUDGET=10 python -m minrank.evolve.evaluator path/to/program.py --no-trace   # score one program
```

SAT is limited by wall-clock time, so the cached reference depends on the machine load when it was computed. The
script computes it on the same cores, with the same number of parallel runs, as the candidate evaluations. Delete
the file's entries (or pass another `SAT_REF`) to recompute after moving machines. Held-out evaluation is unchanged:
`07_eval_heldout.sh` with `PROGRAM=minrank/evolve/runs/evo_hard1/best/best_program.py`; the aggregator compares the
program with SAT at every budget.

## Conventions

- `out[i]` is the bitmask of the messages receiver `i` knows; matrix rows are ints with bit `j` = `A[i, j]`.
- A matrix fits `G` iff `A[i, i] = 1` and `A[i, j] = 0` whenever `j` is not known to `i`.
- **Nothing is reported without verification.** Every method's output is checked by `core/verify.py`; certificates
  store the graph digest and are re-verified by the aggregator. Check one by hand with
  `python -m minrank.core.verify minrank/bench/graphs/<g>.json minrank/certificates/<g>.json`.
- Lower bounds are sizes of *found* acyclic induced subgraphs (re-checked), valid even when CP-SAT times out.
- Splits are frozen: random and planted graphs use seeds 0-3 train, 4 val, 5-7 test; structured graphs are test only.
- Budgets are wall-clock seconds per graph, enforced for every method (SAT encoding included).

## Day-1 findings (preliminary, 4-core container)

- On small graphs (n <= 15, random and structured) the F_2 minrank equals the clique cover in every case tested.
- Planted and orthogonality graphs have certified gaps (e.g. `orth_6`: minrank 6, clique cover 8).
- Bit-flip tabu, annealing and alternating decoding do not find those gaps; SAT does, in seconds, on n <= 50.
- So evolved heuristics are scored on planted/structured families and compared with SAT at equal time.
