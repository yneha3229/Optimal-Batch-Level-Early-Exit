# Prompt: CUDA-graph device for `experiments/local_hw.py` (report item 5.1, second half)

Paste the section below into a coding agent, or use it as a spec when coding it yourself. Work on a new branch made
from `claude/keen-dijkstra-fpe319` (for example `cuda-graph-device`) and push that branch; do not push to
`claude/keen-dijkstra-fpe319` (the `next-steps` agent commits there).

---

## Task

Add a CUDA-graph execution mode to `experiments/local_hw.py`, so the accuracy-matched server tables of report item 5.1
(`analysis/make_report.py`, section 5.1) can be produced for **eager and CUDA graphs**. The eager half is done
(`scripts/run_server_a6000.sh`, results in `results/local_*_a6000_T{128,32}*`, logs `logs/server_*_a6000_T*.log`).
The CUDA-graph half is missing. The old graph numbers (`runner/graph_runner.py`, `results/graph_<task>_gpu.csv`)
used the old, naive selection and are not accuracy-matched, so they cannot be reused.

## What exists, read first

- `experiments/local_hw.py`
  - `class Device` (~line 306): RoBERTa-base built by `simulator.exit_model.build`, fp32, synthetic token ids.
    `run_batch(slot, plan, decide, conf, pred, cost)` runs embed → layers → head at each exit of `plan`; at each
    evaluated exit it copies the confidences to pinned host memory and syncs once; then it calls
    `decide(j, c, n, B, remaining_layers, cost.a, cost.b)` on the **recorded** confidences of the trained model
    (`conf`, `pred` from the traces), drops the rows that exit (`index_select`, compaction), and continues.
    `time_pass` (inherited) times a list of batches.
  - `class LLMDevice(Device)` (~line 374): the same for the LLM classifiers. Out of scope here, but keep it
    working.
  - `make_device()` (~line 483), `tag()`, `cost_path()` (cost files `results/local_cost_<name>_T<T>.json`),
    `cmd_profile` (builds the cost table: full and heads-only time vs batch size), `cmd_select`, `cmd_front`,
    `cmd_time`. `main()` holds the shared CLI.
- `runner/graph_runner.py` (`GraphRunner`): the old CUDA-graph runner. It shows the capture approach that worked:
  - rows are padded up to the next bucket in `BUCKETS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]`;
  - each layer segment is captured once per bucket (`torch.cuda.CUDAGraph`, a shared memory pool);
  - the no-exit baseline is a single graph of the whole model.

  Reuse its ideas, not its code path; its selection protocol is the old one.
- `scripts/run_server_a6000.sh`: the eager pipeline to mirror (profile → select + front on CPU → time on GPU, for
  T in {128, 32}, tasks qqp and mnli, B in {8, 16, 64, 128}, `--n 2048 --rounds 5`).

## What to build

1. **`class GraphDevice(Device)`** in `experiments/local_hw.py`:
   - The constructor signature is the same as `Device`. Same model, dtype (fp32, so it is comparable with the eager
     tables) and synthetic inputs.
   - `run_batch(...)` has the same signature and returns the same thing (the policy's predictions, `[B]` int64).
     The decision logic must stay identical to `Device.run_batch`: one host sync per evaluated exit, recorded
     confidences passed to `decide`, and the same exits taken.
   - Static shapes:
     - At every step, pad the active rows up to the bucket `bucket(n)`.
     - After compaction, gather the kept rows into the static input buffer of the new bucket and fill the padding
       rows with copies of a real row.
     - Read the confidences of the first `n` rows only.
   - Graphs:
     - One graph per (bucket, segment), where a segment is "layers lo..hi then the head at exit e". Capture them
       lazily and cache them by `(bucket, lo, hi, e)`, sharing one pool (`torch.cuda.graph_pool_handle()`).
     - The first segment includes the embedding.
     - The full-model plan (`FULL`) is a single graph per bucket, as in `graph_runner.py`.
   - **Capture must never happen inside a timed region.** Add a `prepare(batch_sizes, plans)` method that captures
     every graph a timing run can need (all buckets ≤ B for every plan used) and runs each once. Call it from
     `cmd_profile` and `cmd_time` before any timing.
2. **CLI:** add `--exec {eager,graph}` (default `eager`) to every subcommand:
   - `make_device` returns `GraphDevice` when `--exec graph` and the task is not an LLM; raise a clear error for
     LLMs.
   - `tag(args)` appends `_graph` when `--exec graph`, so cost files and outputs never overwrite the eager ones
     (`results/local_cost_a6000_graph_T128.json`, and so on).
   - Eager file names must not change.
3. **`scripts/run_server_a6000_graph.sh`:** a copy of `run_server_a6000.sh` that passes `--exec graph` and writes
   `logs/server_*_a6000_graph_T*.log`, with the same arguments otherwise.

## Checks before the full run

- **Equivalence:** for a few batches and plans, `GraphDevice.run_batch` returns exactly the same predictions as
  `Device.run_batch` (decisions come from the recorded traces, so they must match bit for bit).
  - For the full plan, compare the head logits of the graph and eager paths on the same inputs: `allclose`,
    `atol=1e-4`, fp32.
- **Graph timing is sane:**
  - `profile` should show a much smaller per-layer intercept `a` than the eager cost file
    (`results/local_cost_a6000_T128.json`).
  - At B=8 the heads should be a visible share of the time. The old runner measured about 19% on MNLI.
  - Times should be stable across `--rounds` (`real_lo`/`real_hi` within a few %).
- **No recapture during timing:** count graph captures, and assert the count does not change inside `time_pass`.
- `pytest tests/` still passes, and an eager `profile` with `--exec eager` gives the same cost file as before
  (within noise).

## Running it (server)

- Pick a GPU that is free (`nvidia-smi`). GPUs 2 and 3 are in use by another user (a vLLM server), and GPU 4 runs our
  minrank vLLM until the evolve run finishes.
- Run with:

  ```
  GPU_ID=<id> nohup scripts/run_server_a6000_graph.sh > logs/server_a6000_graph.log 2>&1 &
  ```

  Keep the `taskset` pinning of the eager script: timing on cores 0-3,32-35, CPU replay on 4-11,36-43. Never run
  CPU-heavy work on the timing cores.
- Expected time: about the eager run's, roughly 1.5–2 h for both sequence lengths and both tasks.
- Disk has only ~11 GB free; do not save new checkpoints.

## What the result should answer (report 5.1)

- At a matched 1-point drop, under CUDA graphs, is the cost rule with the exit plan within 5% of the best
  method at B ≤ 16?
- Fallback stated in the report: if it is more than 10% behind, report the plan as a partial fix and state the
  settings where a unified exit should be used.
- With eager execution, the plan was 3–6% behind the unified exit at T=32, B ≤ 16 (see `logs/server_time_*_a6000_T32.log`).

Commit on your branch with a message like `local_hw: CUDA-graph device (--exec graph) for report item 5.1`, then
push the branch.
