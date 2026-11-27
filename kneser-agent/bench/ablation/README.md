# Model ablation: blind agents

Each run gives the pipeline's LLM role (Solver, Critic, Verifier) to one model, which answers blind:

- `run.sh CONFIG SPEC SEED` starts the pipeline with `--llm chat:<folder>`; every LLM request is written to
  `chat/<CONFIG>_<graph>/req_NNN.json` and the pipeline waits for `rep_NNN.json`.
- A fresh agent with no other context (Claude Code subagent of the given model and effort) answers the requests. It
  may run only `next_req.py <folder>` and write reply files into that folder; it is told not to open any other file,
  not to search or browse, and not to write code. The prompt it gets is below.
- The Solver and Critic see only the numeric graph summary and check results; the Verifier sees the run result
  without graph name, spec or ground truth.
- Qwen CPU rows (`Qwen2.5-32B-cpu`): Qwen 32B GGUF Q4_K_M on llama.cpp (CPU, cores 16-27), fixed code, `cpu_qwen.sh`
  (budget 7,200 s: about 6.5 min per reply).
- Qwen GPU rows come from the vLLM runs on the GPU server (`--llm local`), same prompts (v3) but the code
  *before* 9081f70 (plan-step results were not shown to the agents): the rerun on the fixed code is blocked
  because CUDA stopped initialising on the server's GPUs at 18:59 UTC on 26 Nov. K_16_4_s3 is the reference
  run; K_14_4_s0 and K_16_4-del0.01_s0 are the benchmark runs.
- `score.py` scores the traces; `score.py --objections` prints every Critic objection for hand labelling into
  `false_objections.json`; `plot.py` draws `model_ablation.png`.

Agent prompt (folder filled in):

> You are acting as the language model behind a running software pipeline. You have no other context, and you must
> not look for any: reason independently and autonomously, only from the data in each request. The ONLY command you
> may run is `python3 next_req.py <folder>`. The ONLY file operation you may do is writing reply files into that same
> folder. Do not read, list, search or open any other file or directory, do not run any other command, do not use web
> search or fetch. Do not write code to compute anything; think it through yourself. Loop: run the command; on DONE
> stop; otherwise treat "system" as your instructions and "prompt" as the input, produce exactly the reply the system
> text asks for (one JSON object), write it to `<folder>/rep_<id as 3 digits>.json`, repeat.

Caveats: the Claude agents run inside Claude Code (not a plain API call), so their LLM seconds include agent overhead
and are not compared with vLLM; one run per cell.

## 26 Nov evening: the 12-cell grid

- Cells: K(16,4) s3/s1, K(16,4)-del0.01 s0/s1, K(14,4) s0/s1, K(13,4) s0/s1, K(15,5) s0/s1, reg(1000,6) s0/s1
  (`start_cells.sh CONFIG` starts the 9 new pipelines of one configuration; the agent prompt is unchanged).
- Qwen on CPU: `cpu_grid.sh MODEL.gguf NAME PARALLEL` (llama.cpp b12780, cores 16-29, pipelines on 30-31, budget
  21,600 s because one reply takes minutes). Qwen3-4B runs with KAGENT_MAX_TOKENS=1500 (trace suffix `-cpu-1500`):
  at the default 600 its replies were cut off (see the analysis in RESULTS.md).
- A first 32B grid start hit a race (the previous server still answered /health): 9 runs got HTTP 503 and fell back
  to default plans; they were deleted, not scored (logs in logs/invalid_32b_race/), and cpu_grid.sh now waits for the
  port and checks /v1/models.
- Labels: `labels.py` -> `label_detail.json` (one record per objection, with the reason); `score.py` counts from it.
- The chart shows per-run rates; the Qwen GPU bar is the 3-run pilot on the old code.
