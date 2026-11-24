#!/usr/bin/env bash
# FunSearch for the MAIS priority function.
#   PROPOSER=gp                                   non-LLM baseline (genetic programming), runs anywhere
#   PROPOSER=anthropic                            Claude via the anthropic SDK (needs ANTHROPIC_API_KEY or `ant auth login`)
#   PROPOSER=anthropic:claude-sonnet-5-5:medium   another model / effort
#   PROPOSER="local:http://127.0.0.1:8000/v1|Qwen/Qwen2.5-Coder-32B-Instruct-AWQ"   vLLM from scripts/minrank/05_vllm_serve.sh
# Compare every LLM run with PROPOSER=gp at the same ITER (number of evaluated programs).
set -euo pipefail
PY=${PY:-python}; PROPOSER=${PROPOSER:-gp}; ITER=${ITER:-300}; PAR=${PAR:-4}; SEED=${SEED:-0}
RUN=${RUN:-$(echo "$PROPOSER" | tr ':/|.' '____')_s$SEED}
PYTHONPATH=. $PY -m pincer.funsearch.loop --proposer "$PROPOSER" --iterations "$ITER" --parallel "$PAR" \
  --seed "$SEED" --out "pincer/funsearch/runs/$RUN" | tee "logs/pincer_funsearch_$RUN.log"
