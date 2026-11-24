#!/usr/bin/env bash
# The LLM policy on the same benchmark as the index policy (it reads the index policy's candidates and estimates
# and picks one). LLM latency counts against the per-graph budget.
#   LLM=anthropic:claude-opus-5-5:low   (default)   or   LLM="local:http://127.0.0.1:8000/v1|<model>"
set -euo pipefail
PY=${PY:-python}; SPLIT=${SPLIT:-test2}; BUDGET=${BUDGET:-30}; WORKERS=${WORKERS:-3}
LLM=${LLM:-anthropic:claude-opus-5-5:low}
PYTHONPATH=. $PY -m pincer.bench --split "$SPLIT" --budget "$BUDGET" --workers "$WORKERS" --methods llm --llm "$LLM" \
  --out "pincer/results/bench_${SPLIT}_b${BUDGET%.*}_llm.jsonl"
PYTHONPATH=. $PY -m pincer.analysis "pincer/results/bench_${SPLIT}_b${BUDGET%.*}.jsonl" \
  "pincer/results/bench_${SPLIT}_b${BUDGET%.*}_llm.jsonl"
