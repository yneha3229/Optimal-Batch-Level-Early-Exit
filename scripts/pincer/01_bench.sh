#!/usr/bin/env bash
# Pincer vs standard pipelines at equal wall-clock budget. Resumable (skips finished (graph, method) pairs).
#   SPLIT=train|test|test2  BUDGET=30  WORKERS=3  METHODS=classic,satdescend,roundrobin,index
# Calibrate the index policy's priors on the train split first (writes pincer/priors.json):
#   SPLIT=train METHODS=roundrobin CALIBRATE=1 bash scripts/pincer/01_bench.sh
set -euo pipefail
PY=${PY:-python}; SPLIT=${SPLIT:-test}; BUDGET=${BUDGET:-30}; WORKERS=${WORKERS:-3}
METHODS=${METHODS:-classic,satdescend,roundrobin,index}
EXTRA=""; [ "${CALIBRATE:-0}" = 1 ] && EXTRA="--calibrate"
PYTHONPATH=. $PY -m pincer.bench --split "$SPLIT" --budget "$BUDGET" --workers "$WORKERS" --methods "$METHODS" $EXTRA
PYTHONPATH=. $PY -m pincer.analysis "pincer/results/bench_${SPLIT}_b${BUDGET%.*}.jsonl"
