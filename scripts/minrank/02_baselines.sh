#!/usr/bin/env bash
# Baseline sweep: every graph x method x budget x seed -> minrank/results/baselines.csv (+ best certificates).
# Resumable: rows already in the CSV are skipped, so rerun after a crash or to add budgets/seeds.
# Usage (repo root):  WORKERS=32 BUDGETS="1 10 60" SEEDS="0 1 2" bash scripts/minrank/02_baselines.sh
# Keep WORKERS <= physical cores: each SAT call starts one extra process.
# Optional: METHODS="dsatur tabucol sat" SPLITS="val test" MAXN=50
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
BUDGETS=${BUDGETS:-"1 10 60"}
SEEDS=${SEEDS:-"0 1 2"}
SPLITS=${SPLITS:-"train val test"}
ARGS=(--split $SPLITS --budgets $BUDGETS --seeds $SEEDS --workers "$WORKERS")
[[ -n "${METHODS:-}" ]] && ARGS+=(--methods $METHODS)
[[ -n "${MAXN:-}" ]] && ARGS+=(--max-n "$MAXN")
mkdir -p logs
$PY -m minrank.run_baselines "${ARGS[@]}" 2>&1 | tee -a logs/minrank_baselines.log
