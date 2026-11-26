#!/usr/bin/env bash
# Option B: exact minrk_2 on structured / planted benchmark graphs by SAT descent with symmetry breaking.
# Warm start = best stored certificate (run 02_baselines.sh first for strong starts) or TabuCol.
# Resumable: graphs already proven optimal in minrank/results/exact.csv are skipped.
#
# Usage (from the repo root):
#   bash scripts/minrank/04_exact.sh                       # defaults below
#   WORKERS=16 TIME_PER_CALL=1800 MAX_N=60 bash scripts/minrank/04_exact.sh
#   FAMILIES="kneser paley" bash scripts/minrank/04_exact.sh
# Env: PY (python), WORKERS, TIME_PER_CALL (s per SAT call), TOTAL_TIME (s per graph, empty = no cap),
#      MAX_N, LB_TIME, MODE (cols|aut|u0|none; cols is fastest), FAMILIES (index.csv family names).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
WORKERS=${WORKERS:-8}
TIME_PER_CALL=${TIME_PER_CALL:-600}
TOTAL_TIME=${TOTAL_TIME:-}
MAX_N=${MAX_N:-60}
LB_TIME=${LB_TIME:-60}
MODE=${MODE:-cols}
FAMILIES=${FAMILIES:-"kneser paley mycielski orthogonality cycle_power circulant cycle planted planted_d"}

mkdir -p logs
[ -f minrank/bench/graphs/index.csv ] || "$PY" -m minrank.bench.instances

extra=()
[ -n "$TOTAL_TIME" ] && extra+=(--total-time "$TOTAL_TIME")
# shellcheck disable=SC2086
"$PY" -m minrank.exact.driver --family $FAMILIES --max-n "$MAX_N" --time-per-call "$TIME_PER_CALL" \
    --lb-time "$LB_TIME" --mode "$MODE" --workers "$WORKERS" "${extra[@]}" 2>&1 | tee -a logs/minrank_exact.log
