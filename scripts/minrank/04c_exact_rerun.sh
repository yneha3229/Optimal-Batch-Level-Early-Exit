#!/usr/bin/env bash
# Option B, second pass: re-solve only the graphs whose last row in minrank/results/exact.csv has status
# call_timeout, with a longer limit per SAT call. Results go to a separate CSV so exact.csv is not appended to.
# Warm start = the stored certificate (so a graph resumes from its best known rank); improved certificates are
# saved to minrank/certificates/ as in 04_exact.sh. Resumable: graphs already proven in $OUT are skipped.
#
# Usage (from the repo root; pinned to the free CPU cores, NUMA node 1):
#   nohup bash scripts/minrank/04c_exact_rerun.sh > /dev/null 2>&1 &
#   TIME_PER_CALL=60 TOTAL_TIME=120 NAMES=cycle_7 OUT=/tmp/x.csv bash scripts/minrank/04c_exact_rerun.sh  # smoke
# Env: PY, CPUS (taskset list, 16-31,48-63), WORKERS (16), TIME_PER_CALL (s per SAT call, 3600),
#      TOTAL_TIME (s per graph, 10800; empty = no cap), LB_TIME (60), MODE (cols), FROM (exact.csv to read),
#      STATUS (statuses to rerun, "call_timeout"), NAMES (optional: further restrict to these graph names),
#      OUT (minrank/results/exact_long.csv), CERTS (minrank/certificates).
# Output: $OUT (exact.csv format, one row per graph), logs/minrank_exact_long.log,
#         logs/minrank_exact_long_status.log (START / OK / FAIL / DONE lines).
set -u
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-$HOME/venv/bin/python}
CPUS=${CPUS:-16-31,48-63}
WORKERS=${WORKERS:-16}
TIME_PER_CALL=${TIME_PER_CALL:-3600}
TOTAL_TIME=${TOTAL_TIME-10800}
LB_TIME=${LB_TIME:-60}
MODE=${MODE:-cols}
FROM=${FROM:-minrank/results/exact.csv}
STATUS=${STATUS:-call_timeout}
OUT=${OUT:-minrank/results/exact_long.csv}
CERTS=${CERTS:-minrank/certificates}
LOG=${LOG:-logs/minrank_exact_long.log}
ST=${ST:-logs/minrank_exact_long_status.log}

mkdir -p logs
extra=()
[ -n "$TOTAL_TIME" ] && extra+=(--total-time "$TOTAL_TIME")
# shellcheck disable=SC2206
[ -n "${NAMES:-}" ] && extra+=(--names $NAMES)
echo "$(date +%F_%T) START exact_rerun from=$FROM status=$STATUS time_per_call=$TIME_PER_CALL total=${TOTAL_TIME:-none} out=$OUT" >> "$ST"
# shellcheck disable=SC2086
if taskset -c "$CPUS" "$PY" -m minrank.exact.driver --rerun-from "$FROM" --rerun-status $STATUS --max-n 1000 \
    --time-per-call "$TIME_PER_CALL" --lb-time "$LB_TIME" --mode "$MODE" --workers "$WORKERS" \
    --out "$OUT" --certs "$CERTS" "${extra[@]}" >> "$LOG" 2>&1; then
  echo "$(date +%F_%T) OK    exact_rerun" >> "$ST"
else
  echo "$(date +%F_%T) FAIL  exact_rerun (exit $?)" >> "$ST"
fi
echo "$(date +%F_%T) DONE" >> "$ST"
