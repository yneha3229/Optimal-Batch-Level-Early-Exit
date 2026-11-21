#!/usr/bin/env bash
# Option D: replay stopping policies over candidate-evaluation traces (CPU only, minutes).
#
# Usage (from the repository root):
#   scripts/minrank/08_cascade_replay.sh                         # real traces from the evolution runs
#   TRACES=path/to/evals.jsonl scripts/minrank/08_cascade_replay.sh
#   SYNTHETIC=400 scripts/minrank/08_cascade_replay.sh           # synthetic traces (no evolution run needed)
# Environment: PY (python), BATCH (batch size, 8), LAMBDAS ("0.001 0.003 0.01 0.03 0.1"), OUT (output dir).
# Outputs: $OUT/summary.csv, $OUT/per_batch.csv, $OUT/anytime.png (if matplotlib is installed).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
BATCH=${BATCH:-8}
LAMBDAS=${LAMBDAS:-"0.001 0.003 0.01 0.03 0.1"}
if [[ -n "${SYNTHETIC:-}" ]]; then
  OUT=${OUT:-minrank/results/cascade_synthetic}
  # shellcheck disable=SC2086
  "$PY" -m minrank.cascade.run_replay --synthetic "$SYNTHETIC" --batch-size "$BATCH" --policies all \
    --lambdas $LAMBDAS --out "$OUT"
else
  TRACES=${TRACES:-minrank/evolve/runs/traces/evals.jsonl}
  OUT=${OUT:-minrank/results/cascade}
  if [[ ! -s "$TRACES" ]]; then
    echo "no traces at $TRACES; run the evolution first or set SYNTHETIC=400" >&2
    exit 1
  fi
  # shellcheck disable=SC2086
  "$PY" -m minrank.cascade.run_replay --traces "$TRACES" --batch-size "$BATCH" --policies all \
    --lambdas $LAMBDAS --out "$OUT"
fi
