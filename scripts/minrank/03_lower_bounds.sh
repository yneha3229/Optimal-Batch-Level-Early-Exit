#!/usr/bin/env bash
# Certified lower bounds (MAIS / independence number by CP-SAT) -> minrank/results/lower_bounds.csv.
# Usage (repo root):  WORKERS=32 LB_TIME=60 bash scripts/minrank/03_lower_bounds.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
LB_TIME=${LB_TIME:-60}
mkdir -p logs
$PY -m minrank.run_baselines --lower-only --lb-time "$LB_TIME" --workers "$WORKERS" 2>&1 | tee logs/minrank_lower_bounds.log
