#!/usr/bin/env bash
# Option B: search for small explicit separations: minrk_2 < clique cover, and minrk_2 != minrk_3.
# Candidates: structured families (+ complements), random G(n,p) for each n in RANDOM_N (RANDOM_COUNT per density),
# and planted/orthogonality positive controls. Separations are written with certificates to
# minrank/results/separations/<graph>.json; the summary to minrank/results/separations.csv (resumable).
#
# Usage (from the repo root):
#   bash scripts/minrank/04b_separations.sh
#   RANDOM_N="10 11 12" RANDOM_COUNT=100 DIRECTED=1 WORKERS=16 bash scripts/minrank/04b_separations.sh
# Env: PY, WORKERS, TIME_PER_CALL, RANDOM_N, RANDOM_COUNT, DIRECTED (1 = directed random graphs), NO_F3 (1 = skip F_3).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
WORKERS=${WORKERS:-8}
TIME_PER_CALL=${TIME_PER_CALL:-120}
RANDOM_N=${RANDOM_N:-"8 9 10 11 12"}
RANDOM_COUNT=${RANDOM_COUNT:-30}

extra=()
[ "${DIRECTED:-0}" = "1" ] && extra+=(--directed)
[ "${NO_F3:-0}" = "1" ] && extra+=(--no-f3)
mkdir -p logs
# shellcheck disable=SC2086
"$PY" -m minrank.exact.separations --random-n $RANDOM_N --random-count "$RANDOM_COUNT" \
    --time-per-call "$TIME_PER_CALL" --workers "$WORKERS" "${extra[@]}" 2>&1 | tee -a logs/minrank_separations.log
