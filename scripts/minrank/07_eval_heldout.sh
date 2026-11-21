#!/usr/bin/env bash
# Score the best evolved program on held-out graphs (val + test) at the baseline budgets (Option A).
# Writes rows in the baseline CSV format to minrank/results/evolved.csv (method "evolved:<id>"); reruns skip
# rows already present.
#
# Usage (from the repository root):
#   RUN=evo1 bash scripts/minrank/07_eval_heldout.sh
#   PROGRAM=path/to/program.py BUDGETS="1 10" SEEDS="0" WORKERS=16 bash scripts/minrank/07_eval_heldout.sh
# Env: RUN or PROGRAM, SPLITS, BUDGETS, SEEDS, WORKERS, PY, OUT.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
SPLITS=${SPLITS:-val test}
BUDGETS=${BUDGETS:-1 10 60}
SEEDS=${SEEDS:-0 1 2}
WORKERS=${WORKERS:-$(( $(nproc) - 1 ))}
OUT=${OUT:-minrank/results/evolved.csv}
if [ -n "${PROGRAM:-}" ]; then SRC=(--program "${PROGRAM}"); else SRC=(--run-dir "minrank/evolve/runs/${RUN:-evo1}"); fi
# shellcheck disable=SC2086
"${PY}" -m minrank.evolve.eval_heldout "${SRC[@]}" --split ${SPLITS} --budgets ${BUDGETS} --seeds ${SEEDS} \
  --workers "${WORKERS}" --out "${OUT}"
