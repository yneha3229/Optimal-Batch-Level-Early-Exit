#!/usr/bin/env bash
# Everything that needs only CPUs, in order: tests, lower bounds, baselines, exact solving (Option B), aggregate.
# Usage (repo root, on the server):  WORKERS=32 nohup bash scripts/minrank/run_cpu_all.sh > logs/minrank_cpu_all.log 2>&1 &
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
D=scripts/minrank
bash $D/00_setup.sh
bash $D/03_lower_bounds.sh
bash $D/02_baselines.sh
[[ -f $D/04_exact.sh ]] && bash $D/04_exact.sh
bash $D/09_aggregate.sh
