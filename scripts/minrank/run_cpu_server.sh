#!/usr/bin/env bash
# Server CPU chain for minrank, pinned to NUMA node 1 so it does not disturb GPU timing on node 0:
# lower bounds -> baseline sweep -> exact (Option B) -> separations (Option B) -> aggregate. Each step is resumable.
# Usage: nohup bash scripts/minrank/run_cpu_server.sh > logs/minrank_cpu_server.log 2>&1 &
set -u
cd "$(git rev-parse --show-toplevel)"
export PY=${PY:-$HOME/venv/bin/python} WORKERS=${WORKERS:-16}
CPUS=${CPUS:-16-31,48-63}
D=scripts/minrank
ST=logs/minrank_status.log
for s in 03_lower_bounds 02_baselines 04_exact 04b_separations 09_aggregate; do
  echo "$(date +%F_%T) START $s" >> $ST
  if taskset -c $CPUS bash $D/$s.sh; then echo "$(date +%F_%T) OK    $s" >> $ST; else echo "$(date +%F_%T) FAIL  $s (exit $?)" >> $ST; fi
done
