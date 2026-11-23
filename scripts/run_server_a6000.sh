#!/usr/bin/env bash
# Report item 5.1, eager half: accuracy-matched server tables on an A6000 with the exit plan, the measured cost table
# and certified selection (experiments/local_hw.py), for the RoBERTa tasks. One cost profile per sequence length,
# then selection + accuracy-speed front (CPU replay), then real timing on held-out batches.
# Usage: GPU_ID=2 nohup scripts/run_server_a6000.sh > logs/server_a6000.log 2>&1 &
# Env: GPU_ID (required), NAME (default a6000), TS (sequence lengths, default "128 32"), TASKS (default "qqp mnli"),
#      TIME_CPUS / CPU_CPUS (taskset lists: timing gets quiet cores on the GPU's NUMA node).
set -u
cd "$(dirname "$0")/.." || exit 1
GPU_ID=${GPU_ID:?set GPU_ID}; NAME=${NAME:-a6000}; TS=${TS:-"128 32"}; TASKS=${TASKS:-"qqp mnli"}
TIME_CPUS=${TIME_CPUS:-0-3,32-35}; CPU_CPUS=${CPU_CPUS:-4-11,36-43}
PY=${PY:-$HOME/venv/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1
ST=logs/server_a6000_status.log
st() { echo "$(date +%F_%T) $*" >> $ST; }
run() { local log=$1; shift; if "$@" > "$log" 2>&1; then st "OK   $log"; else st "FAIL $log (exit $?)"; fi; }
BS="8 16 64 128"
st "START tasks=[$TASKS] T=[$TS] gpu=$GPU_ID"
for T in $TS; do
  [ -s results/local_cost_${NAME}_T$T.json ] || \
    run logs/server_profile_${NAME}_T$T.log taskset -c $TIME_CPUS $PY experiments/local_hw.py profile --task mnli --name $NAME --seq-len $T
  for t in $TASKS; do
    for c in select front; do
      run logs/server_${c}_${t}_${NAME}_T$T.log taskset -c $CPU_CPUS $PY experiments/local_hw.py $c --task $t --name $NAME \
        --seq-len $T --batch-sizes $BS --workers 12 &
    done
  done
  wait
  for t in $TASKS; do
    run logs/server_time_${t}_${NAME}_T$T.log taskset -c $TIME_CPUS $PY experiments/local_hw.py time --task $t --name $NAME \
      --seq-len $T --batch-sizes $BS --n 2048 --rounds 5
  done
done
st "DONE"
