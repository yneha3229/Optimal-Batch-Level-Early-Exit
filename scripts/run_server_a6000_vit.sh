#!/usr/bin/env bash
# Report item 5.1, CIFAR-100 (ViT-B/16) half: eager, then CUDA graphs, on an A6000. Same protocol as the RoBERTa
# scripts (profile -> certified selection + front on the CPU -> timed held-out passes). 224 x 224 images = 197 tokens,
# so outputs carry the tag a6000[_graph]_T197 and the cost file is local_cost_a6000[_graph]_T197_vit.json.
# The graph half starts only if experiments/check_graph_device.py passes on CIFAR-100.
# Usage: GPU_ID=5 nohup scripts/run_server_a6000_vit.sh > logs/server_a6000_vit.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
GPU_ID=${GPU_ID:?set GPU_ID}; NAME=${NAME:-a6000}; T=197; t=cifar100
TIME_CPUS=${TIME_CPUS:-0-3,32-35}; CPU_CPUS=${CPU_CPUS:-4-7,36-39}
PY=${PY:-$HOME/venv/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1
ST=logs/server_a6000_vit_status.log
st() { echo "$(date +%F_%T) $*" >> $ST; }
run() { local log=$1; shift; if "$@" > "$log" 2>&1; then st "OK   $log"; else st "FAIL $log (exit $?)"; return 1; fi; }
BS="8 16 64 128"
st "START cifar100 gpu=$GPU_ID"
for EXM in eager graph; do
  GN=$NAME; [ $EXM = graph ] && GN=${NAME}_graph
  if [ $EXM = graph ]; then
    run logs/check_graph_device_cifar100.log taskset -c $TIME_CPUS $PY experiments/check_graph_device.py --task $t \
      --name $NAME --seq-len $T || { st "graph check failed: skipping the graph half"; break; }
  fi
  [ -s results/local_cost_${GN}_T${T}_vit.json ] || \
    run logs/server_profile_${t}_${GN}_T$T.log taskset -c $TIME_CPUS $PY experiments/local_hw.py profile --exec $EXM \
      --task $t --name $NAME --seq-len $T
  for c in select front; do
    run logs/server_${c}_${t}_${GN}_T$T.log taskset -c $CPU_CPUS $PY experiments/local_hw.py $c --exec $EXM --task $t \
      --name $NAME --seq-len $T --batch-sizes $BS --workers 8 &
  done
  wait
  run logs/server_time_${t}_${GN}_T$T.log taskset -c $TIME_CPUS $PY experiments/local_hw.py time --exec $EXM --task $t \
    --name $NAME --seq-len $T --batch-sizes $BS --n 2048 --rounds 5
done
st "DONE"
