#!/usr/bin/env bash
# LLM-as-classifier: certified selection, matched-accuracy fronts (CPU) and timed runs (GPU) for trained traces.
# Usage: GPU_ID=4 NAME=a6000 scripts/run_llm_cls_hw.sh qwen mnli qqp      (model tag, then tasks)
# Needs results/local_cost_${NAME}_T${T}_<tag>.json from: experiments/local_hw.py profile --task <task>_<tag> --name $NAME
set -u
cd "$(dirname "$0")/.."
TAG=$1; shift
T=${T:-128}; NAME=${NAME:?set NAME}; GPU_ID=${GPU_ID:?set GPU_ID}; PY=~/venv/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1
st() { echo "$(date +%F_%T) $*" >> logs/llm_cls_status.log; }
for t in "$@"; do
  for c in select front; do
    $PY experiments/local_hw.py $c --task ${t}_$TAG --name $NAME --seq-len $T --workers 12 > logs/cls_${c}_${t}_${TAG}_$NAME.log 2>&1 &
  done
done
wait; st "OK select+front $TAG $*"
for t in "$@"; do
  $PY experiments/local_hw.py time --task ${t}_$TAG --name $NAME --seq-len $T --n ${N:-2048} --rounds ${ROUNDS:-5} > logs/cls_time_${t}_${TAG}_$NAME.log 2>&1 \
    && st "OK time ${t}_$TAG" || st "FAIL time ${t}_$TAG"
done
st "DONE $TAG $*"
