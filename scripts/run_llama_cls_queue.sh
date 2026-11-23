#!/usr/bin/env bash
# Second backbone for the LLM-as-classifier result (docs/PLAN_LLM.md section 7): Llama-3.1-8B on MNLI (already
# trained) and QQP (trained here), then cost profile, certified selection, fronts and timed runs on the A6000.
# Waits for scripts/run_server_a6000.sh to finish so the timing GPU is idle.
cd "$(dirname "$0")/.." || exit 1
GPU_ID=${GPU_ID:?set GPU_ID}; PY=~/venv/bin/python; ST=logs/llm_cls_status.log
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID HF_HUB_OFFLINE=1
until grep -q "DONE" logs/server_a6000_status.log; do sleep 60; done
st() { echo "$(date +%F_%T) $*" >> $ST; }
st "START llama queue gpu=$GPU_ID"
taskset -c 0-3,32-35 $PY experiments/local_hw.py profile --task mnli_llama --name a6000 --seq-len 128 > logs/cls_profile_llama_a6000.log 2>&1 \
  && st "OK profile llama" || { st "FAIL profile llama"; exit 1; }
GPU_ID=$GPU_ID NAME=a6000 scripts/run_llm_cls_hw.sh llama mnli
[ -s checkpoints/qqp_llama_cls.pt ] || { $PY decoder/train_cls_exit_lora.py --task qqp --model meta-llama/Llama-3.1-8B-Instruct --tag llama \
  --exits 8 12 16 20 24 28 32 --steps 8000 --bs 16 > logs/cls_qqp_llama.log 2>&1 && st "OK train qqp_llama" || { st "FAIL train qqp_llama"; exit 1; }; }
GPU_ID=$GPU_ID NAME=a6000 scripts/run_llm_cls_hw.sh llama qqp
st "DONE llama queue"
