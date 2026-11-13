#!/bin/bash
# Re-run the real-GPU timing steps that failed in run_final.sh (history-aware policies), then figures.
cd ~/ie624
PY=~/venv/bin/python
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3"
STATUS=logs/final_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/final_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
echo "$(date +%F_%T) gpu redo started (pid $$)" >> $STATUS
step dec_gpu      $GPU $PY decoder/decode_runner.py --regime real_wikitext --batch-sizes 16 64
for t in mnli qqp cifar100; do step gpu_$t $GPU $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 16 64 128; done
for t in wikitext mnli qqp cifar100; do step figs_$t $PY analysis/make_figures.py --task $t; done
echo "$(date +%F_%T) gpu redo finished" >> $STATUS
