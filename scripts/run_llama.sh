#!/bin/bash
# After the GPU redo: encoder timing at B=8 (rho > 1 on real hardware), then the Llama-3.1-8B decoder experiment.
cd ~/ie624
PY=~/venv/bin/python
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3"
STATUS=logs/final_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/final_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
while pgrep -u neha -f "run_gpu_redo.s[h]" > /dev/null; do sleep 60; done
echo "$(date +%F_%T) llama + B8 run started (pid $$)" >> $STATUS
for t in mnli qqp cifar100; do step gpu8_$t $GPU $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 8 --out-suffix _B8; done
step llama_train  $GPU $PY decoder/train_decoder_exits.py --model meta-llama/Llama-3.1-8B-Instruct --exits 12 16 20 24 28 32 --tag llama --bs 4
step llama_calib  $GPU $PY decoder/decode_runner.py --task wikitext_llama --calibrate
step llama_sim    $PY experiments/run_main.py --task wikitext_llama --workers 40
step llama_gpu    $GPU $PY decoder/decode_runner.py --task wikitext_llama --batch-sizes 16 64
step llama_robust $PY experiments/robust_bound.py --task wikitext_llama --regime real_wikitext_llama
step llama_bandit $PY bandit/run_bandit.py --task wikitext_llama --regime real_wikitext_llama --B 64 --T 3000
step figs_wikitext_llama $PY analysis/make_figures.py --task wikitext_llama
echo "$(date +%F_%T) llama + B8 run finished" >> $STATUS
