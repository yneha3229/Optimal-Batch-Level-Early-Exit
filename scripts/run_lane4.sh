#!/bin/bash
# Lane 4 (Llama-3.1-8B) re-run: GPU 4 for training/calibration/B=16 timing, GPU 5 for B=64 timing once lane 5 is done.
cd ~/ie624
PY=~/venv/bin/python
G() { env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 "${@:2}"; }
STATUS=logs/final_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/final_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
echo "$(date +%F_%T) lane4 (llama) restarted (pid $$)" >> $STATUS
step llama_train  G 4 $PY decoder/train_decoder_exits.py --model meta-llama/Llama-3.1-8B-Instruct --exits 12 16 20 24 28 32 --tag llama --bs 4
step llama_calib  G 4 $PY decoder/decode_runner.py --task wikitext_llama --calibrate
step llama_sim    $PY experiments/run_main.py --task wikitext_llama --workers 16
step llama_gpu16  G 4 $PY decoder/decode_runner.py --task wikitext_llama --batch-sizes 16 --out-suffix _B16 &
# lane 5 adds a START and an OK/FAIL line for gpu_cifar100 (the old run left 2): wait for 4
until [ "$(grep -c 'gpu_cifar100' $STATUS)" -ge 4 ] && ! grep -q "START gpu_cifar100" <(tail -n 1 <(grep gpu_cifar100 $STATUS)); do sleep 30; done
step llama_gpu64  G 5 $PY decoder/decode_runner.py --task wikitext_llama --batch-sizes 64 --out-suffix _B64
wait
step llama_merge  $PY scripts/merge_gpu.py wikitext_llama real_wikitext_llama
step llama_robust $PY experiments/robust_bound.py --task wikitext_llama --regime real_wikitext_llama
step llama_bandit $PY bandit/run_bandit.py --task wikitext_llama --regime real_wikitext_llama --B 64 --T 3000
step figs_wikitext_llama $PY analysis/make_figures.py --task wikitext_llama
echo "$(date +%F_%T) lane4 (llama) finished" >> $STATUS
