#!/bin/bash
# Parallel finish on three A6000s (GPU 6 is not usable). Status: logs/final_status.log.
cd ~/ie624
PY=~/venv/bin/python
G() { env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 "${@:2}"; }
STATUS=logs/final_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/final_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
echo "$(date +%F_%T) parallel run started (pid $$)" >> $STATUS

lane3() {   # GPU 3: wait for the running Qwen decode timing, then encoders at B=8
  while pgrep -u neha -f "decode_runner.py --regime real_wikitex[t]" > /dev/null; do sleep 30; done
  echo "$(date +%F_%T) OK    dec_gpu (finished in place)" >> $STATUS
  for t in mnli qqp cifar100; do step gpu8_$t G 3 $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 8 --out-suffix _B8; done
}
lane5() {   # GPU 5: encoders at B=16/64/128
  for t in mnli qqp cifar100; do step gpu_$t G 5 $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 16 64 128 --out-suffix _B16-128; done
}
lane4() {   # GPU 4: Llama-3.1-8B; final timing split B=16 (GPU 4) || B=64 (GPU 5, after lane 5)
  step llama_train  G 4 $PY decoder/train_decoder_exits.py --model meta-llama/Llama-3.1-8B-Instruct --exits 12 16 20 24 28 32 --tag llama --bs 4
  step llama_calib  G 4 $PY decoder/decode_runner.py --task wikitext_llama --calibrate
  step llama_sim    $PY experiments/run_main.py --task wikitext_llama --workers 16
  step llama_gpu16  G 4 $PY decoder/decode_runner.py --task wikitext_llama --batch-sizes 16 --out-suffix _B16 &
  while pgrep -u neha -f "gpu_runner.py --task cifar100 --regime real_cifar100 --batch-sizes 16 64 12[8]" > /dev/null || \
        ! grep -q "gpu_cifar100" $STATUS; do sleep 30; done
  step llama_gpu64  G 5 $PY decoder/decode_runner.py --task wikitext_llama --batch-sizes 64 --out-suffix _B64
  wait
  step llama_merge  $PY scripts/merge_gpu.py wikitext_llama real_wikitext_llama
  step llama_robust $PY experiments/robust_bound.py --task wikitext_llama --regime real_wikitext_llama
  step llama_bandit $PY bandit/run_bandit.py --task wikitext_llama --regime real_wikitext_llama --B 64 --T 3000
}
lane3 & lane5 & lane4 & wait
for t in mnli qqp cifar100; do step merge_$t $PY scripts/merge_gpu.py $t real_$t; done
for t in wikitext wikitext_llama mnli qqp cifar100; do step figs_$t $PY analysis/make_figures.py --task $t; done
echo "$(date +%F_%T) parallel run finished" >> $STATUS
