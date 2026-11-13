#!/bin/bash
# LLM track, second attempt (docs/PLAN_LLM.md): steps 0-4 on one idle A6000. Stops at the first failed step.
# Usage: GPU_ID=5 nohup scripts/run_llm_v2.sh &      (check nvidia-smi first; timing needs an idle GPU)
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID:-5}"
STATUS=logs/llm2_status.log
mkdir -p logs checkpoints data
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/llm2_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; exit 1; fi; }

# Step 0: functional test
step smoke_train $GPU $PY decoder/train_exit_lora.py --tag smoke --steps 20 --bs 2 --train-lines 2000 --eval-windows 16
step smoke_calib $GPU $PY decoder/decode_runner.py --task wikitext_smoke --calibrate

# Step 1: joint training, and the frozen baseline at the same budget
step train_qlora   $GPU $PY decoder/train_exit_lora.py --tag qlora   --rank 16 --steps 6000
step train_qfrozen $GPU $PY decoder/train_exit_lora.py --tag qfrozen --rank 0  --steps 6000

# Step 2: adaptivity vs batch size (CPU). Read results/adaptivity_vs_batch.csv against docs/PLAN_LLM.md before step 3.
step adaptivity $PY analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext_qfrozen wikitext_llama --budgets 0.01 0.03

# Step 3: real wall-clock
step calib $GPU $PY decoder/decode_runner.py --task wikitext_qlora --calibrate
step sim   $PY experiments/run_main.py --task wikitext_qlora --regimes real_wikitext_qlora --batch-sizes 2 4 8 16 64 --workers 32
for B in 2 4 8 16 64; do
  step gpu_B$B $GPU $PY decoder/decode_runner.py --task wikitext_qlora --batch-sizes $B --repeats 3 --out-suffix _B$B
done

# Step 4: robustness bound and bandit on the new traces (CPU)
step robust $PY experiments/robust_bound.py --task wikitext_qlora --regime real_wikitext_qlora
step bandit $PY bandit/run_bandit.py --task wikitext_qlora --regime real_wikitext_qlora --B 8 --T 3000
echo "$(date +%F_%T) ALL DONE" >> $STATUS
