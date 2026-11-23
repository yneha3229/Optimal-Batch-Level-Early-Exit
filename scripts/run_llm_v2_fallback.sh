#!/bin/bash
# Step 1 fallback (docs/PLAN_LLM.md): rank 64, 12000 steps, 128 positions. Waits for the "train" stage to finish first.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}
STATUS=logs/llm2_status.log
until grep -q "STAGE train DONE\|FAIL" $STATUS; do sleep 60; done
grep -q "FAIL" $STATUS && exit 1
echo "$(date +%F_%T) START train_qlora64" >> $STATUS
if env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU_ID:?} $PY decoder/train_exit_lora.py --tag qlora64 --rank 64 \
     --steps 12000 --positions 128 > logs/llm2_train_qlora64.log 2>&1; then
  echo "$(date +%F_%T) OK    train_qlora64" >> $STATUS
else echo "$(date +%F_%T) FAIL  train_qlora64 (exit $?)" >> $STATUS; exit 1; fi
