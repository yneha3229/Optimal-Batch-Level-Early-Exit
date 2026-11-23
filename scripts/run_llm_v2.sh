#!/bin/bash
# LLM track, second attempt (docs/PLAN_LLM.md). One stage per call; read the gate table of that stage in
# docs/PLAN_LLM.md section 4 before starting the next one.
#   GPU_ID=<n> nohup scripts/run_llm_v2.sh train &    steps 0-1: functional test, joint training, frozen baseline
#              nohup scripts/run_llm_v2.sh adapt &    step 2: adaptivity vs batch size (CPU)
#   GPU_ID=<n> nohup scripts/run_llm_v2.sh time  &    step 3: real wall-clock (GPU must be idle)
#              nohup scripts/run_llm_v2.sh post  &    step 4: robustness bound and bandit (CPU)
# GPU_ID has no default: pick an idle A6000 with nvidia-smi (GPU 5 is reserved for the accuracy-matched rerun).
cd "$(dirname "$0")/.." || exit 1
STAGE=$1
PY=${PY:-$HOME/venv/bin/python}
STATUS=logs/llm2_status.log
SANITY_MIN=0.97
case "$STAGE" in
  train|time) [ -n "$GPU_ID" ] || { echo "set GPU_ID (an idle A6000; see nvidia-smi)" >&2; exit 2; } ;;
  adapt|post) ;;
  *) echo "usage: [GPU_ID=<n>] $0 train|adapt|time|post" >&2; exit 2 ;;
esac
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU_ID"
mkdir -p logs checkpoints data
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/llm2_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; exit 1; fi; }
# The runner prints "sanity: ... final-exit token agreement X"; fail the stage if X is below SANITY_MIN.
sanity() { local v; v=$(grep -o "final-exit token agreement [0-9.]*" logs/llm2_$1.log | tail -1 | grep -o "[0-9.]*$")
           if [ -n "$v" ] && awk -v v="$v" -v m=$SANITY_MIN 'BEGIN { exit !(v >= m) }'; then
             echo "$(date +%F_%T) OK    sanity $1 = $v" >> $STATUS
           else echo "$(date +%F_%T) FAIL  sanity $1 = ${v:-missing} (need $SANITY_MIN)" >> $STATUS; exit 1; fi; }

case "$STAGE" in
train)
  # Step 0: functional test (training, then every runner path: stops, per-sample exits, KV propagation)
  step smoke_train $GPU $PY decoder/train_exit_lora.py --tag smoke --steps 20 --bs 2 --train-lines 2000 --eval-windows 16
  step smoke_decode $GPU $PY decoder/decode_runner.py --task wikitext_smoke --max-windows 8 --smoke
  sanity smoke_decode
  git checkout traces/wikitext_smoke_val.parquet results/train_wikitext_smoke_exit_accuracy.csv   # tracked files of the first attempt
  # Step 1: joint training, and the frozen baseline at the same budget
  step train_qlora   $GPU $PY decoder/train_exit_lora.py --tag qlora   --rank 16 --steps 6000
  step train_qfrozen $GPU $PY decoder/train_exit_lora.py --tag qfrozen --rank 0  --steps 6000
  ;;
adapt)
  # Step 2: adaptivity vs batch size, budget on accuracy and on fidelity
  T="wikitext_qlora wikitext_qfrozen wikitext wikitext_llama"
  step adaptivity     $PY analysis/adaptivity_vs_batch.py --tasks $T --budgets 0.01 0.03
  step adaptivity_fid $PY analysis/adaptivity_vs_batch.py --tasks $T --metric fidelity --budgets 0.05 0.10
  ;;
time)
  # Step 3: real wall-clock
  step calib $GPU $PY decoder/decode_runner.py --task wikitext_qlora --calibrate
  sanity calib
  step sim   $PY experiments/run_main.py --task wikitext_qlora --regimes real_wikitext_qlora --batch-sizes 2 4 8 16 64 --workers 32
  for B in 2 4 8 16 64; do
    step gpu_B$B $GPU $PY decoder/decode_runner.py --task wikitext_qlora --batch-sizes $B --repeats 3 --out-suffix _B$B
  done
  ;;
post)
  # Step 4: robustness bound and bandit on the new traces
  step robust $PY experiments/robust_bound.py --task wikitext_qlora --regime real_wikitext_qlora
  step bandit $PY bandit/run_bandit.py --task wikitext_qlora --regime real_wikitext_qlora --B 8 --T 3000
  ;;
esac
echo "$(date +%F_%T) STAGE $STAGE DONE" >> $STATUS
