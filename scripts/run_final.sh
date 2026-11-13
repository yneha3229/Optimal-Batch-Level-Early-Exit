#!/bin/bash
# Final run, fully detached (setsid nohup scripts/run_final.sh &). Strictly sequential so that no CPU-heavy step
# overlaps a GPU timing step. Status: logs/final_status.log; per-step logs: logs/final_<step>.log.
cd ~/ie624
PY=~/venv/bin/python
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3"
STATUS=logs/final_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/final_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
echo "$(date +%F_%T) final run started (pid $$)" >> $STATUS
while pgrep -u neha -f "decoder/train_decoder_exit[s]" > /dev/null; do sleep 60; done

# --- real decoder (large rho): calibrate the runner, simulate with its costs, then time on the GPU
step dec_calib     $GPU $PY decoder/decode_runner.py --calibrate
step dec_sim       $PY experiments/run_main.py --task wikitext --workers 40
step dec_gpu       $GPU $PY decoder/decode_runner.py --regime real_wikitext --batch-sizes 16 64
step dec_robust    $PY experiments/robust_bound.py --task wikitext --regime real_wikitext
step dec_bandit    $PY bandit/run_bandit.py --task wikitext --regime real_wikitext --B 64 --T 3000

# --- encoders: calibrate the runner, simulate all regimes (+ real), then time on the GPU
for t in mnli qqp cifar100; do
  step calib_$t  $GPU $PY runner/gpu_runner.py --task $t --calibrate
  step sim_$t    $PY experiments/run_main.py --task $t --workers 40
  step gpu_$t    $GPU $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 16 64 128
  step robust_$t $PY experiments/robust_bound.py --task $t --regime dec_eager
done
step bandit_qqp_dec      $PY bandit/run_bandit.py --task qqp --regime dec_eager --B 64 --T 3000
step bandit_cifar100_dec $PY bandit/run_bandit.py --task cifar100 --regime dec_eager --B 64 --T 3000
step bandit_mnli_dec     $PY bandit/run_bandit.py --task mnli --regime dec_eager --B 64 --T 3000
step bandit_mnli_enc     $PY bandit/run_bandit.py --task mnli --regime enc_eager_meas --B 64 --T 3000

for t in wikitext mnli qqp cifar100; do step figs_$t $PY analysis/make_figures.py --task $t; done
step tests $PY -m pytest tests simulator/test_simulator.py -q
echo "$(date +%F_%T) final run finished" >> $STATUS
