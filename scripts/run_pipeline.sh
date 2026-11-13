#!/bin/bash
# Everything that remains, fully detached from any terminal (launch with: setsid nohup scripts/run_pipeline.sh &).
# Order: wait for training/attack + FunSearch -> simulations for all tasks -> bandit runs -> real-GPU calibration
# and wall-clock timing on GPU 3 (only once nothing else uses it) -> figures. Each step logs to logs/pipeline_*.log;
# a failed step is recorded and the pipeline moves on.
cd ~/ie624
PY=~/venv/bin/python
GPU="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3"
STATUS=logs/pipeline_status.log
step() { local name=$1; shift; echo "$(date +%F_%T) START $name" >> $STATUS
         if "$@" > logs/pipeline_$name.log 2>&1; then echo "$(date +%F_%T) OK    $name" >> $STATUS
         else echo "$(date +%F_%T) FAIL  $name (exit $?)" >> $STATUS; fi; }
wait_for() { while pgrep -u neha -f "$1" > /dev/null; do sleep 60; done; }

echo "$(date +%F_%T) pipeline started (pid $$)" >> $STATUS

# 1. simulations that only need finished traces (CPU)
wait_for "experiments/run_main.py"
wait_for "scripts/train_text.sh"
step sim_qqp      $PY experiments/run_main.py --task qqp --regimes enc_eager_meas dec_eager --workers 40
step bandit_mnli_dec $PY bandit/run_bandit.py --task mnli --regime dec_eager --B 64 --T 3000
step bandit_mnli_enc $PY bandit/run_bandit.py --task mnli --regime enc_eager_meas --B 64 --T 3000
wait_for "scripts/train_vision.sh"
step sim_cifar100 $PY experiments/run_main.py --task cifar100 --regimes enc_eager_meas dec_eager --workers 40
step bandit_cifar100_dec $PY bandit/run_bandit.py --task cifar100 --regime dec_eager --B 64 --T 3000

# 2. FunSearch result -> include the evolved program in the MNLI comparison
wait_for "openevolve-run"
step evolve_extract $PY evolve/extract.py --task mnli
pkill -u neha -x ollama   # free GPU 3: stops only this user's private Ollama (the shared service runs as other users)

# 3. real GPU: calibrate the runner's own cost model, re-select policies with it, then time them
for t in mnli qqp cifar100; do
  [ -f checkpoints/$t.pt ] || continue
  step calib_$t   $GPU $PY runner/gpu_runner.py --task $t --calibrate
  step simreal_$t $PY experiments/run_main.py --task $t --workers 40
  step gpu_$t     $GPU $PY runner/gpu_runner.py --task $t --regime real_$t --batch-sizes 16 64 128
done

# 4. figures + tables
for t in mnli qqp cifar100; do step figs_$t $PY analysis/make_figures.py --task $t; done
step tests $PY -m pytest tests -q
echo "$(date +%F_%T) pipeline finished" >> $STATUS
