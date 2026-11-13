#!/usr/bin/env bash
# Accuracy-speed fronts (CPU replay with the measured GTX 1650 cost tables). Run after scripts/run_local.sh.
cd "$(dirname "$0")/.." || exit 1
for T in 128 16; do for task in mnli qqp; do
  python3 experiments/local_hw.py front --task $task --seq-len $T --batch-sizes 8 16 64 --workers 7 2>&1 | grep -v -i warn > logs/local_front_${task}_T$T.log
  echo "front $task T$T done $(date)" >> logs/local_front_status.log
done; done
echo "ALL DONE $(date)" >> logs/local_front_status.log
