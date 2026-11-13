#!/usr/bin/env bash
# Local-hardware verification on the GTX 1650: selection (CPU) -> real timing (GPU, nothing else running) -> re-split check.
cd "$(dirname "$0")/.." || exit 1
run() { python3 experiments/local_hw.py "$@" 2>&1 | grep -v -i warn; }
for T in 16 32 128; do for task in qqp mnli; do
  [ -s results/local_select_${task}_cuda_T$T.json ] || run select --task $task --seq-len $T --batch-sizes 8 16 64 --workers 7 > logs/local_select_${task}_T$T.log
done; done
echo "select done $(date)" > logs/local_status.log
for T in 16 32 128; do for task in qqp mnli; do
  n=1920; r=5; [ $T = 128 ] && n=768 && r=3
  run time --task $task --seq-len $T --batch-sizes 8 16 64 --n $n --rounds $r > logs/local_time_${task}_T$T.log
  echo "time $task T$T done $(date)" >> logs/local_status.log
done; done
for task in qqp mnli; do
  run resplit --task $task --seq-len 16 --batch-sizes 8 64 --seeds 30 --workers 7 > logs/local_resplit_${task}_T16.log
  echo "resplit $task done $(date)" >> logs/local_status.log
done
echo "ALL DONE $(date)" >> logs/local_status.log
