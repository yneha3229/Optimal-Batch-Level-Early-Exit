#!/usr/bin/env bash
# Report item 5.1 (CUDA graphs): wait until one of the candidate A6000s can initialise CUDA, then run the
# correctness check of the graph device and, if it passes, the timed graph runs (scripts/run_server_a6000_graph.sh).
# On 23-24 Nov no idle A6000 could start CUDA (GPU 5 "requires reset", GPUs 2 and 4 fail cuInit).
# Usage: setsid nohup scripts/wait_gpu_then_graph.sh > logs/wait_gpu_graph.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}
CANDIDATES=${CANDIDATES:-"5 4"}     # ours (5) and the one the user approved (4); never 2 or 3
while true; do
  for g in $CANDIDATES; do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    rc=$(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$g $PY -c 'import ctypes;print(ctypes.CDLL("libcuda.so.1").cuInit(0))' 2>/dev/null)
    if [ "$rc" = "0" ] && [ -n "$used" ] && [ "$used" -lt 2000 ]; then
      echo "$(date -u +%F_%T) GPU $g usable (cuInit 0, ${used} MiB used)"
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      if CUDA_VISIBLE_DEVICES=$g taskset -c 0-3,32-35 $PY experiments/check_graph_device.py --task mnli > logs/check_graph_device_gpu$g.log 2>&1; then
        echo "$(date -u +%F_%T) graph device check passed; starting timed runs"
        GPU_ID=$g CPU_CPUS=4-7,36-39 scripts/run_server_a6000_graph.sh > logs/server_a6000_graph.log 2>&1
        echo "$(date -u +%F_%T) graph runs finished: $(tail -1 logs/server_a6000_graph_status.log)"
        GPU_ID=$g scripts/run_server_a6000_vit.sh > logs/server_a6000_vit.log 2>&1
        echo "$(date -u +%F_%T) CIFAR-100 runs finished: $(tail -1 logs/server_a6000_vit_status.log)"
        $PY analysis/make_report.py > logs/make_report_after_gpu.log 2>&1 && echo "$(date -u +%F_%T) report rebuilt"
      else
        echo "$(date -u +%F_%T) graph device check FAILED, see logs/check_graph_device_gpu$g.log"
      fi
      exit 0
    fi
  done
  sleep 300
done
