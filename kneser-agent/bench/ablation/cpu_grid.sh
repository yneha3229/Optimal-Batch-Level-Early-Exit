#!/usr/bin/env bash
# Ablation grid with a GGUF model on CPU (llama.cpp): bench/ablation/cpu_grid.sh MODEL.gguf NAME [PARALLEL]
# Server on cores 16-29 (14 threads), pipelines on 30-31 (16 cores in total). Skips cells whose trace exists.
set -u
cd "$(dirname "$0")/../.."
M=$1; NAME=$2; PAR=${3:-6}
LS=$HOME/llm_cpu/llamacpp/llama-b12780/llama-server
LOG=../logs; PORT=8002
taskset -c 16-29 "$LS" -m "$M" --host 127.0.0.1 --port $PORT -t 14 --parallel "$PAR" -c $((PAR * 8192)) \
  > "$LOG/kneser_agent_llamacpp_${NAME}.log" 2>&1 &
SRV=$!; echo $SRV > "$LOG/kneser_agent_llamacpp.pid"
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; kill -0 $SRV || { echo "server died"; exit 1; }; done
echo "server ready $(date -u)"
export KAGENT_LLM_TIMEOUT=6000 HF_HUB_OFFLINE=1
PIDS=""
for c in "K(16,4) 3" "K(16,4) 1" "K(16,4)-del0.01 0" "K(16,4)-del0.01 1" "K(14,4) 0" "K(14,4) 1" \
         "K(13,4) 0" "K(13,4) 1" "K(15,5) 0" "K(15,5) 1" "reg(1000,6) 0" "reg(1000,6) 1"; do
  set -- $c
  g=$(printf %s "$1" | sed 's/(/_/; s/)//; s/,/_/')_s$2
  [ -f "traces/${g}_${NAME}.jsonl" ] && { echo "skip $g"; continue; }
  taskset -c 30-31 ~/venv/bin/python -m kagent.pipeline "$1" --seed "$2" --llm "local:http://127.0.0.1:$PORT/v1|$NAME" \
    --budget 21600 > "$LOG/kneser_agent_cpugrid_${NAME}_${g}.log" 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
kill $SRV
echo "grid done $(date -u)"
