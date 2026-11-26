#!/usr/bin/env bash
# Qwen2.5-Coder-32B (GGUF Q4_K_M) on CPU with llama.cpp, for the model ablation while the GPUs fail cuInit (26 Nov).
# Server: cores 16-27 (NUMA node 1), 3 parallel slots; pipeline: cores 28-31. Same code/prompts as the Claude rows.
set -u
cd "$(dirname "$0")/../.."
M=$HOME/llm_cpu/models/qwen2.5-coder-32b-instruct-q4_k_m.gguf
LS=$HOME/llm_cpu/llamacpp/llama-b12780/llama-server
LOG=../logs
while [ ! -f "$M" ]; do sleep 10; done
echo "model ready $(date -u)"
taskset -c 16-27 "$LS" -m "$M" --host 127.0.0.1 --port 8002 -t 12 --parallel 3 -c 24576 --alias qwen32b-q4km \
  > $LOG/kneser_agent_llamacpp_server.log 2>&1 &
echo $! > $LOG/kneser_agent_llamacpp.pid
until curl -sf http://127.0.0.1:8002/health >/dev/null; do sleep 5; kill -0 "$(cat $LOG/kneser_agent_llamacpp.pid)" || { echo "server died"; exit 1; }; done
echo "server ready $(date -u)"
export KAGENT_LLM_TIMEOUT=3000
L="local:http://127.0.0.1:8002/v1|Qwen2.5-Coder-32B-Instruct-Q4_K_M-cpu"
PIDS=""
for a in "K(16,4) 3" "K(16,4)-del0.01 0" "K(14,4) 0"; do
  set -- $a
  taskset -c 28-31 ~/venv/bin/python -m kagent.pipeline "$1" --seed "$2" --llm "$L" --budget 7200 \
    > "$LOG/kneser_agent_cpuqwen_$(printf %s "$1" | tr -c 'A-Za-z0-9' _)s$2.log" 2>&1 &
  PIDS="$PIDS $!"
done
for p in $PIDS; do wait $p; done
echo "runs done $(date -u)"
