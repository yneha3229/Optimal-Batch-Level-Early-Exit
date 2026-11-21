#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for OpenEvolve (Option A) on one GPU and wait until it answers.
#
# Usage (from the repository root):
#   GPU=5 PORT=8000 MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ bash scripts/minrank/05_vllm_serve.sh
# Env: GPU (CUDA index, PCI order), PORT, MODEL, MAX_LEN, GPU_UTIL, PY (python executable), LOG.
# The server keeps running in the background (PID in logs/minrank_vllm_${PORT}.pid). Stop it with:
#   kill "$(cat logs/minrank_vllm_${PORT}.pid)"
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU=${GPU:-0}
PORT=${PORT:-8000}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}
MAX_LEN=${MAX_LEN:-16384}
GPU_UTIL=${GPU_UTIL:-0.90}
PY=${PY:-python}
mkdir -p logs
LOG=${LOG:-logs/minrank_vllm_${PORT}.log}

if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "a server already answers on port ${PORT}:"; curl -s "http://127.0.0.1:${PORT}/v1/models"; echo; exit 0
fi
echo "GPU ${GPU} must be idle; current use:"; nvidia-smi --id="${GPU}" --query-gpu=memory.used,utilization.gpu --format=csv || true

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="${GPU}" nohup "${PY}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" --port "${PORT}" --max-model-len "${MAX_LEN}" --gpu-memory-utilization "${GPU_UTIL}" \
  > "${LOG}" 2>&1 &
echo $! > "logs/minrank_vllm_${PORT}.pid"
echo "vLLM starting (pid $(cat "logs/minrank_vllm_${PORT}.pid")), log: ${LOG}"

for _ in $(seq 1 180); do  # up to 15 minutes (model download + load)
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "ready on http://127.0.0.1:${PORT}/v1 serving ${MODEL}"; exit 0
  fi
  if ! kill -0 "$(cat "logs/minrank_vllm_${PORT}.pid")" 2>/dev/null; then
    echo "vLLM exited; last log lines:"; tail -n 30 "${LOG}"; exit 1
  fi
  sleep 5
done
echo "timed out waiting for vLLM; see ${LOG}"; exit 1
