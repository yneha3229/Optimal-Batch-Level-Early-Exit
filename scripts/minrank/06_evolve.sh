#!/usr/bin/env bash
# Run OpenEvolve on the minrank seed program (Option A). Needs a running LLM server (05_vllm_serve.sh).
#
# Usage (from the repository root):
#   RUN=evo1 ITER=300 PORT=8000 MODEL=Qwen/Qwen2.5-Coder-32B-Instruct-AWQ bash scripts/minrank/06_evolve.sh
# Env: RUN (output name), ITER, PORT, MODEL, PY, EVO_BUDGET (s per graph), EVO_WORKERS (parallel graphs per
#      candidate), REF_WORKERS, REF_COVER_TIME, REF_LB_TIME, API_BASE (overrides PORT).
# Output: minrank/evolve/runs/$RUN/ (OpenEvolve checkpoints, best/best_program.py),
#         minrank/evolve/runs/$RUN/traces/evals.jsonl (one record per candidate x graph; input of Option D).
# Resume: CHECKPOINT=minrank/evolve/runs/$RUN/checkpoints/checkpoint_100 bash scripts/minrank/06_evolve.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-evo1}
ITER=${ITER:-300}
PORT=${PORT:-8000}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}
API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
PY=${PY:-python}
OUT=minrank/evolve/runs/${RUN}
mkdir -p "${OUT}" logs

export MINRANK_RUN="${RUN}"
export MINRANK_TRACE_DIR="${OUT}/traces"
export MINRANK_EVO_BUDGET=${EVO_BUDGET:-2.0}
export MINRANK_EVO_WORKERS=${EVO_WORKERS:-4}

[ -f minrank/bench/graphs/index.csv ] || "${PY}" -m minrank.bench.instances
# Reference (strong clique cover + certified lower bound) for the evolution graphs; computed once, cached.
"${PY}" -m minrank.evolve.reference --split train val --family planted planted_d gnp --max-n 50 \
  --cover-time "${REF_COVER_TIME:-10}" --lb-time "${REF_LB_TIME:-30}" --workers "${REF_WORKERS:-8}"

# Sanity: the seed must pass its own evaluator before spending GPU time.
"${PY}" -m minrank.evolve.evaluator minrank/evolve/seed_program.py --no-trace | tail -n 12

curl -sf "${API_BASE}/models" >/dev/null || { echo "no LLM server at ${API_BASE}; run 05_vllm_serve.sh first"; exit 1; }

if command -v openevolve-run >/dev/null 2>&1; then OE=(openevolve-run); else OE=("${PY}" -m openevolve.cli); fi
EXTRA=()
[ -n "${CHECKPOINT:-}" ] && EXTRA+=(--checkpoint "${CHECKPOINT}")
"${OE[@]}" minrank/evolve/seed_program.py minrank/evolve/evaluator.py \
  --config minrank/evolve/config.yaml --iterations "${ITER}" --output "${OUT}" \
  --api-base "${API_BASE}" --primary-model "${MODEL}" "${EXTRA[@]}" 2>&1 | tee "logs/minrank_evolve_${RUN}.log"
echo "best program: ${OUT}/best/best_program.py ; traces: ${MINRANK_TRACE_DIR}/evals.jsonl"
