#!/usr/bin/env bash
# Re-targeted OpenEvolve run (Option A, second attempt): evolve on the HARD training graphs, with fitness relative to
# SAT at the same time budget per graph (evaluator mode MINRANK_EVO_TARGET=sat; see minrank/evolve/evaluator.py).
# Needs a running LLM server (05_vllm_serve.sh). Does not touch the default run's settings or outputs.
#
# Usage (from the repository root):
#   RUN=evo_hard1 PORT=8000 bash scripts/minrank/06b_evolve_hard.sh
#   FAMILIES=planted_d MIN_N=50 MAX_N=50 EVO_BUDGET=10 bash scripts/minrank/06b_evolve_hard.sh   # n=50 planted_d only
#   PREPARE_ONLY=1 bash scripts/minrank/06b_evolve_hard.sh     # references + seed check, no LLM needed
# Env: RUN, ITER, PORT, MODEL, API_BASE, PY, CPUS (taskset list for the evaluator and the SAT reference, default the
#      evolve cores 12-15,44-47), FAMILIES (comma list, planted_d,planted), MIN_N (50), MAX_N (100),
#      HARD (auto | all | comma list of graph names; auto = SAT neither proves optimality nor reaches the lower bound),
#      EVO_BUDGET (s per graph, 10; the SAT reference uses the same), EVO_WORKERS (4), REF_COVER_TIME (10),
#      REF_LB_TIME (30), SAT_REF (reference file, minrank/evolve/reference_sat.json), CHECKPOINT, PREPARE_ONLY.
# Output: minrank/evolve/runs/$RUN/ (checkpoints, best/best_program.py), minrank/evolve/runs/$RUN/traces/evals.jsonl
#         (records carry ref_rank = SAT rank and target = "sat"), logs/minrank_evolve_$RUN.log.
# The SAT reference is wall-clock limited, so it is computed once (cached in $SAT_REF, keyed by graph, budget and
# seed) with EVO_WORKERS parallel runs on CPUS: the same load the candidates see during evolution.
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-evo_hard1}
ITER=${ITER:-150}
PORT=${PORT:-8000}
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-32B-Instruct-AWQ}
API_BASE=${API_BASE:-http://127.0.0.1:${PORT}/v1}
PY=${PY:-python}
CPUS=${CPUS:-12-15,44-47}
FAMILIES=${FAMILIES:-planted_d,planted}
OUT=minrank/evolve/runs/${RUN}
mkdir -p "${OUT}" logs

export MINRANK_RUN="${RUN}"
export MINRANK_TRACE_DIR="${OUT}/traces"
export MINRANK_EVO_TARGET=sat
export MINRANK_EVO_FAMILIES="${FAMILIES}"
export MINRANK_EVO_MIN_N=${MIN_N:-50}
export MINRANK_EVO_MAX_N=${MAX_N:-100}
export MINRANK_EVO_HARD=${HARD:-auto}
export MINRANK_EVO_BUDGET=${EVO_BUDGET:-10}
export MINRANK_EVO_WORKERS=${EVO_WORKERS:-4}
export MINRANK_SAT_REFERENCE=${SAT_REF:-minrank/evolve/reference_sat.json}

[ -f minrank/bench/graphs/index.csv ] || "${PY}" -m minrank.bench.instances
# Clique cover + certified lower bound (same settings as 06_evolve.sh; adds the n > 50 entries, keeps existing ones).
# shellcheck disable=SC2046
taskset -c "${CPUS}" "${PY}" -m minrank.evolve.reference --split train val --family $(tr , ' ' <<< "${FAMILIES}") \
  --max-n "${MINRANK_EVO_MAX_N}" --cover-time "${REF_COVER_TIME:-10}" --lb-time "${REF_LB_TIME:-30}" \
  --workers "${MINRANK_EVO_WORKERS}"
# SAT at the evaluation budget, once per graph (cached).
# shellcheck disable=SC2046
taskset -c "${CPUS}" "${PY}" -m minrank.evolve.reference_sat --split train val \
  --family $(tr , ' ' <<< "${FAMILIES}") --min-n "${MINRANK_EVO_MIN_N}" --max-n "${MINRANK_EVO_MAX_N}" \
  --budget "${MINRANK_EVO_BUDGET}" --workers "${MINRANK_EVO_WORKERS}" --out "${MINRANK_SAT_REFERENCE}"

# Sanity: the seed must pass its own evaluator on the hard set before spending GPU time (prints the selected graphs).
taskset -c "${CPUS}" "${PY}" -m minrank.evolve.evaluator minrank/evolve/seed_program.py --no-trace | tail -n 30
[ -n "${PREPARE_ONLY:-}" ] && exit 0

curl -sf "${API_BASE}/models" >/dev/null || { echo "no LLM server at ${API_BASE}; run 05_vllm_serve.sh first"; exit 1; }

if command -v openevolve-run >/dev/null 2>&1; then OE=(openevolve-run); else OE=("${PY}" -m openevolve.cli); fi
EXTRA=()
[ -n "${CHECKPOINT:-}" ] && EXTRA+=(--checkpoint "${CHECKPOINT}")
taskset -c "${CPUS}" "${OE[@]}" minrank/evolve/seed_program.py minrank/evolve/evaluator.py \
  --config minrank/evolve/config_hard.yaml --iterations "${ITER}" --output "${OUT}" \
  --api-base "${API_BASE}" --primary-model "${MODEL}" "${EXTRA[@]}" 2>&1 | tee "logs/minrank_evolve_${RUN}.log"
echo "best program: ${OUT}/best/best_program.py ; traces: ${MINRANK_TRACE_DIR}/evals.jsonl"
