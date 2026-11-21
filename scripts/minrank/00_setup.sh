#!/usr/bin/env bash
# Install Python dependencies for minrank/ and run the unit tests.
# Usage (repo root):  bash scripts/minrank/00_setup.sh
# Env: PY (python executable, default: python), EXTRAS=1 to also install optional packages (OpenEvolve, vLLM, cvxpy).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
$PY -m pip install -r minrank/requirements.txt pandas matplotlib
if [[ "${EXTRAS:-0}" == "1" ]]; then
  $PY -m pip install openevolve cvxpy
  $PY -m pip install vllm   # GPU server only
fi
$PY -m pytest minrank/tests -q
