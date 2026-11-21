#!/usr/bin/env bash
# Write the frozen benchmark (172 graphs + index.csv) to minrank/bench/graphs/. Already committed; rerun only if the
# generators change (certificates for changed graphs are then replaced automatically: digests differ).
# Usage (repo root):  bash scripts/minrank/01_bench.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
$PY -m minrank.bench.instances --out minrank/bench/graphs
