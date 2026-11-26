#!/usr/bin/env bash
# Build the summary tables and figures from everything in minrank/results/ and minrank/certificates/.
# Fails (non-zero exit) if any stored certificate does not verify against the current graphs.
# Usage (repo root):  bash scripts/minrank/09_aggregate.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY=${PY:-python}
$PY -m minrank.analysis.aggregate
