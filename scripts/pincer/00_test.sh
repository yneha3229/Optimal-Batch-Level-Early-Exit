#!/usr/bin/env bash
# Install Pincer's extra requirements (on top of minrank/requirements.txt) and run its tests.
set -euo pipefail
PY=${PY:-python}
$PY -m pip install -q -r pincer/requirements.txt
PYTHONPATH=. $PY -m pytest pincer/tests -q
