#!/bin/bash
# One blind model run on the full set (20 benchmark + 3 large graphs), pinned to one core so compute times compare.
# usage: bench/ablation/full_run.sh CONFIG SPEC CORE     (e.g. opus-med "K(21,2)" 2); tag CONFIG-full, seed 0
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$HERE/chat/$1-full_$(printf %s "$2" | tr -c 'A-Za-z0-9' _)s0"
rm -rf "$D"; mkdir -p "$D"; cd "$HERE/../.."
OMP_NUM_THREADS=1 taskset -c "$3" python -m kagent.pipeline "$2" --seed 0 --llm "chat:$D" --budget 3600 \
  --tag "$1-full" > "$D/pipeline.log" 2>&1
touch "$D/DONE"
