#!/bin/bash
# One blind model run: the pipeline writes its LLM requests into a folder of their own; a fresh agent answers them.
# usage: bench/ablation/run.sh CONFIG SPEC SEED      (e.g. opus-high "K(14,4)" 0)
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$HERE/chat/$1_$(printf %s "$2" | tr -c 'A-Za-z0-9' _)s$3"
mkdir -p "$D"; cd "$HERE/../.."
python -m kagent.pipeline "$2" --seed "$3" --llm "chat:$D" --budget 3600 --tag "$1" > "$D/pipeline.log" 2>&1
touch "$D/DONE"
