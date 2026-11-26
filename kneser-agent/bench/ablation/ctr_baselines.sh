#!/bin/bash
# Container baselines on the full set, one core each: standard methods + Kushagra's script (core 0) and the agent
# without LLM (core 1). Output: results/baselines_ctr.jsonl and rows tagged "ctr" in results/runs.jsonl.
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1
SPECS=$(python3 -c "import json;d=json.load(open('bench/specs.json'));print(' '.join(d['benchmark']+d['large']))")
LARGE=$(python3 -c "import json;print(' '.join(json.load(open('bench/specs.json'))['large']))")
BENCH=$(python3 -c "import json;print(' '.join(json.load(open('bench/specs.json'))['benchmark']))")
(taskset -c 0 python bench/baselines.py run --limit 60 --par 1 --out results/baselines_ctr.jsonl $BENCH
 taskset -c 0 python bench/baselines.py run --limit 120 --par 1 --out results/baselines_ctr.jsonl $LARGE) \
  > bench/ablation/chat/ctr_baselines.log 2>&1 &
(for s in $SPECS; do taskset -c 1 python -m kagent.pipeline "$s" --llm none --budget 3600 --tag ctr | tail -1; done) \
  > bench/ablation/chat/ctr_agent.log 2>&1 &
wait
