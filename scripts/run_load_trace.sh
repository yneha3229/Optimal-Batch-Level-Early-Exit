#!/usr/bin/env bash
# Report item 5.3 / DIRECTION.md step 3: fluctuating load (batch size 1..128 from an arrival trace).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}; CPUS=${CPUS:-4-11,36-43}; ST=logs/load_trace_status.log
for t in qqp mnli cifar100; do
  extra=(--name a6000 --seq-len 128); [ $t = cifar100 ] && extra=(--cost-json results/runner_cost_cifar100.json --seq-len 197)
  taskset -c $CPUS $PY experiments/load_trace.py --task $t "${extra[@]}" --workers 12 2>&1 | grep -v -i warn > logs/load_trace_$t.log \
    && echo "$(date +%F_%T) OK   $t" >> $ST || echo "$(date +%F_%T) FAIL $t" >> $ST
done
echo "$(date +%F_%T) DONE" >> $ST
