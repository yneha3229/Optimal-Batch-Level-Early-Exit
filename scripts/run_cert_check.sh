#!/usr/bin/env bash
# Report item 5.2: finite-sample (HB, WSR) and label-free certificates, and calibration-data fractions, on all tasks.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}; CPUS=${CPUS:-4-11,36-43}; W=${W:-14}; SEEDS=${SEEDS:-30}
ST=logs/cert_check_status.log
for t in qqp mnli cifar100; do
  extra=(--name a6000 --seq-len 128); [ $t = cifar100 ] && extra=(--cost-json results/runner_cost_cifar100.json --seq-len 197)
  echo "$(date +%F_%T) START $t" >> $ST
  if taskset -c $CPUS $PY experiments/cert_check.py --task $t "${extra[@]}" --seeds $SEEDS --workers $W 2>&1 | grep -v -i warn > logs/cert_check_$t.log; then
    echo "$(date +%F_%T) OK    $t" >> $ST; else echo "$(date +%F_%T) FAIL  $t" >> $ST; fi
done
echo "$(date +%F_%T) DONE" >> $ST
