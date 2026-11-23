#!/usr/bin/env bash
# Report item 5.4: bandit with ten seeds and 95% intervals, probe-rate ablation (t^-1/3, t^-1/2, constant), warm start.
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}; CPUS=${CPUS:-4-11,36-43}; ST=logs/bandit_seeds_status.log
for t in mnli qqp cifar100; do for r in dec_eager real_$t; do
  ( taskset -c $CPUS $PY bandit/run_bandit.py --task $t --regime $r --B 64 --T 3000 --seeds 0 1 2 3 4 5 6 7 8 9 --ablation \
      --tag _10seeds 2>&1 | grep -v -i warn > logs/bandit_10seeds_${t}_$r.log \
    && echo "$(date +%F_%T) OK   $t $r" >> $ST || echo "$(date +%F_%T) FAIL $t $r" >> $ST ) &
done; done
wait; echo "$(date +%F_%T) DONE" >> $ST
