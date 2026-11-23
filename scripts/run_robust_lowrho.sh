#!/usr/bin/env bash
# Report item 5.5: robustness in the compute-bound (low rho) regimes, with attackers beyond the certificate (k > m).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}; CPUS=${CPUS:-4-11,36-43}; ST=logs/robust_lowrho_status.log
for t in mnli qqp cifar100; do for r in enc_eager_meas enc_graph real_$t; do
  ( taskset -c $CPUS $PY experiments/robust_bound.py --task $t --regime $r --B 64 --ks 0 1 2 4 8 16 32 --tag _lowrho \
      2>&1 | grep -v -i warn > logs/robust_lowrho_${t}_$r.log \
    && echo "$(date +%F_%T) OK   $t $r" >> $ST || echo "$(date +%F_%T) FAIL $t $r" >> $ST ) &
done; done
wait; echo "$(date +%F_%T) DONE" >> $ST
