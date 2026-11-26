#!/usr/bin/env bash
# Report item 5.5, accuracy-matched: every policy tuned by certified selection (robust_bound.py --tune cert), so all
# are within the 1-point budget on held-out; low-rho regimes plus dec_eager, attackers beyond the certificate (k > m).
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-$HOME/venv/bin/python}; CPUS=${CPUS:-16-31,48-63}; ST=logs/robust_cert_status.log
for t in mnli qqp cifar100; do for r in enc_eager_meas enc_graph real_$t dec_eager; do
  ( taskset -c $CPUS $PY experiments/robust_bound.py --task $t --regime $r --B 64 --ks 0 1 2 4 8 16 32 --tune cert \
      --tag _cert 2>&1 | grep -v -i warn > logs/robust_cert_${t}_$r.log \
    && echo "$(date +%F_%T) OK   $t $r" >> $ST || echo "$(date +%F_%T) FAIL $t $r" >> $ST ) &
done; done
wait; echo "$(date +%F_%T) DONE" >> $ST
