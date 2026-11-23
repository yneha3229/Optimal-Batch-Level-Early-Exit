#!/usr/bin/env bash
# MNLI held-out overshoot checks: fixed split at several delta, fluctuating load and re-split check at delta = 0.05.
cd "$(dirname "$0")/.." || exit 1; P=$HOME/venv/bin/python; C=4-11,36-43; ST=logs/delta_check_status.log
taskset -c $C $P experiments/delta_check.py --task mnli --name a6000 --workers 12 2>&1 | grep -vi warn > logs/delta_check_mnli.log && echo "$(date +%F_%T) OK delta_check mnli" >> $ST
taskset -c $C $P experiments/load_trace.py --task mnli --name a6000 --workers 12 --delta 0.05 --tag _d05 2>&1 | grep -vi warn > logs/load_trace_mnli_d05.log && echo "$(date +%F_%T) OK load_trace mnli d05" >> $ST
taskset -c $C $P experiments/cert_check.py --task mnli --name a6000 --seeds 30 --workers 12 --methods t --delta 0.05 --tag _d05 2>&1 | grep -vi warn > logs/cert_check_mnli_d05.log && echo "$(date +%F_%T) OK cert_check mnli d05" >> $ST
echo "$(date +%F_%T) DONE" >> $ST
