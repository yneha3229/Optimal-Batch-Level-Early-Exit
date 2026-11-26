#!/usr/bin/env bash
# Rerun the steps affected by the calibration-cache fix in experiments/local_hw.py (make(): the cost rule's
# calibration table was cached under id(cal_tr), which Python reuses after garbage collection, so resplit and
# cert_check results depended on worker scheduling). Only the families theory / theory+plan change.
# Writes results/local_resplit_{qqp,mnli}_cuda_T16.csv (Table 10), results/cert_check_*.csv (Table 13), then the
# figures and the report (docs/IE624_progress_report_v2.{html,pdf}).
# Detached:  setsid nohup scripts/rerun_cache_fix.sh > logs/cache_fix_run.log 2>&1 < /dev/null &
# Resumes:   steps logged OK in logs/cache_fix_status.log are skipped (RESUME=0 reruns all).
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv/bin/python}; W=${W:-16}
ST=logs/cache_fix_status.log; touch "$ST"
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1
run() {
  local name=$1; shift
  if [ "${RESUME:-1}" = 1 ] && grep -q "OK $name\$" "$ST"; then echo "skip $name"; return; fi
  local t0=$SECONDS
  if nice -n 10 "$@" > "logs/cache_fix_$name.log" 2>&1; then
    echo "$(date +%F_%T) OK $name" >> "$ST"; echo "$(date +%T) OK $name ($((SECONDS - t0)) s)"
  else
    echo "$(date +%F_%T) FAIL $name" >> "$ST"; echo "$(date +%T) FAIL $name"; exit 1
  fi
}
for t in qqp mnli; do
  run resplit_$t $PY experiments/local_hw.py resplit --task $t --seq-len 16 --batch-sizes 8 64 --seeds 30 --workers $W
done
for t in qqp mnli cifar100; do
  extra="--name a6000 --seq-len 128"; [ $t = cifar100 ] && extra="--cost-json results/runner_cost_cifar100.json --seq-len 197"
  run cert_check_$t $PY experiments/cert_check.py --task $t $extra --seeds 30 --workers $W
done
run cert_check_mnli_d05 $PY experiments/cert_check.py --task mnli --name a6000 --seeds 30 --workers $W --methods t --delta 0.05 --tag _d05
run report_figures $PY analysis/report_figures.py
run make_report $PY analysis/make_report.py
echo "done"
