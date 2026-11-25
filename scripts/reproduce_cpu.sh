#!/usr/bin/env bash
# Regenerate every CPU-derived number of the early-exit report from the committed traces and GPU cost tables, into
# results_repro/, and diff each output against results/ (docs/REPRODUCE_EARLY_EXIT.md, section 3).
#
# Nothing under results/, figures/ or docs/ is written. The scripts write to <repo>/results (their ROOT is the parent
# of their own directory), so they run in a shadow copy of the code, results_repro/.root/, whose results/ starts as a
# copy of the committed results/ with the files of the step deleted first. Its inputs are the GPU-measured files only
# (cost tables, timed runs, traces), which are listed per step below.
#
# Usage:   scripts/reproduce_cpu.sh                    all groups (about 4-6 h on 6 free cores; cert is 2-3 h of it)
#          ONLY="theory gtx a6000" scripts/reproduce_cpu.sh     selected groups
#          W=12 scripts/reproduce_cpu.sh               more worker processes (default 6)
#          RESUME=0 scripts/reproduce_cpu.sh           rerun every step (default: skip steps logged OK in timing.tsv,
#                                                      so the same command continues after a crash or reboot)
# Detached (survives the shell that started it):
#          setsid nohup scripts/reproduce_cpu.sh > results_repro/run.log 2>&1 < /dev/null &
# Groups (in run order): theory gtx a6000 delta load robust bandit theory58 transfer cert report
# Output:  results_repro/<file> (regenerated files), results_repro/logs/<step>.log, results_repro/timing.tsv,
#          results_repro/diff_summary.csv and results_repro/DIFF.md (MATCH / CLOSE / MISMATCH / MISSING per file).
# Tolerance: MATCH = every number within atol 1e-9 + rtol 1e-6; CLOSE = within atol 1e-4 + rtol 1e-3 (analysis/diff_repro.py).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$REPO/results_repro}
PY=${PY:-$HOME/venv/bin/python}
W=${W:-6}
ALL="theory gtx a6000 delta load robust bandit theory58 transfer cert report"
ONLY=${ONLY:-$ALL}
SH=$OUT/.root
export CUDA_VISIBLE_DEVICES="" PYTHONHASHSEED=0 HF_HUB_OFFLINE=1      # CPU only; never touches a GPU

mkdir -p "$OUT/logs"
if [ ! -d "$SH" ]; then
  mkdir -p "$SH/figures"
  for d in analysis attacks bandit configs decoder docs experiments profiling runner simulator tests theory; do
    cp -r "$REPO/$d" "$SH/$d"
  done
  ln -s "$REPO/traces" "$SH/traces"
  cp -r "$REPO/results" "$SH/results"           # inputs; each step deletes its own outputs before it runs
fi
TIMING=$OUT/timing.tsv; [ -s "$TIMING" ] || printf "step\tstatus\tseconds\tfinished\n" > "$TIMING"
DIFF_ITEMS=()

# step NAME GROUP "OUTPUT FILES (relative to results/)" COMMAND...
step() {
  local name=$1 group=$2 outs=$3; shift 3
  if [ "${RESUME:-1}" = 1 ] && awk -F'\t' -v s="$name" '$1 == s && $2 == "OK" {f = 1} END {exit !f}' "$TIMING"; then
    for f in $outs; do DIFF_ITEMS+=("$group:$f"); [ -e "$SH/results/$f" ] && cp "$SH/results/$f" "$OUT/$f"; done
    echo "$(date +%T) skip $name (OK in timing.tsv)"; return
  fi
  for f in $outs; do rm -f "$SH/results/$f"; DIFF_ITEMS+=("$group:$f"); done
  local t0=$SECONDS st=OK
  ( cd "$SH" && nice -n 10 "$@" ) > "$OUT/logs/$name.log" 2>&1 || st=FAIL
  printf "%s\t%s\t%s\t%s\n" "$name" "$st" $((SECONDS - t0)) "$(date +%F_%T)" >> "$TIMING"
  echo "$(date +%T) $st $name ($((SECONDS - t0)) s)"
  for f in $outs; do [ -e "$SH/results/$f" ] && mkdir -p "$(dirname "$OUT/$f")" && cp "$SH/results/$f" "$OUT/$f"; done
}
LH="$PY experiments/local_hw.py"

for G in $ONLY; do case $G in
theory)   # 5.8: exact dynamic-programming checks on small synthetic instances (input: none)
  step verify_58 theory "theory58_checks.json" $PY theory/verify_58.py ;;
gtx)      # Tables 7-10, Figure 4: GTX 1650 cost tables (input: results/local_cost_cuda_T{16,32,128}.json, GPU-measured)
  for T in 16 32 128; do for t in qqp mnli; do
    step select_${t}_cuda_T$T gtx "local_select_${t}_cuda_T$T.json" $LH select --task $t --seq-len $T --batch-sizes 8 16 64 --workers $W
  done; done
  for T in 128 16; do for t in mnli qqp; do
    step front_${t}_cuda_T$T gtx "local_front_${t}_cuda_T$T.csv" $LH front --task $t --seq-len $T --batch-sizes 8 16 64 --workers $W
  done; done
  for t in qqp mnli; do
    step resplit_${t}_cuda_T16 gtx "local_resplit_${t}_cuda_T16.csv" $LH resplit --task $t --seq-len 16 --batch-sizes 8 64 --seeds 30 --workers $W
  done ;;
a6000)    # Tables 11-12, 15-16 (selection and fronts; the timing itself needs the GPU), ViT selection
          # (input: results/local_cost_a6000_T{128,32}.json, local_cost_a6000_T128_{llama,qwen}.json, local_cost_a6000_T197_vit.json)
  for T in 128 32; do for t in qqp mnli; do for c in select front; do
    ext=json; [ $c = front ] && ext=csv
    step ${c}_${t}_a6000_T$T a6000 "local_${c}_${t}_a6000_T$T.$ext" $LH $c --task $t --name a6000 --seq-len $T --batch-sizes 8 16 64 128 --workers $W
  done; done; done
  for m in qwen llama; do for t in qqp mnli; do for c in select front; do
    ext=json; [ $c = front ] && ext=csv
    step ${c}_${t}_${m}_a6000_T128 a6000 "local_${c}_${t}_${m}_a6000_T128.$ext" $LH $c --task ${t}_$m --name a6000 --seq-len 128 --workers $W
  done; done; done
  for c in select front; do
    ext=json; [ $c = front ] && ext=csv
    step ${c}_cifar100_a6000_T197 a6000 "local_${c}_cifar100_a6000_T197.$ext" $LH $c --task cifar100 --name a6000 --seq-len 197 --batch-sizes 8 16 64 128 --workers $W
  done ;;
delta)    # 5.2 text, MNLI fixed split at delta = 0.10/0.05/0.02 (input: local_cost_a6000_T128.json)
  step delta_check_mnli delta "delta_check_mnli.csv" $PY experiments/delta_check.py --task mnli --name a6000 --workers $W ;;
load)     # Table 14 (input: local_cost_a6000_T128.json, runner_cost_cifar100.json)
  for t in qqp mnli cifar100; do
    extra="--name a6000 --seq-len 128"; [ $t = cifar100 ] && extra="--cost-json results/runner_cost_cifar100.json --seq-len 197"
    step load_trace_$t load "load_trace_$t.csv load_trace_${t}_levels.csv load_trace_${t}_selected.csv load_trace_${t}_batches.csv" \
      $PY experiments/load_trace.py --task $t $extra --workers $W
  done
  step load_trace_mnli_d05 load "load_trace_mnli_d05.csv load_trace_mnli_d05_levels.csv load_trace_mnli_d05_selected.csv load_trace_mnli_d05_batches.csv" \
    $PY experiments/load_trace.py --task mnli --name a6000 --workers $W --delta 0.05 --tag _d05 ;;
robust)   # 5.5 (input: runner_cost_*.json, p6_exit_overhead_*.json, main_*_selected.csv; real_<task> = A6000 runner cost)
  for t in mnli qqp cifar100; do for r in enc_eager_meas enc_graph real_$t dec_eager; do
    step robust_cert_${t}_$r robust "robust_${t}_${r}_B64_cert.csv robust_violations_${t}_${r}_B64_cert.csv" \
      $PY experiments/robust_bound.py --task $t --regime $r --B 64 --ks 0 1 2 4 8 16 32 --tune cert --tag _cert
  done; done ;;
bandit)   # 5.4, ten seeds (input: as robust). The 25 MB per-step traces stay in results_repro/.root (not compared)
  for t in mnli qqp cifar100; do for r in dec_eager real_$t; do
    step bandit_10seeds_${t}_$r bandit "bandit_slopes_${t}_${r}_B64_10seeds.csv" \
      $PY bandit/run_bandit.py --task $t --regime $r --B 64 --T 3000 --seeds 0 1 2 3 4 5 6 7 8 9 --ablation --tag _10seeds
  done; done ;;
theory58) # 5.8, real-trace test of the corrected rule (2) (input: the GTX 1650 and A6000 cost tables)
  for t in qqp mnli; do
    for T in 16 32 128; do step theory58_${t}_cuda_T$T theory58 "theory58_front_${t}_cuda_T$T.csv" \
      $PY experiments/theory58_front.py --task $t --name cuda --seq-len $T --plans --workers $W; done
    for T in 128 32; do step theory58_${t}_a6000_T$T theory58 "theory58_front_${t}_a6000_T$T.csv" \
      $PY experiments/theory58_front.py --task $t --name a6000 --seq-len $T --plans --workers $W; done
  done ;;
transfer) # Table 2 row 6 (transfer of tuned parameters, simulator) and 5.7 (adaptivity vs batch size)
  step transfer_test transfer "transfer_mnli_all.csv transfer_qqp_all.csv transfer_cifar100_all.csv transfer_pairs.csv transfer_summary.csv" \
    $PY experiments/transfer_test.py --workers $W
  step adaptivity_vs_batch transfer "adaptivity_vs_batch.csv" \
    $PY analysis/adaptivity_vs_batch.py --tasks wikitext_qlora wikitext wikitext_llama ;;
cert)     # Table 13 (input: local_cost_a6000_T128.json, runner_cost_cifar100.json). The slowest group
  for t in qqp mnli cifar100; do
    extra="--name a6000 --seq-len 128"; [ $t = cifar100 ] && extra="--cost-json results/runner_cost_cifar100.json --seq-len 197"
    step cert_check_$t cert "cert_check_$t.csv cert_check_${t}_summary.csv" \
      $PY experiments/cert_check.py --task $t $extra --seeds 30 --workers $W
  done
  step cert_check_mnli_d05 cert "cert_check_mnli_d05.csv cert_check_mnli_d05_summary.csv" \
    $PY experiments/cert_check.py --task mnli --name a6000 --seeds 30 --workers $W --methods t --delta 0.05 --tag _d05 ;;
report)   # Figures 1-4 and the report HTML from the regenerated files (plus the GPU-timed inputs)
  step report_figures report "" $PY analysis/report_figures.py
  for f in report2_cost_gtx1650 report2_server_speedups report2_local_speedups report2_resplit report2_front; do
    DIFF_ITEMS+=("report:../figures/$f.png"); mkdir -p "$OUT/figures"; cp "$SH/figures/$f.png" "$OUT/figures/" 2>/dev/null
  done
  step make_report report "" $PY analysis/make_report.py
  DIFF_ITEMS+=("report:../docs/IE624_progress_report_v2.html"); mkdir -p "$OUT/docs"
  cp "$SH/docs/IE624_progress_report_v2.html" "$OUT/docs/" 2>/dev/null ;;
*) echo "unknown group $G"; exit 2 ;;
esac; done

SUF=""; [ "$ONLY" = "$ALL" ] || SUF="_$(echo $ONLY | tr ' ' '_')"
[ ${#DIFF_ITEMS[@]} -gt 0 ] && $PY "$REPO/analysis/diff_repro.py" --repro "$SH/results" --ref "$REPO/results" \
  --files "${DIFF_ITEMS[@]}" --out "$OUT/diff_summary$SUF.csv" --md "$OUT/DIFF$SUF.md"
echo "done; see $OUT/DIFF_*.md"
