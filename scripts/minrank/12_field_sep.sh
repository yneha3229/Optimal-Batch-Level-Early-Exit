#!/usr/bin/env bash
# Driver for the field-separation runs (docs/MINRANK_THEORY.md, "Field separation runs"). Meant to run detached
# (setsid nohup) so it survives a lost shell; it does not survive a reboot.
#   1. wait for the F_3 sanity case K(7,2), r = 6 (already started) to write its row to fq.csv;
#   2. if it is SAT, start the four F_q cases (one detached process each, 36000 s); otherwise skip task B;
#   3. while any job runs: every DRAT_EVERY s DRAT-check new UNSAT CNFs (10_drat_check.sh), every COMMIT_EVERY s
#      commit the outputs to the current branch and push it;
#   4. when every job has ended: a last DRAT check, commit and push.
# Vars: PY, SANITY_PID, A_PID (pids of the already running sanity and F_2 jobs), B_CORES (4 cores, one per case),
#       DRAT_CORES, DRAT_JOBS, TLIM_B (s per F_q case), DRAT_EVERY, COMMIT_EVERY.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-$HOME/venv/bin/python}
SANITY_PID=${SANITY_PID:?}; A_PID=${A_PID:?}
B_CORES=(${B_CORES:-5 6 7 36}); DRAT_CORES=${DRAT_CORES:-37-39}; DRAT_JOBS=${DRAT_JOBS:-2}
TLIM_B=${TLIM_B:-36000}; DRAT_EVERY=${DRAT_EVERY:-1800}; COMMIT_EVERY=${COMMIT_EVERY:-10800}
FQ=minrank/results/field_sep/fq.csv; UNSAT_DIR=minrank/results/unsat_cnf/field_sep
log() { echo "$(date -u +%FT%TZ) $*"; }

alive() { local p; for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }

commit_push() {
  git add minrank/exact/field_sep.py scripts/minrank/12_field_sep.sh logs/field_sep_* minrank/results/field_sep \
    minrank/results/drat_checks.csv 2>/dev/null
  [ -d "$UNSAT_DIR" ] && git add "$UNSAT_DIR"
  [ -d minrank/certificates/field_sep ] && git add minrank/certificates/field_sep
  if git diff --cached --quiet; then log "commit: nothing new"; return; fi
  git commit -q -m "field separation: partial results $(date -u +'%d %b %H:%M') UTC ($1)" && log "committed $(git rev-parse --short HEAD)"
  git push -q origin HEAD 2>&1 | tail -2; log "pushed"
}

drat() {
  [ -n "$(find "$UNSAT_DIR" -name '*.cnf.gz' 2>/dev/null)" ] || return
  log "DRAT check of $UNSAT_DIR"
  JOBS=$DRAT_JOBS CORES=$DRAT_CORES WORK=$HOME/drat_work_fieldsep bash scripts/minrank/10_drat_check.sh "$UNSAT_DIR" \
    >> logs/field_sep_drat.log 2>&1
  log "DRAT pass done"
}

log "waiting for the sanity case (pid $SANITY_PID)"
while alive "$SANITY_PID"; do sleep 30; done
sanity=$(awk -F, '$1=="kneser_7_2" && $2==3 && $3==6 {r=$4} END {print r}' "$FQ" 2>/dev/null)
log "sanity K(7,2) F_3 r=6: ${sanity:-no row}"
PIDS=("$A_PID")
if [ "$sanity" = SAT ]; then
  i=0
  for c in 9,2,3,8 11,2,3,10 11,2,5,10 7,2,5,6; do
    tag=${c//,/_}
    setsid nohup taskset -c "${B_CORES[$i]}" "$PY" -m minrank.exact.field_sep --case "$c" --time-limit "$TLIM_B" \
      > "logs/field_sep_fq_$tag.log" 2>&1 < /dev/null &
    sleep 2; pid=$(pgrep -nf "field_sep --case $c ")
    log "started case $c on core ${B_CORES[$i]} (pid $pid)"; PIDS+=("$pid"); i=$((i + 1))
  done
else
  log "sanity case is not SAT: the one-hot encoding is too slow; task B stopped"
fi
commit_push "task B started"

last_drat=$(date +%s); last_commit=$(date +%s)
while alive "${PIDS[@]}"; do
  sleep 60; now=$(date +%s)
  if (( now - last_drat >= DRAT_EVERY )); then drat; last_drat=$(date +%s); fi
  if (( now - last_commit >= COMMIT_EVERY )); then commit_push "running"; last_commit=$(date +%s); fi
done
log "all jobs ended"
drat
commit_push "all jobs ended"
log "driver done"
