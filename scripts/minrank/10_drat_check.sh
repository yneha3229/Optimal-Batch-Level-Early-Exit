#!/usr/bin/env bash
# Independent check of UNSAT results (WORK.md T4): for every gzipped DIMACS file written by --dimacs-dir
# (minrank/exact/families.py, driver.py --mode indep), solve it again with the CaDiCaL binary, write a binary DRAT
# proof, and check the proof with drat-trim. The proof is deleted right after its check (disk is short); one CSV row
# per CNF records the sha256 of the uncompressed CNF, the solver and checker results and times.
# A row with check=VERIFIED means: this exact CNF is UNSAT, checked by drat-trim. Soundness of the symmetry-breaking
# unit clauses inside the CNF is the lemma in minrank/exact/symmetry.py (tested by brute force); the CNF header
# records the graph digest, the rank r and the fixed set I.
#
# Usage: scripts/minrank/10_drat_check.sh [CNF files or directories...]   (default: minrank/results/unsat_cnf)
# Vars: JOBS (parallel checks, default 8), CORES (taskset list, default 16-31,48-63), WORK (proof scratch dir,
#       default ${TMPDIR:-/tmp}/drat_work), TLIM (drat-trim time limit s, default 40000),
#       CADICAL, DRATTRIM (binaries; default ~/tools/...), OUT (default minrank/results/drat_checks.csv).
# Files already VERIFIED in OUT with the same sha256 are skipped.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
JOBS=${JOBS:-8}; CORES=${CORES:-16-31,48-63}; TLIM=${TLIM:-40000}
WORK=${WORK:-${TMPDIR:-/tmp}/drat_work}; OUT=${OUT:-minrank/results/drat_checks.csv}
CADICAL=${CADICAL:-$HOME/tools/cadical/build/cadical}; DRATTRIM=${DRATTRIM:-$HOME/tools/drat-trim/drat-trim}
for b in "$CADICAL" "$DRATTRIM"; do [ -x "$b" ] || { echo "missing $b (see WORK.md T4)" >&2; exit 1; }; done
mkdir -p "$WORK"
[ -s "$OUT" ] || echo "file,sha256,cnf_vars,cnf_clauses,solver,solve_s,check,check_s,proof_mb" > "$OUT"
[ $# -gt 0 ] || set -- minrank/results/unsat_cnf
mapfile -t FILES < <(find "$@" -name '*.cnf.gz' -printf '%s %p\n' | sort -n | cut -d' ' -f2-)   # small first

check_one() {
  local f=$1 base cnf prf sha hdr t0 t1 t2 rc sres cres
  base=$(basename "$f" .cnf.gz); cnf="$WORK/$base.cnf"; prf="$WORK/$base.drat"
  gzip -dc "$f" > "$cnf"
  sha=$(sha256sum "$cnf" | cut -d' ' -f1)
  if grep -q ",$sha,.*,VERIFIED," "$OUT" 2>/dev/null; then rm -f "$cnf"; return; fi
  hdr=$(grep -m1 '^p cnf' "$cnf" | awk '{print $3","$4}')
  t0=$(date +%s.%N)
  "$CADICAL" -q --binary=true "$cnf" "$prf" > "$WORK/$base.solver.log" 2>&1; rc=$?
  t1=$(date +%s.%N)
  case $rc in 20) sres=UNSAT ;; 10) sres=SAT ;; *) sres="ERR$rc" ;; esac
  if [ "$sres" = UNSAT ]; then
    timeout $((TLIM + 60)) "$DRATTRIM" "$cnf" "$prf" -t "$TLIM" > "$WORK/$base.check.log" 2>&1   # exit code is not 0 on success
    if tr -d '\r' < "$WORK/$base.check.log" | grep -q '^s VERIFIED'; then cres=VERIFIED   # drat-trim prints \r before it
    elif grep -q 'TIMEOUT' "$WORK/$base.check.log"; then cres=TIMEOUT
    else cres=FAILED; fi
  else cres=SKIPPED; fi
  t2=$(date +%s.%N)
  local mb; mb=$(du -m "$prf" 2>/dev/null | cut -f1)
  echo "$f,$sha,$hdr,$sres,$(echo "$t1 - $t0" | bc),$cres,$(echo "$t2 - $t1" | bc),${mb:-0}" >> "$OUT"
  echo "$(date +%T) $base: $sres, $cres" >&2
  rm -f "$cnf" "$prf"
}
export -f check_one; export WORK OUT CADICAL DRATTRIM TLIM
printf '%s\n' "${FILES[@]}" | taskset -c "$CORES" xargs -P "$JOBS" -I{} bash -c 'check_one "$@"' _ {}
echo "done: $(grep -c VERIFIED "$OUT") verified, $(grep -c -E 'FAILED|TIMEOUT|ERR' "$OUT") not" >&2
