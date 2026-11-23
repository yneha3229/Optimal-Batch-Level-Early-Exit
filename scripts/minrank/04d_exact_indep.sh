#!/usr/bin/env bash
# Option B with the independent-set symmetry breaking (symmetry.py mode "indep"; see WORK.md).
#   PART=bench    : the benchmark graphs not proven in exact.csv (status call_timeout) -> minrank/results/exact_indep.csv
#   PART=families : named families, ascending from alpha                              -> minrank/results/families.csv
# The CNF of every UNSAT call goes to $CNF (gzipped DIMACS) for the DRAT check (WORK.md T4).
# Vars: PY, PART, WORKERS, LIMIT (s per SAT call), CORES (taskset list), CNF, SPECS (families only).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}; PART=${PART:-bench}; WORKERS=${WORKERS:-8}; LIMIT=${LIMIT:-1800}; CORES=${CORES:-16-31,48-63}
CNF=${CNF:-minrank/results/unsat_cnf}
DEFAULT_SPECS="$(for m in 5 6 7 8 9 10 11 12 13; do echo -n "kneser:$m,2 "; done) kneser:7,3 kneser:8,3 kneser:9,3 kneser:10,3 kneser:9,4
mycielski:3 mycielski:4 mycielski:5 mycielski:6 mycielski:7
paley:5 paley:13 paley:17 paley:29 paley:37 paley:41 paley:53 paley:61
orthogonality:4 orthogonality:5 orthogonality:6 orthogonality:7
$(for n in 9 11 13 15 17 19 21 23 25 27 29 31; do for k in 2 3; do [ $n -gt $((2*k+2)) ] && echo -n "cycle_power:$n,$k "; done; done)"
SPECS=${SPECS:-$DEFAULT_SPECS}
case "$PART" in
  bench)
    taskset -c "$CORES" "$PY" -m minrank.exact.driver --rerun-from minrank/results/exact.csv --rerun-status call_timeout \
      --max-n 1000 --mode indep --time-per-call "$LIMIT" --total-time $((3 * LIMIT)) --lb-time 60 --workers "$WORKERS" \
      --out minrank/results/exact_indep.csv --certs minrank/certificates --dimacs-dir "$CNF/bench" ;;
  families)
    # shellcheck disable=SC2086
    taskset -c "$CORES" "$PY" -m minrank.exact.families --spec $SPECS --time-per-call "$LIMIT" --lb-time 60 \
      --workers "$WORKERS" --out minrank/results/families.csv --certs minrank/certificates --dimacs-dir "$CNF/families" ;;
  *) echo "PART must be bench or families" >&2; exit 1 ;;
esac
