#!/usr/bin/env bash
# Long portfolio run on every minrank instance still open on 24 Nov 2026 (docs/MINRANK_RESULTS.md, "Open").
# Four members (solver x symmetry mode) run the same specs independently; the first to decide a rank wins.
# Floors are lower bounds proven earlier: UNSAT runs that were DRAT-checked (Paley), alpha, or the star-identity
# theorem (K(9,4) >= 57, docs/MINRANK_THEORY.md section 6). Each member has its own CSV and certificate directory
# (seeded with the stored certificates) so that parallel writes cannot collide; merge with
# `python -m minrank.run_baselines`-style keep_best afterwards (see the results doc).
# Usage: TPC=28800 nohup scripts/minrank/11_open_portfolio.sh > logs/minrank_open_portfolio.log 2>&1 &
set -u
cd "$(dirname "$0")/../.." || exit 1
PY=${PY:-$HOME/venv/bin/python}
TPC=${TPC:-28800}
OUT=minrank/results/open_portfolio
mkdir -p $OUT
SPECS="mycielski:7 paley:29 paley:37 paley:41 paley:53 paley:61 circulant:41,1,5,8 kneser:9,4 kneser:11,3 kneser:13,3 kneser:10,4"
FLOORS="paley:29=7 paley:37=7 paley:41=8 paley:53=8 paley:61=8 kneser:9,4=57"
MEMBERS="cadical153:indep:8-18 cadical195:indep:19-29 cadical195:indep_cols:40-50 glucose42:indep:51-61"
for m in $MEMBERS; do
  IFS=: read -r solver mode cores <<< "$m"
  tagm=${solver}_${mode}
  mkdir -p $OUT/certs_$tagm
  cp -n minrank/certificates/*.json $OUT/certs_$tagm/ 2>/dev/null
  DIM=""
  [ "$tagm" = "cadical195_indep" ] && DIM="--dimacs-dir $OUT/unsat_cnf"
  taskset -c $cores $PY -m minrank.exact.families --spec $SPECS --floor $FLOORS --time-per-call $TPC \
    --mode $mode --solver $solver --workers 11 --out $OUT/$tagm.csv --certs $OUT/certs_$tagm $DIM \
    > logs/minrank_open_$tagm.log 2>&1 &
  echo "started $tagm pid $! on cores $cores"
done
wait
echo DONE
