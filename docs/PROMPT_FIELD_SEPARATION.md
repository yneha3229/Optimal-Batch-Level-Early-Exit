# Prompt: overnight field-separation SAT runs (give this to a Claude Code session on the server)

Copy everything below the line into a Claude Code session running on the GPU server (CPU-only work; no GPU needed).

---

You are working in the repository `Optimal-Batch-Level-Early-Exit` on our shared server. This is a CPU-only,
long-running SAT job. Work carefully, report honestly, and never invent results.

## Setup and rules

1. `git fetch origin && git checkout feature/kneser-minrank-theorem && git pull`, then create a new branch
   `feature/field-separation` from it. Commit as the repository's configured git user. Do NOT add any
   `Co-Authored-By` or `Claude-Session` lines to commit messages. Push only to `feature/field-separation`.
2. Read `proof.md` (sections 1, 2 and 4, especially Proposition 4.5) and `minrank/README.md`. Run
   `bash scripts/minrank/00_setup.sh` (tests must pass).
3. Check free cores with `nproc` and `uptime`; other people use this machine. Use at most 16 cores in total, pinned
   with `taskset` if others are busy. Run everything with `nohup` so it survives a disconnect, and log to `logs/`.

## Background (what we want)

Theorem 1 of `proof.md`: over F_p, `rank(I + (-1)^(k-1) D) = alpha = C(n-1,k-1)` for the Kneser graph K(n,k) iff
`q | n-2k` (q = least power of p with q >= k). So `minrk_{F_p}(K(n,k)) = alpha` for those p. We want to know whether
other fields really need more: a **field separation** is a proof that `minrk_{F_r}(K(n,k)) > alpha` for some other
prime r. We already have one: K(7,2) has minrk 7 over F_2 and 6 over F_3 (DRAT-verified).

Earlier runs in a 4-core container timed out after 3000 s per SAT call (`minrank/results/field_sep/families_f2.csv`):
K(9,2) over F_2 is 8 or 9; K(11,2) over F_2 is 10 or 11.

## Task list (in priority order)

**A. F_2 runs (XOR encoding with independent-set symmetry breaking; produces CNFs for DRAT).**

```bash
nohup python -m minrank.exact.families --spec kneser:9,2 kneser:11,2 kneser:13,2 \
  --time-per-call 36000 --lb-time 60 --workers 3 \
  --out minrank/results/field_sep/families_f2_long.csv \
  --dimacs-dir minrank/results/unsat_cnf/field_sep > logs/field_sep_f2_long.log 2>&1 &
```

The SAT call at r = alpha decides each case: UNSAT means `minrk_2 = alpha + 1` (a separation, since the theorem gives
alpha over F_5, F_7, F_3 for n = 9, 11, 13 respectively); SAT means `minrk_2 = alpha` (no separation; save the matrix).

**B. F_3 and F_5 runs (one-hot encoding in `minrank/exact/sat_fq.py`).** Create `minrank/exact/field_sep.py`:
a CLI with a `if __name__ == "__main__":` guard (required: the solver uses multiprocessing), taking
`--case m,k,q,r` (repeatable) and `--time-limit`. For each case call `rank_at_most_fq(kneser(m, k), r, q, time_limit)`
and print/append one CSV row to `minrank/results/field_sep/fq.csv`:
`graph,q,r,result(SAT|UNSAT|TIMEOUT),seconds`. On SAT, verify the returned matrix with `verify_fq` and save it as
JSON under `minrank/certificates/field_sep/`. On UNSAT, also write the exact CNF (`encode_fq(g, r, q)`) gzipped
in DIMACS to `minrank/results/unsat_cnf/field_sep/` so it can be DRAT-checked. Run, one process per case:

| case `m,k,q,r` | meaning | theorem says |
|---|---|---|
| `9,2,3,8` | is minrk_3(K(9,2)) = 8? | alpha is reached over F_5, not proven for F_3 |
| `11,2,3,10` | is minrk_3(K(11,2)) = 10? | alpha over F_7 |
| `11,2,5,10` | is minrk_5(K(11,2)) = 10? | alpha over F_7 |
| `7,2,5,6` | is minrk_5(K(7,2)) = 6? | alpha over F_3 |

Time limit 36000 s each (`timeout 37000`). Sanity check first (must be SAT quickly or within the limit):
`7,2,3,6` (Theorem 1 gives an explicit matrix of rank 6 over F_3). If the sanity case times out, report that the
one-hot encoding is too slow and stop task B.

**C. DRAT-check every UNSAT.** `bash scripts/minrank/10_drat_check.sh minrank/results/unsat_cnf/field_sep`
(needs CaDiCaL and drat-trim under `~/tools`; see the script header). Only a `VERIFIED` row counts as proven.

## Reporting

- Commit results, logs, CNFs of UNSAT calls (gzipped), certificates and `minrank/exact/field_sep.py` to
  `feature/field-separation` and push (`git push -u origin feature/field-separation`). Commit partial results every
  few hours so nothing is lost.
- Add a short section "Field separation runs (date, machine)" to `docs/MINRANK_THEORY.md` with a table:
  graph, field, r, result, seconds, DRAT status, and what it means (separation / no separation / undecided).
- If a separation is proven and DRAT-verified, also add it to Proposition 4.5 of `proof.md` and to the table in
  `paper/isit/main.tex` (Section "Broadcast rate and new cases"), in the same style.
- Final message: a 10-line summary with every case's outcome, total CPU-hours, and anything left undone. A timeout
  is an acceptable outcome; say so plainly. Never report a separation without a VERIFIED DRAT row.
