# Prompt for Claude Cowork: next steps toward publication (written 26 Nov 2026)

Copy everything below the line into the Cowork session. Tasks are in priority order; do them in order.

---

You are working in the repository `yneha3229/Optimal-Batch-Level-Early-Exit` (branch `main` is up to date with all
feature branches merged). Report honestly, never invent results or citations, and say plainly when something timed
out or could not be checked.

## Rules

- Start each task with `git fetch origin && git checkout main && git pull`, then create a branch named
  `feature/<short-name>` for it (names given below). Push only to that branch. Do not merge into `main`.
- Commit as the repository's configured git user. Do NOT add `Co-Authored-By` or `Claude-Session` lines.
- On the shared server: kill processes by PID only (never `pkill -f`), pin CPU jobs with `taskset`, use at most 16
  cores, run long jobs with `setsid nohup` and log to `logs/`. Never touch other users' GPU processes.

## Task 1 (most important): novelty check for the Kneser minrank theorem — `feature/minrank-lit-check`

Our main publishable result is Theorem 1 of `proof.md`: over F_p, rank(I + (-1)^(k-1) D) = C(n-1,k-1) for the
Kneser graph K(n,k) iff q | n-2k (q = least power of p with q >= k), hence minrk_{F_p}(K(n,k)) = alpha and
beta(K(n,k)) = alpha for all but finitely many n. Read `proof.md` sections 1, 2 and 4 and `docs/MINRANK_LIT.md` first.

Find and read (full text, not abstracts) the closest prior work and decide whether any of it already contains
Theorem 1, Theorem 2 (the explicit code) or Corollary 3/4.4:

1. Hu and Tang, arXiv:2607.06480.
2. Brouwer and van Eijl, "On the p-rank of the adjacency matrices of strongly regular graphs" (1992), and any
   Brouwer work on p-ranks of Kneser or Johnson graphs.
3. Ducey et al. on Smith normal forms / diagonal forms of Kneser and Johnson graphs (we use their result in
   Appendix A); Wilson (1990) on diagonal forms of inclusion matrices; Frankl, Linial-type p-rank results on
   disjointness matrices.
4. Golovnev and Haviv, and Haviv, on minrank of Kneser graphs / index coding on Kneser graphs.
5. Blasiak–Kleinberg–Lubetzky (broadcast rate) and any paper computing beta for Kneser graphs.
6. Search arXiv and Google Scholar for: "p-rank" "Kneser", "disjointness matrix" "rank" "mod p",
   "minrank" "Kneser graph", "index coding" "Kneser", "Haemers bound" "Kneser".

Write `docs/MINRANK_LIT_CHECK.md`: for each paper, the exact statement closest to ours (quote it with theorem
number and page), and a verdict: "contains our result", "implies it with a short argument", "related, not
implying", or "could not access". End with an overall verdict and the sentences we must add to the related-work
paragraph of `paper/isit/main.tex`. If a paper already contains Theorem 1, say so in the first line. Do not edit
`proof.md` or the paper in this task.

## Task 2: field-separation runs — `feature/field-separation` (existing branch, continue on it)

Read section 13 of `docs/MINRANK_THEORY.md` and `docs/PROMPT_FIELD_SEPARATION.md`. Run 2 of the F_2 cases
K(9,2), K(11,2), K(13,2) (r = alpha) started 26 Nov 02:40 UTC with a 36000 s limit.

1. Check whether the processes still run (by PID, from `logs/field_sep_driver.log`). If they finished, record the
   outcome in the section 13 table. DRAT-check every UNSAT with `scripts/minrank/10_drat_check.sh`; only a
   VERIFIED row counts.
2. Improve the F_q encoding in `minrank/exact/sat_fq.py`: fix the alpha x alpha identity block on a maximum
   independent set (as the F_2 `indep` mode does) and add symmetry breaking. The success criterion is the sanity
   case `7,2,3,6` returning SAT in under 10 minutes; verify its matrix with `verify_fq`.
3. Only if the sanity case passes, run task B of `docs/PROMPT_FIELD_SEPARATION.md` (the four cases).
4. Any DRAT-verified separation goes into Proposition 4.5 of `proof.md` and the table in `paper/isit/main.tex`.

## Task 3: make the ISIT paper submission-ready — `feature/isit-paper`

1. Compile `paper/isit/main.tex` (pdflatex + bibtex). It must fit 5 pages plus references in IEEEtran conference
   format. Fix every warning about undefined references or citations.
2. Check every statement and number in the paper against `proof.md`: theorem statements, the exceptional n list of
   Cor 4.4, the examples K(10,3)/F_2 (36 vs 40), K(13,4)/F_5 (220 vs 238.3), K(15,4)/F_7 (364 vs 455),
   K(7,2) (6 over F_3, 7 over F_2). List any mismatch.
3. Check every bibliography entry exists (authors, title, venue, year) — no invented references.
4. Add the related-work sentences from Task 1 once that file exists.
5. Prepare an arXiv version (`paper/arxiv/`): the same paper without the page limit, with the full proof,
   Appendix A (second proof via Ducey et al.) and Appendix B (k = 3 inclusion-matrix proof), and a short
   "Computational verification" section citing `verify_star_basis.py` (318 cases), `verify_independent.py`
   (82 cases, two methods) and the 6666-case diagonal-form check.
6. Commit the compiled PDFs too.

## Task 4: early-exit paper evidence — `feature/early-exit-final`

Read `docs/DIRECTION.md` (section 7) and `docs/RESULTS.md`. The claim is: which batch-level exit rule wins flips
with batch size, execution mode and task, and our cost-aware rule (re-certified per setting) stays near the best
everywhere. Weak points a reviewer will attack, in order:

1. CUDA-graph results are not accuracy-matched on MNLI (theory drops 1.8–2.5 points). Rerun the CUDA-graph timing
   with every method tuned to the same accuracy budget (1 point) on held-out data, for MNLI, QQP and CIFAR-100, at
   B = 8, 16, 64, 128. Output one table: speedup at matched accuracy, best method per cell, our gap to the best.
2. MNLI is over budget (1.25–1.48 points) under the default delta. Use the fixed split and a fresh random split
   (3 seeds) to report the mean and spread, so the over-budget result can be shown as split variance or fixed.
3. Throughput under realistic arrival: the fluctuating-load replay with requests per second and p50/p99 latency, not
   only speedup.
4. One more LLM classifier model (e.g. a 3B–8B model already cached) on QQP and MNLI, timed.

Write results to `results/` and a summary to `docs/RESULTS_FINAL.md` with a "claims table": each claim, the
number, the file, and whether it holds. Do not tune on the held-out split.

## Final report

A short message per task: what was done, the key numbers, what failed or timed out, and the branch name.
