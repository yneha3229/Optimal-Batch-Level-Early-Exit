# Minrank over F_2: work split (started 23 Nov 2026)

**Goal.** Turn the independent-set symmetry breaking and the exact values it gives into a short paper. That means a
SAT/CP workshop paper or an arXiv note, plus ISIT/ITW if we lead with index coding. Two people work in parallel:

| | Who | Branch |
|---|---|---|
| Teammate 1 | Claude | `claude/keen-dijkstra-fpe319` |
| Teammate 2 | you | `minrank-t2`, made from `claude/keen-dijkstra-fpe319`; merge it back when a task is done |

**File ownership (avoids merge conflicts).** Only edit your own files. If you need a change in the other person's
file, write it in the status table at the bottom.

| Teammate 1 (Claude) | Teammate 2 (you) |
|---|---|
| `minrank/exact/symmetry.py`, `minrank/exact/driver.py`, `minrank/exact/families.py` (new) | `minrank/bench/generators.py` (append only, do not change existing generators: benchmark digests depend on them) |
| `minrank/tests/test_indep.py` (new) | `minrank/tests/test_generators_new.py` (new) |
| `minrank/results/exact_indep.csv`, `minrank/results/families.csv`, `minrank/certificates/` | `docs/MINRANK_THEORY.md`, `docs/MINRANK_LIT.md` (new) |
| `scripts/minrank/04d_*.sh`, `docs/MINRANK_RESULTS.md` (new) | `~/tools/` (CaDiCaL and drat-trim, outside the repo), `scripts/minrank/10_drat_check.sh` (new) |

The early-exit report is due on 26 Nov, so the CUDA-graph device (`docs/PROMPT_CUDA_GRAPH_DEVICE.md`, your branch)
comes before anything here.

## 1. What we have (preliminary, in `minrank/results/indep_prelim/`)

**Lemma (independent-set symmetry breaking).** Let I be an independent set of an undirected G with |I| = s <= r.
- Every fitting A has A[I, I] = Id. So for A = U V^T with U, V in F_2^{n x r}, U_I has full row rank.
- Some W in GL(r) gives U_I W = [Id_s | 0]. Then V_I W^{-T} = [Id_s | X].
- The stabiliser of [Id | 0] is W' = [[Id, 0], [C, D]], and W' = [[Id, 0], [X^T, Id]] sends X to 0.
- Hence, without loss of generality, U_I = V_I = [Id_s | 0]. This fixes 2 s r variables.

The directed case works the same with an acyclic induced I, because A[I, I] is unit triangular, hence invertible:
fix U_I = [Id | 0] and V_I[:, s:] = 0. Do not combine this with the old u_0 = e_1 fix.

**Effect.**
- **Speed:** on the 24 benchmark graphs that the old encoding (u_0 = e_1 plus column lex) could not settle in 600 s
  or 3600 s per call, most new steps take under a second.
- **Soundness:** 1,080 SAT/UNSAT checks against known values, with 0 mismatches. These cover 29 proven benchmark
  graphs and 511 small graphs.

**Exact minrk_2 values so far (all proven: UNSAT at minrank - 1, SAT at minrank).**

| Graph | n | alpha | minrk_2 | Clique cover | Note |
|---|---|---|---|---|---|
| Kneser K(n,2), n = 5, 6, 7, 8, 10, 12 | 10-66 | n-1 | 5, 5, 7, 7, 9, 11 | equal to minrank | n-1 for even n, n for odd n, which is the chromatic index of K_n |
| K(7,3) | 35 | 15 | 18 | 18 | rank 17 UNSAT in 44 s |
| **K(8,3)** | 56 | 21 | **22** | **28** | rank 21 UNSAT in 822 s and in 898 s (two runs), with the same independent set (vertices 0-20) |
| K(9,3) | 84 | 28 | 28 | 28 | trivial: 3 divides 9 (Baranyai), see T2 (0) |
| **K(10,3)** | 120 | 36 | **36 = alpha** | **>= 40** (cliques have at most 3 disjoint triples) | found 23 Nov in 5.6 s; explicit matrix in `minrank/certificates/kneser_10_3.json`, checked with plain Python; no UNSAT proof needed because alpha is the lower bound |
| Mycielski M3-M6 | 5, 11, 23, 47 | 2, 5, 11, 23 | 3, 6, 12, 24 | equal to minrank | = alpha + 1 = (n+1)/2; M6 rank 23 UNSAT in 504-538 s |
| Paley 5, 13, 17 | | 2, 3, 3 | 3, 5, 6 | equal to minrank | |
| Cycle powers C(n,2), C(n,3), 9 <= n <= 27, plus C(29,2) and C(31,3) | | | equal to the clique cover | | under 0.1 s each |
| planted n = 50, s1 and s3-s6 | 50 | 6-7 | 8 | > 8 | equal to the planted rank |
| planted_d n = 50 (directed) | 50 | 8 | upper bounds only: s1 13, s3 14, s4 14, s6 15 | | not proven |

K(8,3) (22 vs 28) and K(10,3) (36 vs >= 40) are the named graphs where minrank is strictly below the clique cover. Along K(n,3): 18 = alpha + 3 = cover (n = 7), alpha + 1 (n = 8), alpha = cover (n = 9), alpha < cover (n = 10).

**Still running (scratch, Claude's session, started about 07:00 UTC):**
- Sweep: K(9,2), K(11,2), K(9,3), and Paley 29, 37, 41.
- Open benchmark graphs: Paley 29-53, circ_41, and the directed planted graphs.
- Cross-check: K(8,3) at rank 21 with glucose4 and a different maximum independent set (the star at element 7).

## 2. Teammate 1 (Claude)

| # | Task | Output |
|---|---|---|
| C1 | `--mode indep` and `indep_cols` (the independent-set fix plus column lex on the free columns) in `symmetry.py` and the driver. Tests against brute force, undirected and directed | `rank_at_most_sym(g, r, limit, mode="indep", indep=I)` |
| C2 | Family runner: `python -m minrank.exact.families --spec kneser:8,3 johnson:8,3 ... --time-per-call 900`. It ascends from alpha to the clique cover. Each spec `name:a,b` calls `generators.<name>(a, b)`, so any new generator with integer arguments works without changes on my side | `minrank/results/families.csv`, certificates |
| C3 | Rerun every unproven benchmark graph with `--mode indep` | `minrank/results/exact_indep.csv` |
| C4 | `--dimacs-dir`: write the CNF of every UNSAT step, as input for the DRAT check (T4) | `.cnf` files, gzipped, outside git when large |
| C5 | Results write-up: tables, timings old vs new encoding, soundness checks | `docs/MINRANK_RESULTS.md` |

## 3. Teammate 2 (you)

**T1. New graph families (code).** Append to `minrank/bench/generators.py`:
- **Requirements:**
  - Deterministic.
  - A unique `name` encoding the arguments.
  - A `family` string and a params dict, as the existing generators have.
  - Integer arguments only, so that C2 can call them.
- **Generators:**
  - `johnson(m, k)`: k-subsets of [m], adjacent iff |A ∩ B| = k - 1.
  - `generalized_kneser(m, k, s)`: adjacent iff |A ∩ B| <= s. s = 0 is `kneser`; keep a different name prefix, `gkneser_m_k_s`.
  - `odd_graph(k)`: K(2k+1, k), name `odd_k`.
  - `co_kneser(m, k)`: the complement of K(m, k) (k-sets adjacent iff they intersect). Complements are a separate
    question, because minrk(G) minrk(Ḡ) >= n.
  - Optionally `hamming(d, q, dist)`: words of F_q^d adjacent iff their Hamming distance is in a set; the hypercube
    family. These are classical Haemers-bound test cases.
- **Tests** (`minrank/tests/test_generators_new.py`): vertex counts, regularity, known independence numbers (for
  example alpha(J(m,2)) = floor(m/2) and alpha(K(m,k)) = C(m-1, k-1) for m >= 2k), symmetric adjacency, no loops.
- **Then run:** `python -m minrank.exact.families --spec johnson:8,3 gkneser:9,3,1 ...` once C2 lands. Pin it with
  `taskset` to free cores (section 4).

**T2. Theory** (`docs/MINRANK_THEORY.md`). Turning patterns into theorems is what makes this a paper.
- (0) **When Kneser is trivial (found 23 Nov).** If k | n, Baranyai's theorem splits the k-subsets of [n] into
  C(n-1, k-1) parallel classes, each a clique of K(n,k). So the clique cover equals alpha and minrk = alpha, as for
  K(9,3) = 28 and K(n,2) with n even. **The open cases are k not dividing n**: K(n,2) with n odd, K(7,3), K(8,3),
  K(10,3), K(11,3), K(10,4), ... There the cover is at least ceil(C(n,k) / floor(n/k)) > alpha.
- (a) **K(n,2), n odd.** The cover is n (the chromatic index of K_n) and alpha = n - 1; minrk = n is proven for
  n = 5 and 7 (9, 11 and 13 are running). Conjecture: minrk_2(K(n,2)) = n for every odd n, i.e. rank n - 1 is
  impossible. Prove it.
- (b) **Mycielski.** minrk_2(M_k) = alpha + 1 = (n+1)/2 for k = 3..6. The upper bound is the clique cover. Is the
  lower bound alpha + 1 true for all k? A Mycielski-step argument may work.
- (c) **K(8,3) = 22, the main target.** The rank-22 matrix will be in `minrank/certificates/kneser_8_3.json`, with
  rows as hex bitmasks over the 56 vertices in `combinations(range(8), 3)` order.
  - Look for structure: blocks, which triples share rows, the action of S_8.
  - A construction that generalises to K(n,3) would give minrk_2(K(n,3)) < clique cover for all n >= 8 with
    3 not dividing n. That would be the headline theorem.
  - K(7,3) = 18 = cover, but K(8,3) = 22 < 28 and K(10,3) = 36 = alpha < 40. What changes between 7 and 8? Is there
    a rank-alpha construction for K(10,3) (the certificate) that generalises, e.g. to n = 1 mod 3 or n even? K(10,3) (alpha 36, cover 40),
    K(11,3) (alpha 45, cover 55) and K(10,4) (alpha 84, cover 105) are running.

**T3. Literature and novelty** (`docs/MINRANK_LIT.md`). For each paper, note what it computes and whether it covers
our values or our trick. End with a one-paragraph novelty verdict.
- Is fixing a basis on an independent set (or anything similar) known in SAT/ILP encodings of minrank, index coding
  or low-rank matrix problems? Check:
  - SAT for matrix multiplication schemes (Heule, Kauers, Seidl), which use basis/symmetry fixing;
  - clique fixing in SAT colouring (Van Gelder);
  - SAT/ILP for index coding.
- Are F_2 minrank (Haemers number) values of Kneser, Mycielski, Paley or cycle-power graphs published? Check:
  - Haemers 1979/1981;
  - the index-coding tables of Arbabjolfaei and Kim (all problems up to 5 users);
  - Haviv's and Golovnev-Haviv's work on the minrank and orthogonality dimension of Kneser-type graphs (the earlier
    scan found K(8,3) <= 37 from a general bound, nothing exact);
  - Alipour-Gohari and coauthors: exact Haemers numbers of J(n,3) via capacity bounds, a direct precedent.

**T4. DRAT proofs (tooling).** My session was not allowed to download and build tools, so please run this yourself:

```bash
mkdir -p ~/tools && cd ~/tools
git clone --depth 1 https://github.com/arminbiere/cadical && (cd cadical && ./configure && make -j8)
git clone --depth 1 https://github.com/marijnheule/drat-trim && (cd drat-trim && gcc -O2 -o drat-trim drat-trim.c)
```

- **The check:** `cadical --no-binary f.cnf f.drat` then `drat-trim f.cnf f.drat`. Wrap it in
  `scripts/minrank/10_drat_check.sh <dir-of-cnf>`; it should write one line per CNF (sha256, status, seconds) to
  `minrank/results/drat_checks.csv`.
- **Disk:** `/home` has about 10 GB free, and the K(8,3) rank-21 proof may be several GB. Use binary DRAT, check it
  right away and delete it; store only the check log and the sha256 of the CNF. Small proofs (< 5 MB gzipped) can be
  committed under `minrank/certificates/drat/`.
- **What a check proves:** that the CNF, including the symmetry-breaking unit clauses, is UNSAT. Soundness of those
  clauses is the lemma in section 1, so the paper needs both.

## 4. Machine rules (same as `docs/NEXT_STEPS.md` section 2)

- **CPU:** pin everything with `taskset`.

  | Cores | In use for |
  |---|---|
  | 0-3, 32-35 | the K(8,3) cross-check on core 3 (normally GPU timing, keep them quiet when you time) |
  | 4-11, 36-43 | Claude: large Kneser graphs (`families_big.csv`) |
  | 12-15, 44-47, 30-31, 62-63 | Claude: benchmark rerun (`exact_indep.csv`) |
  | 16-29, 48-61 | Claude: families (`families.csv`) |
  | 0-2, 32-35 | free for short jobs; ask before long ones |
- **Processes:** kill by PID only, never with `pkill -f` / `pgrep -f` patterns.
- **GPUs:** GPUs 2 and 3 belong to another user; never touch them. vLLM on GPU 4 (PID 3843328) is ours and can be stopped
  for the CUDA-graph timing.
- **Disk:** about 10 GB free on `/home`. No checkpoints, no large files in git (> 20 MB).
- **Environment:** `~/venv/bin/python`. Tests: `python -m pytest minrank/tests -q`.

## 5. Status (update when you finish or start something)

| Date (UTC) | Who | Task | Status |
|---|---|---|---|
| 23 Nov 07:40 | Claude | WORK.md, preliminary results pushed | done |
| 23 Nov 08:30 | Claude | C1 (indep mode, tests), C2 (`families.py`), C4 (`--dimacs-dir`), `04d_exact_indep.sh` | done; 315 tests pass |
| 23 Nov 08:30 | Claude | C2/C3 runs: `logs/minrank_families.log`, `logs/minrank_families_big.log`, `logs/minrank_exact_indep.log` | running (scratch runs stopped; the old-encoding rerun stopped, superseded) |
| | you | T1-T4 | not started |
