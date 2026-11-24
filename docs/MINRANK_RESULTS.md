# Minrank over F_2: results (WORK.md task C5), 24 Nov 2026

All values are exact over F_2 unless marked. Data: `minrank/results/*.csv`; certificates (verified fitting matrices)
in `minrank/certificates/`; proofs and theory in `docs/MINRANK_THEORY.md`; literature in `docs/MINRANK_LIT.md`.

## 1. Method

**Encoding.** We encode "a fitting matrix of rank ≤ r exists" as M = U V^T with U, V in F_2^{n×r}, using AND gates and
XOR chains. Each diagonal entry is constrained to 1, and each non-edge to 0.

**Independent-set fix** (`--mode indep`, `minrank/exact/symmetry.py`). Let I be an acyclic set: for undirected
graphs, an independent set.
- If I induces no arc, we fix U_I = V_I = [Id | 0].
- Otherwise we fix U_I = [Id | 0] and V_I[:, |I|:] = 0.

This is sound because GL(r) acts on (U, V) and M[I, I] is unit triangular. It is tested against brute force on
607 acyclic sets of 57 graphs, 196 of them with arcs (`minrank/tests/test_indep.py`).

**Search.**
- The lower bound is alpha, from the acyclic set CP-SAT finds (`lower.mais`).
- The upper bound is the best of the clique cover, I + A(G), and the stored certificate.
- The search ascends from alpha (or from a known floor). The first SAT value is the minrank.
- On a timeout it descends from the upper bound.

The runner is `python -m minrank.exact.families`.

**Old vs new encoding** (`exact.csv` / `exact_long.csv` against `exact_indep.csv`; same graphs, per-call limit
3,600 s for the old, ≤ 3,600 s for the new):

| graph | old (column lex only) | new (independent-set fix) |
|---|---|---|
| K(7,2), M5, C_31^3, 6 planted undirected n = 50 | open after 3,600 s | proven in < 2 s each |
| M6 | open after 3,600 s | proven 24 in 293 s |
| K(8,3) | best 28, open | **proven 22** in 737 s |
| 8 planted directed n = 50 | best 14–16, open | all proven 8; 4 in < 3 s, 4 in 769–3,408 s |
| Paley 29–53, circ_41 | open | still open |

So 20 of the 24 graphs that both encodings attempted are proven by the new encoding, and none by the old.

**Soundness checks.**
- 67/67 UNSAT CNFs from the 23 Nov runs have a DRAT proof checked by drat-trim (`minrank/results/drat_checks.csv`).
  These include K(8,3) at rank 21 (950 s, 869 MB proof) and M6 at rank 23.
- The 24 Nov UNSAT CNFs (new families, planted n = 100 at rank 9) are checked by the same script; see the same CSV.
- K(8,3) rank 21 was confirmed UNSAT three times: two solvers, two independent sets.
- Every SAT answer is re-verified as a fitting matrix of the claimed rank (`core/verify.py`).

## 2. Exact values

**Kneser graphs K(n,k)** (alpha = C(n-1, k-1); the cover is the clique cover number):

| graph | alpha | cover | minrk_2 | how |
|---|---|---|---|---|
| K(n,2), n even | n - 1 | n - 1 | n - 1 | cover = alpha |
| K(n,2), n odd ≥ 5 | n - 1 | n | **n** | **Theorem 9.2 (star identity)**; SAT for n = 5, 7 |
| K(7,3) | 15 | 18 | 18 | SAT |
| K(8,3) | 21 | 28 | **22** | SAT + DRAT (rank 21 UNSAT); I + D gives 22 |
| K(9,3), K(12,3) | 28, 55 | = alpha | alpha | Baranyai |
| K(n,3), n ≡ 2 mod 4 | C(n-1,2) | > alpha if 3 ∤ n | **alpha** | rank_2(I + D) = alpha (Theorem 1 of section 6); computed for n = 10, 14, 22 |
| K(n,3), n ≡ 0 mod 4 | | | alpha or alpha + 1 | Theorem 1; n = 8 is alpha + 1 |
| K(12,4), K(16,4) | alpha | | alpha | Baranyai |
| K(9,4) | 56 | 63 | 57..63 | ≥ 57 by Theorem 9.2; open (portfolio running) |
| K(10,4) | 84 | 105 | 84..93 | I + D gives 93; open (portfolio running) |
| K(11,3) | 45 | 55 | 45..55 | open (portfolio running); over F_5 it is 45 |
| K(13,3) | 66 | 72 | 66..72 | open (portfolio running); over F_7 it is 66 |
| K(16,3), K(20,3) | 105, 171 | | alpha or alpha + 1 | I + D gives alpha + 1; SAT at n = 560 is out of reach (~10^8 clauses) |
| K(14,5) | 715 | ≥ 1001 | 715 or 716 | I + D gives 716 |

**Other families** (`families.csv`, `families_new.csv`; every value proven unless an interval is given):

| family | values |
|---|---|
| Mycielski M3..M6 | 3, 6, 12, 24 (= alpha + 1); M7 in [47, 48], decided by the reduction of section 10 (running) |
| Paley P5, P13, P17 | 3, 5, 6 (= cover) |
| Paley 29, 37, 41, 53, 61 | [7,8], [7,10], [8,9], [8,11], [8,13] (portfolio running) |
| orthogonality graphs k = 4..7 | 4, 5, 6, 7 |
| cycle powers C_n^2, C_n^3 (n = 9..31 odd) | all proven, 28 graphs (`families.csv`) |
| circulant C_41(1,5,8) | [13, 21] (portfolio running) |
| Johnson J(m,2), m = 6..9 | m - 2 (= cover) |
| Johnson J(7,3) | 9 (= cover) |
| Johnson J(8,3) | [11, 12] |
| generalized Kneser (|A ∩ B| ≤ 1): (7,3), (8,3), (9,3) | 6, 7, 7 (cover; alpha = cover for (9,3)) |
| odd graph O_3 = K(7,3) | 18 |
| co-Kneser (7,3), (8,3) | 3, 4 (= cover) |
| hypercubes Q3..Q6, H(3,3), H(2,4), distance graphs of Q4, Q5 | = cover |

Among the new families, none has minrk_2 below the clique cover. The Kneser graphs with k ∤ n are the only family
here where minrank beats the cover.

**Planted benchmark.**
- With extra_p = 0 the planted graphs are trivial: I + A is the planted matrix (`docs/MINRANK_THEORY.md` section 5).
- With extra_p = 0.1 (`exact_planted_e01.csv`):
  - n = 30 and n = 50: 32/32 proven;
  - directed n = 100: 8/8 proven = 10;
  - undirected n = 100: all 8 are 10. Pincer (unmerged branch) found the rank-10 matrices and we verified them with
    our checker. Rank 9 is UNSAT with our encoding and an independent fixed set, in 2–25 s per graph
    (`minrank/results/planted_n100_check/`).

## 3. Theory (details in `docs/MINRANK_THEORY.md`)

- **I + D construction.** I + D fits K(n,k) over every field. Its F_2 rank is alpha for k = 3 and n ≡ 2 (mod 4)
  (section 6, Theorem 1). The general-k, all-primes version is on branch `feature/kneser-minrank-theorem` (section 11);
  we checked it independently on 17 cases. Corollary: the Haemers bound is tight on K(n,3) for every n ≥ 9.
- **Star identity (section 9).** A rank-alpha fitting matrix of K(n,k) satisfies n M = k M² over F_2. Hence
  minrk_2 > alpha when k is even and n is odd, and minrk_2(K(n,2)) = n for every odd n.
- **Mycielski reduction (section 10).** minrk_2(mu(G)) = |V(G)| is equivalent to a rank-free SAT instance on G.
- **beta(K(8,3)).** It is 21 or 22. Pincer's symmetric Shannon LP enumerates 2^n subsets, so it is limited to
  n ≤ 16; K(8,3) has 56 vertices. Undecided.

## 4. Still open (24 Nov)

- K(9,4), K(10,4), K(11,3), K(13,3), Paley 29–61, C_41(1,5,8): portfolio of 4 solver/mode members, 8 h per call
  (`scripts/minrank/11_open_portfolio.sh`, `minrank/results/open_portfolio/`).
- M7: the reduction runs with an 8 h limit (`logs/minrank_mycielski_red_M7.log`).
- J(8,3) at rank 11.
- K(n,3) for n ≡ 0 (mod 4), n ≥ 12, 3 ∤ n (K(16,3), K(20,3)): alpha or alpha + 1. SAT is infeasible at this size;
  a proof would need the involution structure of section 9.
- beta(K(8,3)): 21 or 22.
