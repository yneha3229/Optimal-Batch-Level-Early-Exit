# Minrank of Kneser graphs over F_2: construction and conjectures (WORK.md T2), 23 Nov 2026

> **Update 24 Nov 2026.** The k = 3 pattern of section 2 is now **proven** (section 6, Theorem 1), and it extends to
> odd characteristic: rank_p(I + D) = alpha whenever p | n - 6 (Theorem 2). Together with Baranyai this gives
> **min over prime fields of minrk_p(K(n,3)) = alpha = Theta(K(n,3)) for every n >= 9** (Corollary 3). Data for
> k = 2, 3, 4 over F_2 ... F_19: `minrank/results/kneser_prank_k{2,3,4}.csv`, script `minrank/analysis/kneser_prank.py`.
> Literature status: section 8.


Data scripts: `minrank/baselines/identity_adj.py` (the construction). The exploration scripts behind the tables
(group-invariant SAT search, rank sweeps) are summarised here and can be re-run on request.

## 1. The construction: I + D

Let D be the adjacency matrix of K(n,k) (D_ST = 1 iff S ∩ T = ∅). The matrix **I + D** has ones on the diagonal
and zeros exactly on intersecting pairs, so it fits K(n,k):

> **minrk_2(K(n,k)) <= rank_2(I + D).**

**Why this matrix.** S_n acts on K(n,k) transitively on ordered pairs of disjoint sets, so the only S_n-invariant
fitting matrices over F_2 are I and I + D.

**How it was found.** We asked the SAT solver for a rank-22 fitting matrix of K(8,3) and a rank-36 one of K(10,3)
that is invariant under a chosen group H <= S_n. All 22 groups tried admit one, in about 1 s and 6 s: from C_7, C_8
and C_9 up to PSL(2,7), PGL(2,7), PGL(2,9), S_7, S_9 and S_5 x S_5. **Every solution found is exactly I + D**, and
each is symmetric. So the optimal matrices the solver had found for these graphs are explained completely by the
symmetric one.

**Algebra (checked numerically for all cases in section 2).** Inclusion-exclusion gives
[S ∩ T = ∅] = sum_{i=0}^{k} (-1)^i C(|S ∩ T|, i), i.e. D = sum_{i=0}^{k} (-1)^i W_i^T W_i over the integers, where W_i
is the inclusion matrix (i-subsets x k-subsets). The i = k term is I. Mod 2 this gives

> **I + D = M^T M (mod 2)**, with M = [W_0; W_1; ...; W_{k-1}] stacked,

so rank_2(I + D) <= rank_2(M), and rank_2(W_{t,k}) mod p is known (Wilson's rank formula). The rank of a Gram matrix
M^T M over F_2 also depends on how much of the row space of M is self-orthogonal; that is where n mod 4 enters.

## 2. Data: rank_2(I + D) against alpha = C(n-1,k-1) and the clique cover

The clique cover is >= ceil(C(n,k) / floor(n/k)), with equality when k | n (Baranyai). Values marked * are proven
minrank values (SAT, with DRAT-checked UNSAT where needed); the rest are upper bounds.

| k | n | alpha | rank_2(I + D) | cover >= | minrk_2 |
|---|---|---|---|---|---|
| 2 | odd n (5..29) | n-1 | n | n | **n for every odd n** (Corollary 14.3) |
| 2 | even n (6..30) | n-1 | n-1 | n-1 | n-1 (= alpha) |
| 3 | 7 | 15 | 21 | 18 | 18* (SAT finds a non-invariant matrix below I + D) |
| 3 | 8 | 21 | **22** | 28 | **22*** |
| 3 | 9 | 28 | 36 | 28 | 28 (Baranyai) |
| 3 | 10 | 36 | **36** | 40 | **36*** (= alpha) |
| 3 | 12 | 55 | 56 | 55 | 55 (Baranyai) |
| 3 | 14 | 78 | **78** | 91 | **78*** (= alpha) |
| 3 | 16 | 105 | 106 | 112 | 105 or 106 |
| 3 | 20 | 171 | 172 | 190 | 171 or 172 |
| 3 | 22 | 210 | **210** | 220 | **210*** (= alpha) |
| 3 | odd n (9..23) | C(n-1,2) | C(n,2) | | no gain |
| 4 | 10 | 84 | 93 | 105 | <= 93 |
| 4 | 14 | 286 | 299 | 334 | <= 299 |
| 4 | 18 | 680 | 697 | 765 | <= 697 |
| 4 | 12, 16 | alpha | alpha | alpha | alpha (Baranyai) |
| 5 | 12 | 330 | 384 | 396 | <= 384 |
| 5 | 14 | 715 | **716** | 1001 | 715 or 716 |

Patterns (k = 3, verified for 7 <= n <= 24; k = 4 for 9 <= n <= 18; k = 5 for 11 <= n <= 16):
- **k = 3:** rank_2(I + D) = alpha for n ≡ 2 (mod 4), alpha + 1 for n ≡ 0 (mod 4), and C(n,2) for odd n.
- **k = 4:** alpha for n ≡ 0 (mod 4), alpha + (n - 1) for n ≡ 2 (mod 4), and C(n,3) or C(n,3) - 1 for odd n.
- **k = 2:** n - 1 for even n, n for odd n (the chromatic index of K_n, equal to the cover).

## 3. What this gives

**Theorem (if the k = 3 pattern is proven).** For n ≡ 2 (mod 4), minrk_2(K(n,3)) = alpha = C(n-1,2), and for
3 ∤ n this is strictly below the clique cover (n = 10, 14, 22, 26, ...). Since alpha is a lower bound, the proof
needs only rank_2(I + D) = C(n-1,2), a statement about inclusion matrices mod 2, and no SAT.

**Corollaries and remarks.**
1. For n ≡ 0 (mod 4), 3 ∤ n: minrk_2(K(n,3)) in {alpha, alpha + 1}. For n = 8 SAT proves alpha + 1 (DRAT-checked);
   n = 16, 20 are open (is it always alpha + 1?).
2. K(14,5): within 1 of alpha (716 vs 715), while the cover is >= 1001.
3. **Haemers bound.** minrk_F(G) is Haemers' upper bound on the Shannon capacity, and Theta(K(n,k)) = alpha (Lovász).
   So the F_2 Haemers bound is tight on K(n,3) for n ≡ 2 (mod 4) and off by exactly one on K(8,3). To check against
   the literature: the 2-ranks of Kneser-type matrices may already be known (Smith normal forms / p-ranks of
   inclusion and disjointness matrices: Wilson 1990; Brouwer and van Eijl 1992 for strongly regular graphs).
4. **Index coding.** For n ≡ 2 (mod 4), the binary linear index code from I + D is optimal among all codes
   (beta >= alpha). For K(8,3) the best binary linear code has length 22 while beta >= 21: is beta(K(8,3)) 21 or 22,
   or fractional? (Pincer's Shannon LP with symmetry reduction could answer this.)

## 4. Next steps for the proof

1. Prove rank_2(M^T M) = C(n-1,2) for k = 3 and n ≡ 2 (mod 4). Ingredients:
   - rank_2(W_{2,3}) by Wilson's formula: the sum over i <= 2 with C(3-i, 2-i) odd of C(n,i) - C(n,i-1), i.e.
     i = 0 (C(3,2) = 3) and i = 2 (C(1,0) = 1), giving 1 + C(n,2) - n = C(n-1,2) = alpha for every n (matches the
     computed values: 21, 28, 36, 45, 55 for n = 8..12). So the row space of W_2 alone already has dimension alpha;
     the question is what W_0 and W_1 add and what the Gram form loses;
   - the contribution of W_1, and the radical of the Gram form (self-orthogonal part of the row space of M) depending
     on n mod 4.
2. Do the same for n ≡ 0 (mod 4) (alpha + 1) and for k = 4.
3. Lower bounds: decide K(16,3) at rank 105 with SAT, now that the upper bound is 106 (n = 560 is large; try with the
   independent-set fix and the S_n-orbit structure).

## 5. A caveat found on the way: the planted benchmark graphs are trivial for I + A

`minrank.bench.generators.planted` makes the graph's edges exactly the support of the planted matrix when
extra_p = 0, and every planted graph in the benchmark has e0.0. Then I + A(G) is the planted matrix, so
rank_2(I + A) = planted rank on all 48 planted graphs (n = 30, 50, 100, directed and undirected), in microseconds.
Consequences:
- the upper-bound side of every "planted" result (baselines, the evolve gate, Pincer's planted rows) is matched by a
  one-line baseline;
- our planted results stand only on the lower-bound side (UNSAT at rank 7 for the undirected n = 50 graphs).

Fix: regenerate planted graphs with extra_p > 0 (extra edges add free entries, so I + A no longer equals the
planted matrix), and add I + A as a baseline everywhere.

**Done (24 Nov).**
- `minrank/bench/instances.py` adds an extra_p = 0.1 copy of every planted family: 48 graphs `*_e0.1_*`, same
  sizes and seed splits. The 172 old graphs are unchanged (digests checked).
  - I + A on the new graphs has rank 44-50 at n = 50 and 99-100 at n = 100. Even extra_p = 0.05 already gives 36-43
    at n = 50.
- `identity_adj` is a baseline method in `minrank/run_baselines.py` (budget-free, one row per graph), and it has been
  run on all 220 graphs (`minrank/results/baselines.csv`).
  - On the old e0.0 planted graphs it strictly beats every other baseline on 26 of 48 and ties the rest.
  - On K(8,3) it gives 22 against 28 for the best heuristic.
- Pincer: `quick` now offers I + A, and the `classic` / `satdescend` baselines get it too.
  - Pincer's benchmark already used planted graphs with extra_p = 0.05.
  - I + A beats no method's upper bound on any of its 61 test / test2 graphs, so **Pincer's published numbers stand
    unchanged**.
- Baselines and exact SAT on the new graphs: `minrank/results/baselines_planted_e01.csv`,
  `minrank/results/exact_planted_e01.csv`.
  - Baselines: 9 methods, budgets of 1 and 10 s, seeds 0-1.
  - Exact: SAT with the independent-set fix, 300 s per call.
  - Results, 8 graphs per row:

| family (e0.1) | planted r | best baseline (mean) | I + A (mean) | exact |
|---|---|---|---|---|
| undirected n = 30 | 6 | 5.9 | 25.8 | 8/8 proven (one graph has minrank 5 < r) |
| directed n = 30 | 6 | 6.0 | 25.5 | 8/8 proven = 6 |
| undirected n = 50 | 8 | 8.1 | 47.6 | 8/8 proven = 8 (UNSAT at 7) |
| directed n = 50 | 8 | 13.1 | 48.9 | 8/8 proven = 8 (probe at alpha, ~1 s each) |
| undirected n = 100 | 10 | 12.9 | 99.0 | 2/8 proven = 10; 6 open, SAT stuck at 13 although minrk <= 10 by construction |
| directed n = 100 | 10 | 21.2 | 99.5 | 8/8 proven = 10 (probe at alpha, ~18 s each) |

  The undirected n = 100 graphs are now the hard planted instances: the true value is known to be at most 10, and
  every heuristic and the SAT descent stop at 12-13.


## 6. Proofs for k = 3 (24 Nov 2026)

Notation: n >= 7, D = adjacency matrix of K(n,3), alpha = C(n-1,2). Coordinates of F^N, N = 1 + n + C(n,2), are
indexed by the subsets A of [n] with |A| <= 2; M is the N x C(n,3) matrix with M[A, S] = [A subset of S] (that is,
M = [W_0; W_1; W_2]), and C is its column space. For |S cap T| = j, inclusion-exclusion gives the integer identity

    [S cap T = 0]  =  1 - j + C(j,2) - C(j,3),   hence   I + D = W_0^T W_0 - W_1^T W_1 + W_2^T W_2  =  M^T Q M

over Z, with Q = diag(+1 on W_0, -1 on W_1, +1 on W_2). Mod 2, Q = identity and I + D = M^T M.

**General fact.** For a nondegenerate diagonal Q, rank(M^T Q M) = dim C - dim R, where R = C cap ker(M^T Q) and
dim R = dim C^perp - rank(B_Q restricted to C^perp), B_Q(g, g') = g^T Q g'. (y -> Qy maps R onto the radical of B_Q on
C^perp, because C = (C^perp)^perp.) So everything reduces to an (n+1)-dimensional form.

### Theorem 1 (F_2). For n >= 7,
rank_2(I + D) = C(n,2) for odd n, = **alpha** for n = 2 (mod 4), = alpha + 1 for n = 0 (mod 4).
Hence **minrk_2(K(n,3)) = alpha for n = 2 (mod 4)**, strictly below the clique cover when 3 does not divide n
(n = 10, 14, 22, 26, 34, ...), and minrk_2(K(n,3)) is alpha or alpha + 1 for n = 0 (mod 4).

*Lemma 1 (C^perp over F_2).* C^perp = { f : f(empty) = a, f({i}) = b, f({i,j}) = a + b + x_i + x_j } with a, b in F_2,
x in F_2^n, and dim C^perp = n + 1 (so rank_2 M = C(n,2)).
Proof. f is in C^perp iff for every 3-set S: f(empty) + sum_{i in S} f(i) + sum_{e subset S, |e|=2} f(e) = 0. Read f on
pairs as a 1-cochain g on the full simplex on [n] and h(S) = f(empty) + sum_{i in S} f(i); the condition is
delta g = h. Then delta h = 0 on every 4-set Q: delta h(Q) = 4 f(empty) + 3 sum_{i in Q} f(i) = sum_{i in Q} f(i) = 0.
For n >= 5, two 4-sets differing in one element give f(i) = f(j), so f(i) = b for all i and h = a + b is constant. The
constant cochain a + b is a solution (a triangle has 3 edges) and the simplex has H^1 = 0, so the solutions are
g_ij = (a + b) + x_i + x_j, with x defined modulo the all-ones vector: dimension 2 + (n - 1) = n + 1. QED

*Lemma 2 (the radical).* With c = a + b, X = sum x_i, P = <x, x'>, the form on C^perp is
B(f, f') = aa' + n bb' + C(n,2) cc' + (n-1)(c X' + c' X) + (n-2) P + X X'.
- n = 2 (mod 4): n, n - 2 even, C(n,2), n - 1 odd: B = aa' + (c + X)(c' + X'), two independent forms, rank 2.
- n = 0 (mod 4): C(n,2) even: B = aa' + cX' + c'X + XX'; Gram matrix of (a, c, X) is [[1,0,0],[0,0,1],[0,1,1]],
  determinant 1, rank 3.
- n odd: n - 2 odd, n - 1 even; B = (aa' + bb' or ab' + a'b) + (P + XX'). The matrix I + J on F_2^n has kernel
  spanned by the all-ones vector (n odd), which is exactly the gauge, so the x-part has rank n - 1: total n + 1.
So dim R = n - 1, n - 2, 0 respectively, and rank_2(I + D) = C(n,2) - dim R. QED (Theorem 1)

### Theorem 2 (odd characteristic). Let p >= 5 be a prime with **p | n - 6**. Then rank_p(I + D) = alpha, hence
**minrk_p(K(n,3)) = alpha**. Conversely, if p does not divide n - 6 then rank_p(I + D) > alpha.
Proof. For p >= 5, Wilson's rank formula gives rank_p W_{2,3} = C(n,2) (all of C(3,2), C(2,1), C(1,0) are units).
The rows of W_0 and W_1 add nothing: summing the rows of W_2 gives 3 W_0 (a 3-set has three pairs), and summing the
rows {i, j} over j != i gives 2 (row i of W_1) (a 3-set containing i has two pairs through i). 2 and 3 are units,
so dim C = rank_p M = C(n,2) and dim C^perp = n + 1. C^perp consists of the vectors
g(empty) = -3(t + c), g({i}) = t - 2 x_i, g({i,j}) = c + x_i + x_j, with (c, t, x) modulo (c - 2s, t + 2s, x + s 1)
(each 3-set condition reads -3t - 3c + 3t - 2 sum_S x + 3c + 2 sum_S x = 0; the parametrisation has dimension n + 1).
With Q = (+, -, +):
B_Q = 9(t + c)(t' + c') - [n tt' - 2(t X' + t' X) + 4P] + [C(n,2) cc' + (n-1)(c X' + c' X) + (n-2) P + X X'].
The coefficient of P is **n - 6**. If p | n - 6, B_Q depends only on (c, t, X), a 3-dimensional space containing the
non-zero gauge direction (-2, 2, n) in its radical, so rank B_Q <= 2, dim R >= n - 1 and
rank_p(I + D) <= C(n,2) - (n - 1) = alpha; alpha is a lower bound for every fitting matrix, so equality holds.
If p does not divide n - 6, restrict B_Q to {c = t = 0, X = 0} (dimension n - 1, meets the gauge only in 0): it is
(n - 6) P on sum-zero vectors, of rank >= n - 2 >= 3, so dim R <= n - 2 and rank_p(I + D) >= alpha + 1. QED
(Theorem 1's case n = 2 (mod 4) is the same statement for p = 2 with "4 | n - 6".)

### Corollary 3 (the Haemers bound is tight for K(n,3), n >= 9). For every n >= 9 there is a prime p with
minrk_p(K(n,3)) = alpha = Theta(K(n,3)) (Lovász): if n - 6 has a prime factor p >= 5 use Theorem 2; if 4 | n - 6 use
Theorem 1; otherwise n - 6 = 2^e 3^f with e <= 1 and f >= 1, so 3 | n and the clique cover is already alpha
(Baranyai). The only open cases are n = 7, 8: over F_2 the minrank is 18 and 22 (SAT, DRAT-checked), above
alpha = 15 and 21; other fields are not decided. New exact values below the clique cover include K(11,3) over F_5,
K(13,3) over F_7, K(16,3) over F_5, K(17,3) over F_11, K(19,3) over F_13, K(20,3) over F_7.

All statements above are also checked numerically: `python -m minrank.analysis.kneser_prank --k 3 --n 7-16 --p 2
--gram` (both lemmas) and `minrank/results/kneser_prank_k3.csv` (96 (n, p) cases, n = 7..22, p <= 19: the
divisibility rule predicts every hit and every miss, including p = 3).

## 7. Other fields and other k (data, 24 Nov 2026)

Sweep of rank_p(I + (-1)^(k+1) D) for p in {2, ..., 19} (the sign is the one that makes the Gram identity work; the
only S_n-invariant fitting matrices are lambda I + mu D, and mu = (-1)^(k+1) was the best mu in every case we tried):

| k | range | rule rank_p = alpha iff p | n - 2k (odd p) / 4 | n - 2k (p = 2) | exceptions |
|---|---|---|---|
| 2 | n = 5..30 | holds for odd p (e.g. minrk_3(K(7,2)) = 6 < cover 7, minrk_5(K(9,2)) = 8) | p = 2: alpha for **every even n** (stronger) |
| 3 | n = 7..22 | holds in all 96 cases (Theorems 1, 2) | none |
| 4 | n = 9..16 | holds for p = 2 (n = 12, 16), p = 5 (n = 13: 220 < cover 239), p = 7 (n = 15: 364 < 455) | p = 3 (n = 11, 14): alpha + 1 |

Conjecture (general k): for a prime p > k with p | n - 2k, and for p = 2 with 4 | n - 2k, minrk_p(K(n,k)) = alpha.
The k = 3 proof should generalise (M = [W_0; ...; W_{k-1}], Q = diag((-1)^i)); the coefficient playing the role of
n - 6 is presumably n - 2k. Not attempted yet.

What this does **not** settle: the F_2 values for n = 0 (mod 4) (alpha or alpha + 1; SAT says alpha + 1 for n = 8),
odd n over F_2, K(9,4) over F_2 (57..63, I + D gives 83), and the index-coding rate beta(K(8,3)) (21 or 22).

## 8. Literature status of sections 6-7 (24 Nov 2026, full texts read)

Read in full (PDFs supplied by the user): Golovnev-Haviv (arXiv:2002.08580v2), Haviv 2018 (arXiv:1802.03920v2),
Haemers 1979 (IEEE T-IT 25(2)), Sason (arXiv:2608.06573v5). Their reference lists were scanned as well.

| Paper | What it does with Kneser graphs | Relation to Theorems 1-2 |
|---|---|---|
| Golovnev-Haviv, Thm 3.2 (m = 1) | Matrix M_AB = q(\|A cap B\|) with q(x) = C(x-1, s-1). Since q(0) = (-1)^(s-1), q(1) = ... = q(s-1) = 0 and q(s) = 1, **this is exactly I + (-1)^(s-1) D, our matrix (same sign)**. Bound: real rank <= sum_{i<s} C(d,i), so minrk_F <= that for every finite field (via rank mod p <= real rank, Fact 3.5) | They never compute its rank mod p. For k = 3 their bound is 1 + n + C(n,2); the true F_p rank is alpha = C(n-1,2) when p \| n - 6 (Theorem 2) or 4 \| n - 6 (p = 2). The F_2 remark (Lempel) is about orthogonality dimension of triangle-free graphs, unrelated |
| Haviv 2018, Prop 4.3 (T = {0}) | m(x) = prod_{j=1}^{s-1} (x - j), a scalar multiple of I + (-1)^(s-1) D again; minrk_p(K(d,s)) <= C(d, s-1) for primes p > s (Lemma 4.2 bounds the real rank) | Again only the generic rank: C(n,2) for k = 3, versus alpha = C(n,2) - (n-1) for p \| n - 6. Says nothing about p = 2 (needs p > s) |
| Haemers 1979 | Uses the adjacency matrix of K(8,2) (over R) only to build the Schläfli graph example | No overlap |
| Sason 2026 | Theta and fractional Haemers numbers of lexicographic products with Kneser factors; Theta(K(n,k)) = alpha cited from Lovász Thm 13 | No minrank values of Kneser graphs; no overlap |

**Verdict.** The matrix is known (Golovnev-Haviv, Haviv); the exact p-rank of it, the n - 6 divisibility rule, the
exact values minrk_p(K(n,3)) = alpha, and Corollary 3 (tightness of the Haemers bound for all n >= 9) appear in none of
these papers or their cited works. Framing for the paper: *"the polynomial-method matrix of Golovnev-Haviv and Haviv
for Kneser graphs has p-rank exactly alpha whenever p | n - 6; their bounds C(n,2) and 1 + n + C(n,2) are off by
n - 1"*. Cite both prominently.

Still worth checking before submission (not read): Wilson 1990 (diagonal form; used for rank_p W_{2,3}),
Frankl-Wilson / Ray-Chaudhuri-Wilson style p-rank results on intersection matrices, and recent work on Smith forms
of Kneser / disjointness matrices (arXiv:2310.09227, which treats disjointness as inclusion in the complement).
These could contain the p-rank of I + D in another guise; it is a narrower risk than before.

## 9. Independent verification of section 6 (24 Nov 2026)

`python -m minrank.analysis.verify_kneser_proof --nmax 22` (log: `logs/verify_kneser_proof.log`) checks each step of
the proofs separately, with ranks from FLINT (python-flint), not from our own elimination code:
- the integer identity I + D = W_0^T W_0 - W_1^T W_1 + W_2^T W_2, and that I + D fits K(n,3);
- rank_p M = C(n,2);
- the explicit bases of C^perp (Lemma 1 over F_2, and the (c, t, x) parametrisation over F_p) lie in C^perp and span it;
- their Gram matrices equal the closed forms B and B_Q term by term;
- rank B is as in Lemma 2;
- rank_p(I + D) = rank M - dim C^perp + rank B_Q;
- the iff of Theorems 1 and 2.

Coverage:
- n = 7..22 and p in {2, 3, 5, 7, 11, 13, 17, 19, 23}: 128 (n, p) cases, all checks pass. p = 3 is computed but not
  covered by the proof.
- The arithmetic of Corollary 3 is checked for 9 <= n <= 10^6.

Re-reading the proof found one missing sentence, now added to Theorem 2: why W_0 and W_1 do not enlarge the row space
for p >= 5. No step was found to be wrong. A human line-by-line check is still required before submission.

## 10. Final literature check: Wilson 1990 and Ducey et al. 2024 (24 Nov 2026, full texts read)

- **Wilson 1990** (Europ. J. Combin. 11, 609-615) gives the diagonal form and p-rank of the inclusion matrix W_{t,k}
  for t <= min{k, v - k}. This is exactly the rank formula used in Theorem 2, and the condition holds for t = 2,
  k = 3, v = n >= 5. The paper says nothing about Kneser graphs, disjointness matrices, Gram matrices or minrank.
  No overlap: it is a tool we cite.
- **Ducey, Engelthaler, Gathje, Jones, Pfaff, Plute** (arXiv:2310.09227v2, *Integer diagonal forms for subset
  intersection relations*) is the closest prior work.
  - Their Theorem 4.1 reduces the Smith group of any integer matrix in the Bose-Mesner algebra of the Johnson
    scheme to (k+1-s) x (k+1-s) blocks M_s, valid for n >= 3k - 1. Our I + D is in that algebra.
  - So the p-ranks in Theorems 1-2 **can be derived** from their framework by a finite computation. We checked
    that their route gives exactly our ranks for n = 8..22 and nine primes (`check_ducey_route` in
    `verify_kneser_proof.py`).
  - They work out only the Kneser adjacency matrix (lambda = 0, Thm 4.11) and the Kneser Laplacian (lambda = degree,
    Thm 4.13), not A + I (lambda = -1). They never mention minrank, the Haemers bound, Shannon capacity or index coding.

**Final verdict on novelty.**
1. The rank statement (rank_p(I + D) = alpha iff p | n - 6, or 4 | n - 6 for p = 2) is not stated in any paper we
   read. It is, however, a consequence of the Ducey et al. machinery, so a referee may call it routine given that
   paper. Cite them, present our proof as a short self-contained one, and optionally rederive it from their
   Theorem 4.1.
2. The minrank and information-theoretic consequences are new as far as we can tell:
   - exact minrk_p(K(n,3)) = alpha;
   - the Golovnev-Haviv / Haviv matrix has p-rank alpha, below their bounds by n - 1 or more;
   - Corollary 3: the Haemers bound is tight on K(n,3) for all n >= 9;
   - the n - 2k conjecture for general k.
   This is now the headline.
3. Opportunity: their framework may make the **general-k conjecture provable**. The rank of I + (-1)^(k+1) D reduces
   to the p-ranks of k + 1 small explicit blocks, and that is the way to a stronger paper.

## 11. The general-k theorem (24 Nov 2026)

**Theorem 4.** Let k >= 2, n > 2k, and let p > k be a prime with p | n - 2k. Then

    rank_p( I + (-1)^(k-1) D ) = C(n-1, k-1) = alpha,   so   minrk_p(K(n,k)) = alpha = Theta(K(n,k)).

(D = adjacency matrix of K(n,k). I + (-1)^(k-1) D is exactly the Golovnev-Haviv matrix q(|A cap B|),
q(x) = C(x-1, k-1).) It contains Theorem 2 (k = 3) and the odd-p data of section 7 for k = 2 and 4. The exceptions
seen there, p = 3 for k = 4, have p <= k and are outside the theorem.

**Proof.** Write eps = (-1)^(k-1), theta_j = (-1)^j C(n-k-j, k-j) (the eigenvalues of D), and
mu_s = C(n,s) - C(n,s-1).

*Step 1: reduction to small blocks (Ducey et al., arXiv:2310.09227, Def. 3.1, Thm 3.4, Lemma 4.10).*
- For n >= 3k - 1, L = D - lambda I has an integer diagonal form made of the diagonal forms of the blocks
  M_s(i,j) = -lambda delta_ij + C(j-s, i-s) (-1)^i C(n-k-j, k-j), with s <= i, j <= k.
- Each block M_s occurs with multiplicity mu_s - mu_{s-1}.
- Take lambda = -eps: then I + eps D = eps L. Our hypotheses give n - 2k >= p >= k + 1, so n >= 3k + 1 and the
  theorem applies.
- The p-rank of an integer matrix is the number of diagonal-form entries not divisible by p. Hence
  rank_p(I + eps D) = sum_{s=0}^{k} (mu_s - mu_{s-1}) rank_p(M_s).

*Step 2: the blocks.*
- Let T(i,j) = C(j-s, i-s) (upper Pascal) and Lam = diag((-1)^i). Pascal inversion gives T^{-1} = Lam T Lam, so
  Lam T = T^{-1} Lam.
- Hence T M_s = -lambda T + diag(theta_j), and rank_p M_s = rank_p(T + diag(eps theta_j)): for k odd this is direct
  (lambda = -1); for k even multiply by -1.
- Rewrite the diagonal entries with (-1)^r C(x, r) = C(r - x - 1, r), where r = k - j and w = 2k - n - 1:
  eps theta_j = -(-1)^(k-j) C(n-k-j, k-j) = -C(w, k - j).
- If p | n - 2k then w = -1 (mod p). Since k - j <= k < p, C(w, r) is a polynomial in w with p-invertible
  denominator r!, so C(w, k-j) = C(-1, k-j) = (-1)^(k-j) (mod p).
- With m = k - s + 1 and v = j - s, the block is congruent mod p to N_m = P_m - diag((-1)^(m-1-v)), where
  P_m(u,v) = C(v,u). It no longer depends on n.

*Step 3, Lemma: rank N_m = floor(m/2) over every field F with char F != 2.*
- P_m is the matrix of the shift (Ef)(x) = f(x+1) on polynomials of degree < m, in the basis x^0, ..., x^(m-1)
  (column v holds the coefficients of (x+1)^v).
- diag((-1)^(m-1-v)) is the matrix of f(x) -> sigma f(-x), with sigma = (-1)^(m-1).
- So ker N_m = { f : f(x+1) = sigma f(-x) }. Put g(z) = f(z - 1/2); the condition becomes g(z) = sigma g(-z).
- So g is even (m odd) or odd (m even), of degree < m: a space of dimension ceil(m/2). Hence
  rank = m - ceil(m/2) = floor(m/2).

*Step 4: summation.*
- rank_p = sum_s (mu_s - mu_{s-1}) floor((k-s+1)/2).
- Summation by parts turns this into sum of mu_s over the s with k - s odd, because
  floor((t+1)/2) - floor(t/2) = [t odd].
- That sum is C(n,k-1) - C(n,k-2) + C(n,k-3) - ... = C(n-1, k-1) = alpha.
- alpha is a lower bound for every fitting matrix, so minrk_p(K(n,k)) = alpha. QED

**Corollary 5 (Haemers bound).**
- If n - 2k has a prime factor p > k, or k | n (Baranyai: then the clique cover is alpha), then
  min_F minrk_F(K(n,k)) = alpha = Theta(K(n,k)) (Lovász).
- For each k this covers all n except those with n - 2k a product of primes <= k and k not dividing n. That is a
  density-zero set, e.g. for k = 4: n = 9, 10, 11, 14, 17, 26, ...
- k = 3 is complete for n >= 9, because Theorem 1 (p = 2) covers the remaining n = 2 (mod 4).
- The cases where Theorem 4 gives alpha strictly below the clique cover are exactly the n with k not dividing n.
  Examples: K(13,4) over F_5 (220 < 239), K(15,4) over F_7 (364 < 455), K(17,5) over F_7.

**What remains open.**
- p = 2 for k >= 4. The data say alpha iff 4 | n - 2k for k = 4, and every even n works for k = 2.
- Primes 2 < p <= k (for k = 4 and p = 3 the rank is alpha + 1).
- The n where no prime p > k divides n - 2k and k does not divide n.

**Checks.** `python -m minrank.analysis.verify_general_k` (log: `logs/verify_general_k.log`):
- [lemma] for m <= 40 and odd p <= 31;
- [reduce] the block formula equals the directly computed rank for k = 2..5, all n in range and p <= 13;
- [direct] Theorem 4 on 56 full matrices, k = 2..5;
- [blocks] the block formula gives alpha in all 439 cases with k <= 12 and n <= 80.
All pass.

**Novelty after this.** Step 1 is Ducey et al.'s theorem, cited and used as a black box. Steps 2-4 (the reduction
mod p to an n-independent Pascal-type block, the parity lemma, and the alternating sum to alpha) do not appear in
the papers read. The statement of Theorem 4 and Corollary 5 for general k was found in none of the six papers read.

## 12. All primes, including p = 2 and p <= k (24 Nov 2026)

**Theorem 6.** Let p be a prime, k >= 2, n > 2k, and let q = p^c be the smallest power of p with q >= k. If
**q | n - 2k**, then rank_p(I + (-1)^(k-1) D) = C(n-1, k-1) = alpha, hence minrk_p(K(n,k)) = alpha.
- p > k: q = p, so this is Theorem 4.
- p = 2: q = 2 for k = 2 (every even n), 4 for k = 3, 4 (Theorem 1 and the data of section 7), 8 for k = 5..8, and
  so on.
- 2 < p <= k: for example k = 4, p = 3 gives q = 9 (alpha at n = 17, 26, ...; alpha + 1 at n = 11, 14, as observed).
- k = 3, p = 3 gives q = 3, i.e. 3 | n.

**Proof.** Steps 1, 2 and 4 of Theorem 4 are unchanged. Only two points depend on p:

*(a) The diagonal mod p.* The block entries are 1 - C(w, r) with w = 2k - n - 1 and r = k - j in {0, ..., k}.
- For r < q, C(x, r) mod p is periodic in x with period q: C(x + q, r) = sum_i C(q, i) C(x, r - i), and p | C(q, i)
  for 0 < i < q.
- q | n - 2k means w = -1 (mod q), so C(w, r) = C(-1, r) = (-1)^r (mod p) for every r < q.
- Since q >= k, this covers all r <= k except r = k when k = q is itself a power of p. That entry sits only in the
  block s = 0 (position v = 0, the coefficient of x^0) and can take any value mod p.

*(b) The parity lemma in every characteristic.* N_m is the matrix of f -> f(x+1) - sigma f(-x) on polynomials of
degree < m, with sigma = (-1)^(m-1).
- char != 2: rank floor(m/2) (Theorem 4, Step 3).
- char 2: N_m = P_m + I is the matrix of f -> f(x+1) + f(x). Its kernel is the polynomials invariant under
  x -> x + 1. Since F_2[x] = F_2[y] + x F_2[y] with y = x^2 + x, and x -> x + 1 fixes y and sends x to x + 1, the
  invariants are F_2[y]. Those of x-degree < m have y-degree <= (m-1)/2, which is ceil(m/2) dimensions, so the rank
  is floor(m/2).
- The boundary entry of (a) changes only column 0 of the s = 0 block, by a multiple of e_0 = the constant 1. Here
  m = k + 1 >= 3, and the column of x is (x + 1) - sigma(-x) = 1 + (1 + sigma) x:
  - if 1 + sigma = 0 in F (char 2, or sigma = -1), that column is e_0 itself;
  - in the remaining case (char != 2 and sigma = +1, i.e. k = q even, so p = 2) it does not arise.
  So e_0 already lies in the span of the other columns, and changing column 0 by a multiple of e_0 leaves the
  column space, hence the rank, unchanged.

With every block of rank floor(m/2), Step 4 gives exactly alpha. QED

**Corollary 7 (Haemers bound, all k).** For every k >= 2 and all but finitely many n > 2k:
min_F minrk_F(K(n,k)) = alpha = Theta(K(n,k)).
- Proof: n is settled when q_p | n - 2k for some prime p (Theorem 6), or k | n (Baranyai).
- If neither holds, N = n - 2k has v_p(N) < c_p for every prime p. Then no prime p >= k divides N (for those
  c_p = 1), so N divides prod_{p < k} p^(c_p - 1), a constant depending on k.
- The exceptional n (computed; `n | k` removed):

| k | exceptional n |
|---|---|
| 2 | 5 (Petersen) |
| 3 | 7, 8 |
| 4 | 9, 10, 11, 14 |
| 5 | 11, 12, 13, 14, 16, 22 |
| 6 | 13, 14, 15, 16, 17, 22, 27, 32 |
| 7 | 15-20, 24, 26, 29, 34, 44, 74 |

For these n the F_2 values are known in a few cases: K(5,2) = 5, K(7,3) = 18, K(8,3) = 22 (SAT, DRAT-checked), all
above alpha. Whether some other field reaches alpha there is open.

**Is the condition also necessary?**
- For this matrix, yes numerically: the block formula gives rank_p = alpha if and only if q | n - 2k in all 6666
  cases with k <= 12, 3k - 1 <= n <= 120, p <= 13 (738 hits). Not proven.
- It says nothing about other fitting matrices. minrk_p could still be alpha by another matrix when the condition
  fails.

**Checks.** `python -m minrank.analysis.verify_all_primes` (log `logs/verify_all_primes.log`), all pass:
- the char-2 lemma, with and without the flipped entry, for m <= 64;
- the odd-p lemma with an arbitrary boundary entry;
- the periodicity of C(w, r) mod p;
- the iff in 6666 cases;
- full-matrix ranks for new predictions: K(17,4) over F_3 = 560 = alpha (q = 9), K(18,5) over F_2 = 2380 = alpha
  (q = 8), K(12,4), K(16,4) and K(20,4) over F_2 = alpha, and K(11,4), K(14,4) over F_3, K(18,4) over F_2 > alpha,
  as predicted.

## 13. Field separation runs (25 Nov 2026, shared 64-core server, CPU only)

Goal: proofs that minrk_{F_r}(K(n,k)) > alpha for a prime r while Theorem 1 of `proof.md` gives alpha over another
prime p (q_p(k) | n - 2k). For k = 2 and odd n, Theorem 1 gives alpha over every prime p | n - 4 and never over F_2,
so an F_2 UNSAT at r = alpha (cover = alpha + 1) is a separation. Only a VERIFIED DRAT row counts as proven.

Setup: at most 12 cores used (other users and the minrank portfolio share the machine): task A on cores 0-3, the F_3
sanity case on core 4, DRAT on cores 37-39. Jobs run detached (`setsid nohup`), driven by
`scripts/minrank/12_field_sep.sh` (log `logs/field_sep_driver.log`), which also commits partial results.

| graph | field | r | result | seconds | DRAT | meaning |
|---|---|---|---|---|---|---|
| K(7,2) | F_3 | 6 | TIMEOUT | 3600 | n/a | sanity case: Theorem 1 gives an explicit rank-6 matrix (checked: rank_3(I - D) = 6), yet the one-hot SAT encoding did not find one in 1 h. Task B stopped, as specified. |
| K(9,2) | F_2 | 8 | run 1 killed by the 25 Nov 20:16 UTC server reboot after ~24,000 s, no answer; run 2 started 26 Nov 02:40 UTC (36000 s limit) | | | Theorem 1: minrk_5 = 8; UNSAT would give minrk_2 = 9 (separation) |
| K(11,2) | F_2 | 10 | as K(9,2): run 1 killed after ~24,000 s, run 2 running | | | Theorem 1: minrk_7 = 10 |
| K(13,2) | F_2 | 12 | as K(9,2): run 1 killed after ~24,000 s, run 2 running | | | Theorem 1: minrk_3 = 12 |

Not run (task B, stopped by the sanity rule): K(9,2) F_3 r = 8, K(11,2) F_3 r = 10, K(11,2) F_5 r = 10,
K(7,2) F_5 r = 6. The one-hot encoding (8,379 variables, 31,737 clauses for K(7,2), r = 6, q = 3) cannot even find a
known solution on the smallest case in an hour, so 36000 s runs on 36- and 55-vertex graphs would very likely time
out too. A better F_q encoding (e.g. fixing the alpha x alpha identity block on a maximum independent set, as the F_2
`indep` mode does, or a log/order encoding of field elements) is the prerequisite.

Commands:

```bash
python -m minrank.exact.families --spec kneser:9,2 kneser:11,2 kneser:13,2 --time-per-call 36000 --lb-time 60 \
  --workers 3 --out minrank/results/field_sep/families_f2_long.csv --dimacs-dir minrank/results/unsat_cnf/field_sep
python -m minrank.exact.field_sep --case 7,2,3,6 --time-limit 3600       # sanity case
bash scripts/minrank/10_drat_check.sh minrank/results/unsat_cnf/field_sep
```

## 14. The star identity: minrk_2(K(n,k)) > alpha for k even and n odd (24 Nov)

**Lemma 14.1.** Let n >= 2k + 1, and let M fit K(n,k) over F_2 with rank_2(M) = alpha = C(n-1, k-1). Then for every
point a in [n], with S_a = {T : a in T} the star of a,

> M = M[:, S_a] · M[S_a, :],  and summing over a,  **n · M = k · M² (mod 2).**

*Proof.*
- The star S_a is an independent set of size alpha: its sets pairwise intersect. So M[S_a, S_a] = I, and the alpha
  rows indexed by S_a are linearly independent. Since rank M = alpha, they are a basis of the row space.
- Write row e as M_e = Σ_{T in S_a} c_T M_T. Evaluating at a column T' in S_a gives M_{e,T'} = c_{T'}. Hence
  M = M[:, S_a] M[S_a, :].
- Summing over the n points, each k-set T appears in exactly k stars:
  n M = Σ_a Σ_{T ∋ a} M[:, T] M[T, :] = k M². ∎

**Theorem 14.2.** For k even and n odd (n >= 2k + 1), minrk_2(K(n,k)) >= alpha + 1.

*Proof.* If the rank were alpha, Lemma 14.1 gives M = n·M = k·M² = 0 over F_2. This contradicts M_TT = 1. ∎

**Corollary 14.3.** minrk_2(K(n,2)) = n for every odd n >= 5.
- The lower bound n is Theorem 14.2.
- The upper bound is the clique cover: the chromatic index of K_n is n for odd n.
- This closes K(9,2), K(11,2) and K(13,2) (SAT timed out on them at 1,800 s per call) and every larger odd n. It
  also gives K(9,4) >= 57 without SAT; the rank-56 UNSAT call took 1,800+ s and its DRAT proof was too large to
  check on this disk.

**Structure for odd k (Lemma 14.1).**
- A rank-alpha fitting matrix is M = I + X with X supported on the Kneser edges.
- If n is odd, M² = M, i.e. X² = X.
- If n is even, M² = 0, i.e. X² = I: X is an involution and rank(I + X) = alpha.
- The I + D constructions of section 1 (n = 10, 14, 18, 22) and every stored rank-alpha certificate satisfy this:
  `python -m minrank.analysis.star_identity` checks all 17 rank-alpha matrices we have, M = M[:, S_a] M[S_a, :] for
  every a, and n M = k M² (`minrank/results/star_identity.json`).
- For k odd and n ≡ 0 (mod 4), e.g. K(8,3) = alpha + 1, a proof of the lower bound would have to show that no
  involution X supported on the Kneser edges has rank(I + X) = alpha. This is open; for n = 8 SAT decides it.

The same argument works for any graph with a family of maximum independent sets, each of size minrk, that covers
every vertex the same number r of times. With N sets: N·M = r·M². If N is odd and r even, minrk > alpha.

## 15. Mycielski graphs: a rank-free reduction (24 Nov)

M_k = mu(M_{k-1}) consists of the originals V, the shadows U = {u_y} and the apex w. The shadows are independent and
|U| = |V| = alpha(M_k).

**Lemma 15.1.** minrk_2(mu(G)) = |V(G)| (for |V| >= 2 alpha(G)) iff there exist
- A, B in F_2^{V x V}, both supported on the ordered edges of G, and
- x, z in F_2^V,
such that:
1. A B^T fits G;
2. A z = 0 and B x = 0;
3. ⟨x, z⟩ = 1.

*Proof.*
- (⇒) If rank M = |U|, the U-rows are a basis, so M = M[:, U] M[U, :]. Put A = M[V, U], B = M[U, V]^T,
  x = M[w, U] and z = M[U, w]. The zero pattern of M between U and V gives the supports. The fitting conditions on
  V ∪ {w} are conditions 1–3, since w is adjacent to no original.
- (⇐) The matrix [A; I; x^T] · [B^T | I | z] fits mu(G) and has rank at most |V|. ∎

The resulting SAT instance has no rank variables and only 2|E(G)| + 2|V| free bits. For M7, built on G = M6 with
47 vertices and 236 edges, that is 1,038 bits, against 95 · 47 · 2 = 8,930 for the rank encoding.
- Code: `minrank/exact/mycielski_reduction.py`.
- Test: `minrank/tests/test_mycielski_reduction.py`, which checks agreement with the direct encoding on random small
  graphs.
- M3, M4 and M5 are UNSAT in about 1 s, which agrees with the known minrk = alpha + 1.
- M6 and M7 are running with 8 h per instance; results go to `minrank/results/mycielski_reduction.json`.
- The conjecture "minrk_2(M_k) = alpha + 1 for every k" is equivalent to: for every Mycielski graph G, whenever A B^T
  fits G with supports on E(G), ker A and ker B are orthogonal.

## 16. The general-k p-rank theorem of the other session (branch `feature/kneser-minrank-theorem`, `proof.md`)

**Claim.** Let p be a prime and q the least power of p with q >= k. If q | n - 2k, then rank_p(I + (-1)^{k-1} D) = alpha,
so minrk over F_p of K(n,k) equals alpha. The argument goes through the diagonal form of Ducey et al.
- For k = 3 it is Theorems 1-2 of section 6 (already on this branch); the branch extends them to every k and every
  prime, as conjectured in section 7.
- It also gives the claimed corollary: the Haemers bound (minimum over fields) is tight on K(n,k) for all but
  finitely many n.

**Independent check (24 Nov).** We used our own Gaussian elimination mod p, not the branch's scripts, on 17 cases:
- the theorem holds on (n,k,p) = (7,2,3), (10,2,3), (11,2,7), (9,3,3), (12,3,3), (11,3,5), (13,3,7), (10,3,2),
  (14,3,2), (12,4,2), (16,4,2), (13,4,5), and (17,4,3), the q = 9 case;
- where q ∤ n - 2k, the rank was above alpha in every case tried: (9,2,7), (11,3,2), (12,3,2), (13,3,3).
- Over F_5, minrk(K(11,3)) = 45 = alpha, while over F_2 it is open between 45 and 55.

That branch was merged into main on 26 Nov; its general-k theorem is section 11 above (Theorem 4).

**Theorem 14.2 is consistent with it.** For p = 2 and k even, q >= k is even, so q | n - 2k forces n to be even.

**Novelty of sections 14-15 (not yet checked).** The star-identity argument is short, and statements such as
"minrk_2(K(n,2)) = n for odd n" may already be known, for instance in the index-coding literature on the
complement of the triangular graph. This has to be checked before it is presented as new.
