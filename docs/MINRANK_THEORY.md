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
| 2 | odd n (5..29) | n-1 | n | n | n* for n = 5, 7; <= n otherwise |
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
Proof. For p >= 5, Wilson's rank formula gives rank_p W_{2,3} = C(n,2) (all of C(3,2), C(2,1), C(1,0) are units), so
dim C = C(n,2) and dim C^perp = n + 1. C^perp consists of the vectors
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
