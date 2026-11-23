# Minrank of Kneser graphs over F_2: construction and conjectures (WORK.md T2), 23 Nov 2026

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
