# Minrank literature check (WORK.md T3), 23 Nov 2026

Question: are (a) our exact F_2 minrank values for Kneser-type graphs, and (b) the independent-set symmetry breaking
for the SAT encoding, already known? Method: web searches plus reading two papers in full (text extracted from the
PDFs). This is a first pass, not a systematic review; the "not yet read" list at the end matters.

**Convention.** Haemers, Haviv and Golovnev-Haviv use the same definition as `minrank/`: M represents G if
M_ii != 0 and M_ij = 0 for distinct non-adjacent i, j (Golovnev-Haviv, Definition 1.8; Haviv 2018, Definition 4).
So minrk_F(K(n,k)) is the smallest rank of a matrix with nonzero diagonal and zeros on pairs of intersecting
k-sets, which is what our solver computes.

## What the literature has

| Source | What it gives for Kneser-type graphs | Compared with our values |
|---|---|---|
| Golovnev, Haviv, *The (generalized) orthogonality dimension of (generalized) Kneser graphs: bounds and applications*, Theorem 3.2 | For K<(d,s,m) (s-subsets of [d], adjacent iff intersection < m) and every finite field: minrk <= sum_{i=0}^{s-m} C(d,i), by a symmetric matrix. Kneser K(d,s) = K<(d,s,1), so minrk_2(K(d,s)) <= sum_{i=0}^{s-1} C(d,i) | K(8,3) <= 37, K(10,3) <= 56, K(n,2) <= n + 1. Ours: 22, 36, and n - 1 / n. The general bound is above even the clique cover (28, 40) |
| Haviv, *On minrank and the Lovász theta function* (APPROX/RANDOM 2018), Propositions 13, 14 | Polynomial-method upper bounds for K(d,s,T) over F_p for primes p > s (and one family with p = (s+1)/2); aims at graphs with small theta and large minrank | Not F_2 for s = 3 (needs p > s); no exact values |
| Same two papers, Fact 3.10 / Lemma 5 | minrk_F(G) * minrk_F(complement of G) >= n | Used in our complement lower bound for Paley graphs |
| Golovnev-Haviv citing [24] (Haviv) | Over R, orthogonality dimension equals the chromatic number on all Kneser graphs | Different field and different convention (orthogonality on adjacent pairs); does not transfer to F_2 minrank |
| Bukh, Cox, *On a fractional version of Haemers' bound* (IEEE T-IT 2019) | Fractional Haemers bound; implies bounds on generalized orthogonality dimension of K(d,s) over R | Different quantity (fractional); a possible tool for lower bounds on beta, not on minrk_2 |
| Beniamini, *The F_2-rank and size of graphs* | F_2 rank of the adjacency matrix of L(K_n) | Different quantity (fixed adjacency matrix, not minimum over fitting matrices) |
| Lovász 1979 | theta(K(n,k)) = C(n-1,k-1) = alpha, so the Shannon capacity of K(n,k) is alpha | minrk_F(G) is Haemers' upper bound on the Shannon capacity of G. So K(8,3) = 22 > Theta = 21 says the F_2 Haemers bound is not tight on K(8,3), while K(10,3) = 36 = Theta is tight. (To re-check against Lovász's paper before citing.) |

**SAT / symmetry breaking.** The searches found no SAT or CNF encoding of minrank at all. The general SAT symmetry-breaking
literature on graph search addresses vertex permutations:
- complete symmetry-breaking predicates for small graphs (Codish et al., arXiv:1511.08205);
- SAT Modulo Symmetries (CP 2024).

Neither addresses the GL(r) symmetry of the factors U, V. The closest analogue we know of is fixing colours on a
clique in SAT encodings of graph colouring (Van Gelder 2008). That is folklore-level and conceptually the same move:
use a structure every solution must contain to normalise the symmetry. The SAT literature on matrix multiplication
schemes (Heule, Kauers, Seidl) also uses symmetry fixing; **not checked yet**.

## Verdict

- **(a) Values.** We found no published exact F_2 minrank for any of K(7,3), K(8,3), K(10,3), the odd K(n,2), or the
  Mycielski graphs, and the published general upper bounds (37 and 56 for K(8,3) and K(10,3)) are far from ours.
  K(8,3) = 22 and K(10,3) = 36 below the clique cover look new, as far as this pass can tell.
- **(b) Method.** The independent-set normalisation itself is natural and close to known colouring tricks, so it
  should be presented as "we are not aware of prior use for minrank; it is the analogue of clique fixing in
  colouring encodings", with the soundness lemma and the measured speedups, not as a major methodological novelty.
- A paper should lead with the explicit separations (K(8,3), K(10,3)) and their certificates, use the method as the
  enabling tool, and aim for a construction or theorem explaining them (T2).

## Not yet read (do before writing a paper)

1. Haemers 1979 / 1981 (the original minrank papers; Kneser examples?).
2. *Shannon capacity and related graph invariants for lexicographic products* (arXiv:2608.06573): determines
   capacities of products with Kneser graphs using Haemers-type bounds, so it may contain minrank values of Kneser graphs.
3. Arbabjolfaei and Kim, *Fundamentals of Index Coding*: tables of small index coding problems.
4. Alipour, Gohari et al. on exact Haemers numbers of Johnson-type graphs (precedent for exact values; found in the
   earlier scan, not re-read here).
5. Heule-Kauers-Seidl symmetry breaking for matrix multiplication, to position the method.

Sources:
- Golovnev, Haviv: https://golovnev.org/papers/kneser.pdf
- Haviv 2018: https://drops.dagstuhl.de/storage/00lipics/lipics-vol116-approx-random2018/LIPIcs.APPROX-RANDOM.2018.13/LIPIcs.APPROX-RANDOM.2018.13.pdf
- Bukh, Cox: https://arxiv.org/abs/1802.00476
- Beniamini: https://arxiv.org/pdf/2206.11625
- Lexicographic products: https://arxiv.org/pdf/2608.06573
- Codish et al.: https://arxiv.org/pdf/1511.08205v1
- SAT Modulo Symmetries (CP 2024): https://drops.dagstuhl.de/storage/00lipics/lipics-vol307-cp2024/LIPIcs.CP.2024.37/LIPIcs.CP.2024.37.pdf
