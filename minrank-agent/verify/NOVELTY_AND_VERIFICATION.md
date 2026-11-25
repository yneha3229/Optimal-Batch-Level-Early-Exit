# Kneser minrank theorem: independent verification and literature check (25 Nov 2026)

## Verification (`verify/check_theorem.py`, written independently of the proof's own scripts)

| check | cases | failures |
|---|---|---|
| Lemma 2.4 (arithmetic): (2.1) holds ⇔ q_p(k) ∣ N, k ≤ 12, p ≤ 13, N ≤ 400 | 26 400 | 0 |
| Theorem 1: rank_p(I+εD) = C(n−1,k−1) ⇔ q_p(k) ∣ n−2k, direct rank mod p, C(n,k) ≤ 1300, k ≤ 5, p ≤ 13 | 348 | 0 |
| Theorem 2: the explicit code, simulated with random messages, every receiver decodes | 60 | 0 |
| Prop. 4.5: no rank-6 F₂ code for K(7,2) (certified CNF, already DRAT-checked) re-solved by CaDiCaL and Glucose | 1 | 0 (UNSAT) |
| Petersen K(5,2) (our exception for k=2): Blasiak–Kleinberg–Lubetzky, Fact 5.6, give β = 5 > α = 4 | — | consistent |

## Literature read (full text, searched for minrank / p-rank / index coding / Kneser)

| paper | what it has | overlap |
|---|---|---|
| Golovnev–Haviv, ToC 2022 (arXiv 2002.08580) | uses the same matrix (entries C(\|A∩B\|−1, k−1)); bounds its **real** rank by Σ_{i<k} C(n,i) | upper bound only, no exact value, no field condition |
| Haviv, APPROX 2018 (1802.03920); Haviv 2018 (1811.11488) | minrank upper bounds Σ C(d,i) for generalized Kneser graphs over large F_p; topological lower bounds on orthogonality dimension | no exact minrank of K(n,k) |
| Ducey et al., ALCO 2024 (2310.09227) | diagonal forms of D − λI for any λ (n ≥ 3k−1); explicit only for λ = 0 and Laplacians (k = 2, 3) | the λ = ±1 case (our H) is never stated; our Appendix A derives it from their framework, so the "if" direction is derivable from their work in principle |
| Hu–Tang, 2026 (2607.06480) | symmetric minimum rank (all edges nonzero) over **infinite** fields of the **complement** of threshold Kneser graphs | different parameter, graph and fields |
| Sason, 2026 (2608.06573) | Shannon capacity of lexicographic products of Kneser graphs, via Θ(K(n,k)) = ϑ = α (Lovász) | no minrank |
| Blasiak–Kleinberg–Lubetzky 2010 (1004.1379) | "the exact value of β(G) remained unknown for every graph G except those for which α = χ̄"; Petersen β = 5 | supports novelty of exact β for k ∤ n |

## Assessment

Not found anywhere: the exact minrank of K(n,k) over F_p, the if-and-only-if p-rank condition for I + εD, the
explicit code and decoder, and β(K(n,k)) = C(n−1,k−1) for k ∤ n.

Open risks to close before any public claim:
1. **k = 2.** K(n,2) is strongly regular (complement of the triangular graph T(n)); p-ranks of A − aI for strongly
   regular graphs are studied by Brouwer–van Eijl (1992) and in Brouwer–Haemers, *Spectra of Graphs*. The k = 2 case
   may follow from there. Both sources were unreachable from this environment.
2. **"If" direction via Ducey et al.** It is a consequence of their diagonal-form theorem with λ = −ε (they never
   state it). The new parts are then: the stated result, the "only if", the elementary one-page proof, the code and
   decoder, and the index-coding consequences.
3. A specialist read (e.g. an author of Golovnev–Haviv or Ducey et al.) is the most reliable final check.
