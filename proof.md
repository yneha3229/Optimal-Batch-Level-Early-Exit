# Exact minrank of Kneser graphs over prime fields, and an explicit optimal index code

## Summary

For integers $`n > 2k \ge 4`$, let $`K(n,k)`$ be the Kneser graph (vertices: the $`k`$-subsets of $`[n]`$; edges: disjoint
pairs) and $`D`$ its adjacency matrix. Put $`\varepsilon=(-1)^{k-1}`$ and

```math
H \;=\; I + \varepsilon D, \qquad H_{A,B} = \binom{|A\cap B|-1}{k-1}.
```

$`H`$ fits $`K(n,k)`$ over every field; it is the matrix of Golovnev–Haviv, who bounded its rank by $`\sum_{i<k}\binom ni`$.

Here $`\alpha(G)`$ is the independence number. For every graph and field $`\alpha(G)\le\mathrm{minrk}_{\mathbb F}(G)`$, and
$`\alpha(K(n,k))=\binom{n-1}{k-1}`$ (Erdős–Ko–Rado). Write $`q=q_p(k)`$ for the smallest power of the prime $`p`$ with
$`q\ge k`$.

> **Theorem 1.** For every prime $`p`$:
> ```math
> \mathrm{rank}_{\mathbb F_p} H = \binom{n-1}{k-1} \iff q_p(k) \mid n-2k .
> ```
> In particular, if $`q_p(k)\mid n-2k`$ then $`\mathrm{minrk}_{\mathbb F_p}(K(n,k))=\alpha(K(n,k))`$.

> **Theorem 2 (explicit optimal index code).** If $`q_p(k)\mid n-2k`$, the following scalar linear index code over $`\mathbb F_p`$
> for the side-information graph $`K(n,k)`$ uses $`\alpha=\binom{n-1}{k-1}`$ transmissions, which is optimal among **all**
> (also non-linear) index codes. For every $`k`$-set $`A`$ containing the element $`1`$, broadcast
> ```math
> y_A \;=\; x_A + \varepsilon \sum_{B\cap A=\emptyset} x_B .
> ```
> A receiver $`A\ni1`$ decodes $`x_A`$ from $`y_A`$. A receiver $`A'\not\ni1`$ computes
> $`\varepsilon\sum_{A\ni 1,\,A\cap A'=\emptyset} y_A = x_{A'}+\varepsilon\sum_{B\cap A'=\emptyset}x_B`$ and subtracts its side information.

> **Corollary 3 (broadcast rate).** For every $`k\ge2`$ and all but finitely many $`n`$, the broadcast rate of the index
> coding problem on $`K(n,k)`$ is $`\beta = \binom{n-1}{k-1}`$, attained by a scalar linear code over a suitable prime
> field. The field cannot be fixed in advance: $`\mathrm{minrk}_{\mathbb F_2}(K(7,2)) = 7 > 6 = \mathrm{minrk}_{\mathbb F_3}(K(7,2))`$.

**What is new, precisely.**

- Known: the lower bound $`\alpha\le\mathrm{minrk}`$, the matrix $`H`$, and the equality $`\mathrm{minrk}=\alpha`$ when
  $`k\mid n`$ (over every field, via Baranyai's clique cover, Proposition 4.1).
- New: the exact $`p`$-rank of $`H`$ with an if-and-only-if condition; the explicit code and decoder; infinitely many $`n`$
  with $`k\nmid n`$, where no clique cover (even fractional) reaches $`\alpha`$, for every $`k`$ (Corollary 4.3); and the
  resulting broadcast rate.
- The proof (§2) is elementary and self-contained: about one page. A second, independent proof through the integer
  diagonal forms of Ducey et al. is in Appendix A, and a third for $`k=3`$ in Appendix B.
- Not claimed: anything about fields other than the $`\mathbb F_p`$ in question, or $`\mathrm{minrk}>\alpha`$ when
  $`q\nmid n-2k`$ (Theorem 1 says only that $`H`$ itself misses $`\alpha`$ then).

The Shannon capacity plays no role; see Remark 5.4.

---

## 1. Definitions

**Minrank.** A matrix $`M\in\mathbb F^{V\times V}`$ *fits* a graph $`G`$ if $`M_{vv}\neq0`$ for all $`v`$ and $`M_{uv}=0`$
whenever $`u\neq v`$ are non-adjacent. $`\mathrm{minrk}_{\mathbb F}(G)`$ is the least rank of a fitting matrix.

1. $`\alpha(G)\le\mathrm{minrk}_{\mathbb F}(G)`$: on an independent set a fitting matrix restricts to a diagonal matrix
   with nonzero diagonal.
2. $`\mathrm{minrk}_{\mathbb F}(G)\le\bar\chi(G)`$, the clique cover number.

**Index coding.** In the index coding problem with side-information graph $`G`$, receiver $`v`$ wants $`x_v`$ and knows
$`x_u`$ for every neighbour $`u`$. The optimal length of a scalar linear code over $`\mathbb F`$ is
$`\mathrm{minrk}_{\mathbb F}(G)`$ (Bar-Yossef–Birk–Jayram–Kol), and the broadcast rate $`\beta(G)`$ over all codes
satisfies $`\alpha(G)\le\beta(G)\le\mathrm{minrk}_{\mathbb F}(G)`$.

**Kneser graphs.** By Erdős–Ko–Rado, the largest independent sets of $`K(n,k)`$ (intersecting families) have size
$`\binom{n-1}{k-1}`$; one is the star $`\mathcal S=\{A: 1\in A\}`$.

**The matrix $`H`$.** $`H_{AA}=1`$; $`H_{AB}=\varepsilon`$ if $`A\cap B=\emptyset`$; $`H_{AB}=0`$ otherwise. Equivalently
$`H_{AB}=\binom{t-1}{k-1}`$ with $`t=|A\cap B|`$, since $`\binom{-1}{k-1}=\varepsilon`$, $`\binom{t-1}{k-1}=0`$ for
$`1\le t\le k-1`$ and $`\binom{k-1}{k-1}=1`$. So $`H`$ fits $`K(n,k)`$.

**Conventions.** $`\binom xr=x(x-1)\cdots(x-r+1)/r!`$ for every integer $`x`$ and $`r\ge0`$. $`N=n-2k\ge1`$.
$`\mathrm{rank}_p`$ is rank over $`\mathbb F_p`$.

---

## 2. Proof of Theorem 1

**Lemma 2.1 (the star is a basis candidate).** $`H[\mathcal S,\mathcal S]=I`$. Hence the $`\alpha`$ rows of $`H`$ indexed by
$`\mathcal S`$ are linearly independent over every field, and $`\mathrm{rank}\,H\ge\alpha`$.

*Proof.* Two distinct members of $`\mathcal S`$ share the element 1, so $`H_{AB}=0`$; and $`H_{AA}=1`$. $`\square`$

**Lemma 2.2 (the decoding identity).** For $`A'\notin\mathcal S`$ and any $`k`$-set $`B`$, let
$`R(A',B)=\sum_{A\in\mathcal S,\ A\cap A'=\emptyset}H_{A,B}`$ (an integer). Then

- if $`1\in B`$: $`\varepsilon R(A',B)=H_{A',B}`$;
- if $`1\notin B`$: $`\varepsilon R(A',B)=\binom{N-1+t}{k-1}`$, where $`t=|A'\cap B|`$, while $`H_{A',B}=\binom{t-1}{k-1}`$.

*Proof.* $`R(A',B)=[\,B\in\mathcal S,\ B\cap A'=\emptyset\,]+\varepsilon\,\#\{A\in\mathcal S: A\cap(A'\cup B)=\emptyset\}`$.

- If $`1\in B`$, the count is $`0`$, so $`\varepsilon R=\varepsilon[B\cap A'=\emptyset]`$. As $`B\neq A'`$, this is $`H_{A',B}`$.
- If $`1\notin B`$, the first term is $`0`$. Since $`1\notin A'\cup B`$ and $`|A'\cup B|=2k-t`$, the count is
  $`\binom{n-1-(2k-t)}{k-1}=\binom{N-1+t}{k-1}`$, and $`\varepsilon^2=1`$. $`\square`$

**Proposition 2.3.** Over any field $`\mathbb F_p`$, $`\mathrm{rank}_pH=\alpha`$ if and only if

```math
\binom{N-1+t}{k-1}\equiv\binom{t-1}{k-1}\pmod p\qquad\text{for } t=0,1,\dots,k. \tag{2.1}
```

*Proof.*

- ($`\Leftarrow`$) By Lemma 2.2 and (2.1), every row $`A'\notin\mathcal S`$ equals $`\varepsilon\sum_{A\in\mathcal S,\,A\cap A'=\emptyset}`$
  (row $`A`$). So the rows of $`\mathcal S`$ span the row space, and $`\mathrm{rank}_pH\le\alpha`$. Lemma 2.1 gives equality.
- ($`\Rightarrow`$) If $`\mathrm{rank}_pH=\alpha`$, the independent rows of $`\mathcal S`$ (Lemma 2.1) form a basis of the row space.
  Write row $`A'`$ as $`\sum_{A\in\mathcal S}c_A\,(\text{row }A)`$. Restricting to the columns in $`\mathcal S`$, where
  $`H[\mathcal S,\mathcal S]=I`$, gives $`c_A=H_{A',A}=\varepsilon[A\cap A'=\emptyset]`$. So the identity of Lemma 2.2 holds
  for every $`B`$. Every $`t\in\{0,\dots,k\}`$ occurs as $`|A'\cap B|`$ with $`1\notin A'\cup B`$, because
  $`|A'\cup B|=2k-t\le n-1`$. This gives (2.1). $`\square`$

**Lemma 2.4 (arithmetic).** (2.1) holds if and only if $`q_p(k)\mid N`$.

*Proof.* Let $`g(x)=\binom{x+N}{k-1}-\binom{x}{k-1}`$. By Vandermonde,
$`g(x)=\sum_{i=1}^{k-1}\binom Ni\binom{x}{k-1-i}`$, and (2.1) says $`g(x)\equiv0\pmod p`$ for $`x=-1,0,\dots,k-1`$.

- ($`\Leftarrow`$) Let $`q=p^c\mid N`$. The lowest $`c`$ base-$`p`$ digits of $`N`$ are $`0`$, and every $`0<i<q`$ has a
  nonzero digit among them, so $`\binom Ni\equiv0\pmod p`$ by Lucas' theorem. Since $`k-1<q`$, every term of $`g`$
  vanishes mod $`p`$ for every integer $`x`$.
- ($`\Rightarrow`$) The coefficients of $`g`$ in the basis $`\binom xj`$ are its forward differences at $`0`$:
  $`\binom N{k-1-j}=\Delta^jg(0)=\sum_{i=0}^{j}(-1)^{j-i}\binom ji g(i)`$, for $`0\le j\le k-2`$. These use
  $`g(0),\dots,g(k-2)`$ only, which are $`\equiv0`$. So $`\binom Ni\equiv0\pmod p`$ for $`1\le i\le k-1`$.
  By minimality of $`q=p^c`$, $`p^{a}\le k-1`$ for $`0\le a\le c-1`$. Lucas' theorem gives $`\binom N{p^a}\equiv N_a\pmod p`$,
  where $`N_a`$ is the $`a`$-th base-$`p`$ digit of $`N`$. So $`N_0=\dots=N_{c-1}=0`$, i.e. $`p^c\mid N`$. $`\square`$

*Proof of Theorem 1.* Proposition 2.3 and Lemma 2.4. If $`q\mid N`$, then $`H`$ fits $`K(n,k)`$ with rank $`\alpha`$,
so $`\mathrm{minrk}_{\mathbb F_p}\le\alpha`$; the reverse inequality is fact 1 of §1. $`\blacksquare`$

*Remark.* The proof of Lemma 2.4 uses only $`t=1,\dots,k-1`$ in (2.1); the cases $`t=0`$ and $`t=k`$ then follow.

---

## 3. Proof of Theorem 2

Let $`z_A=\sum_BH_{A,B}x_B=x_A+\varepsilon\sum_{B\cap A=\emptyset}x_B`$, so $`y_A=z_A`$ for $`A\in\mathcal S`$.

- A receiver $`A\in\mathcal S`$ knows every $`x_B`$ with $`B\cap A=\emptyset`$ (its neighbours), so it reads off $`x_A`$.
- For $`A'\notin\mathcal S`$, the row identity proved in Proposition 2.3 gives
  $`z_{A'}=\varepsilon\sum_{A\in\mathcal S,\,A\cap A'=\emptyset}z_A`$ over $`\mathbb F_p`$. The receiver computes this from the
  broadcast, using $`\binom{n-k-1}{k-1}`$ transmissions, and subtracts the known $`\varepsilon\sum_{B\cap A'=\emptyset}x_B`$.

The code has $`|\mathcal S|=\alpha`$ transmissions, and $`\beta(G)\ge\alpha(G)`$ for every index code. $`\blacksquare`$

---

## 4. Which cases are new, and the broadcast rate

**Proposition 4.1 (known: $`k\mid n`$).** If $`k\mid n`$, then $`\mathrm{minrk}_{\mathbb F}(K(n,k))=\alpha`$ over every field.

*Proof.* Baranyai partitions the $`k`$-sets into $`\binom{n-1}{k-1}`$ parallel classes; each is a clique, so
$`\bar\chi=\alpha`$. $`\square`$

**Corollary 4.2.** If $`q_p(k)\mid n-2k`$ for some prime $`p`$, or $`k\mid n`$, then
$`\beta(K(n,k))=\min_p\mathrm{minrk}_{\mathbb F_p}(K(n,k))=\alpha(K(n,k))`$.

**Corollary 4.3 (infinitely many new cases).** For every $`k\ge2`$ there are infinitely many $`n`$ with $`k\nmid n`$ covered
by Theorem 1. For them $`\mathrm{minrk}_{\mathbb F_p}=\alpha<\bar\chi_f\le\bar\chi`$, where $`\bar\chi_f=\binom nk/\lfloor n/k\rfloor`$ is the
fractional clique cover number. So no (fractional) clique-cover code is optimal there.

*Proof.* Take a prime $`p>k`$ (so $`q=p`$, $`\gcd(p,k)=1`$) and $`n=2k+pt`$ with $`t\equiv1\pmod k`$. Then $`n\equiv p\not\equiv0`$.
Cliques of $`K(n,k)`$ have at most $`\lfloor n/k\rfloor`$ vertices and $`K(n,k)`$ is vertex-transitive, so
$`\bar\chi_f=\binom nk/\lfloor n/k\rfloor>\binom nk\cdot\frac kn=\alpha`$ when $`k\nmid n`$. $`\square`$

Examples: $`K(10,3)`$ over $`\mathbb F_2`$ ($`36`$ vs. $`\bar\chi_f=40`$); $`K(13,4)`$ over $`\mathbb F_5`$ ($`220`$ vs. $`238.3`$);
$`K(15,4)`$ over $`\mathbb F_7`$ ($`364`$ vs. $`455`$); $`K(7,2)`$ over $`\mathbb F_3`$ ($`6`$ vs. $`7`$).

**Corollary 4.4 (finitely many exceptions).** For fixed $`k`$, only finitely many $`n>2k`$ are not covered by Corollary 4.2.

*Proof.* If $`N=n-2k`$ is not covered, $`v_p(N)<c_p`$ for every prime $`p`$, where $`p^{c_p}=q_p(k)`$. For $`p\ge k`$,
$`c_p=1`$, so $`p\nmid N`$. Hence $`N\mid\prod_{p<k}p^{c_p-1}`$. $`\square`$

| $`k`$ | exceptional $`n`$ |
|---|---|
| 2 | 5 |
| 3 | 7, 8 |
| 4 | 9, 10, 11, 14 |
| 5 | 11, 12, 13, 14, 16, 22 |
| 6 | 13, 14, 15, 16, 17, 22, 27, 32 |

**Proposition 4.5 (the field matters).** $`\mathrm{minrk}_{\mathbb F_2}(K(7,2))=7`$ while $`\mathrm{minrk}_{\mathbb F_3}(K(7,2))=6`$.

*Proof.* $`\mathbb F_3`$: Theorem 1 ($`q_3(2)=3\mid 7-4`$). $`\mathbb F_2`$: $`7`$ is a clique cover; rank $`\le6`$ is refuted by
SAT, with the unsatisfiability proof checked by DRAT (`minrank/results/unsat_cnf/families/kneser_7_2_r6_indep.cnf.gz`,
`minrank/results/drat_checks.csv`). $`\square`$

So the optimal scalar linear code for $`K(7,2)`$ exists over $`\mathbb F_3`$ but not over $`\mathbb F_2`$.

---

## 5. Remarks

1. **Field dependence.** Theorem 1 concerns one field at a time. For example $`\mathrm{rank}_{\mathbb F_3}H=44>36`$ for
   $`K(10,3)`$, and $`\mathrm{minrk}_{\mathbb F_3}(K(10,3))`$ is not determined here.
2. **Small open cases.** SAT with DRAT-checked certificates gives $`\mathrm{minrk}_{\mathbb F_2}(K(7,3))=18`$ and
   $`\mathrm{minrk}_{\mathbb F_2}(K(8,3))=22`$, above $`\alpha=15,21`$.
3. **The case $`k=2`$.** $`K(n,2)`$ is the complement of the line graph of $`K_n`$. Theorem 1 gives $`\mathrm{minrk}_{\mathbb F_p}=n-1`$
   for odd $`n`$ and odd $`p\mid n-4`$. These should be compared with the literature on that graph before being called new.
4. **Shannon capacity (not used).** By Haemers $`\Theta\le\mathrm{minrk}_{\mathbb F}`$, and by Lovász $`\Theta(K(n,k))=\alpha`$.
   So in the cases of Corollary 4.2 Haemers' bound over some $`\mathbb F_p`$ meets the capacity. Theorem 1 with
   $`\alpha\le\Theta`$ also re-derives $`\Theta(K(n,k))=\alpha`$ for these $`n`$ without the theta function.
5. **Literature check (25 Nov 2026).** A web search found no exact minrank or broadcast-rate values for Kneser graphs.
   Related but different: Golovnev–Haviv and Haviv (upper bounds on minrank of generalized Kneser graphs); Ducey et al.
   (integer diagonal forms, Appendix A); Brouwer–van Eijl and Brouwer–Haemers (p-ranks of $`A-aI`$ for strongly regular and
   related graphs); Hu–Tang, arXiv:2607.06480 (symmetric minimum rank of the *complement* of $`K(n,k)`$ over *infinite*
   fields, a different parameter). Full texts of the last two could not be fetched from this environment and should be read
   before submission.

---

## 6. Verification

Ranks computed with FLINT (`python-flint`) and, where stated, with a separate Gaussian elimination.

| Script | What it checks |
|---|---|
| `minrank/analysis/verify_star_basis.py` | Lemma 2.4 for $`k\le12`$, $`p\le13`$, $`N\le400`$; the row identity of Proposition 2.3 $`\iff`$ $`\mathrm{rank}_pH=\alpha`$ $`\iff`$ $`q\mid n-2k`$ in 318 $`(n,k,p)`$ cases with $`\binom nk\le2000`$ |
| `minrank/analysis/verify_independent.py` | for all $`(n,k,p)`$ with $`q\mid n-2k`$, $`\binom nk\le3500`$, $`k\le6`$: $`H`$ fits, the star is independent of size $`\alpha`$, $`\mathrm{rank}_pH=\alpha`$ by two methods. 82 cases; 43 with $`k\mid n`$, 39 without |
| `minrank/analysis/verify_all_primes.py` | Appendix A: Lemmas A.6–A.8, the block formula against Theorem 1 in 6666 cases ($`k\le12`$, $`n\le120`$, $`p\le13`$), full-matrix predictions such as $`K(17,4)/\mathbb F_3`$ and $`K(18,5)/\mathbb F_2`$ |
| `minrank/analysis/verify_general_k.py` | Appendix A for $`p>k`$; the reduction against direct ranks |
| `minrank/analysis/verify_kneser_proof.py` | Appendix B step by step |

---

## Appendix A: a second proof via integer diagonal forms (Ducey et al.)

Notation: $`\theta_j=(-1)^j\binom{n-k-j}{k-j}`$ ($`0\le j\le k`$) are the eigenvalues of $`D`$; $`\mu_s=\binom ns-\binom n{s-1}`$, $`\mu_{-1}=0`$.

**Theorem A.1** (Ducey, Engelthaler, Gathje, Jones, Pfaff, Plute [DEGJPP, Def. 3.1, Thm 3.4, Lemma 4.10]). Let $`n\ge3k-1`$,
$`\lambda\in\mathbb Z`$ and $`L=D-\lambda I`$. For $`0\le s\le k`$ let $`M_s`$ be the integer matrix indexed by $`s\le i,j\le k`$ with
$`M_s(i,j)=-\lambda\delta_{ij}+\binom{j-s}{i-s}(-1)^i\binom{n-k-j}{k-j}`$. Then $`L`$ has a diagonal form made of diagonal forms of the
$`M_s`$, with $`M_s`$ occurring $`\mu_s-\mu_{s-1}`$ times. Hence $`\mathrm{rank}_pL=\sum_s(\mu_s-\mu_{s-1})\mathrm{rank}_pM_s`$.

Take $`\lambda=-\varepsilon`$, so $`L=\varepsilon H`$. If $`q\mid N`$ then $`n\ge3k`$ and the theorem applies.

**Lemma A.2** (Pascal inversion). With $`T(i,j)=\binom{j-s}{i-s}`$ and $`\Lambda=\mathrm{diag}((-1)^i)`$: $`T\Lambda T=\Lambda`$.

**Lemma A.3.** $`\mathrm{rank}_pM_s=\mathrm{rank}_p\big(T+\mathrm{diag}(\varepsilon\theta_j)\big)`$, since $`\varepsilon TM_s=T+\varepsilon\Lambda\,\mathrm{diag}\binom{n-k-j}{k-j}`$ and $`T`$ is unimodular.

**Lemma A.4.** With $`w=2k-n-1`$: $`\varepsilon\theta_j=-\binom w{k-j}`$. So each block is equivalent to
$`B_s=P_m-\mathrm{diag}\big(\binom w{m-1-v}\big)_{v<m}`$, $`m=k-s+1`$, $`P_m(u,v)=\binom vu`$.

**Lemma A.5** (periodicity). For $`q=p^c`$ and $`0\le r<q`$: $`\binom{x+q}r\equiv\binom xr\pmod p`$. If $`q\mid N`$ then $`w\equiv-1\pmod q`$,
so $`B_s\equiv N_m:=P_m-\mathrm{diag}((-1)^{m-1-v})`$, except possibly the $`(0,0)`$ entry of $`B_0`$ when $`k=q`$.

**Lemma A.6** (char $`\ne2`$). $`N_m`$ is the matrix of $`f(x)\mapsto f(x+1)-\sigma f(-x)`$, $`\sigma=(-1)^{m-1}`$, on polynomials of degree $`<m`$.
Its kernel, after $`g(z)=f(z+\frac12)`$, is the even ($`m`$ odd) or odd ($`m`$ even) polynomials: $`\mathrm{rank}N_m=\lfloor m/2\rfloor`$.

**Lemma A.7** (char $`2`$). $`N_m=P_m+I`$ has kernel $`\mathbb F_2[x^2+x]\cap V_m`$, of dimension $`\lceil m/2\rceil`$: $`\mathrm{rank}N_m=\lfloor m/2\rfloor`$.

**Lemma A.8** (the exceptional entry). If $`k=m-1`$ is a power of $`p`$, changing the $`(0,0)`$ entry of $`N_m`$ does not change its rank (column 1 equals $`e_0`$).

**Conclusion.** $`\mathrm{rank}_pH=\sum_{s=0}^k(\mu_s-\mu_{s-1})\lfloor\frac{k-s+1}2\rfloor=\sum_{j=0}^{k-1}(-1)^j\binom n{k-1-j}=\binom{n-1}{k-1}`$
(summation by parts; the last sum is the coefficient of $`x^{k-1}`$ in $`(1+x)^{n-1}`$).

This proof gives only the "if" direction; the "only if" direction was checked through the block formula in 6666 cases
before the proof of §2 was found.

---

## Appendix B: an elementary proof for $`k=3`$ via inclusion matrices

Index $`\mathbb F^{1+n+\binom n2}`$ by the subsets $`X\subseteq[n]`$ with $`|X|\le2`$, and let $`M_{X,S}=[X\subseteq S]`$ for
3-sets $`S`$. Inclusion–exclusion gives $`[S\cap T=\emptyset]=1-j+\binom j2-\binom j3`$ ($`j=|S\cap T|`$), so
$`H=I+D=M^{\mathsf T}QM`$ with $`Q=\mathrm{diag}(+1,-1,+1)`$ on levels $`0,1,2`$. Then
$`\mathrm{rank}(M^{\mathsf T}QM)=\dim C-(\dim C^\perp-\mathrm{rank}B_Q|_{C^\perp})`$, with $`C`$ the column space of $`M`$.

- Over $`\mathbb F_2`$: $`C^\perp=\{f(\emptyset)=a,\ f(\{i\})=b,\ f(\{i,j\})=a+b+x_i+x_j\}`$ has dimension $`n+1`$, and the form
  $`B=aa'+nbb'+\binom n2cc'+(n-1)(cX'+c'X)+(n-2)P+XX'`$ ($`c=a+b`$, $`X=\sum x_i`$, $`P=\langle x,x'\rangle`$) has rank $`2,3,n+1`$
  for $`n\equiv2,0,`$ odd $`\pmod4`$. Hence $`\mathrm{rank}_2H=\binom{n-1}2,\binom{n-1}2+1,\binom n2`$.
- Over $`\mathbb F_p`$, $`p\ge5`$: by Wilson's rank formula $`\dim C=\binom n2`$; the coefficient of $`P`$ in $`B_Q|_{C^\perp}`$ is
  $`n-6`$, and $`\mathrm{rank}_pH=\alpha`$ iff $`p\mid n-6`$.

---

## References

- **[BBJK]** Z. Bar-Yossef, Y. Birk, T. S. Jayram, T. Kol, *Index coding with side information*, IEEE Trans. Inform.
  Theory 57 (2011) 1479–1494.
- **[BvE]** A. E. Brouwer, C. A. van Eijl, *On the p-rank of the adjacency matrices of strongly regular graphs*, J.
  Algebraic Combin. 1 (1992) 329–346.
- **[DEGJPP]** J. E. Ducey, L. Engelthaler, J. Gathje, B. Jones, I. Pfaff, J. Plute, *Integer diagonal forms for
  subset intersection relations*, arXiv:2310.09227.
- **[EKR]** P. Erdős, C. Ko, R. Rado, *Intersection theorems for systems of finite sets*, Quart. J. Math. 12 (1961).
- **[GH]** A. Golovnev, I. Haviv, *The (generalized) orthogonality dimension of (generalized) Kneser graphs: bounds
  and applications*, Theory of Computing 18 (2022); arXiv:2002.08580.
- **[H]** I. Haviv, *On minrank and the Lovász theta function*, APPROX/RANDOM 2018; arXiv:1802.03920.
- **[Hae]** W. Haemers, *On some problems of Lovász concerning the Shannon capacity of a graph*, IEEE Trans. Inform.
  Theory 25 (1979) 231–232.
- **[HT]** T. Hu, Q. Tang, *Minimum rank of complements of Kneser graphs*, arXiv:2607.06480 (2026).
- **[L]** L. Lovász, *On the Shannon capacity of a graph*, IEEE Trans. Inform. Theory 25 (1979) 1–7.
- **[W]** R. M. Wilson, *A diagonal form for the incidence matrices of t-subsets vs. k-subsets*, European J. Combin. 11
  (1990) 609–615.
- Z. Baranyai, *On the factorization of the complete uniform hypergraph* (1975).
