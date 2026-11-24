# The $`p`$-rank of the Golovnev–Haviv matrix and exact minrank of Kneser graphs

## Summary

For integers $`n > 2k \ge 4`$, let $`K(n,k)`$ be the Kneser graph and $`D`$ its adjacency matrix. The integer matrix

```math
H \;=\; I + (-1)^{k-1} D, \qquad H_{A,B} = \binom{|A\cap B|-1}{k-1},
```

fits $`K(n,k)`$ over every field. It is the matrix used by Golovnev–Haviv and by Haviv for upper bounds on minrank.
We prove:

> **Main Theorem.** Let $`p`$ be a prime and let $`q`$ be the smallest power of $`p`$ with $`q \ge k`$.
> If $`q \mid n-2k`$, then
> ```math
> \mathrm{rank}_{\mathbb F_p} H \;=\; \binom{n-1}{k-1} \;=\; \alpha(K(n,k)),
> ```
> and therefore $`\mathrm{minrk}_{\mathbb F_p}(K(n,k)) = \alpha(K(n,k))`$.

> **Corollary.** For every $`k \ge 2`$ and all but finitely many $`n`$, the Haemers bound is tight on $`K(n,k)`$:
> $`\min_{\mathbb F}\mathrm{minrk}_{\mathbb F}(K(n,k)) = \alpha(K(n,k)) = \Theta(K(n,k))`$.

---

## 1. Definitions and notation

**Minrank.** Let $`G`$ be a graph on vertex set $`V`$ and let $`\mathbb F`$ be a field. A matrix $`M\in\mathbb F^{V\times V}`$
*fits* $`G`$ if

- $`M_{vv}\neq 0`$ for all $`v`$, and
- $`M_{uv}=0`$ whenever $`u\neq v`$ are non-adjacent.

The minrank $`\mathrm{minrk}_{\mathbb F}(G)`$ is the minimum rank of a matrix that fits $`G`$.

**Standard facts.**

1. $`\alpha(G)\le \mathrm{minrk}_{\mathbb F}(G)`$. The principal submatrix on an independent set is diagonal
   with nonzero diagonal.
2. $`\mathrm{minrk}_{\mathbb F}(G)\le \bar\chi(G)`$, the clique cover number.
3. (Haemers) $`\Theta(G)\le \mathrm{minrk}_{\mathbb F}(G)`$, where $`\Theta`$ is the Shannon capacity.

**Kneser graphs.** The vertices of $`K(n,k)`$ are the $`k`$-subsets of $`[n]`$, and $`A\sim B`$ iff $`A\cap B=\emptyset`$.
By Erdős–Ko–Rado and Lovász, for $`n\ge 2k`$,

```math
\alpha(K(n,k)) = \Theta(K(n,k)) = \binom{n-1}{k-1}.
```

**The matrix $`H`$.** Put $`\varepsilon = (-1)^{k-1}`$ and $`H = I + \varepsilon D`$. Then $`H_{AA}=1`$ and $`H_{AB}=0`$ when
$`0<|A\cap B|<k`$, so $`H`$ fits $`K(n,k)`$ over every field.

Moreover $`H_{A,B} = q(|A\cap B|)`$ with $`q(x)=\binom{x-1}{k-1}`$:

- $`q(0)=\binom{-1}{k-1}=(-1)^{k-1}`$;
- $`q(1)=\dots=q(k-1)=0`$;
- $`q(k)=1`$.

This is the matrix of Golovnev–Haviv [GH, Thm 3.2] (and, up to the scalar $`(k-1)!`$, of Haviv [H, Prop 4.3]). Those
papers bound its rank from above by $`\sum_{i<k}\binom ni`$, respectively $`\binom n{k-1}`$ for $`p>k`$.

**Conventions.**

- $`\binom{x}{r} = x(x-1)\cdots(x-r+1)/r!`$ for any integer $`x`$ and $`r\ge 0`$; $`\binom{x}{r}=0`$ for $`r<0`$.
- $`\mathrm{rank}_p`$ means rank over $`\mathbb F_p`$.
- $`\theta_j = (-1)^j\binom{n-k-j}{k-j}`$ for $`0\le j\le k`$ (these are the eigenvalues of $`D`$).
- $`\mu_s = \binom ns - \binom n{s-1}`$ for $`s \ge 0`$, and $`\mu_{-1}=0`$.

---

## 2. A cited reduction: Ducey et al.

A *diagonal form* of an integer matrix $`L`$ is a diagonal matrix $`\Delta = ELF`$ with $`E,F`$ unimodular over
$`\mathbb Z`$. Reducing mod $`p`$ keeps $`E,F`$ invertible, so

```math
\mathrm{rank}_p L = \#\{\text{diagonal entries of }\Delta\text{ not divisible by }p\}. \tag{2.1}
```

**Theorem 2.1** (Ducey, Engelthaler, Gathje, Jones, Pfaff, Plute [DEGJPP, Def. 3.1, Thm 3.4, Lemma 4.10]).
Let $`n \ge 3k-1`$, $`\lambda\in\mathbb Z`$, and $`L = D-\lambda I`$ for the Kneser graph $`K(n,k)`$. For $`0\le s\le k`$ let
$`M_s`$ be the $`(k-s+1)\times(k-s+1)`$ integer matrix with rows and columns indexed by $`s\le i,j\le k`$ and

```math
M_s(i,j) = -\lambda\,\delta_{ij} + \binom{j-s}{i-s}(-1)^i\binom{n-k-j}{k-j}.
```

Then $`L`$ has a diagonal form made of diagonal forms of the $`M_s`$, where $`M_s`$ occurs
$`\binom ns-2\binom n{s-1}+\binom n{s-2} = \mu_s-\mu_{s-1}`$ times.

**Corollary 2.2.** For $`n\ge 3k-1`$ and every prime $`p`$:

```math
\mathrm{rank}_p L = \sum_{s=0}^{k} (\mu_s-\mu_{s-1})\,\mathrm{rank}_p M_s.
```

*Proof.* Combine Theorem 2.1 with (2.1), applied to $`L`$ and to each $`M_s`$. $`\square`$

We apply this with $`\lambda = -\varepsilon`$. Then $`D-\lambda I = D+\varepsilon I = \varepsilon H`$, so
$`\mathrm{rank}_p H = \mathrm{rank}_p L`$.

---

## 3. Simplifying the blocks

Fix $`s`$ and write $`T`$ for the upper-triangular Pascal matrix $`T(i,j)=\binom{j-s}{i-s}`$, and
$`\Lambda = \mathrm{diag}((-1)^i)`$, both indexed by $`s\le i,j\le k`$.

**Lemma 3.1** (Pascal inversion). $`T^{-1} = \Lambda T\Lambda`$. Consequently $`T\Lambda T = \Lambda`$.

*Proof.* Put $`u=i-s`$, $`v=j-s`$. The inverse of $`\big(\binom vu\big)_{u,v}`$ is
$`\big((-1)^{u+v}\binom vu\big)_{u,v}`$, because $`\sum_{u\le w\le v}(-1)^{w+v}\binom wu\binom vw = \delta_{uv}`$.
Also $`(-1)^{u+v}=(-1)^{i+j}`$. Multiplying $`T^{-1}=\Lambda T\Lambda`$ on the left by $`T`$ and on the right by $`\Lambda`$
gives $`\Lambda = T\Lambda T`$. $`\square`$

**Lemma 3.2.** With $`\lambda=-\varepsilon`$:
$`\mathrm{rank}_p M_s = \mathrm{rank}_p\big(T + \mathrm{diag}(\varepsilon\theta_s,\dots,\varepsilon\theta_k)\big)`$.

*Proof.*

- Write $`a_j = \binom{n-k-j}{k-j}`$ and $`\mathcal A = \mathrm{diag}(a_j)`$. Then
  $`M_s = \varepsilon I + \Lambda T\mathcal A`$.
- By Lemma 3.1, $`TM_s = \varepsilon T + T\Lambda T \mathcal A = \varepsilon T + \Lambda\mathcal A`$.
- Multiply by $`\varepsilon=\pm1`$: $`\varepsilon T M_s = T + \varepsilon\Lambda\mathcal A = T + \mathrm{diag}(\varepsilon\theta_j)`$.
- $`T`$ is unimodular. $`\square`$

**Lemma 3.3.** Let $`w = 2k-n-1`$. Then $`\varepsilon\theta_j = -\binom{w}{k-j}`$ for $`0\le j\le k`$.

*Proof.*

- For any integer $`x`$ and $`r\ge0`$: $`(-1)^r\binom xr=\binom{r-x-1}{r}`$.
- Put $`r=k-j`$ and $`x = n-k-j`$. Then $`r-x-1 = w`$.
- Hence $`\varepsilon\theta_j = (-1)^{k-1+j}\binom{n-k-j}{k-j} = -(-1)^{r}\binom{x}{r} = -\binom{w}{r}`$. $`\square`$

So, re-indexing by $`v = j-s\in\{0,\dots,m-1\}`$ with $`m=k-s+1`$ and $`r = k-j = m-1-v`$, the block is (up to unimodular
equivalence)

```math
B_s \;=\; P_m - \mathrm{diag}\Big(\tbinom{w}{m-1-v}\Big)_{v=0}^{m-1},
\qquad P_m(u,v)=\tbinom vu. \tag{3.1}
```

**Lemma 3.4** (periodicity). Let $`q=p^c`$ and $`0\le r<q`$. Then $`\binom{x+q}{r}\equiv\binom xr \pmod p`$ for every
integer $`x`$. In particular, if $`x\equiv-1\pmod q`$ then $`\binom xr\equiv\binom{-1}{r}=(-1)^r\pmod p`$.

*Proof.*

- Vandermonde's identity holds as a polynomial identity in $`x`$:
  $`\binom{x+q}{r}=\sum_{i=0}^{r}\binom qi\binom{x}{r-i}`$.
- For $`0<i\le r<q`$, $`p\mid\binom{q}{i}`$ (Kummer; $`q`$ is a power of $`p`$).
- Each $`\binom{x}{r-i}`$ is an integer.
- Iterate the congruence in both directions. $`\square`$

**Corollary 3.5.** Assume $`q\mid n-2k`$, where $`q`$ is the least power of $`p`$ with $`q\ge k`$. Then $`w\equiv-1\pmod q`$,
and modulo $`p`$:

```math
B_s \equiv N_m := P_m - \mathrm{diag}\big((-1)^{m-1-v}\big)_{v=0}^{m-1}
```

with **one possible exception**. If $`k=q`$ (i.e. $`k`$ is itself a power of $`p`$), the entry in block $`s=0`$ at position
$`(0,0)`$ (where $`r=k`$) may differ from that of $`N_{k+1}`$.

*Proof.* $`w = 2k-n-1\equiv -1\pmod q`$. All $`r`$ in (3.1) satisfy $`r\le k\le q`$, and $`r<q`$ unless $`r=k=q`$. Since
$`r=k`$ means $`j=0`$, this happens only for $`s=0`$, $`v=0`$. Apply Lemma 3.4. $`\square`$

---

## 4. The rank of $`N_m`$

Let $`\mathbb F`$ be a field and $`V_m`$ the space of polynomials in $`\mathbb F[x]`$ of degree $`<m`$, with basis
$`x^0,\dots,x^{m-1}`$. Let $`(Ef)(x) = f(x+1)`$ and $`(Rf)(x)=f(-x)`$. Then

- column $`v`$ of $`P_m`$ holds the coefficients of $`(x+1)^v=\sum_u\binom vu x^u`$, so $`P_m`$ is the matrix of $`E`$;
- $`\mathrm{diag}((-1)^{m-1-v}) = \sigma\mathrm{diag}((-1)^v)`$ with $`\sigma=(-1)^{m-1}`$, which is the
  matrix of $`\sigma R`$.

Hence $`N_m`$ is the matrix of $`E-\sigma R`$ on $`V_m`$.

**Lemma 4.1** ($`\mathrm{char}\mathbb F\ne2`$). $`\mathrm{rank} N_m = \lfloor m/2\rfloor`$.

*Proof.*

- $`f\in\ker(E-\sigma R)`$ iff $`f(x+1)=\sigma f(-x)`$ for all $`x`$, i.e. iff $`f(u)=\sigma f(1-u)`$.
- Since $`2`$ is invertible, put $`g(z) = f(z+\tfrac12)`$, which preserves degree. The condition becomes
  $`g(z)=\sigma g(-z)`$.
- If $`m`$ is odd, then $`\sigma=1`$ and $`g`$ is even of degree $`\le m-1`$: the span of $`z^0,z^2,\dots,z^{m-1}`$, of
  dimension $`\tfrac{m+1}{2}`$.
- If $`m`$ is even, then $`\sigma=-1`$ and $`g`$ is odd: the span of $`z^1,z^3,\dots,z^{m-1}`$, of dimension $`\tfrac m2`$.
- Either way $`\dim\ker = \lceil m/2\rceil`$, so the rank is $`\lfloor m/2\rfloor`$. $`\square`$

**Lemma 4.2** ($`\mathrm{char}\mathbb F = 2`$). $`\mathrm{rank} N_m = \lfloor m/2\rfloor`$.

*Proof.*

- In characteristic 2, $`N_m = P_m + I`$ is the matrix of $`f\mapsto f(x+1)+f(x)`$, so
  $`\ker = \{f\in V_m: f(x+1)=f(x)\}`$.
- Let $`y = x^2+x`$. Then $`x`$ is a root of the monic polynomial $`X^2+X+y`$ over $`\mathbb F[y]`$, so
  $`\mathbb F[x]=\mathbb F[y]\oplus x\,\mathbb F[y]`$.
- The substitution $`x\mapsto x+1`$ fixes $`y`$. Writing $`f=a(y)+x\,b(y)`$, we get $`f(x+1)+f(x)=b(y)`$.
- So $`\ker = \mathbb F[y]\cap V_m`$. Since $`\deg_x a(y)=2\deg_y a`$, it consists of the $`a(y)`$ with
  $`\deg_y a\le\lfloor (m-1)/2\rfloor`$, of dimension $`\lceil m/2\rceil`$. $`\square`$

**Lemma 4.3** (the exceptional entry). Let $`m\ge 3`$, let $`k=m-1`$ be a power of $`p`$, and let $`N'`$ be obtained from
$`N_m`$ (over $`\mathbb F_p`$) by replacing the $`(0,0)`$ entry by an arbitrary $`c\in\mathbb F_p`$. Then
$`\mathrm{rank}N' = \mathrm{rank}N_m`$.

*Proof.*

- In both $`N_m`$ and $`N'`$, column $`0`$ is a multiple of $`e_0`$, the coefficient vector of the constant $`1`$.
- Column 1 is the image of $`x`$: $`(x+1)-\sigma(-x) = 1+(1+\sigma)x`$ with $`\sigma=(-1)^{m-1}=(-1)^k`$.
- If $`p=2`$, then $`1+\sigma=0`$ in $`\mathbb F_2`$. If $`p`$ is odd, then $`k=p^c`$ is odd, $`\sigma=-1`$, and again
  $`1+\sigma=0`$. So column 1 equals $`e_0`$.
- Therefore column 0 lies in the span of column 1 in both matrices, and both column spaces equal the span of
  columns $`1,\dots,m-1`$, which the two matrices share. $`\square`$

**Corollary 4.4.** Under the hypothesis of Corollary 3.5, $`\mathrm{rank}_p B_s = \lfloor (k-s+1)/2\rfloor`$ for
every $`0\le s\le k`$.

*Proof.* Use Lemma 4.1 or 4.2 according to $`p`$, and Lemma 4.3 for the exceptional entry. That entry occurs only for
$`s=0`$, where $`m=k+1\ge3`$. $`\square`$

---

## 5. Proof of the Main Theorem

Assume $`q\mid n-2k`$, where $`q`$ is the least power of $`p`$ with $`q\ge k`$, and $`n>2k`$.

**Step 1: the reduction applies.** $`n-2k`$ is a positive multiple of $`q`$, so $`n-2k\ge q\ge k`$ and $`n\ge 3k\ge3k-1`$.
By Corollary 2.2 (with $`\lambda=-\varepsilon`$), Lemma 3.2 and Corollary 4.4:

```math
\mathrm{rank}_p H=\sum_{s=0}^{k}(\mu_s-\mu_{s-1})\,g(s),\qquad g(s)=\Big\lfloor\frac{k-s+1}{2}\Big\rfloor .
```

**Step 2: summation by parts.** Since $`g(k)=0`$,

```math
\sum_{s=0}^{k}(\mu_s-\mu_{s-1})g(s)=\sum_{s=0}^{k-1}\mu_s\big(g(s)-g(s+1)\big).
```

With $`t=k-s\ge1`$, $`g(s)-g(s+1)=\lfloor\tfrac{t+1}2\rfloor-\lfloor\tfrac t2\rfloor`$, which is $`1`$ if $`t`$ is odd and
$`0`$ if $`t`$ is even. Hence

```math
\mathrm{rank}_p H=\sum_{\substack{0\le s\le k-1\\ k-s\text{ odd}}}\Big(\tbinom ns-\tbinom n{s-1}\Big)
=\sum_{j=0}^{k-1}(-1)^j\binom{n}{k-1-j}=\binom{n-1}{k-1}.
```

The last equality is the coefficient of $`x^{k-1}`$ in $`(1+x)^n(1+x)^{-1}=(1+x)^{n-1}`$.

**Step 3: conclusion.**

- $`H`$ fits $`K(n,k)`$, so $`\mathrm{minrk}_{\mathbb F_p}\le\binom{n-1}{k-1}`$.
- $`\alpha(K(n,k))=\binom{n-1}{k-1}\le\mathrm{minrk}_{\mathbb F_p}`$.

Hence equality holds. $`\blacksquare`$

---

## 6. Corollaries

**Corollary 6.1** (Haemers bound). If $`q_p(k)\mid n-2k`$ for some prime $`p`$ (where $`q_p(k)`$ is the least power of
$`p`$ that is $`\ge k`$), or $`k\mid n`$, then

```math
\min_{\mathbb F}\mathrm{minrk}_{\mathbb F}(K(n,k))=\alpha(K(n,k))=\Theta(K(n,k)).
```

*Proof.*

- The first case is the Main Theorem.
- If $`k\mid n`$, Baranyai's theorem partitions the $`k`$-subsets of $`[n]`$ into $`\binom{n-1}{k-1}`$ parallel classes.
  Each class is a clique of $`K(n,k)`$, so $`\bar\chi=\alpha`$.
- Combine with facts 1–3 of §1. $`\square`$

**Corollary 6.2** (finitely many exceptions). For fixed $`k`$, the set of $`n>2k`$ not covered by Corollary 6.1 is
finite.

*Proof.* Suppose $`N=n-2k`$ is not covered. For every prime $`p`$, $`v_p(N)<c_p`$, where $`p^{c_p}=q_p(k)`$. For $`p\ge k`$,
$`c_p=1`$, so $`p\nmid N`$. Hence $`N`$ divides $`\prod_{p<k}p^{\,c_p-1}`$. $`\square`$

The exceptional $`n`$ for small $`k`$ are:

| $`k`$ | exceptional $`n`$ |
|---|---|
| 2 | 5 |
| 3 | 7, 8 |
| 4 | 9, 10, 11, 14 |
| 5 | 11, 12, 13, 14, 16, 22 |
| 6 | 13, 14, 15, 16, 17, 22, 27, 32 |

**Corollary 6.3** (strictly below the clique cover). Each clique of $`K(n,k)`$ has at most $`\lfloor n/k\rfloor`$
vertices, so $`\bar\chi(K(n,k))\ge\binom nk/\lfloor n/k\rfloor`$. This exceeds
$`\alpha=\binom nk\cdot\tfrac kn`$ exactly when $`k\nmid n`$.

So whenever the Main Theorem applies and $`k\nmid n`$, the linear code from $`H`$ is strictly shorter than every
clique-cover code. Examples:

- $`K(10,3)`$ over $`\mathbb F_2`$: $`36<40`$;
- $`K(13,4)`$ over $`\mathbb F_5`$: $`220<239`$;
- $`K(15,4)`$ over $`\mathbb F_7`$: $`364<455`$.

**Index coding.** Since $`\beta(G)\ge\alpha(G)`$ for the broadcast rate, the scalar linear index code over
$`\mathbb F_p`$ given by $`H`$ is optimal among *all* index codes for $`K(n,k)`$ whenever the Main Theorem applies.

---

## 7. Remarks

1. **Sharpness for $`H`$ (computational).** For $`k\le12`$, $`3k-1\le n\le120`$ and $`p\le13`$ (6666 cases), Corollary 2.2
   gives $`\mathrm{rank}_p H=\alpha`$ **if and only if** $`q\mid n-2k`$. The "only if" direction is not proven
   here. It also concerns only $`H`$: other fitting matrices might reach $`\alpha`$.
2. **Small open cases.** For $`(n,k)=(7,3)`$ and $`(8,3)`$, SAT with DRAT-checked certificates gives
   $`\mathrm{minrk}_{\mathbb F_2}=18`$ and $`22`$, above $`\alpha=15,21`$. Other fields are open.
3. **Dependence on [DEGJPP].** Only Theorem 2.1 is imported. Sections 3–5 are self-contained. For $`k=3`$, the
   Appendix gives an independent elementary proof.

---

## 8. Verification

All statements were checked numerically with ranks computed by FLINT (`python-flint`), independently of the proof:

| Script | What it checks |
|---|---|
| `minrank/analysis/verify_kneser_proof.py` | the $`k=3`$ appendix step by step: Gram identity, $`\mathrm{rank}_p M`$, explicit bases of $`C^\perp`$, closed-form Gram matrices, the rank reduction, the iff; 128 $`(n,p)`$ cases; Corollary 6.1 arithmetic for $`k=3`$ up to $`n=10^6`$; agreement with Theorem 2.1 |
| `minrank/analysis/verify_general_k.py` | Lemma 4.1 ($`m\le40`$, $`p\le31`$); Corollary 2.2 against direct ranks ($`k=2..5`$); the theorem for $`p>k`$ on 56 full matrices and 439 block cases |
| `minrank/analysis/verify_all_primes.py` | Lemma 4.2 ($`m\le64`$); Lemma 4.3; Lemma 3.4; the iff in 6666 cases; new full-matrix predictions, e.g. $`K(17,4)`$ over $`\mathbb F_3`$ $`=560=\alpha`$ and $`K(18,5)`$ over $`\mathbb F_2`$ $`=2380=\alpha`$ |

---

## Appendix: an elementary proof for $`k=3`$ (no use of Theorem 2.1)

Let $`k=3`$. Index $`\mathbb F^N`$, $`N=1+n+\binom n2`$, by the subsets $`X\subseteq[n]`$ with $`|X|\le2`$. Let $`M`$ be the
$`N\times\binom n3`$ matrix $`M_{X,S}=[X\subseteq S]`$ and let $`C`$ be its column space. Inclusion–exclusion gives the
integer identity

```math
[S\cap T=\emptyset] = 1-j+\tbinom j2-\tbinom j3\quad(j=|S\cap T|),
\qquad\text{so}\qquad H=I+D=M^{\mathsf T}QM,
```

with $`Q=\mathrm{diag}(+1,-1,+1)`$ on levels $`|X|=0,1,2`$. Mod 2, $`Q=I`$.

**Reduction.** For $`Q`$ nondegenerate, the map $`y\mapsto Qy`$ shows

```math
\mathrm{rank}(M^{\mathsf T}QM)=\dim C-\big(\dim C^\perp-\mathrm{rank} B_Q|_{C^\perp}\big),
\qquad B_Q(g,g')=g^{\mathsf T}Qg'.
```

**$`\mathbb F_2`$, Lemma A.1.** $`C^\perp=\{f: f(\emptyset)=a,\ f(\{i\})=b,\ f(\{i,j\})=a+b+x_i+x_j\}`$, and
$`\dim C^\perp=n+1`$.

*Proof.*

- The condition $`f \perp`$ (column $`S`$) says $`\delta g=h`$, where $`g`$ is $`f`$ on pairs viewed as a 1-cochain on the
  full simplex, and $`h(S)=f(\emptyset)+\sum_{i\in S}f(i)`$.
- Then $`\delta h=0`$, i.e. $`\sum_{i\in Q'}f(i)=0`$ for every 4-set $`Q'`$. For $`n\ge5`$ this forces $`f(i)\equiv b`$, so $`h`$
  is the constant $`a+b`$.
- Since $`H^1`$ of the simplex vanishes, the solutions are $`g=(a+b)+\delta x`$. Here $`x`$ is defined modulo the
  all-ones vector. $`\square`$

**$`\mathbb F_2`$, Lemma A.2.** With $`c=a+b`$, $`X=\sum x_i`$ and $`P=\langle x,x'\rangle`$, the form $`B`$ on $`C^\perp`$ is

```math
B=aa'+n\,bb'+\tbinom n2cc'+(n-1)(cX'+c'X)+(n-2)P+XX'.
```

It has rank $`2`$ if $`n\equiv2`$, rank $`3`$ if $`n\equiv0`$, and rank $`n+1`$ if $`n`$ is odd $`(\bmod 4)`$.

Hence $`\mathrm{rank}_2 H=\binom{n-1}{2}`$, $`\binom{n-1}2+1`$ or $`\binom n2`$ respectively.

**$`\mathbb F_p`$, $`p\ge5`$.**

- Wilson's rank formula [W] gives $`\mathrm{rank}_pW_{2,3}=\binom n2`$.
- Summing rows of $`W_2`$ gives $`3W_0`$ and $`2W_1`$, so $`\dim C=\binom n2`$.
- $`C^\perp`$ is parametrised by $`g(\emptyset)=-3(t+c)`$, $`g(\{i\})=t-2x_i`$, $`g(\{i,j\})=c+x_i+x_j`$, modulo
  $`(c,t,x)\sim(c-2s,t+2s,x+s\mathbf 1)`$.
- Then
  ```math
  B_Q=9(t+c)(t'+c')-\big[n\,tt'-2(tX'+t'X)+4P\big]+\tbinom n2cc'+(n-1)(cX'+c'X)+(n-2)P+XX'.
  ```
- The coefficient of $`P`$ is $`n-6`$. If $`p\mid n-6`$, then $`B_Q`$ factors through the 3-dimensional space of
  $`(c,t,X)`$, which contains the gauge direction in its radical. So $`\mathrm{rank}B_Q\le2`$ and
  $`\mathrm{rank}_pH\le\binom n2-(n-1)=\alpha`$.
- Conversely, if $`p\nmid n-6`$, the restriction of $`B_Q`$ to $`\{c=t=0,\ X=0\}`$ is $`(n-6)P`$, of rank $`\ge n-2`$, so
  $`\mathrm{rank}_pH>\alpha`$.

This reproves the Main Theorem for $`k=3`$ (where $`q=4`$ for $`p=2`$, and $`q=p`$ for $`p\ge5`$).

---

## References

- **[DEGJPP]** J. E. Ducey, L. Engelthaler, J. Gathje, B. Jones, I. Pfaff, J. Plute, *Integer diagonal forms for
  subset intersection relations*, arXiv:2310.09227 (Algebraic Combinatorics).
- **[GH]** A. Golovnev, I. Haviv, *The (generalized) orthogonality dimension of (generalized) Kneser graphs: bounds
  and applications*, arXiv:2002.08580; Theory of Computing 18 (2022).
- **[H]** I. Haviv, *On minrank and the Lovász theta function*, APPROX/RANDOM 2018, arXiv:1802.03920.
- **[Hae]** W. Haemers, *On some problems of Lovász concerning the Shannon capacity of a graph*, IEEE Trans. Inform.
  Theory 25 (1979) 231–232.
- **[L]** L. Lovász, *On the Shannon capacity of a graph*, IEEE Trans. Inform. Theory 25 (1979) 1–7.
- **[W]** R. M. Wilson, *A diagonal form for the incidence matrices of $`t`$-subsets vs. $`k`$-subsets*, European J.
  Combin. 11 (1990) 609–615.
- Baranyai's theorem; Erdős–Ko–Rado theorem.
