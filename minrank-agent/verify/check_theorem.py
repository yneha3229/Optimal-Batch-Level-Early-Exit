"""Independent re-check of the Kneser minrank theorem (does not reuse the proof's scripts).

For each (n, k, p):
  T1  rank_p(H) == C(n-1,k-1)  <=>  q_p(k) | n-2k          (H = I + eps*D, eps = (-1)^(k-1)), by direct rank mod p
  T2  when q | n-2k: the explicit code (broadcast y_A for A containing 1) lets EVERY receiver decode, simulated
      with random messages over F_p
  L24 the arithmetic lemma: binom(N-1+t,k-1) == binom(t-1,k-1) mod p for t=0..k  <=>  q | N   (large range)
"""
import itertools, random, sys
from math import comb
import flint

def binom(x, r):                     # generalised binomial, any integer x
    num = 1
    for i in range(r):
        num *= (x - i)
    den = 1
    for i in range(1, r + 1):
        den *= i
    return num // den

def q_of(p, k):
    q = 1
    while q < k:
        q *= p
    return q

PRIMES = [2, 3, 5, 7, 11, 13]

# ---- L24 over a large range (pure arithmetic, no graphs)
bad = 0; cnt = 0
for k in range(2, 13):
    for p in PRIMES:
        q = q_of(p, k)
        for N in range(1, 401):
            lhs = all((binom(N - 1 + t, k - 1) - binom(t - 1, k - 1)) % p == 0 for t in range(0, k + 1))
            cnt += 1
            bad += lhs != (N % q == 0)
print(f"L24 arithmetic lemma: {cnt} (k,p,N) cases, mismatches {bad}")

# ---- T1 and T2 on actual graphs
def kneser(n, k):
    V = list(itertools.combinations(range(n), k)); S = [frozenset(v) for v in V]
    return V, S

t1 = t1bad = t2 = t2bad = 0
rng = random.Random(7)
for k in range(2, 6):
    for n in range(2 * k + 1, 40):
        if comb(n, k) > 1300:
            continue
        V, S = kneser(n, k); m = len(V); eps = (-1) ** (k - 1)
        alpha = comb(n - 1, k - 1)
        for p in PRIMES:
            ent = []
            for i in range(m):
                for j in range(m):
                    ent.append(1 if i == j else (eps % p if S[i].isdisjoint(S[j]) else 0))
            r = flint.nmod_mat(m, m, ent, p).rank()
            pred = (n - 2 * k) % q_of(p, k) == 0
            t1 += 1; t1bad += (r == alpha) != pred
            if pred:                                   # T2: simulate the explicit code
                x = [rng.randrange(p) for _ in range(m)]
                star = [i for i in range(m) if 0 in S[i]]
                y = {i: (x[i] + eps * sum(x[j] for j in range(m) if S[i].isdisjoint(S[j]))) % p for i in star}
                ok = True
                for a in range(m):
                    if 0 in S[a]:
                        known = sum(x[j] for j in range(m) if S[a].isdisjoint(S[j]))      # side information
                        dec = (y[a] - eps * known) % p
                    else:
                        z = eps * sum(y[i] for i in star if S[i].isdisjoint(S[a])) % p
                        known = sum(x[j] for j in range(m) if S[a].isdisjoint(S[j]))
                        dec = (z - eps * known) % p
                    ok &= dec == x[a]
                t2 += 1; t2bad += not ok
                assert len(star) == alpha
print(f"T1 rank iff condition: {t1} (n,k,p) cases, mismatches {t1bad}")
print(f"T2 explicit code decodes for every receiver: {t2} cases, failures {t2bad}")
