"""Benchmark primes: random primes and special forms (where n-1 or n+1 factors easily).

    python prime-agent/make_numbers.py  -> numbers.json   [{"name", "family", "digits", "n"}]
"""
import json
import math
import random
from pathlib import Path

import cypari2

pari = cypari2.Pari()
pari.allocatemem(2 * 10**9, silent=True)
rng = random.Random(2026)
out = []


def add(family, name, n):
    n = int(n)
    assert pari.ispseudoprime(n), name
    out.append({"name": name, "family": family, "digits": len(str(n)), "n": str(n)})


DIGITS = (50, 100, 200, 300, 400, 500)
for d in DIGITS:                                   # random primes
    for k in range(4):
        add("random", f"rand{d}_{k}", pari.nextprime(rng.randrange(10 ** (d - 1), 10**d)))
for d in DIGITS:                                   # Proth k*2^m+1 and Riesel k*2^m-1
    m = int(d / 0.30103) - 8
    for fam, sign in (("proth", 1), ("riesel", -1)):
        found, k = 0, rng.randrange(101, 999) | 1
        while found < 2:
            n = k * 2**m + sign
            if pari.ispseudoprime(n):
                add(fam, f"{fam}{d}_{k}", n)
                found += 1
            k += 2
for d in (50, 100, 150, 200):                      # safe primes 2q+1 (q prime)
    for k in range(2):
        q = rng.randrange(10 ** (d - 1), 10**d) | 1
        while not (pari.ispseudoprime(q) and pari.ispseudoprime(2 * q + 1)):
            q += 2
        add("safe", f"safe{d}_{k}", 2 * q + 1)
for f in range(12, 330):                           # factorial primes f! +- 1 (searched)
    for sgn in (1, -1):
        n = math.factorial(f) + sgn
        if pari.ispseudoprime(n):
            add("factorial", f"{f}!{'+' if sgn > 0 else '-'}1", n)
P = 1
for p in pari("primes(400)"):                      # primorial primes p# +- 1 (searched)
    P *= int(p)
    for sgn in (1, -1):
        if P > 10**20 and pari.ispseudoprime(P + sgn):
            add("primorial", f"{int(p)}#{'+' if sgn > 0 else '-'}1", P + sgn)
Path(__file__).with_name("numbers.json").write_text(json.dumps(out, indent=0))
print(len(out), "numbers;", {f: sum(o["family"] == f for o in out) for f in sorted({o["family"] for o in out})})
