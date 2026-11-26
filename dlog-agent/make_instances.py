"""Generate the discrete-log benchmark: the answer is stored only for checking, never read by a solver.

Structure families (the whole point of the study: different structure makes different methods the right one):
  generic           g has large prime order -> only sqrt(n) methods or (in F_p*) index calculus can work
  smooth            the order of g is a product of small primes -> Pohlig-Hellman is near-instant at any size
  partial_smooth    order = (smooth part) x (one medium prime) -> PH reduces to one medium sqrt(.) subgroup
  interval          large order, but the exponent is known to lie in a short interval -> kangaroo wins
  index_calculus    F_p* only: large prime-order subgroup where sub-exponential index calculus beats sqrt(n)

Both groups: Fp (integers mod a prime) and EC (elliptic curve over F_p). Sizes are small benchmark sizes.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cypari2

pari = cypari2.Pari()
pari.allocatemem(1 << 31, silent=True)
rng = random.Random(2026)
HERE = Path(__file__).resolve().parent
out = []


def smooth(bits, maxprime):
    """A product of random primes below maxprime, of about `bits` bits."""
    S, primes = 1, []
    while S.bit_length() < bits:
        q = int(pari.prime(rng.randrange(2, maxprime)))
        S *= q
        primes.append(q)
    return S, primes


def fp_from_order(S):
    """A prime p with g of order exactly S in F_p*: find small t with p = S*t + 1 prime, g = r^((p-1)/S)."""
    t = 2
    while True:
        p = S * t + 1
        if pari.isprime(p):
            r = int(pari.znprimroot(p))
            g = pow(r, (p - 1) // S, p)
            return int(p), g
        t += 1


def add(name, family, G, g, h, n, bits, interval=None, x=None):
    from dlog_agent.group import Instance, check

    inst = Instance(name, family, G, g, h, n, bits, interval, x)
    assert check(inst, x), name
    d = {"name": name, "family": family, "bits": bits, "kind": G.kind, "n": str(n),
         "g": json.dumps(g), "h": json.dumps(h), "interval": None if interval is None else str(interval)}
    if G.kind == "Fp":
        d["p"] = str(G.p)
    else:
        d["a"], d["b"], d["p"] = str(G.a), str(G.b), str(G.p)
    out.append(d)


def gen_fp():
    from dlog_agent.group import Fp

    for bits, k in ((40, 2), (48, 2), (56, 2)):            # generic prime order (safe prime subgroup)
        for i in range(k):
            q = int(pari.nextprime(rng.getrandbits(bits)))
            p, g = fp_from_order(q)
            G = Fp(p, q)
            x = rng.randrange(q)
            add(f"fp_generic_{bits}_{i}", "generic", G, g, G.pow(g, x), q, bits, x=x)
    for bits in (120, 200, 400):                           # smooth order, large p
        for i in range(2):
            S, _ = smooth(bits, 50000)
            p, g = fp_from_order(S)
            G = Fp(p, S)
            x = rng.randrange(S)
            add(f"fp_smooth_{bits}_{i}", "smooth", G, g, G.pow(g, x), S, bits, x=x)
    for bits, med in ((120, 40), (160, 44), (200, 48)):    # partial smooth: smooth x one medium prime
        for i in range(2):
            S, _ = smooth(bits - med, 50000)
            Q = int(pari.nextprime(rng.getrandbits(med)))
            n = S * Q
            p, g = fp_from_order(n)
            G = Fp(p, n)
            x = rng.randrange(n)
            add(f"fp_partial_{bits}_{i}", "partial_smooth", G, g, G.pow(g, x), n, bits, x=x)
    for bits, ib in ((120, 40), (200, 44), (400, 48)):     # interval-restricted exponent
        for i in range(2):
            q = int(pari.nextprime(rng.getrandbits(bits)))
            p, g = fp_from_order(q)
            G = Fp(p, q)
            B = rng.randrange(1 << (ib - 1), 1 << ib)
            x = rng.randrange(B)
            add(f"fp_interval_{bits}_{i}", "interval", G, g, G.pow(g, x), q, bits, interval=B, x=x)
    for bits in (60, 70, 80):                              # index calculus beats sqrt(n) in F_p*
        for i in range(2):
            q = int(pari.nextprime(rng.getrandbits(bits)))
            p, g = fp_from_order(q)
            G = Fp(p, q)
            x = rng.randrange(q)
            add(f"fp_indexcalc_{bits}_{i}", "index_calculus", G, g, G.pow(g, x), q, bits, x=x)


def random_curve(pbits):
    p = int(pari.nextprime(rng.getrandbits(pbits)))
    while True:
        a, b = rng.randrange(p), rng.randrange(p)
        if (4 * a**3 + 27 * b**2) % p:
            E = pari.ellinit([a, b], p)
            N = int(pari.ellcard(E))
            return a, b, p, E, N


def ec_point(E, N, want_order):
    """A generator of a cyclic subgroup of order `want_order` (assumes the group is cyclic of order N)."""
    while True:
        P = pari.ellmul(E, E.random(), N // want_order)
        if len(P) == 2 and int(pari.ellorder(E, P)) == want_order:   # len 1 ([0]) is the point at infinity
            return (int(pari.lift(P[0])), int(pari.lift(P[1])))


def classify(N):
    fa = pari.factor(N)
    f = [(int(fa[0][i]), int(fa[1][i])) for i in range(len(fa[0]))]
    biggest = max(q for q, _ in f)
    return f, biggest, biggest.bit_length()


def ec_base(E):
    """A random point and its exact order (the DL lives in the cyclic group it generates)."""
    P = E.random()
    while len(P) != 2:
        P = E.random()
    return (int(pari.lift(P[0])), int(pari.lift(P[1]))), int(pari.ellorder(E, P))


def gen_ec():
    from dlog_agent.group import EC

    made = {"generic": 0, "smooth": 0, "partial_smooth": 0, "interval": 0}
    target = {"generic": 6, "smooth": 4, "partial_smooth": 4, "interval": 4}
    tries = 0
    while sum(made.values()) < sum(target.values()) and tries < 60000:
        tries += 1
        pbits = rng.choice([36, 40, 44, 48])
        a, b, p, E, N = random_curve(pbits)
        g, n = ec_base(E)
        if n < 1 << 20:
            continue
        f, biggest, bb = classify(n)
        nb = n.bit_length()
        G = EC(a, b, p, n)                       # the instance order is the base point's order
        prime_order = len(f) == 1 and f[0][1] == 1
        if prime_order and made["generic"] < target["generic"]:
            x = rng.randrange(n)
            add(f"ec_generic_{pbits}_{made['generic']}", "generic", G, g, G.pow(g, x), n, pbits, x=x)
            made["generic"] += 1
        elif bb <= 17 and made["smooth"] < target["smooth"]:
            x = rng.randrange(n)
            add(f"ec_smooth_{pbits}_{made['smooth']}", "smooth", G, g, G.pow(g, x), n, pbits, x=x)
            made["smooth"] += 1
        elif 18 <= bb <= 30 and bb <= nb - 6 and made["partial_smooth"] < target["partial_smooth"]:
            x = rng.randrange(n)
            add(f"ec_partial_{pbits}_{made['partial_smooth']}", "partial_smooth", G, g, G.pow(g, x), n, pbits, x=x)
            made["partial_smooth"] += 1
        elif prime_order and made["interval"] < target["interval"]:
            B = rng.randrange(1 << 23, 1 << 24)
            x = rng.randrange(B)
            add(f"ec_interval_{pbits}_{made['interval']}", "interval", G, g, G.pow(g, x), n, pbits, interval=B, x=x)
            made["interval"] += 1
        if tries % 1000 == 0:
            print("  ec tries", tries, made, flush=True)
    return made


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE))
    gen_fp()
    made = gen_ec()
    (HERE / "instances.json").write_text(json.dumps(out, indent=0))
    fam = {}
    for o in out:
        fam[(o["kind"], o["family"])] = fam.get((o["kind"], o["family"]), 0) + 1
    print(len(out), "instances")
    for k in sorted(fam):
        print(" ", k, fam[k])
