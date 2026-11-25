"""AKS (Agrawal-Kayal-Saxena 2002) on FLINT polynomial arithmetic: deterministic, polynomial time, impractical.

aks(n) -> (is_prime, info). Steps: perfect power test; smallest r with ord_r(n) > (log2 n)^2; gcd checks up to r;
then for a = 1 .. floor(sqrt(phi(r)) log2 n):  (X + a)^n == X^(n mod r) + a   mod (X^r - 1, n).
"""
from __future__ import annotations

import math
import time

import flint


def _ord_gt(n, r, bound):
    """Is the multiplicative order of n mod r larger than bound?"""
    x = 1
    for _ in range(int(bound)):
        x = x * n % r
        if x == 1:
            return False
    return True


def aks(n: int):
    t0 = time.perf_counter()
    if n < 2:
        return False, {}
    if flint.fmpz(n).is_perfect_power():
        return False, {"step": "perfect power"}
    L = math.log2(n)
    r = 2
    while True:
        if math.gcd(r, n) == 1 and _ord_gt(n % r, r, L * L):
            break
        r += 1
    for a in range(2, min(r, n - 1) + 1):
        if 1 < math.gcd(a, n) < n:
            return False, {"step": "gcd", "r": r}
    if n <= r:
        return True, {"r": r}
    phi = int(flint.fmpz(r).euler_phi())
    A = int(math.isqrt(phi) * L)
    ctx = flint.fmpz_mod_poly_ctx(n)
    mod = ctx([-1] + [0] * (r - 1) + [1])          # X^r - 1
    X = ctx([0, 1])
    rhs_x = X.pow_mod(n % r, mod) if n % r else ctx([1])
    for a in range(1, A + 1):
        lhs = (X + a).pow_mod(n, mod)
        if lhs != rhs_x + a:
            return False, {"step": f"poly a={a}", "r": r}
    return True, {"r": r, "a_checks": A, "seconds": round(time.perf_counter() - t0, 3)}


if __name__ == "__main__":
    import sys

    import cypari2

    pari = cypari2.Pari()
    for d in map(int, sys.argv[1:] or ["8", "12", "16", "20"]):
        p = int(pari.nextprime(10 ** (d - 1) + 12345))
        t = time.perf_counter()
        ok, info = aks(p)
        ta = time.perf_counter() - t
        t = time.perf_counter()
        ok2 = bool(pari.isprime(p))
        tp = time.perf_counter() - t
        print(f"{d:3d} digits: AKS {ok} in {ta:8.2f} s (r = {info.get('r')}, {info.get('a_checks')} polynomial checks) | "
              f"PARI isprime {ok2} in {tp * 1000:.2f} ms | ratio {ta / max(tp, 1e-6):,.0f}x", flush=True)
