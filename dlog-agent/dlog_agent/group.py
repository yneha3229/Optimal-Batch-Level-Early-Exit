"""Finite cyclic groups for the discrete-logarithm study, with one interface so the same algorithms run on both.

A Group provides:  op(a, b), inv(a), identity, pow(a, k), eq(a, b), key(a) (a hashable label), order (of the group).
A discrete-log instance is (G, g, h, n) with g of order n and h in <g>; the answer x has g^x = h, 0 <= x < n.

  Fp        the multiplicative group of integers mod a prime p (native-int modular arithmetic)
  EC        an elliptic curve y^2 = x^3 + a x + b over F_p in affine coordinates (point at infinity = None)

Everything is classical textbook material (Cohen, "A Course in Computational Algebraic Number Theory"; Galbraith,
"Mathematics of Public Key Cryptography"); instances are generated at small benchmark sizes for the study.
"""
from __future__ import annotations

from dataclasses import dataclass


class Fp:
    kind = "Fp"

    def __init__(self, p: int, order: int | None = None):
        self.p = p
        self.identity = 1
        self.order = p - 1 if order is None else order

    def op(self, a, b):
        return a * b % self.p

    def inv(self, a):
        return pow(a, -1, self.p)

    def pow(self, a, k):
        return pow(a, k % (self.p - 1) if k < 0 else k, self.p)

    def eq(self, a, b):
        return a == b

    def key(self, a):
        return a


def _inv(a, p):
    return pow(a, -1, p)


class EC:
    """y^2 = x^3 + a x + b over F_p. Points are (x, y) or None (the identity, point at infinity)."""
    kind = "EC"

    def __init__(self, a: int, b: int, p: int, order: int):
        self.a, self.b, self.p, self.order = a % p, b % p, p, order
        self.identity = None

    def op(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        p = self.p
        x1, y1 = P
        x2, y2 = Q
        if x1 == x2 and (y1 + y2) % p == 0:
            return None
        if P == Q:
            if y1 == 0:
                return None
            lam = (3 * x1 * x1 + self.a) * _inv(2 * y1, p) % p
        else:
            lam = (y2 - y1) * _inv((x2 - x1) % p, p) % p
        x3 = (lam * lam - x1 - x2) % p
        y3 = (lam * (x1 - x3) - y1) % p
        return (x3, y3)

    def inv(self, P):
        return None if P is None else (P[0], (-P[1]) % self.p)

    def pow(self, P, k):  # scalar multiplication k*P
        if k < 0:
            P, k = self.inv(P), -k
        R = None
        while k:
            if k & 1:
                R = self.op(R, P)
            P = self.op(P, P)
            k >>= 1
        return R

    def eq(self, P, Q):
        return P == Q

    def key(self, P):
        return P


@dataclass
class Instance:
    name: str
    family: str
    group: object
    g: object
    h: object
    n: int                 # order of g
    bits: int
    interval: int | None = None   # if set, the exponent is known to lie in [0, interval]
    x_true: int | None = None     # for generation/checking only; a solver never reads it


def check(inst: Instance, x) -> bool:
    """x is a valid discrete log iff g^x = h (and, when an interval is given, x lies in it)."""
    if x is None:
        return False
    G = inst.group
    if inst.interval is not None and not (0 <= x <= inst.interval):
        return False
    return G.eq(G.pow(inst.g, x % inst.n), inst.h)
