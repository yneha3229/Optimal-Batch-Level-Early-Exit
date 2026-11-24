"""Exact test of minrk_2(mu(G)) = |V(G)| through the shadow-basis reduction (docs/MINRANK_THEORY.md section 7).

mu(G) is the Mycielskian: originals V, shadows U = {u_y}, apex w; u_y ~ x iff y ~ x in G, w ~ every u_y. U is
independent of size |V|, which is alpha(mu(G)) when |V| >= 2 alpha(G) (true for the Mycielski graphs M_k).

Lemma. minrk_2(mu(G)) = |V| iff there are A, B in F_2^{V x V}, both supported on the (ordered) edges of G, and
x, z in F_2^V such that
  (i)   A B^T fits G: (A B^T)_vv = 1, (A B^T)_vv' = 0 for v != v' non-adjacent in G;
  (ii)  A z = 0 and B x = 0;
  (iii) <x, z> = 1.
Proof: if rank M = |U| for M fitting mu(G), the rows of U are a basis (M[U,U] = I), so M = M[:,U] M[U,:]. Put
A = M[V,U], B = M[U,V]^T, x = M[w,U], z = M[U,w]; the fitting conditions of M on V u {w} are (i)-(iii), and the
zeros of M between U and V are the supports. Conversely M := [A; I; x^T] [B^T | I | z] (rows/cols V, U, w) fits.
The SAT instance has no rank variable and only 2|E(G)| + 2|V| free bits, so it is much smaller than the rank
encoding of mu(G).

Usage: python -m minrank.exact.mycielski_reduction --k 3 4 5 6 7 --time 3600
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


from ..bench.generators import mycielski
from ..core.graph import Graph, iter_bits

ROOT = Path(__file__).resolve().parents[1]


class CNF:
    def __init__(self) -> None:
        self.top, self.cl = 0, []

    def var(self) -> int:
        self.top += 1
        return self.top

    def and2(self, a: int, b: int) -> int:
        z = self.var()
        self.cl += [[-z, a], [-z, b], [z, -a, -b]]
        return z

    def xor_eq(self, lits: list[int], parity: int) -> None:
        """XOR of lits == parity (Tseitin chain)."""
        if not lits:
            if parity:
                self.cl.append([])
            return
        acc = lits[0]
        for b in lits[1:]:
            x = self.var()
            self.cl += [[-x, acc, b], [-x, -acc, -b], [x, -acc, b], [x, acc, -b]]
            acc = x
        self.cl.append([acc] if parity else [-acc])


def encode(g: Graph) -> tuple[CNF, dict, dict, list, list]:
    n = g.n
    nb = [list(iter_bits(g.out[v])) for v in range(n)]
    c = CNF()
    A = {(v, y): c.var() for v in range(n) for y in nb[v]}
    B = {(v, y): c.var() for v in range(n) for y in nb[v]}
    x = [c.var() for _ in range(n)]
    z = [c.var() for _ in range(n)]
    for v in range(n):
        for v2 in range(n):
            if v != v2 and (g.out[v] >> v2) & 1:
                continue                                    # adjacent: entry free
            common = [y for y in nb[v] if (g.out[v2] >> y) & 1]
            c.xor_eq([c.and2(A[v, y], B[v2, y]) for y in common], 1 if v == v2 else 0)       # (i)
        c.xor_eq([c.and2(A[v, y], z[y]) for y in nb[v]], 0)                                    # (ii) A z = 0
        c.xor_eq([c.and2(B[v, y], x[y]) for y in nb[v]], 0)                                    # (ii) B x = 0
    c.xor_eq([c.and2(x[y], z[y]) for y in range(n)], 1)                                         # (iii)
    return c, A, B, x, z


def build_matrix(g: Graph, model: set, A, B, x, z) -> list[int]:
    """Rows (bitmasks) of the fitting matrix of mu(G) from a model: vertices 0..n-1 originals, n..2n-1 shadows, 2n apex."""
    n = g.n
    a = [[1 if A.get((v, y)) in model else 0 for y in range(n)] for v in range(n)]
    b = [[1 if B.get((v, y)) in model else 0 for y in range(n)] for v in range(n)]
    xs = [1 if x[y] in model else 0 for y in range(n)]
    zs = [1 if z[y] in model else 0 for y in range(n)]
    left = a + [[int(i == j) for j in range(n)] for i in range(n)] + [xs]            # [2n+1, n]
    right = b + [[int(i == j) for j in range(n)] for i in range(n)] + [zs]           # columns' vectors [2n+1, n]
    rows = []
    for r in left:
        m = 0
        for col, cv in enumerate(right):
            if sum(p & q for p, q in zip(r, cv)) & 1:
                m |= 1 << col
        rows.append(m)
    return rows


def _clauses(k: int) -> list[list[int]]:
    """Module-level CNF builder for solve_timed (pickled to a child process)."""
    return encode(mycielski(k - 1))[0].cl


def decide(k: int, limit: float, solver: str = "cadical195") -> dict:
    from .symmetry import solve_timed
    g = mycielski(k - 1)                 # mu(M_{k-1}) = M_k
    c, A, B, x, z = encode(g)            # deterministic numbering: the child builds the same CNF
    t = time.perf_counter()
    res = solve_timed(_clauses, (k,), limit, solver)       # model, False (UNSAT) or None (timeout); child is killed
    out = {"graph": f"mycielski_{k}", "n": 2 * g.n + 1, "alpha": g.n, "vars": c.top, "clauses": len(c.cl),
           "result": "timeout" if res is None else ("UNSAT: minrk >= alpha + 1" if res is False else "SAT: minrk = alpha"),
           "seconds": round(time.perf_counter() - t, 2), "solver": solver, "time_limit": limit}
    if res not in (None, False):
        from ..core.verify import verify
        model = set(l for l in res if l > 0)
        out["verified_rank"] = verify(mycielski(k), build_matrix(g, model, A, B, x, z))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, nargs="+", default=[3, 4, 5, 6, 7])
    ap.add_argument("--time", type=float, default=3600.0)
    ap.add_argument("--solver", default="cadical195")
    a = ap.parse_args()
    res = []
    for k in a.k:
        r = decide(k, a.time, a.solver)
        print(r, flush=True)
        res.append(r)
    out = ROOT / "results/mycielski_reduction.json"
    old = json.loads(out.read_text()) if out.exists() else []
    out.write_text(json.dumps(old + res, indent=1))


if __name__ == "__main__":
    main()
