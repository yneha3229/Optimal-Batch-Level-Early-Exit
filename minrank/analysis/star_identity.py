"""Check of the star identity for Kneser graphs (docs/MINRANK_THEORY.md section 6).

Lemma. Let M fit K(n,k) over F_2 (ones on the diagonal, zeros on intersecting pairs) with rank_2(M) = alpha =
C(n-1, k-1), n >= 2k + 1. For every point a, the star S_a = {T : a in T} is an independent set of size alpha, so the
rows of M indexed by S_a are a basis of the row space and M = M[:, S_a] M[S_a, :]. Summing over a: n M = k M^2.
Hence: k even, n odd is impossible (M = 0), so minrk_2(K(n,k)) >= alpha + 1; for k odd, M^2 = M (n odd) or
M^2 = 0 (n even).

This script verifies the identity M = M[:, S_a] M[S_a, :] for every a, and n M = k M^2, on every stored rank-alpha
certificate (and on I + D where its rank is alpha), and checks that I + D never has rank alpha for k even, n odd.
Run: python -m minrank.analysis.star_identity
"""
from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np

from ..bench.generators import kneser
from ..core.f2 import rank_f2
from ..core.verify import verify
from ..baselines.identity_adj import identity_plus_adjacency_rows

ROOT = Path(__file__).resolve().parents[1]


def to_mat(rows: list[int], n: int) -> np.ndarray:
    return np.array([[(r >> j) & 1 for j in range(n)] for r in rows], np.uint8)


def matmul2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (A.astype(np.int64) @ B.astype(np.int64) % 2).astype(np.uint8)


def check(m: int, k: int, rows: list[int], label: str) -> dict:
    g = kneser(m, k)
    rk = verify(g, rows)
    M = to_mat(rows, g.n)
    sets = list(combinations(range(m), k))
    alpha = comb(m - 1, k - 1)
    out = {"graph": g.name, "source": label, "rank": rk, "alpha": alpha}
    if rk != alpha:
        return out
    star_ok = True
    for a in range(m):
        S = [i for i, T in enumerate(sets) if a in T]
        star_ok &= bool((matmul2(M[:, S], M[S, :]) == M).all())
    M2 = matmul2(M, M)
    out |= {"star_identity": star_ok, "nM_eq_kM2": bool(((m * M) % 2 == (k * M2) % 2).all()),
            "M2_zero": bool((M2 == 0).all()), "M2_eq_M": bool((M2 == M).all())}
    return out


def main() -> None:
    res = []
    for f in sorted((ROOT / "certificates").glob("kneser_*.json")):
        m, k = map(int, f.stem.split("_")[1:3])
        if m < 2 * k + 1:
            continue
        c = json.loads(f.read_text())
        res.append(check(m, k, [int(r, 16) if isinstance(r, str) else int(r) for r in c["rows"]], "certificate"))
    for m, k in [(6, 2), (8, 2), (10, 2), (12, 2), (10, 3), (14, 3), (18, 3), (12, 4), (16, 4), (9, 3), (12, 3)]:
        g = kneser(m, k)
        res.append(check(m, k, identity_plus_adjacency_rows(g), "I+D"))
    # k even, n odd: I + D never reaches alpha (consistent with the lemma)
    odd = []
    for m, k in [(5, 2), (7, 2), (9, 2), (11, 2), (13, 2), (15, 2), (9, 4), (11, 4), (13, 4)]:
        g = kneser(m, k)
        odd.append({"graph": g.name, "alpha": comb(m - 1, k - 1), "rank_I+D": verify(g, identity_plus_adjacency_rows(g))})
    for r in res:
        print(r)
    for r in odd:
        print(r)
    full = [r for r in res if r["rank"] == r["alpha"]]
    assert all(r["star_identity"] and r["nM_eq_kM2"] for r in full)
    assert all(r["rank_I+D"] > r["alpha"] for r in odd)
    out = ROOT / "results/star_identity.json"
    out.write_text(json.dumps({"rank_alpha_checks": res, "k_even_n_odd": odd}, indent=1))
    print(f"{len(full)} rank-alpha matrices satisfy the identity; wrote {out}")


if __name__ == "__main__":
    main()
