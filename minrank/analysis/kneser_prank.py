"""Ranks of I +- D (D = Kneser adjacency) over prime fields, and the checks behind docs/MINRANK_THEORY.md section 6.

For each (k, n, p): r = rank_p(I + (-1)^(k+1) D) (over F_2 the sign does not matter) against alpha = C(n-1, k-1),
the clique-cover bound ceil(C(n,k) / floor(n/k)), and the divisibility prediction
    rank_p = alpha  <=>  p | n - 2k  (p odd)   or   4 | n - 2k  (p = 2),
which is a theorem for k = 3 and p != 3 (MINRANK_THEORY.md Theorem 2) and a conjecture otherwise.
With --gram it also checks the two lemmas of the k = 3 proof over F_2: rank_2 M = C(n,2) for the stacked inclusion
matrix M = [W_0; W_1; W_2], and dim(C cap C^perp) = rank M - rank(I + D).

Usage:
  python -m minrank.analysis.kneser_prank --k 3 --n 7-22 --p 2 3 5 7 11 13 17 19 --out minrank/results/kneser_prank.csv
  python -m minrank.analysis.kneser_prank --k 3 --n 7-16 --p 2 --gram
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from math import comb
from pathlib import Path

import numpy as np


def rank_mod_p(a: np.ndarray, p: int) -> int:
    """Rank over F_p by row reduction (int64; entries stay in [0, p))."""
    a = (a % p).astype(np.int64)
    rows, cols = a.shape
    r = 0
    for c in range(cols):
        nz = np.nonzero(a[r:, c])[0]
        if len(nz) == 0:
            continue
        piv = r + nz[0]
        if piv != r:
            a[[r, piv]] = a[[piv, r]]
        a[r] = (a[r] * pow(int(a[r, c]), p - 2, p)) % p
        col = a[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if len(nzr):
            a[nzr] = (a[nzr] - np.outer(col[nzr], a[r])) % p
        r += 1
        if r == rows:
            break
    return r


def kneser_adjacency(n: int, k: int) -> np.ndarray:
    x = np.zeros((comb(n, k), n), dtype=np.int64)
    for i, s in enumerate(itertools.combinations(range(n), k)):
        x[i, list(s)] = 1
    return (x @ x.T == 0).astype(np.int64)


def stacked_inclusion(n: int, k: int) -> np.ndarray:
    """M = [W_0; W_1; ...; W_{k-1}]: rows = subsets A of size < k, columns = k-sets S, entry [A subset of S]."""
    cols = [frozenset(s) for s in itertools.combinations(range(n), k)]
    rows = [frozenset(a) for t in range(k) for a in itertools.combinations(range(n), t)]
    return np.array([[int(a <= s) for s in cols] for a in rows], dtype=np.int64)


def predicted(n: int, k: int, p: int) -> bool:
    return (n - 2 * k) % 4 == 0 if p == 2 else (n - 2 * k) % p == 0


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, nargs="+", default=[3])
    ap.add_argument("--n", default="7-16", help="e.g. 7-22 or 7,9,11")
    ap.add_argument("--p", type=int, nargs="+", default=[2, 3, 5, 7])
    ap.add_argument("--gram", action="store_true", help="also check the k = 3 lemmas over F_2")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    fields = ["k", "n", "p", "rank", "alpha", "cover_lb", "hit", "predicted", "seconds"]
    w = None
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        f = a.out.open("w", newline="")
        w = csv.DictWriter(f, fields)
        w.writeheader()
    for k in a.k:
        for n in parse_range(a.n):
            if n < 2 * k + 1:
                continue
            d = kneser_adjacency(n, k)
            eye = np.eye(d.shape[0], dtype=np.int64)
            alpha, cover = comb(n - 1, k - 1), -(-comb(n, k) // (n // k))
            for p in a.p:
                t = time.perf_counter()
                r = rank_mod_p(eye + (1 if k % 2 else -1) * d, p)
                row = {"k": k, "n": n, "p": p, "rank": r, "alpha": alpha, "cover_lb": cover, "hit": r == alpha,
                       "predicted": predicted(n, k, p), "seconds": round(time.perf_counter() - t, 2)}
                flag = "" if row["hit"] == row["predicted"] else "  <-- differs from prediction"
                print(" ".join(f"{key}={row[key]}" for key in fields[:-1]) + flag, flush=True)
                if w:
                    w.writerow(row)
            if a.gram and k == 3:
                m = stacked_inclusion(n, 3)
                g = (m.T @ m) % 2
                assert np.array_equal(g, (eye + d) % 2), "I + D != M^T M mod 2"
                rm, rg = rank_mod_p(m, 2), rank_mod_p(g, 2)
                rad = rm - rg
                exp_rad = {1: 0, 3: 0, 2: n - 1, 0: n - 2}[n % 4]
                print(f"  gram n={n}: rank M={rm} (C(n,2)={comb(n, 2)}), radical={rad} (lemma: {exp_rad})", flush=True)
                assert rm == comb(n, 2) and rad == exp_rad
    if w:
        f.close()


if __name__ == "__main__":
    main()
