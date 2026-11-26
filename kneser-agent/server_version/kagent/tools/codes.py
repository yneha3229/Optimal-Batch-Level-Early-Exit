"""Code-rule library. Every rule builds a fitting matrix M (nonzero diagonal, M_uv = 0 when u != v are not adjacent);
the code transmits M[S, :] x for a row basis S and has length rank_F(M) (scalar linear) or rank/t (vector linear).

clique    the clique-cover code (block matrix of all-ones blocks): length = number of cliques over every field.
spectral  M = I + c A over F_p. For an integer eigenvalue th with th != 0 mod p, c = -1/th makes
          rank_p(I + cA) = n - nullity_p(A - th I). Eigenvalues congruent to th mod p can merge into that kernel, so
              rank_pred(p, th) = n - sum{ mult(mu) : mu = th (mod p) }
          orders the (p, th) pairs. The prediction is only a heuristic for the order; the exact rank decides.
          This is minrank-agent's spectral_multifield idea, generalised from a fixed field list to every prime <= n
          and to an explicit per-(p, th) ranking.
blowup    vector-linear: a scalar code of rank r on the t-fold blow-up G^(t) (lambda-agent SAT), rate r/t;
          n * t <= 150 only.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from ..rank import SpectralMatrix, rank_factor

ROOT = Path(__file__).resolve().parents[4]


def primes_upto(m: int) -> list[int]:
    if m < 2:
        return []
    s = np.ones(m + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(m ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return np.nonzero(s)[0].tolist()


def integer_spectrum(spec: dict) -> list[tuple[int, int]]:
    """[(eigenvalue, multiplicity)] for the integer eigenvalues of a describe() spectrum (exact or estimated)."""
    out = []
    key = "mult" if spec.get("multiplicities_exact") else "mult_est"
    for e in spec.get("eigenvalues_all", spec["eigenvalues"]):
        if isinstance(e["value"], int):
            out.append((e["value"], int(e[key])))
    return out


def spectral_candidates(n: int, spectrum: list[tuple[int, int]], primes: list[int] | None = None,
                        lower: int = 0) -> list[dict]:
    """All (p, th) pairs with th != 0 mod p, ranked by predicted rank. Predictions below the certified lower bound
    are impossible as stated (rank >= beta >= lower), which means eigenvalues merged mod p without their eigenspaces
    all joining the kernel; such pairs are ranked after the plausible ones."""
    primes = primes if primes is not None else primes_upto(max(2, n))
    out = []
    for p in primes:
        for th, _ in spectrum:
            if th % p == 0:
                continue
            nul = sum(m for mu, m in spectrum if (mu - th) % p == 0)
            pred = n - nul
            c = (-pow(th % p, -1, p)) % p
            out.append({"p": p, "theta": th, "c": c, "rank_pred": pred, "plausible": pred >= lower})
    out.sort(key=lambda d: (not d["plausible"], d["rank_pred"] if d["plausible"] else lower - d["rank_pred"],
                            d["p"]))
    return out


def spectral_code(A: np.ndarray, p: int, c: int, cap: int | None, seed: int = 0, stop=None) -> dict:
    t0 = time.perf_counter()
    res = rank_factor(SpectralMatrix(A, p, c), cap=cap, seed=seed, stop=stop)
    return {"p": p, "c": c, "rank": res.rank, "S": res.S, "C": res.C, "seconds": round(time.perf_counter() - t0, 3),
            "rounds": res.rounds}


def clique_code_matrix(n: int, cliques: list[list[int]]) -> np.ndarray:
    """The fitting matrix of the clique-cover code (small n only; used by tests)."""
    M = np.zeros((n, n), dtype=np.int64)
    for C in cliques:
        M[np.ix_(C, C)] = 1
    return M


# ------------------------------------------------------------------------------------------------ blowup (SAT)
def blowup_code(A: np.ndarray, t: int, r: int, time_limit: float, fix: list[int] | None = None):
    """Rank-<= r fitting rows (bitmasks over F_2) on G^(t), False if none exists, None on timeout or size."""
    n = A.shape[0]
    if n * t > 150:
        return None
    sys.path.insert(0, str(ROOT / "lambda-agent"))
    from lambda_agent.graph import Digraph, blowup
    from lambda_agent.upper import lift_fix, minrank_sat

    i, j = np.nonzero(A)
    g = Digraph.from_arcs(n, list(zip(i.tolist(), j.tolist())), "g")
    h = blowup(g, t)
    return minrank_sat(h, r, time_limit, lift_fix(fix, t) if fix else None)


def blowup_adjacency(A: np.ndarray, t: int) -> np.ndarray:
    """Adjacency of G^(t) (sub-receiver i*t+a knows j*t+b iff i ~ j), with the block diagonal off (i == j, a != b
    are not side information)."""
    return np.kron(A.astype(np.int8), np.ones((t, t), dtype=np.int8)).astype(bool)


def rows_to_matrix(rows: list[int], N: int) -> np.ndarray:
    M = np.zeros((N, N), dtype=np.int64)
    for i, m in enumerate(rows):
        for j in range(N):
            if (m >> j) & 1:
                M[i, j] = 1
    return M
