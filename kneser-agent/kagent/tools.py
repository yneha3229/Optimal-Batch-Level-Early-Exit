"""Solver-side tools. Each returns plain data; bounds come with certificates.

Lower bound on beta : an independent set S (beta >= alpha >= |S|).
Upper bounds        : a clique cover (classic code), and linear codes M = I + cA over F_p (code length = rank_p M).
Information         : the graph summary for the agents, and the Hoffman bound (an upper bound on alpha).
"""
from __future__ import annotations

import time

import numpy as np

PRIMES = [p for p in range(2, 400) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


# ------------------------------------------------------------------------------------------- summary
def describe(A: np.ndarray, eig_limit: int = 4000, seed: int = 0) -> dict:
    t0 = time.perf_counter()
    N = A.shape[0]
    deg = A.sum(1)
    rng = np.random.default_rng(seed)
    out = {"receivers": int(N), "edges": int(deg.sum() // 2), "degree_min": int(deg.min()),
           "degree_max": int(deg.max()), "regular": bool(deg.min() == deg.max())}
    # common-neighbour counts on sampled adjacent / non-adjacent pairs
    Af = A.astype(np.float32)
    samp = rng.choice(N, size=min(N, 200), replace=False)
    C = (Af[samp] @ Af.T).astype(int)
    adj_cn, non_cn = set(), set()
    for a, v in enumerate(samp):
        row, cn = A[v], C[a]
        js = rng.choice(N, size=min(N, 50), replace=False)
        for j in js:
            if j == v:
                continue
            (adj_cn if row[j] else non_cn).add(int(cn[j]))
    out["common_neighbours_adjacent"] = sorted(adj_cn)[:12]
    out["common_neighbours_nonadjacent"] = sorted(non_cn)[:12]
    out["triangles_through_sampled_vertex"] = int(np.einsum("ij,ij->i", C[:20], Af[samp[:20]]).max() // 2)
    if N <= eig_limit:
        ev = np.linalg.eigvalsh(Af.astype(np.float64))
        r = np.round(ev)
        integral = bool(np.abs(ev - r).max() < 1e-6)
        vals, mult = np.unique(r if integral else np.round(ev, 4), return_counts=True)
        order = np.argsort(-vals)
        out["spectrum_integral"] = integral
        out["eigenvalues"] = [[float(vals[i]) if not integral else int(vals[i]), int(mult[i])] for i in order][:14]
        out["distinct_eigenvalues"] = int(len(vals))
    else:
        out["spectrum"] = f"not computed (N > {eig_limit})"
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out


def hoffman_alpha_bound(summary: dict):
    """alpha <= N * (-lmin) / (d - lmin) for a d-regular graph (needs the spectrum)."""
    if not summary.get("regular") or "eigenvalues" not in summary:
        return None
    lmin = min(v for v, _ in summary["eigenvalues"]) if summary.get("distinct_eigenvalues", 99) <= 14 else None
    if lmin is None:
        return None
    d, N = summary["degree_max"], summary["receivers"]
    return float(N * (-lmin) / (d - lmin))


# ------------------------------------------------------------------------------------------- lower bound
def independent_set(A: np.ndarray, restarts: int = 1, seed: int = 0) -> list[int]:
    """Greedy: repeatedly take the candidate keeping the most candidates (min-degree greedy on the remainder)."""
    N = A.shape[0]
    nonadj = ~A
    np.fill_diagonal(nonadj, False)
    best: list[int] = []
    rng = np.random.default_rng(seed)
    for r in range(restarts):
        cand = np.ones(N, dtype=bool)
        score = nonadj.sum(1).astype(np.int64)
        S = []
        while cand.any():
            idx = np.nonzero(cand)[0]
            s = score[idx].astype(float) + (rng.random(len(idx)) * 0.5 if r else 0)
            v = int(idx[np.argmax(s)])
            S.append(v)
            removed = cand & ~nonadj[v]
            cand &= nonadj[v]
            removed[v] = True
            if removed.any():
                score -= nonadj[:, removed].sum(1)
        if len(S) > len(best):
            best = S
    return best


# ------------------------------------------------------------------------------------------- upper bounds
def clique_cover(A: np.ndarray, seed: int = 0) -> list[list[int]]:
    """Greedy clique cover (= greedy colouring of the complement), largest-degree-first order."""
    N = A.shape[0]
    order = np.argsort(-A.sum(1), kind="stable")
    classes = np.zeros((0, N), dtype=bool)
    members: list[list[int]] = []
    for v in order:
        if len(members):
            ok = ~(classes & ~A[v]).any(1)          # class is a clique together with v
            hit = np.nonzero(ok)[0]
        else:
            hit = []
        if len(hit):
            c = int(hit[0])
        else:
            classes = np.vstack([classes, np.zeros((1, N), dtype=bool)])
            members.append([])
            c = len(members) - 1
        classes[c, v] = True
        members[c].append(int(v))
    return members


def fitting_matrix(A: np.ndarray, p: int, c: int) -> np.ndarray:
    M = (c * A.astype(np.int64)) % p
    np.fill_diagonal(M, 1)
    return M


def rank_mod(M: np.ndarray, p: int) -> int:
    import flint
    N = M.shape[0]
    return flint.nmod_mat(N, N, M.reshape(-1).tolist(), p).rank()


def factor_check(A: np.ndarray, p: int, c: int, S, block: int = 512) -> bool:
    """Is M = M[:,S] @ M[S,:] (mod p) for M = I + cA?  If S is independent (M[S,S] = I) this holds iff
    rank_p(M) = |S|. Exact, by blocked float64 matrix products (entries < p, sums < 2^53)."""
    S = np.asarray(S)
    R = (c * A[S].astype(np.int64)) % p                         # rows S of M
    R[np.arange(len(S)), S] = 1
    R = R.astype(np.float64)
    N = A.shape[0]
    for s in range(0, N, block):
        Mb = (c * A[s:s + block].astype(np.int64)) % p
        idx = np.arange(s, min(N, s + block))
        Mb[idx - s, idx] = 1
        Cb = Mb[:, S].astype(np.float64)
        if not np.array_equal(np.mod(Cb @ R, p).astype(np.int64), Mb):
            return False
    return True


def spectral_candidates(summary: dict, max_p: int | None = None, top: int = 40) -> list[dict]:
    """(p, theta) pairs for M = I + cA with c = -1/theta (mod p), ordered by predicted rank
    N - sum{mult(mu) : mu = theta mod p}. Generic: uses only the integer spectrum."""
    if not summary.get("spectrum_integral"):
        return []
    N = summary["receivers"]
    ev = summary["eigenvalues"]
    if summary.get("distinct_eigenvalues", 0) > len(ev):
        return []
    max_p = max_p or max(3, N)
    out = []
    for p in PRIMES:
        if p > max_p:
            break
        classes: dict[int, int] = {}
        for mu, m in ev:
            classes[mu % p] = classes.get(mu % p, 0) + m
        for res, tot in classes.items():
            if res == 0:
                continue                                    # theta = 0 mod p has no inverse
            theta = next(mu for mu, _ in ev if mu % p == res)
            c = (-pow(theta % p, -1, p)) % p
            out.append({"p": p, "theta": int(theta), "c": int(c), "rank_pred": int(N - tot)})
    out.sort(key=lambda d: (d["rank_pred"], d["p"]))
    return out[:top]


def family_rule(n: int, k: int):
    """The structural rule for K(n,k) (our theorem, used as one rule of the library): the smallest prime p such
    that q | n-2k, q = least power of p >= k; code I + (-1)^(k-1) A over F_p."""
    for p in PRIMES:
        q = 1
        while q < k:
            q *= p
        if n - 2 * k > 0 and (n - 2 * k) % q == 0:
            return {"p": p, "c": (1 if k % 2 == 1 else -1) % p}
    return None


def try_code(A: np.ndarray, p: int, c: int, S, rank_limit: int = 4000) -> dict:
    """Code M = I + cA over F_p. If rank = |S| (S independent) the bracket closes; checked by factor_check.
    Otherwise the exact rank (flint) when N <= rank_limit."""
    t0 = time.perf_counter()
    N = A.shape[0]
    out = {"p": p, "c": c}
    if S is not None and factor_check(A, p, c, S):
        out.update(rank=len(S), meets_lower=True)
    elif N <= rank_limit:
        out.update(rank=rank_mod(fitting_matrix(A, p, c), p), meets_lower=False)
    else:
        out.update(rank=None, meets_lower=False)
    out["seconds"] = round(time.perf_counter() - t0, 3)
    return out
