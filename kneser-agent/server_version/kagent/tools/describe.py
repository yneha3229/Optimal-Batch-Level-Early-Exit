"""`describe`: the graph summary the agents see. Generic invariants only; nothing here knows any graph family.

Spectrum:
  n <= EIGH_MAX: dense symmetric eigensolver (exact up to rounding), multiplicities counted.
  larger n     : Lanczos with full reorthogonalisation from vertex-indicator start vectors e_v. The Krylov space of
                 e_v has dimension = number of distinct eigenvalues seen from v, so when Lanczos breaks down the Ritz
                 values are those eigenvalues. The squared first components of the Ritz vectors are (E_i)_vv; their
                 mean over sampled vertices times n estimates the multiplicity m_i (exact for walk-regular graphs).
                 The estimate is labelled "est" and checked against sum m_i = n, sum m_i th_i = 0, sum m_i th_i^2 = 2|E|.
"""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

EIGH_MAX = 4000


def _int_or_float(x: float):
    r = round(x)
    return int(r) if abs(x - r) < 1e-6 else round(float(x), 6)


def spectrum_dense(A: np.ndarray) -> dict:
    import scipy.linalg as sl

    w = sl.eigh(A.astype(np.float64), eigvals_only=True, driver="evd", check_finite=False)
    w = np.sort(w)[::-1]
    groups: list[list[float]] = []
    for x in w:
        if groups and abs(groups[-1][-1] - x) < 1e-6 * max(1.0, abs(x)):
            groups[-1].append(x)
        else:
            groups.append([x])
    eig = [{"value": _int_or_float(float(np.mean(g))), "mult": len(g)} for g in groups]
    return {"method": "dense eigh (exact up to rounding)", "distinct": len(eig), "eigenvalues": eig,
            "multiplicities_exact": True, "lambda_min": float(w[-1]), "lambda_max": float(w[0])}


def _lanczos(A64: np.ndarray, v0: np.ndarray, kmax: int):
    n = A64.shape[0]
    Q = np.zeros((kmax + 1, n))
    alpha, beta = [], []
    q = v0 / np.linalg.norm(v0)
    Q[0] = q
    scale = None
    for j in range(kmax):
        w = A64 @ Q[j]
        a = float(Q[j] @ w)
        w -= a * Q[j] + (beta[-1] * Q[j - 1] if j else 0.0)
        w -= Q[: j + 1].T @ (Q[: j + 1] @ w)                  # full reorthogonalisation (twice)
        w -= Q[: j + 1].T @ (Q[: j + 1] @ w)
        b = float(np.linalg.norm(w))
        alpha.append(a)
        scale = scale or max(1.0, abs(a))
        if b < 1e-9 * scale:
            return np.array(alpha), np.array(beta), True
        beta.append(b)
        Q[j + 1] = w / b
    return np.array(alpha), np.array(beta[: len(alpha) - 1]), False


def spectrum_lanczos(A: np.ndarray, samples: int = 4, kmax: int = 40, seed: int = 0) -> dict:
    n = A.shape[0]
    A64 = A.astype(np.float64)
    rng = np.random.default_rng(seed)
    weights: dict[int, list[float]] = {}
    values: dict[int, list[float]] = {}
    complete = True
    for v in rng.choice(n, size=min(samples, n), replace=False):
        e = np.zeros(n)
        e[v] = 1.0
        a, b, broke = _lanczos(A64, e, kmax)
        complete &= broke
        T = np.diag(a) + np.diag(b, 1) + np.diag(b, -1)
        th, U = np.linalg.eigh(T)
        for x, u0 in zip(th, U[0]):
            key = int(round(x)) if abs(x - round(x)) < 1e-4 else round(float(x), 3)
            weights.setdefault(key, []).append(float(u0 ** 2))
            values.setdefault(key, []).append(float(x))
    eig = []
    for key in sorted(values, key=lambda k: -np.mean(values[k])):
        w = weights[key] + [0.0] * (samples - len(weights[key]))
        eig.append({"value": _int_or_float(float(np.mean(values[key]))), "mult_est": int(round(n * np.mean(w))),
                    "local_weight_spread": round(float(np.std(w) * n), 2)})
    mult = np.array([e["mult_est"] for e in eig], dtype=float)
    th = np.array([float(e["value"]) for e in eig])
    m2 = float(A.sum())
    checks = {"sum_mult": int(mult.sum()), "n": n, "trace_A": round(float(mult @ th), 3),
              "trace_A2": round(float(mult @ th ** 2), 3), "2|E|": m2}
    ok = complete and int(mult.sum()) == n and abs(mult @ th) < 0.5 and abs(mult @ th ** 2 - m2) < 0.5
    return {"method": f"Lanczos from {samples} vertex vectors (distinct values exact when the Krylov space closes; "
                      "multiplicities estimated from local spectral weights)",
            "krylov_closed": bool(complete), "distinct": len(eig), "eigenvalues": eig,
            "multiplicities_exact": False, "trace_checks_pass": bool(ok), "trace_checks": checks,
            "lambda_min": float(min(th)), "lambda_max": float(max(th))}


def spectrum(A: np.ndarray) -> dict:
    t0 = time.perf_counter()
    out = spectrum_dense(A) if A.shape[0] <= EIGH_MAX else spectrum_lanczos(A)
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def _greedy_clique(A: np.ndarray, rng) -> int:
    n = A.shape[0]
    cand = np.ones(n, dtype=bool)
    size = 0
    while cand.any():
        idx = np.nonzero(cand)[0]
        v = idx[rng.integers(len(idx))]
        size += 1
        cand &= A[v]
    return size


def _greedy_indep(A: np.ndarray, rng) -> int:
    n = A.shape[0]
    cand = np.ones(n, dtype=bool)
    size = 0
    while cand.any():
        idx = np.nonzero(cand)[0]
        v = idx[rng.integers(len(idx))]
        size += 1
        cand &= ~A[v]
        cand[v] = False
    return size


def describe(A: np.ndarray, seed: int = 0, sample_pairs: int = 3000, with_spectrum: bool = True) -> dict:
    """Generic summary of an undirected graph."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    deg = A.sum(1)
    m = int(deg.sum() // 2)
    A32 = A.astype(np.float32)
    out: dict = {"n": n, "m": m, "density": round(2 * m / (n * (n - 1)), 6) if n > 1 else 0.0,
                 "degree_min": int(deg.min()), "degree_max": int(deg.max()),
                 "regular": bool(deg.min() == deg.max())}
    # common neighbours on sampled adjacent / non-adjacent pairs (exact on all pairs if n is small)
    iu, ju = np.nonzero(np.triu(A, 1)) if n <= 1500 else (None, None)
    if iu is not None and len(iu) <= sample_pairs:
        ea, eb = iu, ju
    else:
        rows = rng.integers(n, size=sample_pairs * 4)
        ea, eb = [], []
        for v in rows:
            nb = np.nonzero(A[v])[0]
            if len(nb):
                ea.append(v)
                eb.append(nb[rng.integers(len(nb))])
            if len(ea) >= sample_pairs:
                break
        ea, eb = np.array(ea), np.array(eb)
    cn_adj = (A32[ea] * A32[eb]).sum(1).astype(int) if len(ea) else np.array([], int)
    na, nb_ = [], []
    while len(na) < sample_pairs and n > 2:
        u, v = rng.integers(n, size=2)
        if u != v and not A[u, v]:
            na.append(u)
            nb_.append(v)
        if len(na) == 0 and m == n * (n - 1) // 2:
            break
    cn_non = (A32[na] * A32[nb_]).sum(1).astype(int) if na else np.array([], int)
    exact_pairs = iu is not None and len(iu) <= sample_pairs
    out["common_neighbours_adjacent"] = {"values": dict(sorted(Counter(cn_adj.tolist()).items())),
                                         "pairs": int(len(cn_adj)), "sampled": not exact_pairs}
    out["common_neighbours_nonadjacent"] = {"values": dict(sorted(Counter(cn_non.tolist()).items())),
                                            "pairs": int(len(cn_non)), "sampled": True}
    # triangles: exact sum over edges of common neighbours / 3 when small, else estimated from the edge sample
    if n <= 2000:
        tri = int(round(float(((A32 @ A32) * A32).sum()) / 6))
        out["triangles"] = {"count": tri, "exact": True}
    else:
        est = float(cn_adj.mean()) * m / 3 if len(cn_adj) else 0.0
        out["triangles"] = {"count": int(round(est)), "exact": False, "from_edge_sample": int(len(cn_adj))}
    out["greedy_clique_size"] = max(_greedy_clique(A, rng) for _ in range(8))
    out["greedy_independent_set_size"] = max(_greedy_indep(A, rng) for _ in range(8))
    if with_spectrum:
        sp = spectrum(A)
        sp["eigenvalues"] = sp["eigenvalues"][:12]
        out["spectrum"] = sp
    out["seconds"] = round(time.perf_counter() - t0, 2)
    return out


def summary_text(d: dict) -> str:
    """Compact text for the LLM prompts."""
    lines = [f"n = {d['n']} vertices, m = {d['m']} edges, density {d['density']}",
             f"degree min {d['degree_min']}, max {d['degree_max']}, regular: {d['regular']}",
             f"triangles: {d['triangles']['count']} ({'exact' if d['triangles']['exact'] else 'estimated'})",
             f"common neighbours of adjacent pairs ({d['common_neighbours_adjacent']['pairs']} pairs"
             f"{', sampled' if d['common_neighbours_adjacent']['sampled'] else ''}): value->count "
             f"{d['common_neighbours_adjacent']['values']}",
             f"common neighbours of non-adjacent pairs ({d['common_neighbours_nonadjacent']['pairs']} sampled pairs): "
             f"{d['common_neighbours_nonadjacent']['values']}",
             f"greedy clique size {d['greedy_clique_size']}, greedy independent set size "
             f"{d['greedy_independent_set_size']}"]
    sp = d.get("spectrum")
    if sp is None:
        lines.append("spectrum: unavailable")
    else:
        key = "mult" if sp["multiplicities_exact"] else "mult_est"
        ev = ", ".join(f"{e['value']} (x{e[key]})" for e in sp["eigenvalues"])
        tag = "exact multiplicities" if sp["multiplicities_exact"] else (
            f"ESTIMATED multiplicities, Krylov closed: {sp['krylov_closed']}, trace checks pass: "
            f"{sp['trace_checks_pass']}")
        lines.append(f"spectrum ({sp['distinct']} distinct eigenvalues; {tag}): {ev}")
    return "\n".join(lines)
