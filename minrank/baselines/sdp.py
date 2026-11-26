"""B10 (optional): vector-colouring SDP + hyperplane rounding -> clique cover (Karger-Motwani-Sudan rounding of the
vector chromatic number SDP; the SDP that Chlamtac & Haviv bound by minrank, CPC 2014).

Contract
--------
Conflict graph H = complement of the mutual graph (an edge = the two receivers cannot share a clique). The vector
chromatic number SDP:
    minimise t   s.t.   X PSD (n x n),  X_ii = 1,  X_ij <= t  for every edge ij of H.
If H has a k-colouring then t <= -1/(k-1). Rounding: factor X = W W^T (eigendecomposition in float64, negative
eigenvalues clipped to 0: the solver returns X only approximately PSD), draw h random hyperplanes, colour each
vertex by its sign pattern (2^h classes), then repair: any class containing a conflict edge is split greedily.
Best of `trials` roundings; the colouring is turned into a clique cover (rows) and verified.

Dependency: cvxpy (with SCS or CLARABEL). Imported lazily; `sdp_rows` raises ImportError with an install hint if
it is missing. The SDP has O(n^2) variables, so it is meant for n <= ~100.
"""
from __future__ import annotations

import numpy as np

from ..core.graph import Graph
from ..core.verify import fit_error
from .clique_cover import colors_to_rows, conflict_graph, greedy_order, is_proper, num_colors


def vector_coloring(conf: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve the SDP. Returns (W (n, n) float64 with rows = unit-ish vectors, optimal t)."""
    try:
        import cvxpy as cp
    except ImportError as e:  # pragma: no cover - exercised only without cvxpy
        raise ImportError("B10 needs cvxpy: pip install cvxpy") from e
    n = conf.shape[0]
    iu, ju = np.nonzero(np.triu(conf, 1))
    if len(iu) == 0:  # no conflicts: one clique
        return np.ones((n, 1)), 1.0
    X = cp.Variable((n, n), PSD=True)
    t = cp.Variable()
    cons = [cp.diag(X) == 1, X[iu, ju] <= t]
    prob = cp.Problem(cp.Minimize(t), cons)
    prob.solve(solver="SCS" if "SCS" in cp.installed_solvers() else None, eps=1e-5)
    if X.value is None:
        raise RuntimeError(f"SDP failed: status {prob.status}")
    Xv = np.asarray(X.value, dtype=np.float64)
    Xv = (Xv + Xv.T) / 2.0
    w, Q = np.linalg.eigh(Xv)  # float64 decomposition
    W = Q * np.sqrt(np.clip(w, 0.0, None))[None, :]  # (n, n)
    return W, float(t.value)


def round_hyperplanes(W: np.ndarray, conf: np.ndarray, h: int, rng: np.random.Generator) -> np.ndarray:
    """Sign-pattern colouring with h random hyperplanes, then conflict repair by greedy recolouring."""
    n, d = W.shape
    R = rng.standard_normal((d, h))  # (d, h)
    bits = (W @ R) > 0  # (n, h)
    colors = (bits.astype(np.int64) * (1 << np.arange(h))[None, :]).sum(1)  # (n,)
    # repair: process vertices in a random order; keep a colour only if no earlier-kept neighbour shares it
    order = rng.permutation(n)
    kept = -np.ones(n, dtype=np.int64)
    for v in order:
        c = colors[v]
        if not (conf[v] & (kept == c)).any():
            kept[v] = c
    bad = np.flatnonzero(kept < 0)
    if len(bad):  # recolour the rest greedily on top of the kept classes
        offset = int(kept.max()) + 1 if (kept >= 0).any() else 0
        sub = greedy_order(conf[np.ix_(bad, bad)], np.arange(len(bad)))
        # a fresh colour block per repaired vertex group keeps the colouring proper
        kept[bad] = offset + sub
    return np.unique(kept, return_inverse=True)[1]


def sdp_rows(g: Graph, time_limit: float, rng: np.random.Generator, trials: int = 64) -> list[int]:
    """B10: best rounded clique cover over `trials` (hyperplane counts 1..ceil(log2 n)+1). Verified rows.
    `time_limit` is not enforced inside the SDP solve (cvxpy has no portable wall-clock limit)."""
    conf = conflict_graph(g)
    W, _ = vector_coloring(conf)
    hmax = max(1, int(np.ceil(np.log2(max(g.n, 2)))) + 1)
    best = None
    for k in range(trials):
        c = round_hyperplanes(W, conf, 1 + k % hmax, rng)
        if not is_proper(conf, c):
            raise RuntimeError("rounding produced an improper colouring")
        if best is None or num_colors(c) < num_colors(best):
            best = c
    rows = colors_to_rows(best)
    err = fit_error(g, rows)
    if err:
        raise RuntimeError(f"SDP rounding produced a non-fitting matrix on {g.name}: {err}")
    return rows
