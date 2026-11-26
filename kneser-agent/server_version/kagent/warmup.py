"""Load every numba kernel once in the parent process, so forked workers inherit compiled code instead of each
loading the on-disk cache (1-3 s per worker otherwise)."""
from __future__ import annotations

import numpy as np

_done = False


def warmup() -> None:
    global _done
    if _done:
        return
    from .graphs import kneser
    from .rank import SpectralMatrix, rank_factor
    from .tools.covers import _greedy_weight_clique, clique_cover, conflict_csr
    from .tools.indep import indep_ils

    A = kneser(7, 2)
    indep_ils(A, 0.05)
    clique_cover(A, 0.2, csr=conflict_csr(A), tabu_iters=100)
    _greedy_weight_clique(A, np.ones(A.shape[0]), 0.3, 2)
    rank_factor(SpectralMatrix(A, 3, 1), cap=10)
    _done = True
