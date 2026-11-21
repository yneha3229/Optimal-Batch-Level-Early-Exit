"""Exact linear algebra over F_2 on bit-packed rows (Python ints).

Rows are ints: bit j of rows[i] is A[i, j]. All operations are exact; there is no floating point anywhere, so the
numerical-stability rules for real SVD/QR do not apply.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .graph import bool_to_masks


def rank_f2(rows: Sequence[int]) -> int:
    """Rank over F_2. Gaussian elimination keyed by the highest set bit; O(n * r) big-int XORs."""
    basis: dict[int, int] = {}  # pivot bit -> reduced row with that leading bit
    for row in rows:
        if row < 0:
            raise ValueError("rows must be non-negative ints")
        x = row
        while x:
            p = x.bit_length() - 1
            b = basis.get(p)
            if b is None:
                basis[p] = x
                break
            x ^= b
    return len(basis)


def row_basis_f2(rows: Sequence[int]) -> list[int]:
    """A basis of the row space (the broadcast vectors of the index code)."""
    basis: dict[int, int] = {}
    for row in rows:
        x = row
        while x:
            p = x.bit_length() - 1
            b = basis.get(p)
            if b is None:
                basis[p] = x
                break
            x ^= b
    return list(basis.values())


def product_f2(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """P = U V^T over F_2 as a uint8 (n, m) array.

    U: (n, r) and V: (m, r) with entries in {0, 1}. Integer matmul is exact: entries of U @ V.T are at most r.
    """
    U = np.asarray(U)
    V = np.asarray(V)
    if U.ndim != 2 or V.ndim != 2 or U.shape[1] != V.shape[1]:
        raise ValueError(f"shape mismatch: U {U.shape}, V {V.shape}")
    if U.shape[1] == 0:
        return np.zeros((U.shape[0], V.shape[0]), dtype=np.uint8)
    # int32 accumulation is exact for r < 2**31.
    return ((U.astype(np.int32) @ V.astype(np.int32).T) & 1).astype(np.uint8)  # (n, m)


def rows_from_factors(U: np.ndarray, V: np.ndarray) -> list[int]:
    """Rows of A = U V^T over F_2. Materialises the n x n matrix, which verification needs entry by entry."""
    return bool_to_masks(product_f2(U, V))
