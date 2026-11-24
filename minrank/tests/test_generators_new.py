"""T1 generators (WORK.md): vertex counts, regularity, independence numbers, symmetry, no loops."""
from math import comb

import numpy as np
import pytest

from minrank.bench import generators as gen
from minrank.exact.lower import mais


def adj(g):
    return np.array([[(g.out[i] >> j) & 1 for j in range(g.n)] for i in range(g.n)], bool)


def degrees(g):
    return {bin(x).count("1") for x in g.out}


@pytest.mark.parametrize("g,n,deg", [
    (gen.johnson(6, 2), 15, 8), (gen.johnson(7, 3), 35, 12),
    (gen.generalized_kneser(7, 3, 1), 35, 4 + 18), (gen.odd_graph(3), 35, 4),
    (gen.co_kneser(6, 2), 15, 8), (gen.hamming(3, 2, 1), 8, 3), (gen.hamming(2, 3, 1), 9, 4),
    (gen.hamming(4, 2, 2), 16, 6)])
def test_counts_regular_symmetric(g, n, deg):
    a = adj(g)
    assert g.n == n and degrees(g) == {deg}
    assert (a == a.T).all() and not a.diagonal().any() and not g.directed


def test_special_cases():
    assert gen.generalized_kneser(7, 3, 0).out == gen.kneser(7, 3).out
    assert gen.odd_graph(2).out == gen.kneser(5, 2).out                     # Petersen
    a, b = adj(gen.co_kneser(7, 3)), adj(gen.kneser(7, 3))
    assert ((a ^ b) == ~np.eye(35, dtype=bool)).all()                       # complements
    assert gen.gkneser(8, 3, 1).name == "gkneser_8_3_1"
    with pytest.raises(ValueError):
        gen.generalized_kneser(7, 3, 3)


@pytest.mark.parametrize("g,alpha", [
    (gen.johnson(6, 2), 3), (gen.johnson(7, 2), 3),                         # alpha(J(m,2)) = floor(m/2)
    (gen.odd_graph(3), comb(6, 2)), (gen.kneser(8, 3), comb(7, 2)),         # EKR
    (gen.hamming(3, 2, 1), 4), (gen.co_kneser(6, 2), 3)])                   # bipartite cube; co-Kneser: 6/2 disjoint pairs
def test_independence_numbers(g, alpha):
    lb = mais(g, 20)
    assert lb.value == alpha and lb.proven_max
