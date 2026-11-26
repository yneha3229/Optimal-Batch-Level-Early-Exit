"""The structured-graph benchmark for the minrank agent."""
from __future__ import annotations

from minrank_agent import structgraph as SG


def benchmark():
    g = []
    # Kneser: the family where the field matters most (q_p(k) | n-2k decides the gap-closing field)
    for n, k in [(7, 2), (9, 2), (11, 2), (13, 2), (15, 2), (8, 3), (10, 3), (11, 3), (12, 3),
                 (13, 3), (14, 3), (9, 4), (10, 4), (11, 4), (13, 4)]:
        from math import comb
        if comb(n, k) <= 260:
            g.append(SG.kneser(n, k))
    # Johnson
    for n, k in [(5, 2), (6, 2), (7, 2), (6, 3), (7, 3), (8, 3)]:
        g.append(SG.johnson(n, k))
    # Paley (q = 1 mod 4 prime)
    for q in (5, 13, 17, 29, 37):
        g.append(SG.paley(q))
    # circular / Cayley
    for n in (9, 11, 13, 15):
        g.append(SG.comp_cycle(n))
    for n, r in [(13, 2), (15, 2), (16, 3), (21, 2)]:
        g.append(SG.cycle_power(n, r))
    for n, S in [(13, [1, 5]), (15, [1, 4]), (17, [1, 2, 4]), (16, [1, 6])]:
        g.append(SG.cayley_sum(n, S))
    # random controls (no structure: the gap should stay open)
    for n in (24, 30, 40):
        g.append(SG.random_gnp(n, 0.5, n))
    return g
