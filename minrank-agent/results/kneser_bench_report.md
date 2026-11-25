# Kneser graphs: agent vs standard methods (4 graphs, 120 s limit per method, one thread)

Baseline = largest independent set (CP-SAT) as lower bound, best of clique cover (CP-SAT), fractional clique cover (column generation) and F_2 minrank (SAT) as upper bound; its time is counted as if all standard methods ran in parallel. 'exact' = lower = upper, every bound re-checked.

| graph | receivers | baseline answer | baseline time | agent answer | agent time | field | speed-up |
|---|---|---|---|---|---|---|---|
| K(5,2) (exception) | 10 | [4, 5] | open after 1 s | [4, 5] | 0.00 s | F2 |  |
| K(6,2) | 15 | **5** | 0.23 s | **5** | 0.00 s | F2 | x93 |
| K(7,2) | 21 | [6, 7] | open after 1 s | **6** | 0.00 s | F3 | baseline never closes |
| K(8,2) | 28 | **7** | 0.27 s | **7** | 0.00 s | F2 | x78 |

- exact answers: agent **3/4**, baseline 2/4
- graphs only the agent solves exactly: 1; only the baseline: 0
- where both are exact (2): agent faster on 2, median speed-up x86, min x78.5, max x93
- agent exact answers that disagree with C(n-1,k-1): 0
