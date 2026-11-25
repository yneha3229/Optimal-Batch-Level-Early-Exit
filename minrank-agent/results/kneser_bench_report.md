# Kneser graphs: agent vs standard methods (29 graphs, 120 s limit per method, one thread)

Baseline = largest independent set (CP-SAT) as lower bound, best of clique cover (CP-SAT), fractional clique cover (column generation) and F_2 minrank (SAT) as upper bound; its time is counted as if all standard methods ran in parallel. 'exact' = lower = upper, every bound re-checked.

| graph | receivers | baseline answer | baseline time | agent answer | agent time | field | speed-up |
|---|---|---|---|---|---|---|---|
| K(5,2) (exception) | 10 | [4, 5] | open after 1 s | [4, 5] | 0.00 s | F2 |  |
| K(6,2) | 15 | **5** | 0.23 s | **5** | 0.00 s | F2 | x93 |
| K(7,2) | 21 | [6, 7] | open after 1 s | **6** | 0.00 s | F3 | baseline never closes |
| K(8,2) | 28 | **7** | 0.27 s | **7** | 0.00 s | F2 | x78 |
| K(9,2) | 36 | [8, 9] | open after 1 s | **8** | 0.01 s | F5 | baseline never closes |
| K(10,2) | 45 | **9** | 0.25 s | **9** | 0.01 s | F2 | x22 |
| K(11,2) | 55 | [10, 11] | open after 120 s | **10** | 0.02 s | F7 | baseline never closes |
| K(12,2) | 66 | **11** | 0.23 s | **11** | 0.01 s | F2 | x21 |
| K(13,2) | 78 | [12, 13] | open after 120 s | **12** | 0.03 s | F3 | baseline never closes |
| K(14,2) | 91 | **13** | 0.28 s | **13** | 0.02 s | F2 | x16 |
| K(15,2) | 105 | [14, 15] | open after 120 s | **14** | 0.07 s | F11 | baseline never closes |
| K(16,2) | 120 | **15** | 0.26 s | **15** | 0.04 s | F2 | x6 |
| K(17,2) | 136 | [16, 17] | open after 120 s | **16** | 0.16 s | F13 | baseline never closes |
| K(18,2) | 153 | **17** | 0.42 s | **17** | 0.05 s | F2 | x8 |
| K(19,2) | 171 | [18, 19] | open after 120 s | **18** | 0.13 s | F3 | baseline never closes |
| K(20,2) | 190 | **19** | 0.64 s | **19** | 0.10 s | F2 | x6 |
| K(21,2) | 210 | [20, 21] | open after 120 s | **20** | 0.57 s | F17 | baseline never closes |
| K(22,2) | 231 | **21** | 0.98 s | **21** | 0.19 s | F2 | x5 |
| K(23,2) | 253 | [22, 23] | open after 120 s | **22** | 0.93 s | F19 | baseline never closes |
| K(24,2) | 276 | **23** | 1.77 s | **23** | 0.24 s | F2 | x7 |
| K(7,3) (exception) | 35 | [15, 17.5] | open after 120 s | [15, 20] | 0.03 s | F3 |  |
| K(8,3) (exception) | 56 | [21, 28] | open after 120 s | [21, 22] | 0.05 s | F2 |  |
| K(9,3) | 84 | **28** | 1.27 s | **28** | 0.02 s | F3 | x56 |
| K(10,3) | 120 | **36** | 4.15 s | **36** | 0.04 s | F2 | x105 |
| K(11,3) | 165 | [45, 55] | open after 120 s | **45** | 0.16 s | F5 | baseline never closes |
| K(12,3) | 220 | **55** | 14.86 s | **55** | 0.29 s | F3 | x51 |
| K(13,3) | 286 | [66, 71.5] | open after 120 s | **66** | 0.70 s | F7 | baseline never closes |
| K(9,4) (exception) | 126 | [56, 63] | open after 120 s | [56, 77] | 120.01 s | F3 |  |
| K(10,4) (exception) | 210 | [84, 105] | open after 120 s | [84, 93] | 97.55 s | F2 |  |

- exact answers: agent **24/29**, baseline 13/29
- graphs only the agent solves exactly: 11; only the baseline: 0
- where both are exact (13): agent faster on 13, median speed-up x21, min x5.1, max x105
- agent exact answers that disagree with C(n-1,k-1): 0
