# Benchmark: Pincer vs standard methods

46 graphs (`suite.json`: random networks n = 10-60, caching groups, textbook cases), 30 s limit per graph for every method, one CPU core each. Accuracy is measured against the true optimum of each bound (proved by at least one method; all proofs agree). β exact = the two bounds met, so β is known.

## all graphs (46 graphs)

| method | lower bound optimal | upper bound optimal | β pinned exactly | missing transmissions (sum of gaps to optimum) | median time to final answer | max time to final answer | median time incl. proofs | max time incl. proofs |
|---|---|---|---|---|---|---|---|---|
| Greedy heuristics | 28/46 | 35/46 | 17/46 | 40 | 0.002 s | 0.02 s | 0.002 s | 0.02 s |
| Plain exact search | 45/46 | 46/46 | 24/46 | 1 | 0.015 s | 30.11 s | 0.015 s | 30.11 s |
| CP-SAT (OR-Tools) | 46/46 | 46/46 | 24/46 | 0 | 0.014 s | 1.23 s | 0.014 s | 1.23 s |
| Pincer agent | 46/46 | 46/46 | 24/46 | 0 | 0.002 s | 1.08 s | 0.003 s | 30.00 s |

## small (n <= 20) (20 graphs)

| method | lower bound optimal | upper bound optimal | β pinned exactly | missing transmissions (sum of gaps to optimum) | median time to final answer | max time to final answer | median time incl. proofs | max time incl. proofs |
|---|---|---|---|---|---|---|---|---|
| Greedy heuristics | 17/20 | 20/20 | 11/20 | 4 | 0.001 s | 0.00 s | 0.001 s | 0.00 s |
| Plain exact search | 20/20 | 20/20 | 14/20 | 0 | 0.001 s | 0.01 s | 0.001 s | 0.01 s |
| CP-SAT (OR-Tools) | 20/20 | 20/20 | 14/20 | 0 | 0.005 s | 0.01 s | 0.005 s | 0.01 s |
| Pincer agent | 20/20 | 20/20 | 14/20 | 0 | 0.001 s | 0.00 s | 0.001 s | 0.00 s |

## medium (21-40) (16 graphs)

| method | lower bound optimal | upper bound optimal | β pinned exactly | missing transmissions (sum of gaps to optimum) | median time to final answer | max time to final answer | median time incl. proofs | max time incl. proofs |
|---|---|---|---|---|---|---|---|---|
| Greedy heuristics | 9/16 | 13/16 | 5/16 | 14 | 0.003 s | 0.01 s | 0.003 s | 0.01 s |
| Plain exact search | 16/16 | 16/16 | 9/16 | 0 | 0.025 s | 0.31 s | 0.025 s | 0.31 s |
| CP-SAT (OR-Tools) | 16/16 | 16/16 | 9/16 | 0 | 0.022 s | 0.08 s | 0.022 s | 0.08 s |
| Pincer agent | 16/16 | 16/16 | 9/16 | 0 | 0.004 s | 0.14 s | 0.011 s | 0.80 s |

## large (41-60) (10 graphs)

| method | lower bound optimal | upper bound optimal | β pinned exactly | missing transmissions (sum of gaps to optimum) | median time to final answer | max time to final answer | median time incl. proofs | max time incl. proofs |
|---|---|---|---|---|---|---|---|---|
| Greedy heuristics | 2/10 | 2/10 | 1/10 | 22 | 0.015 s | 0.02 s | 0.015 s | 0.02 s |
| Plain exact search | 9/10 | 10/10 | 1/10 | 1 | 2.338 s | 30.11 s | 2.338 s | 30.11 s |
| CP-SAT (OR-Tools) | 10/10 | 10/10 | 1/10 | 0 | 0.141 s | 1.23 s | 0.141 s | 1.23 s |
| Pincer agent | 10/10 | 10/10 | 1/10 | 0 | 0.154 s | 1.08 s | 5.502 s | 30.00 s |

## Per graph

| graph | n | true MAIS | true cover | Greedy heuristics | Plain exact search | CP-SAT (OR-Tools) | Pincer agent |
|---|---|---|---|---|---|---|---|
| random_n10_p0.2_s1 | 10 | 6 | 6 | [6, 6] 0.00s | [6, 6] 0.00s | [6, 6] 0.00s | [6, 6] 0.00s |
| random_n10_p0.2_s2 | 10 | 7 | 7 | [7, 7] 0.00s | [7, 7] 0.00s | [7, 7] 0.00s | [7, 7] 0.00s |
| random_n10_p0.4_s1 | 10 | 5 | 5 | [5, 5] 0.00s | [5, 5] 0.00s | [5, 5] 0.00s | [5, 5] 0.00s |
| random_n10_p0.4_s2 | 10 | 6 | 6 | [6, 6] 0.00s | [6, 6] 0.00s | [6, 6] 0.00s | [6, 6] 0.00s |
| random_n15_p0.2_s1 | 15 | 12 | 12 | [12, 12] 0.00s | [12, 12] 0.00s | [12, 12] 0.01s | [12, 12] 0.00s |
| random_n15_p0.2_s2 | 15 | 10 | 10 | [10, 10] 0.00s | [10, 10] 0.00s | [10, 10] 0.01s | [10, 10] 0.00s |
| random_n15_p0.4_s1 | 15 | 10 | 10 | [9, 10] 0.00s | [10, 10] 0.00s | [10, 10] 0.01s | [10, 10] 0.00s |
| random_n15_p0.4_s2 | 15 | 8 | 8 | [8, 8] 0.00s | [8, 8] 0.00s | [8, 8] 0.00s | [8, 8] 0.00s |
| random_n20_p0.2_s1 | 20 | 13 | 13 | [13, 13] 0.00s | [13, 13] 0.01s | [13, 13] 0.01s | [13, 13] 0.00s |
| random_n20_p0.2_s2 | 20 | 12 | 12 | [12, 12] 0.00s | [12, 12] 0.01s | [12, 12] 0.01s | [12, 12] 0.00s |
| random_n20_p0.4_s1 | 20 | 10 | 10 | [8, 10] 0.00s | [10, 10] 0.01s | [10, 10] 0.01s | [10, 10] 0.00s |
| random_n20_p0.4_s2 | 20 | 10 | 10 | [9, 10] 0.00s | [10, 10] 0.01s | [10, 10] 0.01s | [10, 10] 0.00s |
| random_n25_p0.2_s1 | 25 | 17 | 17 | [15, 17] 0.00s | [17, 17] 0.01s | [17, 17] 0.02s | [17, 17] 0.00s |
| random_n25_p0.2_s2 | 25 | 14 | 14 | [14, 14] 0.00s | [14, 14] 0.01s | [14, 14] 0.02s | [14, 14] 0.00s |
| random_n25_p0.4_s1 | 25 | 12 | 13 | [12, 13] 0.00s | [12, 13] 0.02s | [12, 13] 0.01s | [12, 13] 0.01s |
| random_n25_p0.4_s2 | 25 | 10 | 11 | [10, 11] 0.00s | [10, 11] 0.02s | [10, 11] 0.02s | [10, 11] 0.01s |
| random_n30_p0.2_s1 | 30 | 16 | 16 | [15, 16] 0.00s | [16, 16] 0.03s | [16, 16] 0.03s | [16, 16] 0.01s |
| random_n30_p0.2_s2 | 30 | 15 | 16 | [15, 16] 0.00s | [15, 16] 0.03s | [15, 16] 0.03s | [15, 16] 0.03s |
| random_n30_p0.4_s1 | 30 | 14 | 14 | [14, 14] 0.00s | [14, 14] 0.02s | [14, 14] 0.03s | [14, 14] 0.00s |
| random_n30_p0.4_s2 | 30 | 11 | 12 | [10, 12] 0.00s | [11, 12] 0.03s | [11, 12] 0.03s | [11, 12] 0.03s |
| random_n40_p0.2_s1 | 40 | 21 | 21 | [20, 22] 0.01s | [21, 21] 0.11s | [21, 21] 0.07s | [21, 21] 0.08s |
| random_n40_p0.2_s2 | 40 | 19 | 21 | [19, 21] 0.01s | [19, 21] 0.31s | [19, 21] 0.08s | [19, 21] 0.80s |
| random_n40_p0.4_s1 | 40 | 14 | 17 | [12, 18] 0.01s | [14, 17] 0.20s | [14, 17] 0.08s | [14, 17] 0.29s |
| random_n40_p0.4_s2 | 40 | 13 | 16 | [10, 17] 0.01s | [13, 16] 0.13s | [13, 16] 0.06s | [13, 16] 0.16s |
| random_n50_p0.2_s1 | 50 | 21 | 24 | [21, 25] 0.02s | [21, 24] 3.02s | [21, 24] 0.16s | [21, 24] 7.09s |
| random_n50_p0.2_s2 | 50 | 23 | 24 | [22, 25] 0.02s | [23, 24] 1.65s | [23, 24] 0.11s | [23, 24] 3.44s |
| random_n50_p0.4_s1 | 50 | 15 | 18 | [14, 20] 0.01s | [15, 18] 1.14s | [15, 18] 0.12s | [15, 18] 2.05s |
| random_n50_p0.4_s2 | 50 | 15 | 18 | [14, 20] 0.01s | [15, 18] 1.06s | [15, 18] 0.12s | [15, 18] 2.19s |
| random_n60_p0.2_s1 | 60 | 28 | 29 | [27, 30] 0.02s | [27, 29] 30.09s | [28, 29] 0.34s | [28, 29] 30.00s |
| random_n60_p0.2_s2 | 60 | 26 | 28 | [24, 29] 0.02s | [26, 28] 30.11s | [26, 28] 0.32s | [26, 28] 30.00s |
| random_n60_p0.4_s1 | 60 | 17 | 21 | [15, 23] 0.01s | [17, 21] 5.52s | [17, 21] 1.23s | [17, 21] 16.76s |
| random_n60_p0.4_s2 | 60 | 17 | 21 | [16, 23] 0.01s | [17, 21] 4.54s | [17, 21] 0.42s | [17, 21] 14.16s |
| groups_3x5_s1 | 15 | 6 | 6 | [6, 6] 0.00s | [6, 6] 0.00s | [6, 6] 0.01s | [6, 6] 0.00s |
| groups_3x5_s2 | 15 | 7 | 7 | [7, 7] 0.00s | [7, 7] 0.00s | [7, 7] 0.01s | [7, 7] 0.00s |
| groups_4x6_s1 | 24 | 7 | 7 | [7, 7] 0.00s | [7, 7] 0.01s | [7, 7] 0.01s | [7, 7] 0.00s |
| groups_4x6_s2 | 24 | 8 | 8 | [8, 8] 0.00s | [8, 8] 0.02s | [8, 8] 0.01s | [8, 8] 0.00s |
| groups_5x6_s1 | 30 | 13 | 13 | [12, 13] 0.00s | [13, 13] 0.03s | [13, 13] 0.02s | [13, 13] 0.02s |
| groups_5x6_s2 | 30 | 10 | 10 | [10, 10] 0.00s | [10, 10] 0.03s | [10, 10] 0.01s | [10, 10] 0.00s |
| groups_6x8_s1 | 48 | 15 | 16 | [14, 16] 0.02s | [15, 16] 1.50s | [15, 16] 0.06s | [15, 16] 3.92s |
| groups_6x8_s2 | 48 | 17 | 17 | [17, 17] 0.01s | [17, 17] 0.92s | [17, 17] 0.06s | [17, 17] 0.01s |
| ring_5 | 5 | 2 | 3 | [2, 3] 0.00s | [2, 3] 0.00s | [2, 3] 0.00s | [2, 3] 0.00s |
| ring_7 | 7 | 3 | 4 | [3, 4] 0.00s | [3, 4] 0.00s | [3, 4] 0.01s | [3, 4] 0.00s |
| ring_9 | 9 | 4 | 5 | [4, 5] 0.00s | [4, 5] 0.00s | [4, 5] 0.00s | [4, 5] 0.00s |
| ring_11 | 11 | 5 | 6 | [5, 6] 0.00s | [5, 6] 0.00s | [5, 6] 0.00s | [5, 6] 0.00s |
| directed_cycle_6 | 6 | 5 | 6 | [5, 6] 0.00s | [5, 6] 0.00s | [5, 6] 0.00s | [5, 6] 0.00s |
| directed_cycle_12 | 12 | 11 | 12 | [11, 12] 0.00s | [11, 12] 0.00s | [11, 12] 0.01s | [11, 12] 0.00s |
