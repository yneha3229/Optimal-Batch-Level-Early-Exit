# Gap agent: results on unseen large graphs

27 test graphs, 30 s each, one core, same opening and seed for every controller. Gap integral = area under gap(t)/gap(0) over the 30 s (lower is better: the gap closed sooner). All final bounds re-checked.

| controller | mean gap integral | final gap (sum) | wins (best integral) | lower bound (sum) | upper bound (sum) | decision time |
|---|---|---|---|---|---|---|
| **learned agent** | 0.519 | 2796 | 15/27 | 1480 | 4276 | 0.12 s |
| online bandit (UCB) | 0.510 | 2662 | 2/27 | 1566 | 4228 | 0.01 s |
| random switching | 0.507 | 2686 | 3/27 | 1568 | 4254 | 0.00 s |
| round robin | 0.515 | 2696 | 1/27 | 1571 | 4267 | 0.00 s |
| fixed pair (annealing + tabu) | 0.504 | 2675 | 10/27 | 1538 | 4213 | 0.00 s |

## By size and family (mean gap integral)

| graphs | learned agent | online bandit (UCB) | random switching | round robin | fixed pair (annealing + tabu) |
|---|---|---|---|---|---|
| random, n = 500 | 0.680 | 0.694 | 0.693 | 0.703 | 0.684 |
| planted, n = 500 | 0.190 | 0.201 | 0.189 | 0.204 | 0.181 |
| layered, n = 500 | 0.618 | 0.509 | 0.507 | 0.497 | 0.556 |
| random, n = 1000 | 0.723 | 0.746 | 0.744 | 0.751 | 0.729 |
| planted, n = 1000 | 0.560 | 0.583 | 0.583 | 0.600 | 0.563 |
| layered, n = 1000 | 0.309 | 0.275 | 0.260 | 0.275 | 0.272 |
| random, n = 2000 | 0.850 | 0.808 | 0.810 | 0.816 | 0.793 |
| planted, n = 2000 | 0.402 | 0.403 | 0.409 | 0.413 | 0.413 |
| layered, n = 2000 | 0.339 | 0.366 | 0.366 | 0.375 | 0.347 |

## How the learned agent spends its time (share of slices)

| graphs | lo_greedy | lo_anneal | lo_descent | lo_kick | hi_dsatur | hi_greedy | hi_tabu |
|---|---|---|---|---|---|---|---|
| random | 0.02 | 0.00 | 0.01 | 0.22 | 0.00 | 0.00 | 0.75 |
| planted | 0.08 | 0.00 | 0.01 | 0.00 | 0.10 | 0.00 | 0.70 |
| layered | 0.02 | 0.00 | 0.04 | 0.00 | 0.00 | 0.00 | 0.93 |
