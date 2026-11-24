# Gap agent: results on unseen large graphs

27 test graphs, 30 s each, one core, same opening and seed for every controller. Gap integral = area under gap(t)/gap(0) over the 30 s (lower is better: the gap closed sooner). All final bounds re-checked.

| controller | mean gap integral | final gap (sum) | wins (best integral) | lower bound (sum) | upper bound (sum) | decision time |
|---|---|---|---|---|---|---|
| **learned agent** | 0.767 | 19400 | 8/27 | 4481 | 23881 | 0.23 s |
| online bandit (UCB) | 0.736 | 17567 | 10/27 | 5776 | 23343 | 0.01 s |
| random switching | 0.770 | 18291 | 0/27 | 5844 | 24135 | 0.00 s |
| round robin | 0.771 | 18560 | 3/27 | 5982 | 24542 | 0.00 s |
| fixed pair (annealing + tabu) | 0.741 | 17710 | 6/27 | 5957 | 23667 | 0.00 s |

## By size and family (mean gap integral)

| graphs | learned agent | online bandit (UCB) | random switching | round robin | fixed pair (annealing + tabu) |
|---|---|---|---|---|---|
| random, n = 3000 | 0.864 | 0.836 | 0.860 | 0.859 | 0.829 |
| planted, n = 3000 | 0.664 | 0.662 | 0.759 | 0.812 | 0.719 |
| layered, n = 3000 | 0.344 | 0.336 | 0.346 | 0.347 | 0.331 |
| random, n = 6000 | 0.921 | 0.892 | 0.937 | 0.943 | 0.893 |
| planted, n = 6000 | 0.898 | 0.900 | 0.942 | 0.948 | 0.903 |
| layered, n = 6000 | 0.579 | 0.617 | 0.632 | 0.586 | 0.576 |
| random, n = 10000 | 0.990 | 0.945 | 0.973 | 0.980 | 0.953 |
| planted, n = 10000 | 0.963 | 0.946 | 0.971 | 0.981 | 0.956 |
| layered, n = 10000 | 0.682 | 0.486 | 0.512 | 0.485 | 0.507 |

## How the learned agent spends its time (share of slices)

| graphs | lo_greedy | lo_anneal | lo_descent | lo_kick | hi_dsatur | hi_greedy | hi_tabu |
|---|---|---|---|---|---|---|---|
| random | 0.06 | 0.35 | 0.02 | 0.00 | 0.00 | 0.32 | 0.25 |
| planted | 0.04 | 0.10 | 0.01 | 0.00 | 0.00 | 0.43 | 0.43 |
| layered | 0.01 | 0.06 | 0.10 | 0.00 | 0.00 | 0.28 | 0.56 |
