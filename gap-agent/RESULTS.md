# Gap agent: results on unseen very large graphs (3000-10000 receivers)

27 test graphs, 30 s each, one core, same opening and seed for every controller. Gap integral = area under gap(t)/gap(0) over the 30 s (lower is better: the gap closed sooner). All final bounds re-checked.

| controller | mean gap integral | final gap (sum) | wins (best integral) | lower bound (sum) | upper bound (sum) | decision time |
|---|---|---|---|---|---|---|
| **learned agent: learned prior + online** | 0.724 | 17245 | 14/27 | 5760 | 23005 | 0.12 s |
| learned agent: offline policy (FQI) | 0.767 | 19400 | 6/27 | 4481 | 23881 | 0.23 s |
| online bandit (UCB) | 0.736 | 17567 | 4/27 | 5776 | 23343 | 0.01 s |
| random switching | 0.770 | 18291 | 0/27 | 5844 | 24135 | 0.00 s |
| round robin | 0.771 | 18560 | 2/27 | 5982 | 24542 | 0.00 s |
| fixed pair (annealing + tabu) | 0.741 | 17710 | 1/27 | 5957 | 23667 | 0.00 s |

## By size and family (mean gap integral)

| graphs | learned agent: learned prior + online | learned agent: offline policy (FQI) | online bandit (UCB) | random switching | round robin | fixed pair (annealing + tabu) |
|---|---|---|---|---|---|---|
| random, n = 3000 | 0.823 | 0.864 | 0.836 | 0.860 | 0.859 | 0.829 |
| planted, n = 3000 | 0.661 | 0.664 | 0.662 | 0.759 | 0.812 | 0.719 |
| layered, n = 3000 | 0.320 | 0.344 | 0.336 | 0.346 | 0.347 | 0.331 |
| random, n = 6000 | 0.879 | 0.921 | 0.892 | 0.937 | 0.943 | 0.893 |
| planted, n = 6000 | 0.885 | 0.898 | 0.900 | 0.942 | 0.948 | 0.903 |
| layered, n = 6000 | 0.579 | 0.579 | 0.617 | 0.632 | 0.586 | 0.576 |
| random, n = 10000 | 0.948 | 0.990 | 0.945 | 0.973 | 0.980 | 0.953 |
| planted, n = 10000 | 0.938 | 0.963 | 0.946 | 0.971 | 0.981 | 0.956 |
| layered, n = 10000 | 0.482 | 0.682 | 0.486 | 0.512 | 0.485 | 0.507 |

## How the learned agent spends its time (share of slices)

| graphs | lo_greedy | lo_anneal | lo_descent | lo_kick | hi_dsatur | hi_greedy | hi_tabu |
|---|---|---|---|---|---|---|---|
| random | 0.03 | 0.03 | 0.08 | 0.04 | 0.03 | 0.07 | 0.72 |
| planted | 0.03 | 0.03 | 0.08 | 0.03 | 0.03 | 0.12 | 0.68 |
| layered | 0.05 | 0.05 | 0.09 | 0.03 | 0.03 | 0.11 | 0.64 |

## Learned agent vs each other controller, graph by graph

Time to match = when the learned agent's gap first reaches the other controller's gap at 30 s.

| other controller | agent better on | mean difference in gap integral (± s.e.) | agent matches its 30 s gap on | median time to match |
|---|---|---|---|---|
| learned agent: offline policy (FQI) | 21/27 | -0.0435 ± 0.0146 | 25/27 | 6.6 s of 30 s |
| online bandit (UCB) | 22/27 | -0.0118 ± 0.0029 | 26/27 | 20.5 s of 30 s |
| random switching | 27/27 | -0.0464 ± 0.0080 | 25/27 | 9.6 s of 30 s |
| round robin | 24/27 | -0.0474 ± 0.0144 | 26/27 | 8.0 s of 30 s |
| fixed pair (annealing + tabu) | 23/27 | -0.0168 ± 0.0071 | 26/27 | 18.7 s of 30 s |
