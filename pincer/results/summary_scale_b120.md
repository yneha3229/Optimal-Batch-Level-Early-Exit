12 graphs with every method; budget 120 s per graph

| method | mean minrank gap | mean beta gap | minrank closed | beta closed | both closed | mean seconds used |
|---|---|---|---|---|---|---|
| classic | 62.67 | 62.667 | 0 | 0 | 0 | 119.7 |
| index | 37.33 | 37.500 | 0 | 0 | 0 | 120.0 |

Per family, mean minrank gap / mean beta gap:

| family | graphs | classic | index |
|---|---|---|---|
| clustered digraph (many SCCs) | 2 | 10.50 / 10.50 | 8.00 / 8.00 |
| planted digraph | 2 | 26.50 / 26.50 | 26.50 / 26.50 |
| planted undirected | 2 | 11.50 / 11.50 | 10.00 / 11.00 |
| random digraph | 6 | 109.17 / 109.17 | 59.83 / 59.83 |

Paired comparison on total gap (minrank gap + beta gap), wins / ties / losses of the row method:

| | classic | index |
|---|---|---|
| classic | - | 0/2/10 |
| index | 10/2/0 | - |

Time to match: seconds until the Pincer run's total gap first drops to the baseline's FINAL total gap (graphs where it does), against the seconds the baseline used:

| Pincer policy | baseline | graphs matched | median Pincer s | median baseline s | median speed-up |
|---|---|---|---|---|---|
| index | classic | 12/12 | 23.03 | 120.0 | 5x |

Which tools moved the bounds (sum of gap reduction over all graphs, excluding the quick pass):

- index: mais_sa 221.0, mais_lazy 78.0, cover_tabu 42.0, sat_up 8.0, mais_order 7.0, sat_down 0.0, partial_cover 0.0, frac_cover 0.0

Per graph (minrank ; beta ; seconds used):

| graph | n | classic | index |
|---|---|---|---|
| clustered_k20_m15_pb0.0002_s7 | 300 | [136,144] ; [136.00,144.00] ; 120s | [137,144] ; [137.00,144.00] ; 120s |
| clustered_k20_m15_pb0.0002_s8 | 300 | [129,142] ; [129.00,142.00] ; 120s | [133,142] ; [133.00,142.00] ; 120s |
| dnp_n200_p0.05_s7 | 200 | [93,168] ; [93.00,168.00] ; 120s | [103,140] ; [103.00,140.00] ; 120s |
| dnp_n200_p0.05_s8 | 200 | [96,171] ; [96.00,171.00] ; 120s | [102,143] ; [102.00,143.00] ; 120s |
| dnp_n200_p0.15_s7 | 200 | [40,97] ; [40.00,97.00] ; 118s | [48,97] ; [48.00,97.00] ; 120s |
| dnp_n200_p0.15_s8 | 200 | [42,98] ; [42.00,98.00] ; 120s | [49,98] ; [49.00,98.00] ; 120s |
| dnp_n400_p0.03_s7 | 400 | [151,348] ; [151.00,348.00] ; 120s | [187,277] ; [187.00,277.00] ; 120s |
| dnp_n400_p0.03_s8 | 400 | [155,350] ; [155.00,350.00] ; 120s | [186,279] ; [186.00,279.00] ; 120s |
| planted_d_n200_r12_e0.03_s7 | 200 | [12,38] ; [12.00,38.00] ; 120s | [12,38] ; [12.00,38.00] ; 120s |
| planted_d_n200_r12_e0.03_s8 | 200 | [12,39] ; [12.00,39.00] ; 120s | [12,39] ; [12.00,39.00] ; 120s |
| planted_n200_r12_e0.03_s7 | 200 | [10,22] ; [10.00,22.00] ; 120s | [12,22] ; [11.00,22.00] ; 120s |
| planted_n200_r12_e0.03_s8 | 200 | [11,22] ; [11.00,22.00] ; 118s | [12,22] ; [11.00,22.00] ; 120s |
