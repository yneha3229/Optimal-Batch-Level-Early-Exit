23 graphs with every method; budget 60 s per graph

| method | mean minrank gap | mean beta gap | minrank closed | beta closed | both closed | mean seconds used |
|---|---|---|---|---|---|---|
| classic | 1.22 | 1.217 | 2 | 2 | 2 | 30.0 |
| satdescend | 0.57 | 1.217 | 16 | 2 | 2 | 11.8 |
| roundrobin | 0.00 | 0.138 | 23 | 21 | 21 | 7.5 |
| index | 0.00 | 0.167 | 23 | 20 | 20 | 7.8 |

Per family, mean minrank gap / mean beta gap:

| family | graphs | classic | satdescend | roundrobin | index |
|---|---|---|---|---|---|
| structured | 23 | 1.22 / 1.22 | 0.57 / 1.22 | 0.00 / 0.14 | 0.00 / 0.17 |

Paired comparison on total gap (minrank gap + beta gap), wins / ties / losses of the row method:

| | classic | satdescend | roundrobin | index |
|---|---|---|---|---|
| classic | - | 0/9/14 | 0/2/21 | 0/2/21 |
| satdescend | 14/9/0 | - | 0/2/21 | 0/2/21 |
| roundrobin | 21/2/0 | 21/2/0 | - | 1/22/0 |
| index | 21/2/0 | 21/2/0 | 0/22/1 | - |

Time to match: seconds until the Pincer run's total gap first drops to the baseline's FINAL total gap (graphs where it does), against the seconds the baseline used:

| Pincer policy | baseline | graphs matched | median Pincer s | median baseline s | median speed-up |
|---|---|---|---|---|---|
| index | classic | 23/23 | 0.01 | 30.0 | 600x |
| index | satdescend | 23/23 | 0.42 | 3.4 | 8x |
| roundrobin | classic | 23/23 | 0.50 | 30.0 | 60x |
| roundrobin | satdescend | 23/23 | 0.54 | 3.4 | 6x |

Which tools moved the bounds (sum of gap reduction over all graphs, excluding the quick pass):

- index: sat_down 32.0, shannon 13.6, frac_cover 10.6, mais_lazy 6.0, cover_tabu 0.0
- roundrobin: shannon 37.6, frac_cover 11.2, mais_order 6.0, cover_tabu 4.0, sat_down 4.0

Per graph (minrank ; beta ; seconds used):

| graph | n | classic | satdescend | roundrobin | index |
|---|---|---|---|---|---|
| circ_13_1-5 | 13 | [4,7] ; [4.00,7.00] ; 30s | [4,7] ; [4.00,7.00] ; 30s | 7 ; 13/2 ; 1s | 7 ; 13/2 ; 1s |
| circ_14_1-3 | 14 | 7 ; 7 ; 30s | 7 ; 7 ; 10s | 7 ; 7 ; 0s | 7 ; 7 ; 0s |
| circ_15_1-4 | 15 | [6,8] ; [6.00,8.00] ; 30s | [6,8] ; [6.00,8.00] ; 30s | 8 ; 15/2 ; 7s | 8 ; 15/2 ; 9s |
| circ_16_1-4 | 16 | [6,8] ; [6.00,8.00] ; 30s | [6,8] ; [6.00,8.00] ; 30s | 8 ; 8 ; 54s | 8 ; 8 ; 53s |
| cycle_11 | 11 | [5,6] ; [5.00,6.00] ; 30s | 6 ; [5.00,6.00] ; 4s | 6 ; 11/2 ; 1s | 6 ; 11/2 ; 1s |
| cycle_11_c | 11 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 11/5 ; 1s | 3 ; 11/5 ; 1s |
| cycle_13 | 13 | [6,7] ; [6.00,7.00] ; 30s | [6,7] ; [6.00,7.00] ; 30s | 7 ; 13/2 ; 1s | 7 ; 13/2 ; 1s |
| cycle_13_c | 13 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 13/6 ; 1s | 3 ; 13/6 ; 2s |
| cycle_15 | 15 | [7,8] ; [7.00,8.00] ; 30s | [7,8] ; [7.00,8.00] ; 30s | 8 ; 15/2 ; 15s | 8 ; 15/2 ; 15s |
| cycle_15_c | 15 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 15/7 ; 20s | 3 ; 15/7 ; 21s |
| cycle_5 | 5 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 5/2 ; 1s | 3 ; 5/2 ; 1s |
| cycle_7 | 7 | [3,4] ; [3.00,4.00] ; 30s | 4 ; [3.00,4.00] ; 3s | 4 ; 7/2 ; 1s | 4 ; 7/2 ; 1s |
| cycle_7_c | 7 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 7/3 ; 1s | 3 ; 7/3 ; 1s |
| cycle_9 | 9 | [4,5] ; [4.00,5.00] ; 30s | 5 ; [4.00,5.00] ; 3s | 5 ; 9/2 ; 1s | 5 ; 9/2 ; 1s |
| cycle_9_c | 9 | [2,3] ; [2.00,3.00] ; 30s | 3 ; [2.00,3.00] ; 3s | 3 ; 9/4 ; 1s | 3 ; 9/4 ; 1s |
| cyclepow_13_2 | 13 | [4,5] ; [4.00,5.00] ; 30s | 5 ; [4.00,5.00] ; 3s | 5 ; 13/3 ; 1s | 5 ; 13/3 ; 1s |
| cyclepow_16_2 | 16 | [5,6] ; [5.00,6.00] ; 30s | 6 ; [5.00,6.00] ; 4s | 6 ; 16/3 ; 61s | 6 ; [5.33,6.00] ; 60s |
| kneser_5_2 | 10 | [4,5] ; [4.00,5.00] ; 30s | 5 ; [4.00,5.00] ; 3s | 5 ; 5 ; 1s | 5 ; 5 ; 1s |
| kneser_6_2 | 15 | 5 ; 5 ; 30s | 5 ; 5 ; 3s | 5 ; 5 ; 1s | 5 ; 5 ; 0s |
| mycielski_4 | 11 | [5,6] ; [5.00,6.00] ; 30s | 6 ; [5.00,6.00] ; 5s | 6 ; 11/2 ; 1s | 6 ; 11/2 ; 1s |
| mycielski_5 | 23 | [11,12] ; [11.00,12.00] ; 30s | [11,12] ; [11.00,12.00] ; 30s | 12 ; [11.00,11.50] ; 2s | 12 ; [11.00,11.50] ; 2s |
| paley_13 | 13 | [3,5] ; [3.00,5.00] ; 30s | 5 ; [3.00,5.00] ; 3s | 5 ; 13/3 ; 1s | 5 ; 13/3 ; 1s |
| paley_17 | 17 | [3,6] ; [3.00,6.00] ; 30s | [3,6] ; [3.00,6.00] ; 30s | 6 ; [3.00,5.67] ; 2s | 6 ; [3.00,5.67] ; 2s |
