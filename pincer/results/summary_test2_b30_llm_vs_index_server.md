26 graphs with every method; budget 30 s per graph

| method | mean minrank gap | mean beta gap | minrank closed | beta closed | both closed | mean seconds used |
|---|---|---|---|---|---|---|
| index | 5.69 | 5.975 | 13 | 6 | 6 | 23.7 |
| llm | 5.77 | 6.056 | 13 | 6 | 6 | 26.2 |

Per family, mean minrank gap / mean beta gap:

| family | graphs | index | llm |
|---|---|---|---|
| clustered digraph (many SCCs) | 4 | 0.75 / 0.62 | 0.75 / 0.62 |
| planted digraph | 2 | 0.00 / 0.00 | 0.00 / 0.00 |
| planted undirected | 2 | 0.00 / 0.92 | 0.00 / 0.91 |
| random digraph | 14 | 9.93 / 10.12 | 10.07 / 10.27 |
| random undirected | 4 | 1.50 / 2.34 | 1.50 / 2.33 |

Paired comparison on total gap (minrank gap + beta gap), wins / ties / losses of the row method:

| | index | llm |
|---|---|---|
| index | - | 3/17/6 |
| llm | 6/17/3 | - |

Which tools moved the bounds (sum of gap reduction over all graphs, excluding the quick pass):

- index: mais_order 68.0, sat_up 67.0, mais_sa 38.0, cover_tabu 16.0, mais_lazy 10.0, frac_cover 6.2, sat_down 3.0, shannon 0.5, partial_cover 0.0
