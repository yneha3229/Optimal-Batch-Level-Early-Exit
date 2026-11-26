# Lambda agent benchmark

119 graphs. One certified lower bound per graph (acyclic set, Shannon LP); each method's upper bound comes from explicit codes. "β pinned" = upper bound equals the lower bound, which proves β = λ₂. 60 s per graph per method, one core; times include the lower bound.

## all (119 graphs)

| method | β pinned | mean gap to lower bound | median time | 90th pct time | max time |
|---|---|---|---|---|---|
| Clique cover | 100/119 | 0.214 | 22 ms | 248 ms | 3.3 s |
| Fractional clique cover | 108/119 | 0.176 | 23 ms | 249 ms | 3.3 s |
| Scalar minrank (plain SAT) | 111/119 | 0.037 | 50 ms | 3.3 s | 63.3 s |
| Vector-linear, plain search | 119/119 | 0.000 | 38.7 s | 45.9 s | 63.3 s |
| Vector-linear, plain search + stop at LB | 119/119 | 0.000 | 31 ms | 595 ms | 4.3 s |
| Lambda agent | 119/119 | 0.000 | 4 ms | 39 ms | 2.4 s |

## random digraphs (80 graphs)

| method | β pinned | mean gap to lower bound | median time | 90th pct time | max time |
|---|---|---|---|---|---|
| Clique cover | 77/80 | 0.037 | 16 ms | 199 ms | 811 ms |
| Fractional clique cover | 77/80 | 0.037 | 20 ms | 201 ms | 812 ms |
| Scalar minrank (plain SAT) | 80/80 | 0.000 | 44 ms | 3.4 s | 60.7 s |
| Vector-linear, plain search | 80/80 | 0.000 | 32.8 s | 45.6 s | 60.7 s |
| Vector-linear, plain search + stop at LB | 80/80 | 0.000 | 18 ms | 188 ms | 907 ms |
| Lambda agent | 80/80 | 0.000 | 3 ms | 5 ms | 48 ms |

## structured (15 graphs)

| method | β pinned | mean gap to lower bound | median time | 90th pct time | max time |
|---|---|---|---|---|---|
| Clique cover | 1/15 | 1.428 | 42 ms | 270 ms | 3.3 s |
| Fractional clique cover | 7/15 | 1.200 | 43 ms | 272 ms | 3.3 s |
| Scalar minrank (plain SAT) | 9/15 | 0.228 | 164 ms | 3.9 s | 63.3 s |
| Vector-linear, plain search | 15/15 | 0.000 | 45.2 s | 49.1 s | 63.3 s |
| Vector-linear, plain search + stop at LB | 15/15 | 0.000 | 139 ms | 1.9 s | 4.3 s |
| Lambda agent | 15/15 | 0.000 | 49 ms | 281 ms | 2.4 s |

## random undirected (24 graphs)

| method | β pinned | mean gap to lower bound | median time | 90th pct time | max time |
|---|---|---|---|---|---|
| Clique cover | 22/24 | 0.042 | 103 ms | 629 ms | 796 ms |
| Fractional clique cover | 24/24 | 0.000 | 104 ms | 630 ms | 798 ms |
| Scalar minrank (plain SAT) | 22/24 | 0.042 | 129 ms | 703 ms | 822 ms |
| Vector-linear, plain search | 24/24 | 0.000 | 45.1 s | 45.8 s | 45.9 s |
| Vector-linear, plain search + stop at LB | 24/24 | 0.000 | 104 ms | 622 ms | 933 ms |
| Lambda agent | 24/24 | 0.000 | 3 ms | 5 ms | 755 ms |

## Per graph

| graph | lower bound | Clique cover | Fractional clique cover | Scalar minrank (plain SAT) | Vector-linear, plain search | Vector-linear, plain search + stop at LB | Lambda agent |
|---|---|---|---|---|---|---|---|
| digraph_n10_p0.3_two0.3_s1 | 9 | 9 (0.0s) | 9 (0.0s) | 9 (0.0s) | 9 (0.1s) | 9 (0.0s) | 9 (0.0s) |
| digraph_n10_p0.3_two0.3_s2 | 6 | 6 (0.1s) | 6 (0.1s) | 6 (0.2s) | 6 (36.3s) | 6 (0.1s) | 6 (0.0s) |
| digraph_n10_p0.3_two0.3_s3 | 7 | 7 (0.2s) | 7 (0.2s) | 7 (5.1s) | 7 (45.6s) | 7 (0.2s) | 7 (0.0s) |
| digraph_n10_p0.3_two0.3_s4 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (0.3s) | 6 (40.8s) | 6 (0.2s) | 6 (0.0s) |
| digraph_n10_p0.3_two0.6_s1 | 5 | 5 (0.8s) | 5 (0.8s) | 5 (0.9s) | 5 (46.0s) | 5 (0.9s) | 5 (0.0s) |
| digraph_n10_p0.3_two0.6_s2 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (19.6s) | 7 (0.1s) | 7 (0.0s) |
| digraph_n10_p0.3_two0.6_s3 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (0.3s) | 6 (40.8s) | 6 (0.1s) | 6 (0.0s) |
| digraph_n10_p0.3_two0.6_s4 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (31.6s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n10_p0.5_two0.3_s1 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (31.7s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n10_p0.5_two0.3_s2 | 8 | 8 (0.2s) | 8 (0.2s) | 8 (54.3s) | 8 (54.3s) | 8 (0.2s) | 8 (0.0s) |
| digraph_n10_p0.5_two0.3_s3 | 8 | 8 (0.6s) | 8 (0.6s) | 8 (60.7s) | 8 (60.7s) | 8 (0.6s) | 8 (0.0s) |
| digraph_n10_p0.5_two0.3_s4 | 8 | 8 (0.1s) | 8 (0.1s) | 8 (4.3s) | 8 (40.4s) | 8 (0.1s) | 8 (0.0s) |
| digraph_n10_p0.5_two0.6_s1 | 5 | 5 (0.6s) | 5 (0.6s) | 5 (0.6s) | 5 (45.7s) | 5 (0.6s) | 5 (0.0s) |
| digraph_n10_p0.5_two0.6_s2 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (31.6s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n10_p0.5_two0.6_s3 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (22.3s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n10_p0.5_two0.6_s4 | 5 | 5 (0.5s) | 5 (0.5s) | 5 (0.6s) | 5 (45.7s) | 5 (0.5s) | 5 (0.0s) |
| digraph_n6_p0.3_two0.3_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (0.8s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.3_two0.3_s2 | 5 | 6 (0.0s) | 6 (0.0s) | 5 (0.7s) | 5 (11.6s) | 5 (0.8s) | 5 (0.0s) |
| digraph_n6_p0.3_two0.3_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (0.7s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n6_p0.3_two0.3_s4 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (12.0s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n6_p0.3_two0.6_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (12.3s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.3_two0.6_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (0.1s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n6_p0.3_two0.6_s3 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n6_p0.3_two0.6_s4 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (13.5s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.5_two0.3_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.5_two0.3_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (11.7s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n6_p0.5_two0.3_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (0.1s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n6_p0.5_two0.3_s4 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.0s) | 3 (34.5s) | 3 (0.0s) | 3 (0.0s) |
| digraph_n6_p0.5_two0.6_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (29.5s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.5_two0.6_s2 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.0s) | 3 (34.2s) | 3 (0.0s) | 3 (0.0s) |
| digraph_n6_p0.5_two0.6_s3 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (14.6s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n6_p0.5_two0.6_s4 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.0s) | 3 (17.9s) | 3 (0.0s) | 3 (0.0s) |
| digraph_n7_p0.3_two0.3_s1 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (38.7s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.3_two0.3_s2 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (10.5s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n7_p0.3_two0.3_s3 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (11.3s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n7_p0.3_two0.3_s4 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (0.1s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.3_two0.6_s1 | 4 | 5 (0.0s) | 5 (0.0s) | 4 (0.1s) | 4 (45.1s) | 4 (0.7s) | 4 (0.0s) |
| digraph_n7_p0.3_two0.6_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (0.2s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.3_two0.6_s3 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n7_p0.3_two0.6_s4 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (11.1s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.5_two0.3_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n7_p0.5_two0.3_s2 | 5 | 6 (0.0s) | 6 (0.0s) | 5 (0.1s) | 5 (26.5s) | 5 (0.7s) | 5 (0.0s) |
| digraph_n7_p0.5_two0.3_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (25.9s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.5_two0.3_s4 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (38.7s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.5_two0.6_s1 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.0s) | 3 (34.8s) | 3 (0.0s) | 3 (0.0s) |
| digraph_n7_p0.5_two0.6_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (11.0s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n7_p0.5_two0.6_s3 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n7_p0.5_two0.6_s4 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.1s) | 3 (34.6s) | 3 (0.0s) | 3 (0.0s) |
| digraph_n8_p0.3_two0.3_s1 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (28.3s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n8_p0.3_two0.3_s2 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (0.1s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n8_p0.3_two0.3_s3 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (10.3s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n8_p0.3_two0.3_s4 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (0.1s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n8_p0.3_two0.6_s1 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.2s) | 5 (45.3s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n8_p0.3_two0.6_s2 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (22.9s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n8_p0.3_two0.6_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.1s) | 5 (20.6s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n8_p0.3_two0.6_s4 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.1s) | 4 (30.3s) | 4 (0.0s) | 4 (0.0s) |
| digraph_n8_p0.5_two0.3_s1 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.1s) | 5 (45.2s) | 5 (0.1s) | 5 (0.0s) |
| digraph_n8_p0.5_two0.3_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (39.5s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n8_p0.5_two0.3_s3 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.2s) | 5 (45.2s) | 5 (0.1s) | 5 (0.0s) |
| digraph_n8_p0.5_two0.3_s4 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (33.8s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n8_p0.5_two0.6_s1 | 4 | 4 (0.1s) | 4 (0.1s) | 4 (0.1s) | 4 (45.2s) | 4 (0.1s) | 4 (0.0s) |
| digraph_n8_p0.5_two0.6_s2 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.1s) | 5 (45.2s) | 5 (0.1s) | 5 (0.0s) |
| digraph_n8_p0.5_two0.6_s3 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.1s) | 5 (45.2s) | 5 (0.1s) | 5 (0.0s) |
| digraph_n8_p0.5_two0.6_s4 | 4 | 4 (0.1s) | 4 (0.1s) | 4 (0.1s) | 4 (45.1s) | 4 (0.1s) | 4 (0.0s) |
| digraph_n9_p0.3_two0.3_s1 | 8 | 8 (0.0s) | 8 (0.0s) | 8 (0.0s) | 8 (25.1s) | 8 (0.0s) | 8 (0.0s) |
| digraph_n9_p0.3_two0.3_s2 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (24.2s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n9_p0.3_two0.3_s3 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (30.1s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n9_p0.3_two0.3_s4 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (4.0s) | 6 (49.3s) | 6 (0.2s) | 6 (0.0s) |
| digraph_n9_p0.3_two0.6_s1 | 7 | 7 (0.1s) | 7 (0.1s) | 7 (3.7s) | 7 (43.7s) | 7 (0.1s) | 7 (0.0s) |
| digraph_n9_p0.3_two0.6_s2 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (9.6s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n9_p0.3_two0.6_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.1s) | 5 (34.1s) | 5 (0.0s) | 5 (0.0s) |
| digraph_n9_p0.3_two0.6_s4 | 7 | 7 (0.0s) | 7 (0.0s) | 7 (0.0s) | 7 (20.9s) | 7 (0.0s) | 7 (0.0s) |
| digraph_n9_p0.5_two0.3_s1 | 7 | 7 (0.1s) | 7 (0.1s) | 7 (4.2s) | 7 (44.4s) | 7 (0.1s) | 7 (0.0s) |
| digraph_n9_p0.5_two0.3_s2 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (4.5s) | 6 (49.5s) | 6 (0.2s) | 6 (0.0s) |
| digraph_n9_p0.5_two0.3_s3 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (23.7s) | 6 (0.0s) | 6 (0.0s) |
| digraph_n9_p0.5_two0.3_s4 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (3.4s) | 6 (48.6s) | 6 (0.2s) | 6 (0.0s) |
| digraph_n9_p0.5_two0.6_s1 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.1s) | 5 (40.1s) | 5 (0.1s) | 5 (0.0s) |
| digraph_n9_p0.5_two0.6_s2 | 5 | 5 (0.2s) | 5 (0.2s) | 5 (0.3s) | 5 (45.4s) | 5 (0.2s) | 5 (0.0s) |
| digraph_n9_p0.5_two0.6_s3 | 5 | 5 (0.2s) | 5 (0.2s) | 5 (0.3s) | 5 (45.3s) | 5 (0.2s) | 5 (0.0s) |
| digraph_n9_p0.5_two0.6_s4 | 5 | 5 (0.1s) | 5 (0.1s) | 5 (0.1s) | 5 (40.2s) | 5 (0.1s) | 5 (0.0s) |
| directed_circulant_10_12 | 8 | 10 (3.3s) | 10 (3.3s) | 8 (63.3s) | 8 (63.3s) | 8 (2.4s) | 8 (2.4s) |
| directed_circulant_5_12 | 3 | 5 (0.0s) | 5 (0.0s) | 3 (0.1s) | 3 (34.7s) | 3 (0.1s) | 3 (0.0s) |
| directed_circulant_6_12 | 4 | 6 (0.0s) | 6 (0.0s) | 4 (0.1s) | 4 (45.1s) | 4 (0.1s) | 4 (0.0s) |
| directed_circulant_7_12 | 5 | 7 (0.0s) | 7 (0.0s) | 5 (0.2s) | 5 (45.2s) | 5 (0.1s) | 5 (0.0s) |
| directed_circulant_7_13 | 4 | 7 (0.0s) | 7 (0.0s) | 4 (0.2s) | 4 (45.2s) | 4 (0.1s) | 4 (0.0s) |
| directed_circulant_8_12 | 6 | 8 (0.1s) | 8 (0.1s) | 6 (3.3s) | 6 (48.4s) | 6 (0.1s) | 6 (0.1s) |
| directed_circulant_9_12 | 7 | 9 (0.2s) | 9 (0.2s) | 7 (60.3s) | 7 (60.3s) | 7 (0.5s) | 7 (0.3s) |
| directed_circulant_9_13 | 6 | 9 (0.2s) | 9 (0.2s) | 6 (3.9s) | 6 (49.1s) | 6 (0.3s) | 6 (0.3s) |
| odd_cycle_5 | 5/2 | 3 (0.0s) | 5/2 (0.0s) | 3 (0.0s) | 5/2 (30.9s) | 5/2 (0.0s) | 5/2 (0.0s) |
| odd_cycle_5_complement | 5/2 | 3 (0.0s) | 5/2 (0.0s) | 3 (0.0s) | 5/2 (30.6s) | 5/2 (0.0s) | 5/2 (0.0s) |
| odd_cycle_7 | 7/2 | 4 (0.0s) | 7/2 (0.0s) | 4 (0.1s) | 7/2 (45.1s) | 7/2 (0.1s) | 7/2 (0.0s) |
| odd_cycle_7_complement | 7/3 | 3 (0.0s) | 7/3 (0.0s) | 3 (0.0s) | 7/3 (32.4s) | 7/3 (0.3s) | 7/3 (0.0s) |
| odd_cycle_9 | 9/2 | 5 (0.2s) | 9/2 (0.2s) | 5 (0.7s) | 9/2 (45.8s) | 9/2 (0.4s) | 9/2 (0.2s) |
| odd_cycle_9_complement | 9/4 | 3 (0.3s) | 9/4 (0.3s) | 3 (0.3s) | 9/4 (36.2s) | 9/4 (4.3s) | 9/4 (0.2s) |
| petersen | 5 | 5 (2.1s) | 5 (2.1s) | 5 (2.8s) | 5 (47.8s) | 5 (1.9s) | 5 (2.0s) |
| undirected_n10_p0.3_s1 | 5 | 5 (0.7s) | 5 (0.7s) | 5 (0.7s) | 5 (45.8s) | 5 (0.6s) | 5 (0.0s) |
| undirected_n10_p0.3_s2 | 6 | 6 (0.2s) | 6 (0.2s) | 6 (0.3s) | 6 (40.8s) | 6 (0.1s) | 6 (0.0s) |
| undirected_n10_p0.3_s3 | 5 | 5 (0.6s) | 5 (0.6s) | 5 (0.7s) | 5 (45.8s) | 5 (0.7s) | 5 (0.0s) |
| undirected_n10_p0.5_s1 | 7/2 | 4 (0.8s) | 7/2 (0.8s) | 4 (0.8s) | 7/2 (45.9s) | 7/2 (0.9s) | 7/2 (0.8s) |
| undirected_n10_p0.5_s2 | 4 | 4 (0.8s) | 4 (0.8s) | 4 (0.8s) | 4 (45.9s) | 4 (0.8s) | 4 (0.0s) |
| undirected_n10_p0.5_s3 | 4 | 4 (0.6s) | 4 (0.6s) | 4 (0.6s) | 4 (45.7s) | 4 (0.6s) | 4 (0.0s) |
| undirected_n7_p0.3_s1 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| undirected_n7_p0.3_s2 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (38.7s) | 5 (0.0s) | 5 (0.0s) |
| undirected_n7_p0.3_s3 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| undirected_n7_p0.5_s1 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.0s) | 3 (35.2s) | 3 (0.0s) | 3 (0.0s) |
| undirected_n7_p0.5_s2 | 3 | 3 (0.0s) | 3 (0.0s) | 3 (0.1s) | 3 (33.7s) | 3 (0.0s) | 3 (0.0s) |
| undirected_n7_p0.5_s3 | 4 | 4 (0.0s) | 4 (0.0s) | 4 (0.0s) | 4 (45.1s) | 4 (0.0s) | 4 (0.0s) |
| undirected_n8_p0.3_s1 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (39.5s) | 5 (0.0s) | 5 (0.0s) |
| undirected_n8_p0.3_s2 | 6 | 6 (0.0s) | 6 (0.0s) | 6 (0.0s) | 6 (33.8s) | 6 (0.0s) | 6 (0.0s) |
| undirected_n8_p0.3_s3 | 5 | 5 (0.0s) | 5 (0.0s) | 5 (0.0s) | 5 (39.5s) | 5 (0.0s) | 5 (0.0s) |
| undirected_n8_p0.5_s1 | 3 | 3 (0.1s) | 3 (0.1s) | 3 (0.1s) | 3 (34.2s) | 3 (0.1s) | 3 (0.0s) |
| undirected_n8_p0.5_s2 | 3 | 3 (0.1s) | 3 (0.1s) | 3 (0.1s) | 3 (34.6s) | 3 (0.1s) | 3 (0.0s) |
| undirected_n8_p0.5_s3 | 7/2 | 4 (0.0s) | 7/2 (0.0s) | 4 (0.1s) | 7/2 (26.9s) | 7/2 (0.1s) | 7/2 (0.0s) |
| undirected_n9_p0.3_s1 | 4 | 4 (0.2s) | 4 (0.2s) | 4 (0.2s) | 4 (45.3s) | 4 (0.2s) | 4 (0.0s) |
| undirected_n9_p0.3_s2 | 4 | 4 (0.2s) | 4 (0.2s) | 4 (0.3s) | 4 (45.3s) | 4 (0.2s) | 4 (0.0s) |
| undirected_n9_p0.3_s3 | 4 | 4 (0.1s) | 4 (0.2s) | 4 (0.2s) | 4 (45.2s) | 4 (0.1s) | 4 (0.0s) |
| undirected_n9_p0.5_s1 | 4 | 4 (0.2s) | 4 (0.2s) | 4 (0.3s) | 4 (45.3s) | 4 (0.2s) | 4 (0.0s) |
| undirected_n9_p0.5_s2 | 4 | 4 (0.2s) | 4 (0.2s) | 4 (0.3s) | 4 (45.3s) | 4 (0.3s) | 4 (0.0s) |
| undirected_n9_p0.5_s3 | 4 | 4 (0.1s) | 4 (0.2s) | 4 (0.2s) | 4 (45.2s) | 4 (0.2s) | 4 (0.0s) |
