# Gap agent v2 (parallel: search + structure/exact + lambda_2 block codes)

28 unseen graphs (3000-10000 receivers), 30 s each. Gap integral = area under gap(t)/gap(0), same reference for all three set-ups (lower = closed sooner). 'pinned' = lower bound = upper bound, i.e. beta = lambda_2 proven. All certificates re-checked.

| family | set-up | mean gap integral | final gap (sum) | pinned | median time to pin | verified |
|---|---|---|---|---|---|---|
| communities | v1, 1 core | 0.641 | 3919.0 | 0/6 | - | 6/6 |
| communities | v1 portfolio, 3 cores | 0.622 | 3842.0 | 0/6 | - | 6/6 |
| communities | **v2, 3 workers** | 0.649 | 3886.0 | 0/6 | - | 6/6 |
| sparse | v1, 1 core | 0.102 | 144.0 | 0/6 | - | 6/6 |
| sparse | v1 portfolio, 3 cores | 0.090 | 114.0 | 1/6 | 6.1 s | 6/6 |
| sparse | **v2, 3 workers** | 0.098 | 150.0 | 0/6 | - | 6/6 |
| random | v1, 1 core | 0.879 | 4337.0 | 0/6 | - | 6/6 |
| random | v1 portfolio, 3 cores | 0.876 | 4318.0 | 0/6 | - | 6/6 |
| random | **v2, 3 workers** | 0.878 | 4323.0 | 0/6 | - | 6/6 |
| planted | v1, 1 core | 0.804 | 3587.0 | 0/6 | - | 6/6 |
| planted | v1 portfolio, 3 cores | 0.799 | 3575.0 | 0/6 | - | 6/6 |
| planted | **v2, 3 workers** | 0.799 | 3597.0 | 0/6 | - | 6/6 |
| layered | v1, 1 core | 0.768 | 2536.0 | 0/4 | - | 4/4 |
| layered | v1 portfolio, 3 cores | 0.641 | 2166.0 | 0/4 | - | 4/4 |
| layered | **v2, 3 workers** | 0.713 | 2196.0 | 0/4 | - | 4/4 |

## What v2's structure and lambda workers contributed

| graph | SCCs | SCCs closed (receivers) | acyclic kernel: receivers before -> after | block-code merges | pieces through lambda-agent | SCCs whose upper bound beats the clique cover | v1 final | v2 final |
|---|---|---|---|---|---|---|---|---|
| communities_n3000_0 | 81 | 38 (215) | 0 -> 0 | 4 | 15 | 3 | 529 .. 845 | 531 .. 842 |
| communities_n3000_1 | 74 | 38 (285) | 0 -> 0 | 9 | 8 | 6 | 551 .. 858 | 553 .. 851 |
| communities_n6000_0 | 152 | 69 (323) | 0 -> 0 | 8 | 8 | 2 | 930 .. 1558 | 933 .. 1569 |
| communities_n6000_1 | 145 | 64 (433) | 0 -> 0 | 8 | 28 | 2 | 1076 .. 1702 | 2181/2 .. 3415/2 |
| communities_n10000_0 | 22 | 7 (46) | 9975 -> 9975 | 6 | 5 | 0 | 1686 .. 2702 | 1698 .. 2715 |
| communities_n10000_1 | 53 | 24 (263) | 9936 -> 9916 | 5 | 10 | 3 | 1622 .. 2648 | 3251/2 .. 5265/2 |
| sparse_n3000_0 | 35 | 34 (34) | 2966 -> 2574 | 0 | 0 | 0 | 1768 .. 1801 | 1761 .. 1802 |
| sparse_n3000_1 | 183 | 182 (186) | 2814 -> 0 | 0 | 4 | 0 | 1935 .. 1942 | 1941 .. 1942 |
| sparse_n6000_0 | 328 | 327 (332) | 5668 -> 3272 | 0 | 5 | 0 | 4039 .. 4107 | 4041 .. 4107 |
| sparse_n6000_1 | 132 | 131 (132) | 5868 -> 2393 | 0 | 1 | 0 | 3373 .. 3400 | 3366 .. 3402 |
| sparse_n10000_0 | 3327 | 3326 (4149) | 5851 -> 0 | 0 | 555 | 0 | 7088 .. 7090 | 7089 .. 7091 |
| sparse_n10000_1 | 2289 | 2288 (2603) | 7397 -> 0 | 0 | 249 | 0 | 6904 .. 6911 | 6909 .. 6913 |
| random_n3000_0 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 24 .. 461 | 24 .. 458 |
| random_n3000_1 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 12 .. 228 | 13 .. 228 |
| random_n6000_0 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 21 .. 779 | 22 .. 776 |
| random_n6000_1 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 33 .. 1196 | 32 .. 1196 |
| random_n10000_0 | 1 | 0 (0) | 10000 -> 10000 | 0 | 0 | 0 | 18 .. 967 | 18 .. 958 |
| random_n10000_1 | 1 | 0 (0) | 10000 -> 10000 | 0 | 0 | 0 | 15 .. 829 | 15 .. 831 |
| planted_n3000_0 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 55 .. 514 | 55 .. 513 |
| planted_n3000_1 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 22 .. 330 | 21 .. 328 |
| planted_n6000_0 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 45 .. 984 | 46 .. 989 |
| planted_n6000_1 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 44 .. 960 | 46 .. 968 |
| planted_n10000_0 | 1 | 0 (0) | 10000 -> 10000 | 0 | 0 | 0 | 49 .. 256 | 48 .. 256 |
| planted_n10000_1 | 1 | 0 (0) | 10000 -> 10000 | 0 | 0 | 0 | 15 .. 773 | 15 .. 774 |
| layered_n3000_0 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 164 .. 604 | 165 .. 606 |
| layered_n3000_1 | 1 | 0 (0) | 3000 -> 3000 | 0 | 0 | 0 | 145 .. 421 | 140 .. 426 |
| layered_n6000_0 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 51 .. 1269 | 375 .. 1254 |
| layered_n6000_1 | 1 | 0 (0) | 6000 -> 6000 | 0 | 0 | 0 | 246 .. 848 | 256 .. 846 |

Missing or failed runs: layered_n10000_0, layered_n10000_1
