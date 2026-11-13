# Main results, qqp, B = 64 (held-out split)


## Decoder-like cost (ρ₆₄≈95)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 91.08 | 0.00 | 1.00× | — | 61.8 | 1.00× |
| Per-sample threshold | 89.90 | 1.19 | 0.94× | — | 59.2 | 1.00× |
| Wait-for-all (no compaction) | 89.90 | 1.19 | 0.94× | — | 59.2 | — |
| Majority vote (DREX) | 90.14 | 0.95 | 1.66× | — | 57.2 | 1.04× |
| Consensus (DREX) | 90.59 | 0.49 | 1.25× | — | 60.2 | 1.17× |
| Unified exit (SkipDecode-style) | 89.89 | 1.20 | 1.87× | — | 54.9 | 1.00× |
| CeeBERT-style bandit | 90.10 | 0.98 | 0.94× | — | 59.9 | 1.00× |
| Difficulty lanes | 89.73 | 1.36 | 1.19× | — | — | — |
| Random search (150) | 89.47 | 1.61 | 1.95× | — | 53.5 | 1.02× |
| Theory rule (ours) | 89.57 | 1.51 | 1.99× | — | 55.0 | 1.01× |
| Theory rule + rescue (ours) | 89.72 | 1.36 | 1.48× | — | 54.4 | 1.04× |
| Theory + learned rescue (ours) | 89.58 | 1.50 | 1.99× | — | 55.0 | 1.01× |
| Label-free bandit (ours) | 91.02 | 0.06 | 1.36× | — | 61.3 | 1.03× |
| Bandit + rescue (ours) | 89.91 | 1.18 | 1.36× | — | 54.0 | 1.03× |

## Encoder cost (measured, ρ₆₄≈0.3)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 91.08 | 0.00 | 1.00× | — | 61.8 | 1.00× |
| Per-sample threshold | 89.90 | 1.19 | 1.54× | — | 59.2 | 1.01× |
| Wait-for-all (no compaction) | 89.90 | 1.19 | 0.90× | — | 59.2 | — |
| Majority vote (DREX) | 90.14 | 0.95 | 1.59× | — | 57.2 | 1.04× |
| Consensus (DREX) | 90.59 | 0.49 | 1.19× | — | 60.2 | 1.17× |
| Unified exit (SkipDecode-style) | 89.89 | 1.20 | 1.79× | — | 54.9 | 1.00× |
| CeeBERT-style bandit | 90.10 | 0.98 | 1.52× | — | 59.9 | 1.01× |
| Difficulty lanes | 89.73 | 1.36 | 1.73× | — | — | — |
| Random search (150) | 89.48 | 1.60 | 2.22× | — | 55.1 | 1.02× |
| Theory rule (ours) | 89.74 | 1.34 | 2.20× | — | 57.1 | 1.02× |
| Theory rule + rescue (ours) | 89.69 | 1.40 | 2.13× | — | 57.0 | 1.03× |
| Theory + learned rescue (ours) | 89.76 | 1.32 | 2.20× | — | 57.2 | 1.02× |
| Label-free bandit (ours) | 90.81 | 0.27 | 1.41× | — | 60.4 | 1.04× |
| Bandit + rescue (ours) | 90.76 | 0.32 | 1.41× | — | 59.9 | 1.04× |

## Real GPU runner cost, qqp

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 91.08 | 0.00 | 1.00× | 1.00× | 61.8 | 1.00× |
| Per-sample threshold | 89.90 | 1.19 | 1.63× | 1.63× | 59.2 | 1.01× |
| Wait-for-all (no compaction) | 89.90 | 1.19 | 0.93× | 0.94× | 59.2 | — |
| Majority vote (DREX) | 90.14 | 0.95 | 1.66× | 1.60× | 57.2 | 1.04× |
| Consensus (DREX) | 90.59 | 0.49 | 1.24× | 1.23× | 60.2 | 1.17× |
| Unified exit (SkipDecode-style) | 89.89 | 1.20 | 1.87× | 1.81× | 54.9 | 1.00× |
| CeeBERT-style bandit | 90.10 | 0.98 | 1.61× | 1.60× | 59.9 | 1.01× |
| Difficulty lanes | 89.73 | 1.36 | 1.82× | — | — | — |
| Random search (150) | 89.48 | 1.60 | 2.33× | 2.30× | 55.1 | 1.02× |
| Theory rule (ours) | 89.79 | 1.29 | 2.27× | 2.16× | 57.1 | 1.03× |
| Theory rule + rescue (ours) | 89.54 | 1.54 | 2.27× | 2.13× | 56.3 | 1.03× |
| Theory + learned rescue (ours) | 89.81 | 1.27 | 2.27× | 2.16× | 57.2 | 1.03× |
| Label-free bandit (ours) | 90.78 | 0.30 | 1.56× | 1.49× | 60.1 | 1.06× |
| Bandit + rescue (ours) | 90.68 | 0.41 | 1.56× | 1.48× | 59.5 | 1.06× |
