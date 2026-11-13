# Main results, mnli, B = 64 (held-out split)


## Decoder-like cost (ρ₆₄≈95)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 87.66 | 0.00 | 1.00× | — | 54.2 | 1.00× |
| Per-sample threshold | 86.64 | 1.02 | 0.94× | — | 54.0 | 1.00× |
| Wait-for-all (no compaction) | 86.64 | 1.02 | 0.94× | — | 54.0 | — |
| Majority vote (DREX) | 86.33 | 1.32 | 1.42× | — | 47.6 | 1.01× |
| Consensus (DREX) | 86.02 | 1.64 | 1.37× | — | 49.0 | 1.08× |
| Unified exit (SkipDecode-style) | 86.59 | 1.06 | 1.40× | — | 48.4 | 1.00× |
| CeeBERT-style bandit | 86.86 | 0.79 | 0.94× | — | 55.2 | 1.00× |
| Difficulty lanes | 85.08 | 2.57 | 0.94× | — | — | — |
| Random search (150) | 86.40 | 1.26 | 1.42× | — | 50.0 | 1.01× |
| Theory rule (ours) | 86.25 | 1.41 | 1.43× | — | 48.4 | 1.01× |
| Theory rule + rescue (ours) | 86.14 | 1.52 | 1.41× | — | 49.5 | 1.00× |
| Theory + learned rescue (ours) | 86.22 | 1.44 | 1.43× | — | 48.1 | 1.01× |
| Label-free bandit (ours) | 87.57 | 0.08 | 1.06× | — | 53.4 | 1.01× |
| Bandit + rescue (ours) | 86.94 | 0.72 | 1.06× | — | 50.7 | 1.01× |
| FunSearch best | 86.21 | 1.45 | 1.41× | — | 50.4 | 1.00× |

## Encoder cost (measured, ρ₆₄≈0.3)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 87.66 | 0.00 | 1.00× | 1.00× | 54.2 | 1.00× |
| Per-sample threshold | 86.64 | 1.02 | 1.28× | 1.39× | 54.0 | 1.01× |
| Wait-for-all (no compaction) | 86.64 | 1.02 | 0.90× | 0.92× | 54.0 | — |
| Majority vote (DREX) | 86.33 | 1.32 | 1.36× | 1.42× | 47.6 | 1.01× |
| Consensus (DREX) | 86.02 | 1.64 | 1.31× | 1.33× | 49.0 | 1.08× |
| Unified exit (SkipDecode-style) | 86.59 | 1.06 | 1.35× | 1.38× | 48.4 | 1.00× |
| CeeBERT-style bandit | 86.86 | 0.79 | 1.27× | 1.38× | 55.2 | 1.01× |
| Difficulty lanes | 85.08 | 2.57 | 1.32× | — | — | — |
| Random search (150) | 85.55 | 2.10 | 1.61× | 1.63× | 49.1 | 1.01× |
| Theory rule (ours) | 85.33 | 2.32 | 1.60× | 1.36× | 49.0 | 1.00× |
| Theory rule + rescue (ours) | 85.33 | 2.32 | 1.60× | 1.60× | 49.0 | 1.00× |
| Theory + learned rescue (ours) | 85.33 | 2.32 | 1.60× | 1.62× | 49.0 | 1.00× |
| Label-free bandit (ours) | 87.49 | 0.17 | 1.21× | 1.27× | 55.4 | 1.04× |
| Bandit + rescue (ours) | 87.34 | 0.31 | 1.21× | 1.26× | 54.6 | 1.04× |
| FunSearch best | 86.21 | 1.45 | 1.55× | — | 50.4 | 1.00× |

## Real GPU runner cost, mnli

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 87.66 | 0.00 | 1.00× | 1.00× | 54.2 | 1.00× |
| Per-sample threshold | 86.64 | 1.02 | 1.28× | 1.36× | 54.0 | 1.01× |
| Wait-for-all (no compaction) | 86.64 | 1.02 | 0.92× | 0.94× | 54.0 | — |
| Majority vote (DREX) | 86.33 | 1.32 | 1.40× | 1.43× | 47.6 | 1.01× |
| Consensus (DREX) | 86.02 | 1.64 | 1.35× | 1.35× | 49.0 | 1.08× |
| Unified exit (SkipDecode-style) | 86.59 | 1.06 | 1.38× | 1.39× | 48.4 | 1.00× |
| CeeBERT-style bandit | 86.86 | 0.79 | 1.27× | 1.36× | 55.2 | 1.01× |
| Difficulty lanes | 85.08 | 2.57 | 1.32× | — | — | — |
| Random search (150) | 85.55 | 2.10 | 1.63× | 1.68× | 49.1 | 1.01× |
| Theory rule (ours) | 85.89 | 1.77 | 1.60× | 1.63× | 49.4 | 1.00× |
| Theory rule + rescue (ours) | 85.89 | 1.77 | 1.60× | 1.62× | 49.4 | 1.00× |
| Theory + learned rescue (ours) | 85.89 | 1.77 | 1.60× | 1.62× | 49.4 | 1.00× |
| Label-free bandit (ours) | 87.48 | 0.18 | 1.22× | 1.26× | 55.3 | 1.04× |
| Bandit + rescue (ours) | 87.33 | 0.32 | 1.22× | 1.24× | 54.6 | 1.04× |
| FunSearch best | 86.21 | 1.45 | 1.57× | 1.62× | 50.4 | 1.00× |
