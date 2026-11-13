# Main results, cifar100, B = 64 (held-out split)


## Decoder-like cost (ρ₆₄≈95)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 93.57 | 0.00 | 1.00× | — | 60.3 | 1.00× |
| Per-sample threshold | 92.76 | 0.81 | 0.94× | — | 60.2 | 1.00× |
| Wait-for-all (no compaction) | 92.76 | 0.81 | 0.94× | — | 60.2 | — |
| Majority vote (DREX) | 92.89 | 0.69 | 1.02× | — | 57.2 | 1.00× |
| Consensus (DREX) | 93.55 | 0.02 | 0.94× | — | 60.1 | 1.01× |
| Unified exit (SkipDecode-style) | 93.57 | 0.00 | 0.94× | — | 60.3 | 1.00× |
| CeeBERT-style bandit | 92.78 | 0.79 | 0.94× | — | 60.2 | 1.00× |
| Difficulty lanes | 93.17 | 0.40 | 0.94× | — | — | — |
| Random search (150) | 92.30 | 1.27 | 1.03× | — | 57.1 | 1.05× |
| Theory rule (ours) | 92.39 | 1.19 | 1.07× | — | 52.8 | 1.03× |
| Theory rule + rescue (ours) | 92.45 | 1.12 | 1.02× | — | 55.9 | 1.04× |
| Theory + learned rescue (ours) | 92.49 | 1.08 | 1.07× | — | 53.3 | 1.03× |
| Label-free bandit (ours) | 93.55 | 0.02 | 0.94× | — | 60.2 | 1.01× |
| Bandit + rescue (ours) | 93.49 | 0.08 | 0.94× | — | 60.0 | 1.01× |

## Encoder cost (measured, ρ₆₄≈0.3)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 93.57 | 0.00 | 1.00× | — | 60.3 | 1.00× |
| Per-sample threshold | 92.76 | 0.81 | 1.24× | — | 60.2 | 1.01× |
| Wait-for-all (no compaction) | 92.76 | 0.81 | 0.90× | — | 60.2 | — |
| Majority vote (DREX) | 92.89 | 0.69 | 0.98× | — | 57.2 | 1.00× |
| Consensus (DREX) | 93.55 | 0.02 | 0.90× | — | 60.1 | 1.01× |
| Unified exit (SkipDecode-style) | 93.57 | 0.00 | 0.90× | — | 60.3 | 1.00× |
| CeeBERT-style bandit | 92.78 | 0.79 | 1.23× | — | 60.2 | 1.01× |
| Difficulty lanes | 93.17 | 0.40 | 1.17× | — | — | — |
| Random search (150) | 92.30 | 1.27 | 1.29× | — | 57.1 | 1.03× |
| Theory rule (ours) | 92.67 | 0.91 | 1.26× | — | 58.3 | 1.02× |
| Theory rule + rescue (ours) | 92.56 | 1.01 | 1.26× | — | 58.4 | 1.02× |
| Theory + learned rescue (ours) | 92.61 | 0.96 | 1.26× | — | 57.9 | 1.02× |
| Label-free bandit (ours) | 92.28 | 1.29 | 1.28× | — | 60.1 | 1.02× |
| Bandit + rescue (ours) | 92.25 | 1.32 | 1.28× | — | 60.2 | 1.02× |

## Real GPU runner cost, cifar100

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 93.57 | 0.00 | 1.00× | 1.00× | 60.3 | 1.00× |
| Per-sample threshold | 92.76 | 0.81 | 1.43× | 1.49× | 60.2 | 1.01× |
| Wait-for-all (no compaction) | 92.76 | 0.81 | 0.96× | 0.97× | 60.2 | — |
| Majority vote (DREX) | 92.89 | 0.69 | 1.05× | 1.05× | 57.2 | 1.00× |
| Consensus (DREX) | 93.55 | 0.02 | 0.96× | 0.98× | 60.1 | 1.01× |
| Unified exit (SkipDecode-style) | 93.57 | 0.00 | 0.96× | 0.98× | 60.3 | 1.00× |
| CeeBERT-style bandit | 92.78 | 0.79 | 1.43× | 1.49× | 60.2 | 1.01× |
| Difficulty lanes | 93.17 | 0.40 | 1.34× | — | — | — |
| Random search (150) | 92.30 | 1.27 | 1.47× | 1.51× | 57.1 | 1.03× |
| Theory rule (ours) | 92.64 | 0.94 | 1.46× | 1.47× | 58.4 | 1.01× |
| Theory rule + rescue (ours) | 92.64 | 0.94 | 1.46× | 1.46× | 58.5 | 1.01× |
| Theory + learned rescue (ours) | 92.62 | 0.95 | 1.46× | 1.47× | 58.3 | 1.01× |
| Label-free bandit (ours) | 92.31 | 1.26 | 1.52× | 1.53× | 60.6 | 1.01× |
| Bandit + rescue (ours) | 92.30 | 1.27 | 1.52× | 1.52× | 60.6 | 1.01× |
