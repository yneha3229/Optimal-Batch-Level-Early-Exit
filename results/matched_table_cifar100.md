# Matched-accuracy comparison, cifar100, B = 64 (held-out Pareto fronts)

Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.


## Decoder-like cost (ρ₆₄≈95) (full model 93.57%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Theory, trajectory-aware (ours) | 0.97× | 1.05× | 1.16× |
| Label-free bandit, 3% target (ours) | — | 1.03× | 1.03× |
| Robust theory, m=B/16 (ours) | 0.98× | 1.03× | 1.15× |
| Theory rule (ours) | 0.99× | 1.02× | 1.12× |
| Theory + learned rescue (ours) | 0.96× | 1.02× | 1.17× |
| Majority vote (DREX) | 0.96× | 1.02× | 1.11× |
| Robust theory, m=B/8 (ours) | 0.96× | 1.01× | 1.15× |
| Full model | 1.00× | 1.00× | 1.00× |
| Theory rule + rescue (ours) | 0.96× | 0.99× | 1.07× |
| Random search (150) | 0.96× | 0.99× | 1.14× |
| Label-free bandit (ours) | 0.94× | 0.94× | 0.94× |
| Bandit + rescue (ours) | 0.94× | 0.94× | 0.94× |
| Consensus (DREX) | 0.94× | 0.94× | 0.94× |
| Per-sample threshold | 0.94× | 0.94× | 0.94× |
| Difficulty lanes | 0.94× | 0.94× | 0.94× |
| Theory, individual exits only | 0.94× | 0.94× | 0.94× |
| CeeBERT-style bandit | 0.94× | 0.94× | 0.94× |
| Wait-for-all (no compaction) | 0.94× | 0.94× | 0.94× |
| Unified exit (SkipDecode-style) | 0.94× | 0.94× | 1.12× |
| Greedy (DREX) | — | — | — |

## Encoder cost (measured, ρ₆₄≈0.3) (full model 93.57%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Theory rule (ours) | 1.20× | 1.26× | 1.33× |
| Theory + learned rescue (ours) | 1.20× | 1.26× | 1.33× |
| Theory, trajectory-aware (ours) | 1.18× | 1.26× | 1.30× |
| Random search (150) | 1.19× | 1.25× | 1.31× |
| Per-sample threshold | 1.18× | 1.25× | 1.32× |
| Difficulty lanes | 1.17× | 1.24× | 1.31× |
| Theory, individual exits only | 1.20× | 1.23× | 1.30× |
| CeeBERT-style bandit | 1.14× | 1.23× | 1.23× |
| Theory rule + rescue (ours) | 1.20× | 1.23× | 1.33× |
| Robust theory, m=B/16 (ours) | 1.08× | 1.22× | 1.33× |
| Robust theory, m=B/8 (ours) | 0.94× | 1.05× | 1.29× |
| Full model | 1.00× | 1.00× | 1.00× |
| Majority vote (DREX) | 0.92× | 0.98× | 1.06× |
| Consensus (DREX) | 0.90× | 0.90× | 0.90× |
| Wait-for-all (no compaction) | 0.90× | 0.90× | 0.90× |
| Unified exit (SkipDecode-style) | 0.90× | 0.90× | 1.08× |
| Label-free bandit (ours) | — | — | 1.28× |
| Bandit + rescue (ours) | — | — | 1.28× |
| Label-free bandit, 3% target (ours) | — | — | — |
| Greedy (DREX) | — | — | — |

## Real GPU runner cost, cifar100 (full model 93.57%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Theory rule (ours) | 1.38× | 1.46× | 1.57× |
| Theory rule + rescue (ours) | 1.38× | 1.46× | 1.55× |
| Theory + learned rescue (ours) | 1.38× | 1.46× | 1.57× |
| Theory, trajectory-aware (ours) | 1.34× | 1.46× | 1.52× |
| Theory, individual exits only | 1.38× | 1.46× | 1.53× |
| Random search (150) | 1.36× | 1.46× | 1.55× |
| Per-sample threshold | 1.34× | 1.45× | 1.56× |
| Difficulty lanes | 1.34× | 1.43× | 1.54× |
| CeeBERT-style bandit | 1.29× | 1.43× | 1.43× |
| Robust theory, m=B/16 (ours) | 1.28× | 1.40× | 1.54× |
| Robust theory, m=B/8 (ours) | 1.09× | 1.12× | 1.48× |
| Majority vote (DREX) | 0.98× | 1.05× | 1.14× |
| Full model | 1.00× | 1.00× | 1.00× |
| Consensus (DREX) | 0.96× | 0.96× | 0.96× |
| Wait-for-all (no compaction) | 0.96× | 0.96× | 0.96× |
| Unified exit (SkipDecode-style) | 0.96× | 0.96× | 1.15× |
| Label-free bandit (ours) | — | — | 1.52× |
| Bandit + rescue (ours) | — | — | 1.52× |
| Label-free bandit, 3% target (ours) | — | — | — |
| Greedy (DREX) | — | — | — |
