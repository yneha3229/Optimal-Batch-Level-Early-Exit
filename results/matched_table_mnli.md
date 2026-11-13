# Matched-accuracy comparison, mnli, B = 64 (held-out Pareto fronts)

Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.


## Decoder-like cost (ρ₆₄≈95) (full model 87.66%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Robust theory, m=B/8 (ours) | — | 1.39× | 1.45× |
| Majority vote (DREX) | 0.94× | 1.39× | 1.42× |
| Random search (150) | 1.00× | 1.37× | 1.42× |
| Robust theory, m=B/16 (ours) | — | 1.36× | 1.41× |
| Theory rule (ours) | — | 1.36× | 1.43× |
| Theory, trajectory-aware (ours) | — | 1.35× | 1.43× |
| Theory + learned rescue (ours) | — | 1.33× | 1.43× |
| Label-free bandit, 3% target (ours) | 1.19× | 1.19× | 1.19× |
| Consensus (DREX) | 1.08× | 1.15× | 1.37× |
| Unified exit (SkipDecode-style) | 1.12× | 1.12× | 1.40× |
| Label-free bandit (ours) | 1.06× | 1.06× | 1.06× |
| Bandit + rescue (ours) | — | 1.06× | 1.06× |
| Theory rule + rescue (ours) | — | 1.05× | 1.43× |
| Full model | 1.00× | 1.00× | 1.00× |
| Theory, individual exits only | — | 1.00× | 1.41× |
| CeeBERT-style bandit | 0.94× | 0.94× | 0.94× |
| Per-sample threshold | 0.94× | 0.94× | 0.94× |
| Difficulty lanes | — | 0.94× | 0.94× |
| Wait-for-all (no compaction) | 0.94× | 0.94× | 0.94× |
| FunSearch best | — | — | 1.41× |
| FunSearch best + rescue | — | — | — |
| Greedy (DREX) | — | — | — |

## Encoder cost (measured, ρ₆₄≈0.3) (full model 87.66%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Random search (150) | 1.24× | 1.46× | 1.55× |
| Theory, trajectory-aware (ours) | — | 1.41× | 1.57× |
| Label-free bandit, 3% target (ours) | — | 1.39× | 1.39× |
| Robust theory, m=B/16 (ours) | — | 1.37× | 1.58× |
| Robust theory, m=B/8 (ours) | — | 1.36× | 1.58× |
| Theory + learned rescue (ours) | — | 1.36× | 1.58× |
| Theory rule (ours) | — | 1.36× | 1.58× |
| Majority vote (DREX) | 0.90× | 1.33× | 1.36× |
| Theory rule + rescue (ours) | — | 1.32× | 1.58× |
| CeeBERT-style bandit | 1.22× | 1.27× | 1.27× |
| Per-sample threshold | 1.24× | 1.26× | 1.32× |
| Theory, individual exits only | — | 1.23× | 1.58× |
| Label-free bandit (ours) | 1.21× | 1.21× | 1.21× |
| Bandit + rescue (ours) | 1.21× | 1.21× | 1.21× |
| Difficulty lanes | — | 1.18× | 1.28× |
| Consensus (DREX) | 1.04× | 1.11× | 1.31× |
| Unified exit (SkipDecode-style) | 1.08× | 1.08× | 1.35× |
| Full model | 1.00× | 1.00× | 1.00× |
| Wait-for-all (no compaction) | 0.90× | 0.90× | 0.90× |
| FunSearch best | — | — | 1.55× |
| FunSearch best + rescue | — | — | — |
| Greedy (DREX) | — | — | — |

## Real GPU runner cost, mnli (full model 87.66%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Random search (150) | 1.24× | 1.49× | 1.58× |
| Theory, trajectory-aware (ours) | — | 1.46× | 1.60× |
| Theory rule (ours) | — | 1.41× | 1.60× |
| Robust theory, m=B/8 (ours) | — | 1.40× | 1.62× |
| Theory + learned rescue (ours) | — | 1.40× | 1.60× |
| Label-free bandit, 3% target (ours) | — | 1.39× | 1.39× |
| Robust theory, m=B/16 (ours) | — | 1.39× | 1.60× |
| Majority vote (DREX) | 0.93× | 1.37× | 1.40× |
| Theory rule + rescue (ours) | — | 1.35× | 1.60× |
| CeeBERT-style bandit | 1.23× | 1.27× | 1.27× |
| Per-sample threshold | 1.24× | 1.26× | 1.32× |
| Theory, individual exits only | — | 1.24× | 1.60× |
| Label-free bandit (ours) | 1.22× | 1.22× | 1.22× |
| Bandit + rescue (ours) | 1.22× | 1.22× | 1.22× |
| Difficulty lanes | — | 1.19× | 1.28× |
| Consensus (DREX) | 1.06× | 1.14× | 1.35× |
| Unified exit (SkipDecode-style) | 1.11× | 1.11× | 1.38× |
| Full model | 1.00× | 1.00× | 1.00× |
| Wait-for-all (no compaction) | 0.92× | 0.92× | 0.92× |
| FunSearch best | — | — | 1.57× |
| FunSearch best + rescue | — | — | — |
| Greedy (DREX) | — | — | — |
