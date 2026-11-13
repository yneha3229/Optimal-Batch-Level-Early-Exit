# Matched-accuracy comparison, qqp, B = 64 (held-out Pareto fronts)

Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.


## Decoder-like cost (ρ₆₄≈95) (full model 91.08%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Robust theory, m=B/16 (ours) | 1.61× | 1.81× | 2.05× |
| Theory + learned rescue (ours) | 1.55× | 1.80× | 1.99× |
| Theory rule (ours) | 1.55× | 1.80× | 1.99× |
| Robust theory, m=B/8 (ours) | 1.58× | 1.79× | 1.94× |
| Theory, trajectory-aware (ours) | 1.56× | 1.79× | 1.98× |
| Majority vote (DREX) | 1.02× | 1.66× | 1.66× |
| Random search (150) | 1.36× | 1.65× | 1.95× |
| Unified exit (SkipDecode-style) | 1.40× | 1.40× | 1.87× |
| Label-free bandit (ours) | 1.36× | 1.36× | 1.36× |
| Consensus (DREX) | 1.25× | 1.25× | 1.45× |
| Theory rule + rescue (ours) | 1.05× | 1.25× | 1.66× |
| Difficulty lanes | 0.99× | 1.19× | 1.23× |
| Theory, individual exits only | 1.15× | 1.16× | 1.45× |
| Full model | 1.00× | 1.00× | 1.00× |
| CeeBERT-style bandit | 0.94× | 0.94× | 0.94× |
| Per-sample threshold | 0.94× | 0.94× | 0.94× |
| Wait-for-all (no compaction) | 0.94× | 0.94× | 0.94× |
| Bandit + rescue (ours) | — | — | 1.36× |
| Label-free bandit, 3% target (ours) | — | — | 1.82× |
| Greedy (DREX) | — | — | — |

## Encoder cost (measured, ρ₆₄≈0.3) (full model 91.08%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Theory + learned rescue (ours) | 1.85× | 2.03× | 2.34× |
| Theory rule (ours) | 1.76× | 2.03× | 2.34× |
| Robust theory, m=B/8 (ours) | 1.77× | 2.02× | 2.33× |
| Robust theory, m=B/16 (ours) | 1.85× | 2.00× | 2.31× |
| Random search (150) | 1.75× | 1.98× | 2.32× |
| Theory, trajectory-aware (ours) | 1.85× | 1.97× | 2.22× |
| Theory rule + rescue (ours) | 1.76× | 1.92× | 2.20× |
| Theory, individual exits only | 1.66× | 1.78× | 2.16× |
| Difficulty lanes | 1.39× | 1.63× | 1.85× |
| Majority vote (DREX) | 0.98× | 1.59× | 1.59× |
| CeeBERT-style bandit | 1.33× | 1.52× | 1.52× |
| Per-sample threshold | 1.41× | 1.50× | 1.64× |
| Bandit + rescue (ours) | 1.41× | 1.41× | 1.41× |
| Label-free bandit (ours) | 1.41× | 1.41× | 1.41× |
| Unified exit (SkipDecode-style) | 1.35× | 1.35× | 1.79× |
| Consensus (DREX) | 1.19× | 1.19× | 1.39× |
| Full model | 1.00× | 1.00× | 1.00× |
| Wait-for-all (no compaction) | 0.90× | 0.90× | 0.90× |
| Label-free bandit, 3% target (ours) | — | — | 1.98× |
| Greedy (DREX) | — | — | — |

## Real GPU runner cost, qqp (full model 91.08%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Robust theory, m=B/16 (ours) | 1.93× | 2.11× | 2.42× |
| Robust theory, m=B/8 (ours) | 1.89× | 2.11× | 2.47× |
| Theory rule (ours) | 1.84× | 2.10× | 2.44× |
| Theory + learned rescue (ours) | 1.95× | 2.10× | 2.44× |
| Random search (150) | 1.83× | 2.07× | 2.43× |
| Theory, trajectory-aware (ours) | 1.94× | 2.06× | 2.34× |
| Theory rule + rescue (ours) | 1.84× | 2.02× | 2.27× |
| Theory, individual exits only | 1.76× | 1.86× | 2.28× |
| Difficulty lanes | 1.47× | 1.71× | 1.96× |
| Majority vote (DREX) | 1.02× | 1.66× | 1.66× |
| CeeBERT-style bandit | 1.40× | 1.61× | 1.61× |
| Per-sample threshold | 1.49× | 1.59× | 1.75× |
| Bandit + rescue (ours) | 1.56× | 1.56× | 1.56× |
| Label-free bandit (ours) | 1.56× | 1.56× | 1.56× |
| Unified exit (SkipDecode-style) | 1.40× | 1.40× | 1.87× |
| Consensus (DREX) | 1.24× | 1.24× | 1.45× |
| Full model | 1.00× | 1.00× | 1.00× |
| Wait-for-all (no compaction) | 0.93× | 0.93× | 0.94× |
| Label-free bandit, 3% target (ours) | — | — | 2.22× |
| Greedy (DREX) | — | — | — |
