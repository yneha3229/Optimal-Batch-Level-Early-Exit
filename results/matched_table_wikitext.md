# Matched-accuracy comparison, wikitext, B = 64 (held-out Pareto fronts)

Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.


## Decoder-like cost (ρ₆₄≈95) (full model 49.15%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Full model | 1.00× | 1.00× | 1.00× |
| Robust theory, m=B/16 (ours) | 0.99× | 0.99× | 0.99× |
| Majority vote (DREX) | 0.98× | 0.99× | 1.01× |
| Robust theory, m=B/8 (ours) | 0.98× | 0.98× | 0.98× |
| Theory, trajectory-aware (ours) | 0.97× | 0.97× | 0.97× |
| Random search (150) | 0.97× | 0.97× | 0.97× |
| Theory, individual exits only | 0.97× | 0.97× | 0.97× |
| Consensus (DREX) | 0.97× | 0.97× | 0.97× |
| Label-free bandit (ours) | 0.97× | 0.97× | 0.97× |
| Label-free bandit, 3% target (ours) | 0.97× | 0.97× | 0.97× |
| Theory rule (ours) | 0.97× | 0.97× | 0.97× |
| Per-sample threshold | 0.97× | 0.97× | 0.97× |
| Wait-for-all (no compaction) | 0.97× | 0.97× | 0.97× |
| Unified exit (SkipDecode-style) | 0.97× | 0.97× | 0.97× |
| CeeBERT-style bandit | — | — | — |
| Greedy (DREX) | — | — | — |

## Real GPU runner cost, wikitext (full model 49.15%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Full model | 1.00× | 1.00× | 1.00× |
| Robust theory, m=B/16 (ours) | 0.94× | 0.97× | 0.97× |
| Majority vote (DREX) | 0.94× | 0.96× | 0.97× |
| Robust theory, m=B/8 (ours) | 0.94× | 0.95× | 0.95× |
| Theory, trajectory-aware (ours) | 0.94× | 0.94× | 0.94× |
| Label-free bandit, 3% target (ours) | 0.94× | 0.94× | 0.94× |
| Theory, individual exits only | 0.94× | 0.94× | 0.94× |
| Consensus (DREX) | 0.94× | 0.94× | 0.94× |
| Label-free bandit (ours) | 0.94× | 0.94× | 0.94× |
| Theory rule (ours) | 0.94× | 0.94× | 0.94× |
| Unified exit (SkipDecode-style) | 0.94× | 0.94× | 0.94× |
| Random search (150) | 0.93× | 0.93× | 0.93× |
| Per-sample threshold | 0.92× | 0.92× | 0.92× |
| Wait-for-all (no compaction) | 0.92× | 0.92× | 0.92× |
| CeeBERT-style bandit | — | — | — |
| Greedy (DREX) | — | — | — |
