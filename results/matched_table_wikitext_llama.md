# Matched-accuracy comparison, wikitext_llama, B = 64 (held-out Pareto fronts)

Best held-out speedup among candidates within 0.5 / 1 / 2 points of the full model's held-out accuracy.


## Decoder-like cost (ρ₆₄≈95) (full model 52.99%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Majority vote (DREX) | 1.00× | 1.00× | 1.03× |
| Theory, trajectory-aware (ours) | 1.00× | 1.00× | 1.00× |
| Robust theory, m=B/8 (ours) | 0.98× | 1.00× | 1.00× |
| Full model | 1.00× | 1.00× | 1.00× |
| Robust theory, m=B/16 (ours) | 0.99× | 0.99× | 0.99× |
| Theory rule (ours) | 0.99× | 0.99× | 0.99× |
| Consensus (DREX) | 0.98× | 0.98× | 0.98× |
| Random search (150) | 0.98× | 0.98× | 0.98× |
| Theory, individual exits only | 0.98× | 0.98× | 0.98× |
| Label-free bandit, 3% target (ours) | 0.98× | 0.98× | 0.98× |
| Label-free bandit (ours) | 0.98× | 0.98× | 0.98× |
| Per-sample threshold | 0.98× | 0.98× | 0.98× |
| Wait-for-all (no compaction) | 0.98× | 0.98× | 0.98× |
| Unified exit (SkipDecode-style) | 0.98× | 0.98× | 0.98× |
| CeeBERT-style bandit | — | — | — |
| Greedy (DREX) | — | — | 1.03× |

## Real GPU runner cost, wikitext_llama (full model 52.99%)

| Policy | ≤0.5 pt | ≤1 pt | ≤2 pt |
|---|---|---|---|
| Full model | 1.00× | 1.00× | 1.00× |
| Majority vote (DREX) | 0.98× | 0.98× | 1.00× |
| Consensus (DREX) | 0.96× | 0.96× | 0.96× |
| Label-free bandit (ours) | 0.95× | 0.95× | 0.95× |
| Label-free bandit, 3% target (ours) | 0.95× | 0.95× | 0.95× |
| Theory rule (ours) | 0.95× | 0.95× | 0.95× |
| Unified exit (SkipDecode-style) | 0.95× | 0.95× | 0.95× |
| Theory, individual exits only | 0.95× | 0.95× | 0.95× |
| Robust theory, m=B/16 (ours) | 0.95× | 0.95× | 0.95× |
| Robust theory, m=B/8 (ours) | 0.95× | 0.95× | 0.95× |
| Random search (150) | 0.95× | 0.95× | 0.95× |
| Per-sample threshold | 0.94× | 0.94× | 0.94× |
| Wait-for-all (no compaction) | 0.94× | 0.94× | 0.94× |
| Theory, trajectory-aware (ours) | 0.92× | 0.92× | 0.92× |
| CeeBERT-style bandit | — | — | — |
| Greedy (DREX) | — | — | 1.01× |
