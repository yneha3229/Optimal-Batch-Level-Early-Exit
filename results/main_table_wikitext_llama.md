# Main results, wikitext_llama, B = 64 (held-out split)


## Decoder-like cost (ρ₆₄≈95)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 52.99 | 0.00 | 1.00× | — | 10.9 | 1.00× |
| Per-sample threshold | 51.98 | 1.01 | 0.98× | — | 10.9 | 1.00× |
| Wait-for-all (no compaction) | 51.98 | 1.01 | 0.98× | — | 10.9 | — |
| Majority vote (DREX) | 52.50 | 0.49 | 1.00× | — | 11.2 | 1.00× |
| Consensus (DREX) | 53.01 | -0.02 | 0.98× | — | 10.9 | 1.01× |
| Unified exit (SkipDecode-style) | 52.99 | 0.00 | 0.98× | — | 10.9 | 1.00× |
| Random search (150) | 51.97 | 1.02 | 0.98× | — | 10.8 | 1.00× |
| Theory rule (ours) | 52.67 | 0.32 | 0.99× | — | 10.6 | 1.00× |
| Label-free bandit (ours) | 52.99 | 0.00 | 0.98× | — | 10.9 | 1.00× |

## Real GPU runner cost, wikitext_llama

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 52.99 | 0.00 | 1.00× | 1.00× | 10.9 | 1.00× |
| Per-sample threshold | 52.99 | 0.00 | 0.94× | 0.92× | 10.9 | 1.00× |
| Wait-for-all (no compaction) | 52.99 | 0.00 | 0.94× | 0.92× | 10.9 | — |
| Majority vote (DREX) | 52.50 | 0.49 | 0.98× | 0.95× | 11.2 | 1.00× |
| Consensus (DREX) | 53.01 | -0.02 | 0.96× | 0.93× | 10.9 | 1.01× |
| Unified exit (SkipDecode-style) | 52.99 | 0.00 | 0.95× | 0.93× | 10.9 | 1.00× |
| Random search (150) | 52.99 | 0.00 | 0.95× | 0.93× | 10.9 | 1.00× |
| Theory rule (ours) | 52.99 | 0.00 | 0.95× | 0.92× | 10.9 | 1.00× |
| Label-free bandit (ours) | 52.99 | 0.00 | 0.95× | 0.92× | 10.9 | 1.00× |
