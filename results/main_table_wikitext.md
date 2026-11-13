# Main results, wikitext, B = 64 (held-out split)


## Decoder-like cost (ρ₆₄≈95)

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 49.15 | 0.00 | 1.00× | — | 9.7 | 1.00× |
| Per-sample threshold | 48.04 | 1.10 | 0.97× | — | 9.8 | 1.00× |
| Wait-for-all (no compaction) | 48.04 | 1.10 | 0.97× | — | 9.8 | — |
| Majority vote (DREX) | 48.51 | 0.64 | 0.99× | — | 9.3 | 1.00× |
| Consensus (DREX) | 49.15 | 0.00 | 0.97× | — | 9.7 | 1.00× |
| Unified exit (SkipDecode-style) | 49.15 | 0.00 | 0.97× | — | 9.7 | 1.00× |
| Random search (150) | 48.02 | 1.12 | 0.97× | — | 9.8 | 1.00× |
| Theory rule (ours) | 49.15 | 0.00 | 0.97× | — | 9.7 | 1.00× |
| Label-free bandit (ours) | 49.15 | 0.00 | 0.97× | — | 9.7 | 1.00× |

## Real GPU runner cost, wikitext

| Policy | Accuracy | Acc. drop (pt) | Sim speedup | Real GPU speedup | Hardest 10% acc | Victim slowdown, 1 attacker |
|---|---|---|---|---|---|---|
| Full model | 49.15 | 0.00 | 1.00× | 1.00× | 9.7 | 1.00× |
| Per-sample threshold | 49.15 | 0.00 | 0.92× | 0.92× | 9.7 | 1.00× |
| Wait-for-all (no compaction) | 49.15 | 0.00 | 0.92× | 0.90× | 9.7 | — |
| Majority vote (DREX) | 48.51 | 0.64 | 0.96× | 0.93× | 9.3 | 1.00× |
| Consensus (DREX) | 49.15 | 0.00 | 0.94× | 0.91× | 9.7 | 1.00× |
| Unified exit (SkipDecode-style) | 49.15 | 0.00 | 0.94× | 0.91× | 9.7 | 1.00× |
| Random search (150) | 49.15 | 0.00 | 0.93× | 0.91× | 9.7 | 1.00× |
| Theory rule (ours) | 49.15 | 0.00 | 0.94× | 0.90× | 9.7 | 1.00× |
| Label-free bandit (ours) | 49.15 | 0.00 | 0.94× | 0.90× | 9.7 | 1.00× |
