"""Spearman correlation of SAT time with fixed-set features (pincer/fixset_study.py output).
    python -m pincer.fixset_analysis pincer/results/fixset_*.jsonl"""
import json
import sys

from scipy.stats import spearmanr

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path)]
    secs = [r["seconds"] for r in rows]
    print(f"{rows[0]['graph']}: {len(rows)} sets, |I| = {rows[0]['size']}, SAT time min {min(secs):.1f} "
          f"median {sorted(secs)[len(secs) // 2]:.1f} max {max(secs):.1f} s, timeouts "
          f"{sum(r['status'] == 'timeout' for r in rows)}")
    for k in ("deg_mean", "deg_max", "deg_min", "covered", "private", "nbr_mean", "nbr_max"):
        rho, p = spearmanr([r[k] for r in rows], secs)
        print(f"   {k:9s} rho = {rho:+.2f}  (p = {p:.3f})")
