"""Quick check of the robust (trimmed) and trajectory-aware theory rules: clean speedup at matched accuracy and
victim latency under k slowdown inputs, B = 64. Usage: ~/venv/bin/python experiments/robust_check.py --task cifar100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_main import all_costs  # noqa: E402
from simulator import policies as P  # noqa: E402
from simulator.attacks import hardest_pool, victim_experiment  # noqa: E402
from simulator.simulator import load_traces, simulate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cifar100")
    ap.add_argument("--B", type=int, default=64)
    args = ap.parse_args()
    path = ROOT / f"traces/{args.task}_val.parquet"
    ev, ho = load_traces(path, "evolve"), load_traces(path, "heldout")
    att_p = ROOT / f"traces/{args.task}_attack.parquet"
    if att_p.exists():
        attack, clean, kind = load_traces(att_p), ho, "deepsloth"
    else:
        pool = hardest_pool(ho, 0.02)
        attack, clean, kind = ho.subset(pool), ho.subset(np.setdiff1d(np.arange(len(ho)), pool)), "hardest_2pct"
    c1, c2 = P.Calibration(ev), P.Calibration2D(ev)
    B = args.B
    lams = np.logspace(-3.5, 1.5, 50)
    rows = []
    for reg in ("enc_eager_meas", "dec_eager"):
        cost = all_costs(args.task)[reg]
        full_ev = simulate(P.never_exit(), ev, cost, B, 150, seed=1, exits_enabled=False)["accuracy"]
        full_ho = simulate(P.never_exit(), ho, cost, B, 150, seed=2, exits_enabled=False)["accuracy"]
        variants = {"theory": (c1, 0.0), "theory_robust": (c1, 1 / 16), "theory_hist": (c2, 0.0),
                    "theory_hist_robust": (c2, 1 / 16)}
        for name, (cal, tf) in variants.items():
            best = None
            for lam in lams:
                d = P.theory_rule(lam, cal, ev.exit_layers, cost, trim_frac=tf)
                r = simulate(d, ev, cost, B, 150, seed=1)
                if r["accuracy"] >= full_ev - 0.01 and (best is None or r["speedup"] > best[1]):
                    best = (lam, r["speedup"])
            d = P.theory_rule(best[0], cal, ev.exit_layers, cost, trim_frac=tf)
            h = simulate(d, ho, cost, B, 150, seed=2)
            att = victim_experiment(d, clean, attack, cost, B, ks=(0, 1, 2, 4, 8), n_batches=150, seed=7)
            rows.append({"regime": reg, "policy": name, "lam": best[0], "ho_speedup": h["speedup"],
                         "ho_acc_drop": full_ho - h["accuracy"],
                         **{f"victim_k{a['k']}": a["victim_rel_full"] for a in att}})
        for name, d in [("majority", None), ("threshold", None)]:
            fam = P.majority if name == "majority" else P.threshold
            best = None
            for tau in np.linspace(0.4, 0.995, 40):
                r = simulate(fam(tau), ev, cost, B, 150, seed=1)
                if r["accuracy"] >= full_ev - 0.01 and (best is None or r["speedup"] > best[1]):
                    best = (tau, r["speedup"])
            h = simulate(fam(best[0]), ho, cost, B, 150, seed=2)
            att = victim_experiment(fam(best[0]), clean, attack, cost, B, ks=(0, 1, 2, 4, 8), n_batches=150, seed=7)
            rows.append({"regime": reg, "policy": name, "lam": best[0], "ho_speedup": h["speedup"],
                         "ho_acc_drop": full_ho - h["accuracy"],
                         **{f"victim_k{a['k']}": a["victim_rel_full"] for a in att}})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(f"task {args.task}, B={B}, attack inputs: {kind}")
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
