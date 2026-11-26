"""Certified, detection-free robustness of the m-trimmed theory rule, and its price.

Theorem (pathwise). Let C be the clean inputs of a batch and K the k <= m slowdown inputs. Every input's gain depends
only on its own confidences, the trimmed sum satisfies TS_m(C u K) <= TS_{m-k}(C), and the remnant cost grows with
|K|. Hence, at every exit, if the clean batch would stop under the (m-k)-trimmed rule, the attacked batch stops under
the m-trimmed rule too:
        depth(C u K; m)  <=  depth(C; m - k)        for every batch, every k <= m.
So the victims' depth (and latency, up to b*k per layer) is certified by a clean-batch quantity, with no attacker
detection. The price: the m-trimmed rule ignores the m largest gains on clean batches too.

This script (a) checks the inequality on every simulated batch (violations must be 0), (b) reports victims' latency
vs k for m in {0, B/16, B/8} and the baselines, (c) reports clean held-out speedup at matched accuracy for each m.
Usage: ~/venv/bin/python experiments/robust_bound.py --task cifar100 --regime dec_eager
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_main import TASK_EXITS, all_costs  # noqa: E402
from simulator import policies as P  # noqa: E402
from simulator.attacks import hardest_pool, victim_experiment  # noqa: E402
from simulator.simulator import Traces, load_traces, run_batch, simulate  # noqa: E402
from scipy import stats  # noqa: E402

ALPHA, DELTA = 0.01, 0.10            # accuracy budget and failure probability, as in experiments/local_hw.py


def tune(make, ev, cost, B, grid, full_ev):
    best = None
    for x in grid:
        r = simulate(make(x), ev, cost, B, 150, seed=1)
        if r["accuracy"] >= full_ev - 0.01 and (best is None or r["speedup"] > best[1]):
            best = (x, r["speedup"])
    return best[0]


def tune_cert(make, fit, cert, cost, B, grid, compact=True, alpha=ALPHA, delta=DELTA):
    """Certified tuning (as local_hw.select_family): order the grid by speedup on the fit split (safest first),
    then fixed-sequence testing on disjoint batches of the cert split: certify while the one-sided t-test on
    per-batch accuracy drop rejects 'drop > alpha' at level delta; return the last certified value (None if none).
    Valid at level delta without a multiple-testing correction because the order is fixed before cert is seen."""
    sp = [simulate(make(x), fit, cost, B, 100, seed=1)["speedup"] for x in grid]
    order = np.argsort(sp, kind="stable")
    rows = np.random.default_rng(3).permutation(len(cert))
    batches = [rows[i * B:(i + 1) * B] for i in range(len(cert) // B)]
    full_ok = [float((cert.pred[r, -1] == cert.label[r]).mean()) for r in batches]
    best = None
    for i in order:
        d = np.array([f - float((run_batch(make(grid[i]), cert, r, cost, compact=compact).pred == cert.label[r]).mean())
                      for f, r in zip(full_ok, batches)])
        m, sd, K = d.mean(), d.std(ddof=1), len(d)
        ok = m < alpha if sd == 0 else (alpha - m) / (sd / np.sqrt(K)) > stats.t.ppf(1 - delta, K - 1)
        if not ok:
            break
        best = grid[i]
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cifar100")
    ap.add_argument("--regime", default="dec_eager")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--batches", type=int, default=300)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 4, 8],
                    help="attacker counts; values above m test an attacker outside the certificate")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    ap.add_argument("--tune", choices=["naive", "cert"], default="naive",
                    help="naive: fastest value within 1 point on the whole evolve split (not accuracy-matched on "
                         "held-out); cert: certified selection on a fit 40%% / cert 60%% split of evolve")
    args = ap.parse_args()
    B, ex = args.B, TASK_EXITS.get(args.task)
    path = ROOT / f"traces/{args.task}_val.parquet"
    ev, ho = load_traces(path, "evolve", exit_layers=ex), load_traces(path, "heldout", exit_layers=ex)
    att_p = ROOT / f"traces/{args.task}_attack.parquet"
    if att_p.exists():
        attack, clean, kind = load_traces(att_p, exit_layers=ex), ho, "deepsloth"
    else:
        pool = hardest_pool(ho, 0.02)
        attack, clean, kind = ho.subset(pool), ho.subset(np.setdiff1d(np.arange(len(ho)), pool)), "hardest_2pct"
    cost = all_costs(args.task)[args.regime]
    compact = args.task not in ("wikitext",)
    if args.tune == "cert":
        perm = np.random.default_rng(0).permutation(len(ev))
        fit, cert = ev.subset(perm[: len(ev) * 2 // 5]), ev.subset(perm[len(ev) * 2 // 5:])
    cal = P.Calibration(fit if args.tune == "cert" else ev)
    full_ev = simulate(P.never_exit(), ev, cost, B, 150, seed=1, exits_enabled=False)["accuracy"]
    full_ho = simulate(P.never_exit(), ho, cost, B, 150, seed=2, exits_enabled=False)["accuracy"]
    lams = np.logspace(-3.5, 1.5, 50)
    ks = tuple(args.ks)
    rows, viol_rows = [], []
    for m in (0, B // 16, B // 8):
        tf = m / B
        mk = lambda l, tf=tf: P.theory_rule(l, cal, ev.exit_layers, cost, trim_frac=tf)
        lam = (tune_cert(mk, fit, cert, cost, B, lams, compact) if args.tune == "cert"
               else tune(mk, ev, cost, B, lams, full_ev))
        if lam is None:                                   # nothing certified: the safest value (exits least)
            lam = lams[int(np.argmin([simulate(mk(l), fit, cost, B, 100, seed=1)["speedup"] for l in lams]))]
        rule = lambda mm: P.theory_rule(lam, cal, ev.exit_layers, cost, trim_frac=mm / B)
        h = simulate(rule(m), ho, cost, B, 150, seed=2)
        att = victim_experiment(rule(m), clean, attack, cost, B, ks=ks, n_batches=args.batches, seed=7, compact=compact)
        rows.append({"policy": f"theory m={m}", "m": m, "lam": lam, "clean_speedup": h["speedup"],
                     "clean_acc_drop": full_ho - h["accuracy"], **{f"k{a['k']}": a["victim_rel_full"] for a in att}})
        # pathwise check of depth(C u K; m) <= depth(C; m - k)
        both = Traces.concat(clean, attack)
        rng = np.random.default_rng(11)
        for k in [k for k in ks if 0 < k <= m]:
            fixed = lambda mm: (lambda e, c, n, B_, left, a, b, r=rule(mm): r(e, c, n, B, left, a, b))
            n_viol = 0
            for _ in range(args.batches):
                C = rng.choice(len(clean), B - k, replace=False)
                K = len(clean) + rng.choice(len(attack), k, replace=len(attack) < k)
                d_att = run_batch(rule(m), both, np.concatenate([C, K]), cost, compact=compact).exit_of.max()
                d_cln = run_batch(fixed(m - k), both, C, cost, compact=compact).exit_of.max()
                n_viol += int(d_att > d_cln)
            viol_rows.append({"m": m, "k": k, "batches": args.batches, "violations": n_viol})
    for name, fam, grid in [("majority", P.majority, np.linspace(0.4, 0.995, 40)),
                            ("consensus", P.consensus, np.linspace(0.05, 0.995, 40)),
                            ("threshold", P.threshold, np.linspace(0.4, 0.995, 40))]:
        x = (tune_cert(fam, fit, cert, cost, B, grid, compact) if args.tune == "cert"
             else tune(fam, ev, cost, B, grid, full_ev))
        if x is None:
            x = grid[int(np.argmin([simulate(fam(g), fit, cost, B, 100, seed=1)["speedup"] for g in grid]))]
        h = simulate(fam(x), ho, cost, B, 150, seed=2)
        att = victim_experiment(fam(x), clean, attack, cost, B, ks=ks, n_batches=args.batches, seed=7, compact=compact)
        rows.append({"policy": name, "m": np.nan, "lam": x, "clean_speedup": h["speedup"],
                     "clean_acc_drop": full_ho - h["accuracy"], **{f"k{a['k']}": a["victim_rel_full"] for a in att}})
    df, vd = pd.DataFrame(rows), pd.DataFrame(viol_rows)
    tag = f"{args.task}_{args.regime}_B{B}{args.tag}"
    df.to_csv(ROOT / f"results/robust_{tag}.csv", index=False)
    vd.to_csv(ROOT / f"results/robust_violations_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"{tag}, attack inputs: {kind}; victims' latency / full-model latency for k attackers")
    print(df.round(3).to_string(index=False))
    print("pathwise bound check:", vd.to_dict("records"))


if __name__ == "__main__":
    main()
