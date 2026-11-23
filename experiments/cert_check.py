"""Report item 5.2: a finite-sample certificate for certified selection, a label-free version, and the amount of
calibration data a rule needs.

experiments/local_hw.py certifies a candidate with a one-sided t-test on batch means (asymptotic). This script
re-runs its re-split validity check with three certificates and two risks:

  certificates (H0: risk > alpha, rejected at level delta, on the K disjoint batch means of the cert split)
    t     the t-test of local_hw.py (asymptotic; the reference)
    hb    Hoeffding-Bentkus (Bates et al. 2021): finite-sample, uses only the range of the batch loss
    wsr   Waudby-Smith-Ramdas betting bound (as in RCPS): finite-sample and variance-adaptive
  risks
    drop      accuracy drop vs the full model, batch means in [-1, 1] (mapped to [0, 1] for hb / wsr)
    disagree  disagreement with the full model, in [0, 1]; label-free (calibration needs no labels), and since
              drop <= disagree on every batch it also certifies the accuracy budget, conservatively

and repeats the selection with a fraction of the calibration data (fit and cert parts subsampled alike), for the
cost rule and for random search over a three-parameter rule.

Each job: random re-split of the validation traces into fit 20% / cert 30% / test 50%, certified selection (the
fixed-sequence procedure of local_hw.select_family), then accuracy drop and speedup of the choice on test.
A violation is a test drop above alpha. Speedups come from a measured cost table (replay, not timed).

Usage:
  python experiments/cert_check.py --task qqp --name a6000 --seeds 30 --workers 12
  python experiments/cert_check.py --task cifar100 --cost-json results/runner_cost_cifar100.json --seeds 30
Output: results/cert_check_<task>.csv and a summary table on stdout (results/cert_check_<task>_summary.csv).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import local_hw as lh  # noqa: E402
from simulator.simulator import Traces, load_traces  # noqa: E402

DELTA = lh.DELTA
CFG: dict = {"method": "t"}          # certificate used by the patched lh.certified (set per job)


# ------------------------------------------------------------------------------------------ certificates
def _kl(a: float, b: float) -> float:
    a = min(max(a, 1e-12), 1 - 1e-12)
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def hb_pvalue(x: np.ndarray, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: E[x] >= alpha, x in [0, 1] i.i.d. (Bates, Angelopoulos, Lei, Malik,
    Jordan 2021, 'Distribution-free, risk-controlling prediction sets', Prop. 3)."""
    n, r = len(x), float(np.mean(x))
    if alpha >= 1:
        return 0.0
    if alpha <= 0:
        return 1.0
    p_h = math.exp(-n * _kl(min(r, alpha), alpha)) if r < alpha else 1.0
    p_b = math.e * stats.binom.cdf(math.ceil(n * r), n, alpha)
    return float(min(p_h, p_b, 1.0))


def wsr_reject(x: np.ndarray, alpha: float, delta: float) -> bool:
    """Waudby-Smith-Ramdas betting test of H0: E[x] >= alpha for x in [0, 1] (predictable plug-in bets as in RCPS,
    Bates et al. 2021, Sec. 3.1.3). Rejects when the capital of a bettor who bets that the mean is below alpha
    reaches 1/delta (Ville's inequality makes this valid at any sample size)."""
    n = len(x)
    if alpha <= 0:
        return False
    cap, s, ss_dev, mu_prev = 1.0, 0.0, 0.0, 0.5
    var_prev = 0.25
    for i, xi in enumerate(x, start=1):
        nu = min(1.0, math.sqrt(2 * math.log(1 / delta) / (n * var_prev)))
        cap *= 1 - nu * (xi - alpha)
        if cap >= 1 / delta:
            return True
        if cap <= 0:
            return False
        s += xi
        mu = (0.5 + s) / (i + 1)
        ss_dev += (xi - mu_prev) ** 2
        var_prev = (0.25 + ss_dev) / (i + 1)
        mu_prev = mu
    return False


def certified(r: dict, key: str = "drop", alpha: float = lh.ALPHA, delta: float | None = None) -> bool:
    delta = CFG.get("delta", DELTA) if delta is None else delta
    m = CFG["method"]
    if m == "t":
        return _t_cert(r, key, alpha, delta)
    x = np.asarray(r[key + "_b"], float)
    a = alpha
    if key == "drop":                                   # batch drop in [-1, 1] -> [0, 1]
        x, a = (x + 1) / 2, (alpha + 1) / 2
    if m == "hb":
        return hb_pvalue(x, a) <= delta
    if m == "wsr":
        order = np.random.default_rng(len(x)).permutation(len(x))   # fixed order (batches are already random)
        return wsr_reject(x[order], a, delta)
    raise KeyError(m)


_t_cert = lh.certified


def evaluate(fam, params, cal_tr, tr, batches, cost) -> dict:
    """lh.evaluate plus the per-batch values the finite-sample certificates need."""
    plan, decide = lh.make(fam, params, cal_tr, cost)
    lat, pred = lh.replay(decide, lh.sub_traces(tr, plan), batches, cost)
    rows = np.stack(batches)
    full_ok = tr.pred[rows, -1] == tr.label[rows]
    ok = pred == tr.label[rows]
    d = full_ok.mean(1) - ok.mean(1)
    dis = (pred != tr.pred[rows, -1]).mean(1)
    return {"drop": float(d.mean()), "drop_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "K": len(d),
            "disagree": float(dis.mean()), "disagree_sd": float(dis.std(ddof=1)) if len(d) > 1 else 0.0,
            "drop_b": d, "disagree_b": dis,
            "speedup": float(lh.full_ms(cost, len(batches[0])) / lat.mean()), "acc": float(ok.mean())}


lh.certified = certified        # select_family looks both names up in lh's globals
lh.evaluate = evaluate


# ------------------------------------------------------------------------------------------ cost
def load_cost(args) -> lh.TCost:
    if args.cost_json:                                   # a server runner cost file (runner/gpu_runner.py --calibrate)
        d = json.loads(Path(args.cost_json).read_text())
        pts = [{"B": p["B"], "full_ms": p.get("full_ms", p.get("full_ms_per_batch"))} for p in d["points"]]
        d = {"device": Path(args.cost_json).stem, "seq_len": args.seq_len, "a": d["a"], "b": d["b"],
             "head": d["head"], "points": pts}
        return lh.TCost(d, "table")
    return lh.load_cost(args)


# ------------------------------------------------------------------------------------------ jobs
G: dict = {}


def _job(a):
    seed, B, fam, method, key, frac = a
    CFG["method"] = method
    tr, cost = G["all"], G["cost"]
    n = len(tr)
    perm = np.random.default_rng(1000 + seed).permutation(n)
    fit_i, cert_i, test_i = perm[:n // 5], perm[n // 5:n // 2], perm[n // 2:]
    if frac < 1:
        fit_i, cert_i = fit_i[: int(frac * len(fit_i))], cert_i[: int(frac * len(cert_i))]
    fit, cert, test = tr.subset(fit_i), tr.subset(cert_i), tr.subset(test_i)
    out = {"seed": seed, "B": B, "family": fam, "method": method, "key": key, "frac": frac,
           "n_fit": len(fit), "n_cert": len(cert), "K_cert": len(cert) // B}
    if len(cert) // B < 2:
        out.update(n_certified=0, test_drop=0.0, test_disagree=0.0, speedup=1.0, chose=False)
        return out
    r = lh.select_family(fam, fit, cert, cost, B, seed=seed, key=key)
    p = r["cert_params"]
    if p is None:                                        # nothing certified: run the full model
        out.update(n_certified=0, test_drop=0.0, test_disagree=0.0, speedup=1.0, chose=False)
        return out
    e = evaluate(fam, p, fit, test, lh.disjoint_batches(len(test), B, seed + 7), cost)
    out.update(n_certified=r["n_certified"], test_drop=e["drop"], test_disagree=e["disagree"],
               speedup=e["speedup"], chose=True, params=json.dumps(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qqp")
    ap.add_argument("--name", default="a6000", help="cost file results/local_cost_<name>_T<seq-len>.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--cost-json", default=None, help="use a runner cost file instead (e.g. CIFAR-100)")
    ap.add_argument("--cost-model", default="table")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
    ap.add_argument("--families", nargs="+", default=["theory", "theory+plan", "random_collective", "threshold", "unified"])
    ap.add_argument("--methods", nargs="+", default=["t", "hb", "wsr"])
    ap.add_argument("--keys", nargs="+", default=["drop", "disagree"])
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1])
    ap.add_argument("--frac-families", nargs="+", default=["theory", "random_collective"],
                    help="families for which fractions below 1 are run (the report's question)")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--delta", type=float, default=DELTA, help="failure probability of the certificate")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    CFG["delta"] = args.delta
    lh.set_task(args.task)
    path = ROOT / f"traces/{args.task}_val.parquet"
    G["all"] = Traces.concat(load_traces(path, "evolve", exit_layers=list(lh.LAYERS)),
                             load_traces(path, "heldout", exit_layers=list(lh.LAYERS)))
    G["all"].attacker = None
    G["cost"] = load_cost(args)
    jobs = [(s, B, f, m, k, fr) for s in range(args.seeds) for B in args.batch_sizes for f in args.families
            for m in args.methods for k in args.keys for fr in args.fracs
            if fr == 1.0 or (f in args.frac_families and k == "drop")]
    print(f"{args.task}: {len(G['all'])} rows, {len(jobs)} jobs", flush=True)
    with Pool(args.workers) as pool:
        res = pool.map(_job, jobs, chunksize=4)
    df = pd.DataFrame(res)
    df.to_csv(ROOT / f"results/cert_check_{args.task}{args.tag}.csv", index=False)
    a = lh.ALPHA
    g = df.groupby(["frac", "key", "method", "family", "B"]).agg(
        viol=("test_drop", lambda d: float((d > a).mean())), drop_pts=("test_drop", lambda d: 100 * d.mean()),
        drop_p90_pts=("test_drop", lambda d: 100 * np.quantile(d, 0.9)), speedup=("speedup", "mean"),
        chose=("chose", "mean")).reset_index()
    g.to_csv(ROOT / f"results/cert_check_{args.task}{args.tag}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(g.round(3).to_string(index=False))
    print("\nviolation rate (test drop > alpha), all B pooled; delta =", args.delta)
    print(df.groupby(["frac", "key", "method"]).test_drop.apply(lambda d: float((d > a).mean())).unstack().round(3))
    print("\nmean speedup, all B pooled")
    print(df.groupby(["frac", "key", "method"]).speedup.mean().unstack().round(3))


if __name__ == "__main__":
    main()
