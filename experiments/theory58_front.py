"""Report item 5.8, empirical side: do the two corrections of rule (2) that the theory suggests help on real traces?

theory/THEORY_58.md shows that rule (2) as implemented (input j leaves iff its gain to the LAST exit is below the
saving to the last exit, all inputs priced at today's marginal cost c(n) - c(n-1)) has two faults:
  LA   it compares only with the last exit. In the compute-bound limit the optimal rule (monotone case) is
       "leave iff no later exit is worth reaching", the same all-exits lookahead that rule (3) uses;
  SEQ  it prices every leaving input at today's marginal cost. When k inputs leave together the real saving is
       c(n) - c(n - k), which is less than k (c(n) - c(n-1)) for a convex cost (the floor of Section 4.3).
       Fix: inputs leave one at a time in increasing score while the next one's own marginal saving exceeds it.
Families (rule (3) unchanged in all of them): theory (as in the report), theory_la, theory_seq, theory_la_seq, and
the same four with the exit plan (+plan). Protocol and outputs as `local_hw.py front` (certified selection on
the evolve split at each accuracy target, scored once on held-out; cost-table replay on the CPU, no GPU).

Usage: python experiments/theory58_front.py --task qqp --name a6000 --seq-len 128 --workers 12
Output: results/theory58_front_<task>_<name>_T<T>.csv
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_hw as lh  # noqa: E402

VARIANTS = {"theory": (False, False), "theory_la": (True, False), "theory_seq": (False, True),
            "theory_la_seq": (True, True)}


def theory_v(lam: float, cal, layers: list[int], cost, la: bool, seq: bool):
    """Rule (2) with the corrections switched on by la / seq, then rule (3) exactly as lh.theory_c."""
    E = len(layers)
    lay = np.asarray(layers, float)

    def marg(m: int) -> float:
        return cost.c(m) - cost.c(m - 1) if m >= 2 else cost.c(2) - cost.c(1)

    def decide(e, confs, n, B, left, a, b):
        F = lh.full_ms(cost, B)
        pc = cal.p_correct(e, confs)
        g = np.maximum(pc - pc[:, [e]], 0.0)
        # score = gain per layer that must beat lam * marginal / F * B; LA: best over later exits, else last exit
        span = lay[e + 1:] - lay[e]
        score = (g[:, e + 1:] / span).max(1) if la else g[:, E - 1] / left
        thr = lam / F * B
        if not seq:
            mask = score < thr * marg(n)
        else:
            mask = np.zeros(n, bool)
            k = 0
            for i in np.argsort(score, kind="stable"):
                if score[i] < thr * marg(n - k):
                    mask[i] = True
                    k += 1
                else:
                    break
        rest = ~mask
        nr = int(rest.sum())
        if nr == 0:
            return mask, False
        gs = g[rest].sum(0) / B
        for e2 in range(e + 1, E):
            if gs[e2] >= lam * ((layers[e2] - layers[e]) * cost.c(nr) + (e2 - e) * cost.head) / F:
                return mask, False
        return mask, True
    return decide


_make, _grid = lh.make, lh.grid


def make(fam: str, params: dict, cal_tr, cost):
    base = fam.replace("+plan", "")
    if base not in VARIANTS:
        return _make(fam, params, cal_tr, cost)
    plan = tuple(params.get("plan", lh.ALL))
    key = ("cal", id(cal_tr), plan)
    if key not in lh.G:
        lh.G[key] = lh.P.Calibration(lh.sub_traces(cal_tr, plan))
    return plan, theory_v(params["lam"], lh.G[key], [lh.LAYERS[e] for e in plan], cost, *VARIANTS[base])


def grid(fam: str):
    base = fam.replace("+plan", "")
    if base in VARIANTS:
        return _grid("theory+plan" if fam.endswith("+plan") else "theory")
    return _grid(fam)


lh.make, lh.grid = make, grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="qqp")
    ap.add_argument("--name", default="a6000")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--plans", action="store_true", help="also the +plan families (31x the grid)")
    a = ap.parse_args()
    a.device, a.cost_model, a.exec = a.name, "table", "eager"
    lh.set_task(a.task)
    ev = lh.load_traces(lh.ROOT / f"traces/{a.task}_val.parquet", "evolve", exit_layers=list(lh.LAYERS))
    lh.G["ho"] = lh.load_traces(lh.ROOT / f"traces/{a.task}_val.parquet", "heldout", exit_layers=list(lh.LAYERS))
    lh.G["fit"], lh.G["cert"] = lh.split_evolve(ev, 0)
    lh.G["cost"], lh.G["task"], lh.G["tag"] = lh.load_cost(a), a.task, lh.tag(a)
    m = lh.G["ho"].conf[:, :-1].mean(1)
    lh.G["hard"] = m <= np.quantile(m, 0.25)
    fams = list(VARIANTS) + ([f + "+plan" for f in VARIANTS] if a.plans else [])
    jobs = [(f, B) for B in a.batch_sizes for f in fams]
    with Pool(a.workers) as pool:
        res = pool.map(lh._front_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rs in res for r in rs])
    out = lh.ROOT / f"results/theory58_front_{a.task}_{lh.tag(a)}.csv"
    df.to_csv(out, index=False)
    print(df[df.alpha_pts == 1.0][["B", "family", "ho_speedup", "ho_drop_pts", "hard_drop_pts"]].round(3)
          .to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
