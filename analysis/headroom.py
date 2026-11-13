"""How much room is there above the theory rule? Hindsight bounds + difficulty-sorted batches. Held-out, B=64, simulator."""
import sys
import numpy as np
sys.path.insert(0, "/home/neha/ie624")
import os; os.chdir("/home/neha/ie624")
from simulator import policies as P
from simulator.simulator import load_traces, simulate, draw_batches, full_latency
from experiments.run_main import all_costs

B, TOL = 64, 0.01
lams = np.logspace(-3, 1.5, 36)
mus = np.concatenate([[0], np.logspace(-3, 2, 60)])


def best(points, full):
    ok = [s for a, s in points if a >= full - TOL]
    return max(ok) if ok else float("nan")


def oracle(tr, cal, batches, cost, mode):
    """mode: depth = one hindsight depth per batch, everyone exits there (labels known)
             label = hindsight depth + per-input exits, labels known
             traj  = same, but errors replaced by calibrated expected error (knows future confidences, not labels)"""
    L = np.array(tr.exit_layers, float); E = len(L)
    F = full_latency(tr.exit_layers, cost, B)
    pts = []
    for mu in mus:
        lat, corr = 0.0, 0
        for rows in batches:
            wrong = ~tr.correct[rows]                                    # [B,E]
            if mode == "traj":
                est = np.stack([1 - cal.p_correct(e, tr.conf[rows, e])[:, e] for e in range(E)], 1)
            else:
                est = wrong.astype(float)
            bestJ = None
            for s in range(E):
                if mode == "depth":
                    ex = np.full(B, s)
                else:
                    per = cost.b * L[None, :s + 1] / F + mu * est[:, :s + 1] / B
                    ex = per.argmin(1)
                # latency with compaction: layer segment k is run by inputs whose exit >= k
                n_seg = np.array([(ex >= k).sum() for k in range(s + 1)])
                seg = np.diff(np.concatenate([[0], L[:s + 1]]))
                t = (seg * (cost.a + cost.b * n_seg)).sum() + cost.head * (s + 1 if s < E - 1 else E)
                J = t / F + mu * est[np.arange(B), ex].sum() / B
                if bestJ is None or J < bestJ[0]:
                    bestJ = (J, t, (~wrong[np.arange(B), ex]).sum())
            lat += bestJ[1]; corr += bestJ[2]
        pts.append((corr / (B * len(batches)), F * len(batches) / lat))
    return pts


def theory(tr, cal, batches, cost):
    return [(r["accuracy"], r["speedup"]) for r in
            (simulate(P.theory_rule(l, cal, tr.exit_layers, cost), tr, cost, B, len(batches), batches=batches) for l in lams)]


print(f"{'task':9s} {'regime':14s} full  | theory  sorted512 sortedAll | depth-orc  traj-orc  label-orc")
for task in ["mnli", "qqp", "cifar100"]:
    ev = load_traces(f"traces/{task}_val.parquet", "evolve"); ho = load_traces(f"traces/{task}_val.parquet", "heldout")
    cal = P.Calibration(ev)
    costs = all_costs(task)
    batches = draw_batches(len(ho), B, 150, 0)
    rows = np.concatenate(batches)
    full = ho.correct[rows, -1].mean()
    def resort(pool):   # regroup the same rows into batches of similar exit-0 confidence, within pools of `pool` batches
        out = []
        for i in range(0, len(batches), pool):
            r = np.concatenate(batches[i:i + pool]); r = r[np.argsort(ho.conf[r, 0])]
            out += [r[j:j + B] for j in range(0, len(r), B)]
        return out
    for reg in [f"real_{task}", "dec_eager"]:
        c = costs[reg]
        th = best(theory(ho, cal, batches, c), full)
        s8 = best(theory(ho, cal, resort(8), c), full)
        sA = best(theory(ho, cal, resort(len(batches)), c), full)
        o = {m: best(oracle(ho, cal, batches, c, m), full) for m in ["depth", "traj", "label"]}
        print(f"{task:9s} {reg:14s} {100*full:.1f} | {th:5.2f}   {s8:5.2f}     {sA:5.2f}    |  {o['depth']:5.2f}     {o['traj']:5.2f}    {o['label']:5.2f}", flush=True)
