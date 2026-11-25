"""Exact checks of the claims in theory/THEORY_58.md (report item 5.8) on small synthetic instances.

Model (M1-M3 in the write-up): each input's confidence bin X_e follows its own Markov chain over the evaluated exits
e = 0..E-1 (inputs independent); p[e][x] = P(correct | exit at e in bin x). The calibration table is exact:
q[e][x][e'] = E[p[e'][X_e'] | X_e = x]. Leaving at exit e in bin x earns p[e][x] / B; running the layers from exit e
to exit e+1 with n inputs active costs lam * ((l[e+1] - l[e]) c(n) + h) (F = 1). At the last exit everyone leaves.

Checks:
  T1  collective stopping with the active set fixed: rule (3) equals the optimum whenever the problem is monotone
      (checked on the reachable states), for any cost c; it can be strictly worse when it is not monotone;
  T2  compute-bound limit (c(n) = b n, h = 0): the problem splits per input; the all-exits lookahead
      ("exit j iff no later exit is worth reaching") is optimal in the monotone case; rule (2), which compares only
      with the last exit, exits at or before it and can lose;
  P3  floor cost c(n) = max(c0, b n): the simultaneous rule (2) over-counts the saving; the instance of the write-up,
      and the sequential (joint-saving) version that fixes it;
  J   the full joint problem (individual exits + collective stop) for B = 3: gap of rules (2)+(3) to the optimum
      for affine and floor costs.
Run: python theory/verify_58.py  (CPU, under a minute)
"""
from __future__ import annotations

import itertools
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

TOL = 1e-10


# ------------------------------------------------------------------------------------------- instances
class Inst:
    def __init__(self, rng, E=4, K=3, diminishing=True):
        self.E, self.K = E, K
        self.l = np.cumsum(rng.integers(1, 3, E))          # layers of the evaluated exits
        # transitions: confidence tends to rise (upper-triangular-ish), random otherwise
        self.P = []
        for _ in range(E - 1):
            M = rng.random((K, K)) ** 2 + np.triu(np.ones((K, K))) * rng.random()
            self.P.append(M / M.sum(1, keepdims=True))
        # accuracy per exit and bin: increasing in bin; increasing in depth with diminishing steps if asked
        base = np.sort(rng.random(K)) * 0.5 + 0.2
        steps = np.sort(rng.random(E - 1) * 0.3)[::-1] if diminishing else rng.random(E - 1) * 0.3
        self.p = [np.clip(base + s, 0, 1) for s in np.concatenate([[0], np.cumsum(steps)])]
        self.p = [np.clip(pe + rng.normal(0, 0.05, K), 0, 1) for pe in self.p]
        # q[e][x][e'] = E[p_e'(X_e') | X_e = x]
        self.q = []
        for e in range(E):
            row = np.zeros((K, E))
            M = np.eye(K)
            for e2 in range(e, E):
                if e2 > e:
                    M = M @ self.P[e2 - 1]
                row[:, e2] = M @ self.p[e2]
            self.q.append(row)


def successors(inst, e, xs):
    """Joint successor states of the inputs xs (tuple of bins) from exit e to e+1, with probabilities."""
    for ys in itertools.product(range(inst.K), repeat=len(xs)):
        pr = 1.0
        for x, y in zip(xs, ys):
            pr *= inst.P[e][x, y]
        if pr > 0:
            yield ys, pr


# ------------------------------------------------------------------- T1: collective stopping, R fixed
def collective_values(inst, xs0, B, lam, c, h):
    """(optimal value, value of rule (3), monotone?) for stopping the fixed set of inputs together."""
    E, n = inst.E, len(xs0)
    step = lambda e: lam * ((inst.l[e + 1] - inst.l[e]) * c(n) + h)

    def stop_val(e, xs):
        return sum(inst.p[e][x] for x in xs) / B

    @lru_cache(None)
    def V(e, xs):
        if e == E - 1:
            return stop_val(e, xs)
        cont = -step(e) + sum(pr * V(e + 1, ys) for ys, pr in successors(inst, e, xs))
        return max(stop_val(e, xs), cont)

    def rule3_stops(e, xs):
        for e2 in range(e + 1, E):
            gain = sum(inst.q[e][x][e2] - inst.p[e][x] for x in xs) / B
            cost = sum(step(k) for k in range(e, e2))
            if gain >= cost:
                return False
        return True

    def ola_stops(e, xs):
        gain = sum(inst.q[e][x][e + 1] - inst.p[e][x] for x in xs) / B
        return gain < step(e)

    @lru_cache(None)
    def W(e, xs):  # value of rule (3)
        if e == E - 1 or rule3_stops(e, xs):
            return stop_val(e, xs)
        return -step(e) + sum(pr * W(e + 1, ys) for ys, pr in successors(inst, e, xs))

    # monotone: once the one-step lookahead says stop, it says stop at every reachable successor
    monotone = True
    frontier = {xs0}
    for e in range(E - 1):
        nxt = set()
        for xs in frontier:
            for ys, _ in successors(inst, e, xs):
                nxt.add(ys)
                if e + 1 < E - 1 and ola_stops(e, xs) and not ola_stops(e + 1, ys):
                    monotone = False
        frontier = nxt
    return V(0, xs0), W(0, xs0), monotone


# ------------------------------------------------------------------- T2: per-input, compute-bound
def single_values(inst, x0, lam, b):
    """One input, cost lam * b per layer it runs (B = 1 scale). Returns optimal, all-lookahead, rule (2), monotone."""
    E = inst.E
    step = lambda e: lam * b * (inst.l[e + 1] - inst.l[e])

    @lru_cache(None)
    def V(e, x):
        if e == E - 1:
            return inst.p[e][x]
        return max(inst.p[e][x], -step(e) + inst.P[e][x] @ np.array([V(e + 1, y) for y in range(inst.K)]))

    def all_la(e, x):
        return all(inst.q[e][x][e2] - inst.p[e][x] < lam * b * (inst.l[e2] - inst.l[e]) for e2 in range(e + 1, E))

    def rule2(e, x):
        return inst.q[e][x][E - 1] - inst.p[e][x] < lam * b * (inst.l[E - 1] - inst.l[e])

    def ola(e, x):
        return inst.q[e][x][e + 1] - inst.p[e][x] < step(e)

    def W(rule):
        @lru_cache(None)
        def w(e, x):
            if e == E - 1 or rule(e, x):
                return inst.p[e][x]
            return -step(e) + inst.P[e][x] @ np.array([w(e + 1, y) for y in range(inst.K)])
        return w(0, x0)

    monotone = all(not ola(e, x) or all(ola(e + 1, y) for y in range(inst.K) if inst.P[e][x, y] > 0)
                   for e in range(E - 2) for x in range(inst.K))
    submart = all(inst.q[e][x][e + 1] >= inst.p[e][x] - 1e-12 for e in range(E - 1) for x in range(inst.K))

    @lru_cache(None)
    def bound(e, x):  # E[lam b (L - l_{tau+1})] for the exit tau of rule (2) (0 if it reaches the last exit)
        if e == E - 1:
            return 0.0
        if rule2(e, x):
            return lam * b * (inst.l[E - 1] - inst.l[e + 1])
        return float(inst.P[e][x] @ np.array([bound(e + 1, y) for y in range(inst.K)]))
    # where rule (2) exits but the optimum continues / vice versa (over all states, not only reachable ones)
    early = sum(rule2(e, x) and not all_la(e, x) for e in range(E - 1) for x in range(inst.K))
    late = sum(all_la(e, x) and not rule2(e, x) for e in range(E - 1) for x in range(inst.K))
    return V(0, x0), W(all_la), W(rule2), monotone, early, late, submart, bound(0, x0)


# ------------------------------------------------------------------- J: full joint problem, small B
def joint_values(inst, xs0, lam, c, h, seq=False):
    """Optimal policy (any subset leaves, or all stop, at each exit) vs rules (2)+(3) as implemented.
    seq=True: rule (2) with the joint saving, inputs leaving one at a time in increasing gain."""
    E, B = inst.E, len(xs0)

    def step(e, n):
        return lam * ((inst.l[e + 1] - inst.l[e]) * c(n) + h) if n > 0 else 0.0

    @lru_cache(None)
    def V(e, xs):  # xs: bins of the inputs still active, sorted
        if not xs:
            return 0.0
        if e == E - 1:
            return sum(inst.p[e][x] for x in xs) / B
        best = -np.inf
        n = len(xs)
        for keep in itertools.product((0, 1), repeat=n):
            out = sum(inst.p[e][x] for x, k in zip(xs, keep) if not k) / B
            rest = tuple(x for x, k in zip(xs, keep) if k)
            if rest:
                cont = -step(e, len(rest)) + sum(pr * V(e + 1, tuple(sorted(ys)))
                                                 for ys, pr in successors(inst, e, rest))
            else:
                cont = 0.0
            best = max(best, out + cont)
        return best

    left = lambda e: inst.l[E - 1] - inst.l[e]

    def decide(e, xs):
        n = len(xs)
        g = [inst.q[e][x][E - 1] - inst.p[e][x] for x in xs]
        if not seq:
            marg = c(n) - c(n - 1) if n >= 2 else c(2) - c(1)
            leave = [gi / B < lam * marg * left(e) for gi in g]
        else:  # leave in increasing gain while the next input's joint saving exceeds its gain
            order = np.argsort(g)
            leave = [False] * n
            k = 0
            for i in order:
                if n - k >= 2 and g[i] / B < lam * (c(n - k) - c(n - k - 1)) * left(e):
                    leave[i] = True
                    k += 1
                else:
                    break
        rest = tuple(x for x, lv in zip(xs, leave) if not lv)
        if not rest:
            return leave, True
        nr = len(rest)
        for e2 in range(e + 1, E):
            gs = sum(inst.q[e][x][e2] - inst.p[e][x] for x in rest) / B
            if gs >= lam * ((inst.l[e2] - inst.l[e]) * c(nr) + (e2 - e) * h):
                return leave, False
        return leave, True

    @lru_cache(None)
    def W(e, xs):
        if not xs:
            return 0.0
        if e == E - 1:
            return sum(inst.p[e][x] for x in xs) / B
        leave, stop = decide(e, xs)
        out = sum(inst.p[e][x] for x, lv in zip(xs, leave) if lv or stop) / B
        if stop:
            return out
        rest = tuple(x for x, lv in zip(xs, leave) if not lv)
        return out - step(e, len(rest)) + sum(pr * W(e + 1, tuple(sorted(ys))) for ys, pr in successors(inst, e, rest))

    return V(0, tuple(sorted(xs0))), W(0, tuple(sorted(xs0)))


# ------------------------------------------------------------------- P3: the floor counterexample
def floor_example(nk=8, b=1.0, lam=1.0, layers_left=1.0, eps=1e-3):
    """n = nk + 1 inputs at their last decision before the final exit; one hard input (large gain), nk easy inputs,
    each with gain/B just below the marginal threshold lam * b * layers_left. c(n) = max(c0, b n) with c0 = nk b.
    Simultaneous rule (2) sends all nk easy inputs out (each sees marginal b); the batch keeps running for the hard
    one, so the saving is c(nk+1) - c(1) = b per layer, and nk gains are lost. Returns (objective of simultaneous
    rule (2), of the sequential version, of keeping everyone), relative to keeping everyone."""
    B = nk + 1
    c0 = nk * b
    c = lambda n: max(c0, b * n) if n > 0 else 0.0
    g_easy = (lam * b * layers_left - eps) * B           # gain (accuracy) of carrying one easy input to the end
    keep_all = 0.0
    simult = -nk * g_easy / B + lam * (c(B) - c(1)) * layers_left
    seq = -1 * g_easy / B + lam * (c(B) - c(B - 1)) * layers_left        # only the first exit saves anything
    return simult - keep_all, seq - keep_all, lam * b * layers_left


def main():
    rng = np.random.default_rng(0)
    res = {}
    # T1
    stats = {"mono": 0, "mono_gap": 0, "nonmono": 0, "nonmono_gap": 0, "max_gap": 0.0}
    for t in range(400):
        inst = Inst(rng, E=4, K=3, diminishing=bool(t % 2))
        B = 4
        xs0 = tuple(rng.integers(0, 3, 3))
        lam = float(10 ** rng.uniform(-3, -1))
        a, b, h = rng.random(), rng.random(), 0.2 * rng.random()
        c = (lambda n, a=a, b=b: a + b * n) if t % 3 else (lambda n, a=a, b=b: max(a * 3, b * n))
        v, w, mono = collective_values(inst, xs0, B, lam, c, h)
        gap = v - w
        assert gap > -TOL
        key = "mono" if mono else "nonmono"
        stats[key] += 1
        if gap > 1e-9:
            stats[key + "_gap"] += 1
            stats["max_gap"] = max(stats["max_gap"], gap / max(v, 1e-9))
    print(f"T1 collective, R fixed: {stats['mono']} monotone instances, rule (3) suboptimal in {stats['mono_gap']}; "
          f"{stats['nonmono']} non-monotone, suboptimal in {stats['nonmono_gap']} (max relative gap "
          f"{stats['max_gap']:.2%})")
    assert stats["mono_gap"] == 0
    res["T1"] = stats

    # T2
    s = {"mono": 0, "la_gap": 0, "r2_gap": 0, "r2_late": 0, "nonmono": 0, "nm_la_gap": 0, "r2_loss": [],
         "bound_n": 0, "bound_viol": 0, "bound_ratio": []}
    for t in range(1000):
        inst = Inst(rng, E=5, K=3, diminishing=True)
        lam = float(10 ** rng.uniform(-2.5, -0.5))
        x0 = int(rng.integers(0, 3))
        v, w_la, w_r2, mono, early, late, submart, bnd = single_values(inst, x0, lam, b=1.0)
        if mono and submart:
            s["bound_n"] += 1
            s["bound_viol"] += (v - w_r2) > bnd + 1e-12
            s["bound_ratio"].append((v - w_r2) / bnd if bnd > 0 else 0.0)
        assert v - w_la > -TOL and v - w_r2 > -TOL
        if mono:
            s["mono"] += 1
            s["la_gap"] += v - w_la > 1e-9
            s["r2_gap"] += v - w_r2 > 1e-9
            s["r2_late"] += late > 0
            s["r2_loss"].append(v - w_r2)
        else:
            s["nonmono"] += 1
            s["nm_la_gap"] += v - w_la > 1e-9
    loss = np.array(s["r2_loss"])
    print(f"T2 compute-bound, one input: {s['mono']} monotone instances; all-exits lookahead suboptimal in "
          f"{s['la_gap']}; rule (2) suboptimal in {s['r2_gap']} (mean loss {loss.mean():.4f}, max {loss.max():.4f} "
          f"in accuracy units); rule (2) ever later than the lookahead in {s['r2_late']}; "
          f"{s['nonmono']} non-monotone, lookahead suboptimal in {s['nm_la_gap']}")
    br = np.array(s.pop("bound_ratio"))
    print(f"T2 loss bound (monotone, gains >= 0): {s['bound_n']} instances, violated in {s['bound_viol']}; "
          f"loss / bound mean {br.mean():.2f}, max {br.max():.2f}")
    assert s["la_gap"] == 0 and s["r2_late"] == 0 and s["bound_viol"] == 0
    s["bound_ratio_mean"], s["bound_ratio_max"] = float(br.mean()), float(br.max())
    res["T2"] = {k: v for k, v in s.items() if k != "r2_loss"} | {"r2_loss_mean": float(loss.mean()),
                                                                  "r2_loss_max": float(loss.max())}

    # P3
    sim, seq, unit = floor_example()
    print(f"P3 floor, n_k = 8: simultaneous rule (2) changes the objective by {sim / unit:+.2f} x (lam b layers_left), "
          f"the sequential version by {seq / unit:+.3f} x")
    assert sim < 0 < seq
    res["P3"] = {"n_k": 8, "simultaneous": sim / unit, "sequential": seq / unit}

    # J
    for name, mk in (("affine", lambda a, b: (lambda n: a + b * n if n > 0 else 0.0)),
                     ("floor", lambda a, b: (lambda n: max(a, b * n) if n > 0 else 0.0))):
        gaps, gaps_seq, opt_hits, seq_hits = [], [], 0, 0
        for t in range(150):
            inst = Inst(rng, E=4, K=3, diminishing=True)
            xs0 = tuple(rng.integers(0, 3, 3))
            lam = float(10 ** rng.uniform(-2.5, -0.5))
            a, b = rng.random() * 3, rng.random()
            c = mk(a, b)
            v, w = joint_values(inst, xs0, lam, c, h=0.1)
            _, ws = joint_values(inst, xs0, lam, c, h=0.1, seq=True)
            gaps.append(v - w)
            gaps_seq.append(v - ws)
            opt_hits += v - w < 1e-9
            seq_hits += v - ws < 1e-9
        gaps, gaps_seq = np.array(gaps), np.array(gaps_seq)
        print(f"J {name}, B = 3: rules (2)+(3) optimal in {opt_hits}/150, mean gap {gaps.mean():.4f}, max "
              f"{gaps.max():.4f}; sequential (2) optimal in {seq_hits}/150, mean gap {gaps_seq.mean():.4f}, max "
              f"{gaps_seq.max():.4f}")
        res["J_" + name] = {"n": 150, "rules_optimal": int(opt_hits), "gap_mean": float(gaps.mean()),
                            "gap_max": float(gaps.max()), "seq_optimal": int(seq_hits),
                            "seq_gap_mean": float(gaps_seq.mean()), "seq_gap_max": float(gaps_seq.max())}
    out = Path(__file__).resolve().parents[1] / "results/theory58_checks.json"
    out.write_text(json.dumps(res, indent=1, default=int))
    print("wrote", out)


if __name__ == "__main__":
    main()
