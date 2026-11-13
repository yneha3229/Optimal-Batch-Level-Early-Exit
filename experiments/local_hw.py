"""Local-hardware verification (GTX 1650 / CPU): cost profile, exit plans, certified selection, real wall-clock.

What is real and what is replayed
  * Wall-clock: real. Every transformer layer, exit head, softmax, host sync, index_select compaction and the
    policy's own decide() call are executed and timed on this machine.
  * Exit decisions and predictions: replayed from the recorded traces of the trained models
    (traces/<task>_val.parquet). The checkpoints are not on this machine, so the timed network has the trained
    model's architecture, shapes and dtype but randomly initialised weights; the confidences it computes are
    discarded after the host sync and replaced by the trained model's recorded confidences for the same rows.
    Accuracy numbers are therefore exact for the trained model, and timing is exact for this hardware.

Subcommands
  profile   time the full model and the all-heads model vs batch size; fit c(n) = a + b n and the head cost.
  select    choose parameters per family on the evolve split (fit part -> candidate path, cert part -> certified
            choice by fixed-sequence testing), using the cost measured by `profile`.
  time      execute the selected policies on held-out batches on the device; report real speedup.
  resplit   validity check of the certified selection against the naive protocol over random re-splits.
  front     speed at matched held-out accuracy: certified selection at several accuracy targets (CPU replay).

Usage
  python3 experiments/local_hw.py profile --device cuda
  python3 experiments/local_hw.py select --task qqp --device cuda
  python3 experiments/local_hw.py time --task qqp --device cuda
  python3 experiments/local_hw.py resplit --task qqp --device cuda
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import policies as P  # noqa: E402
from simulator.simulator import Cost, Traces, load_traces  # noqa: E402

LAYERS = [2, 4, 6, 8, 10, 12]
NUM_LABELS = {"mnli": 3, "qqp": 2}
FRONT_ALPHAS = [0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03]
ALPHA, DELTA = 0.01, 0.10          # accuracy budget (drop vs full model) and failure probability
G: dict = {}


# ------------------------------------------------------------------------------------------ cost models
class TCost:
    """Per-layer cost c(n). kind='table': measured points, made monotone and interpolated linearly in n.
    kind='affine': a + b n. Policies that take (a, b) receive the local linearisation at the current n:
    b_eff = c(n) - c(n-1) (what one input leaving saves per layer), a_eff = c(n) - b_eff n."""
    skip = 0.0
    exit_pass = 0.0

    def __init__(self, d: dict, kind: str = "table") -> None:
        from sklearn.isotonic import IsotonicRegression
        self.name, self.kind = f"local_{d['device']}_T{d['seq_len']}_{kind}", kind
        self.a, self.b, self.head = d["a"], d["b"], d["head"]
        B = np.array([p["B"] for p in d["points"]], float)
        y = np.array([p["full_ms"] for p in d["points"]]) / LAYERS[-1]
        y = IsotonicRegression().fit_transform(B, y)
        n = np.arange(0, 257, dtype=float)
        tab = np.interp(n, B, y)
        hi = n > B[-1]
        tab[hi] = y[-1] + (n[hi] - B[-1]) * (y[-1] - y[-2]) / (B[-1] - B[-2])
        self.tab = tab if kind == "table" else self.a + self.b * n
        self.tab[0] = 0.0

    def c(self, n):
        return self.tab[n]

    def lin(self, n: int) -> tuple[float, float]:
        if self.kind == "affine":
            return self.a, self.b
        b = self.tab[n] - self.tab[n - 1] if n >= 2 else self.tab[2] - self.tab[1]
        return float(self.tab[n] - b * n), float(b)


# ------------------------------------------------------------------------------------------ cost-model replay
def sub_traces(tr: Traces, plan: tuple[int, ...]) -> Traces:
    """Traces restricted to the exits in `plan` (indices into LAYERS). The last one is terminal."""
    p = list(plan)
    return Traces(tr.conf[:, p], tr.pred[:, p], tr.label, [LAYERS[e] for e in p])


def replay(decide, tr: Traces, batches: list[np.ndarray], cost, with_n: bool = False):
    """Cost-model latency per batch [K] and returned prediction per sample [K, B]. Only exits in tr.exit_layers
    are evaluated (each costs cost.head); the last one is terminal. Same control flow as the device runner.
    with_n: also return the active count in each segment [K, E] (0 after the batch ended)."""
    L, E = tr.exit_layers, len(tr.exit_layers)
    lat, preds = np.zeros(len(batches)), []
    nseg = np.zeros((len(batches), E), np.int64)
    for k, rows in enumerate(batches):
        B = len(rows)
        ids = np.arange(B)
        out = np.empty(B, np.int64)
        t, prev = 0.0, 0
        for e, layer in enumerate(L):
            n = len(ids)
            t += (layer - prev) * cost.c(n) + cost.head
            nseg[k, e] = n
            prev = layer
            if e == E - 1:
                out[ids] = tr.pred[rows[ids], e]
                break
            mask, stop = decide(e, tr.conf[rows[ids], e], n, B, LAYERS[-1] - layer, cost.a, cost.b)
            leave = np.ones(n, bool) if stop else np.asarray(mask, bool)
            out[ids[leave]] = tr.pred[rows[ids[leave]], e]
            ids = ids[~leave]
            if len(ids) == 0:
                break
        lat[k] = t
        preds.append(out)
    return (lat, np.stack(preds), nseg) if with_n else (lat, np.stack(preds))


def latency_of(nseg: np.ndarray, layers: list[int], cost) -> float:
    """Total cost-model latency of recorded segment counts under another cost model (same decisions)."""
    seg = np.diff([0] + list(layers))
    return float((cost.c(nseg) * seg).sum() + cost.head * (nseg > 0).sum())


def full_ms(cost, B: int) -> float:
    return LAYERS[-1] * cost.c(B) + cost.head


def disjoint_batches(n: int, B: int, seed: int, limit: int | None = None) -> list[np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    k = n // B if limit is None else min(n // B, limit)
    return [perm[i * B:(i + 1) * B] for i in range(k)]


# ------------------------------------------------------------------------------------------ candidates
PLANS = [tuple(s) + (5,) for r in range(1, 6) for s in itertools.combinations(range(5), r)]   # 31 plans
ALL = (0, 1, 2, 3, 4, 5)
TAUS = np.round(np.linspace(0.40, 0.995, 36), 4)
LAMS = np.round(np.logspace(-3.5, 1.5, 40), 7)
FAMILIES = ["threshold", "threshold+plan", "majority", "unified", "random_collective", "theory", "theory+plan"]


def theory_c(lam: float, cal, layers: list[int], cost: "TCost"):
    """The theory rule for an arbitrary measured per-layer cost c(n) (simulator/policies.py::theory_rule is the
    special case c(n) = a + b n). At exit e with n active inputs, batch size B, full-model latency F:
      input j leaves alone   iff  g_j(e -> last) / B  <  lam * (c(n) - c(n-1)) * layers_left / F
      the rest R stop        iff  for every later evaluated exit e':
                                  sum_{j in R} g_j(e -> e') / B  <  lam * [(l_e' - l_e) c(|R|) + (e' - e) head] / F
    Only exits in `layers` are evaluated, so a plan with fewer exits pays fewer heads."""
    E = len(layers)

    def decide(e, confs, n, B, left, a, b):
        F = full_ms(cost, B)
        pc = cal.p_correct(e, confs)
        g = np.maximum(pc - pc[:, [e]], 0.0)
        marg = cost.c(n) - cost.c(n - 1) if n >= 2 else cost.c(2) - cost.c(1)
        mask = g[:, E - 1] / B < lam * marg * left / F
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


def never(e, c, n, B, left, a, b):
    return np.zeros(n, bool), False


def make(fam: str, params: dict, cal_tr: Traces, cost: Cost):
    """(plan, decide) for one candidate. cal_tr: the traces calibration tables are fitted on."""
    plan = tuple(params.get("plan", ALL))
    if fam in ("threshold", "threshold+plan"):
        return plan, P.threshold(params["tau"])
    if fam == "majority":
        return plan, P.majority(params["tau"])
    if fam == "unified":
        return (params["exit_idx"],), never
    if fam == "random_collective":
        return plan, P.param_collective(params["tau"], params["frac_cut"], params["conf_floor"])
    if fam in ("theory", "theory+plan"):
        key = ("cal", id(cal_tr), plan)
        if key not in G:
            G[key] = P.Calibration(sub_traces(cal_tr, plan))
        return plan, theory_c(params["lam"], G[key], [LAYERS[e] for e in plan], cost)
    raise KeyError(fam)


def grid(fam: str) -> list[dict]:
    if fam == "threshold":
        return [{"tau": float(t)} for t in TAUS]
    if fam == "threshold+plan":
        return [{"tau": float(t), "plan": p} for p in PLANS for t in TAUS[::2]]
    if fam == "majority":
        return [{"tau": float(t)} for t in TAUS]
    if fam == "unified":
        return [{"exit_idx": e} for e in range(5)]
    if fam == "random_collective":
        rng = np.random.default_rng(0)
        return [dict(tau=float(rng.uniform(0.5, 1.0)), frac_cut=float(rng.uniform(0, 0.5)),
                     conf_floor=float(rng.uniform(0.3, 0.95))) for _ in range(150)]
    if fam == "theory":
        return [{"lam": float(l)} for l in LAMS]
    if fam == "theory+plan":
        return [{"lam": float(l), "plan": p} for p in PLANS for l in LAMS[::2]]
    raise KeyError(fam)


def evaluate(fam: str, params: dict, cal_tr: Traces, tr: Traces, batches, cost: Cost) -> dict:
    """Batch-level accuracy drop vs the full model and cost-model speedup of one candidate on `batches` of tr."""
    plan, decide = make(fam, params, cal_tr, cost)
    lat, pred = replay(decide, sub_traces(tr, plan), batches, cost)
    rows = np.stack(batches)
    full_ok = tr.pred[rows, -1] == tr.label[rows]
    ok = pred == tr.label[rows]
    d = full_ok.mean(1) - ok.mean(1)                      # per-batch drop, in [-1, 1]
    dis = (pred != tr.pred[rows, -1]).mean(1)             # per-batch disagreement with the full model (label-free)
    return {"drop": float(d.mean()), "drop_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "K": len(d),
            "disagree": float(dis.mean()), "disagree_sd": float(dis.std(ddof=1)) if len(d) > 1 else 0.0,
            "speedup": float(full_ms(cost, len(batches[0])) / lat.mean()), "acc": float(ok.mean())}


def certified(r: dict, key: str = "drop", alpha: float = ALPHA, delta: float = DELTA) -> bool:
    """One-sided t-test on batch means: reject H0 'risk > alpha' at level delta."""
    m, sd, K = r[key], r[key + "_sd"], r["K"]
    if sd == 0:
        return m < alpha
    return (alpha - m) / (sd / np.sqrt(K)) > stats.t.ppf(1 - delta, K - 1)


def pareto_path(rows: list[dict]) -> list[int]:
    """Indices on the (risk asc, speedup desc) frontier, ordered from safest to fastest."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["drop"], -rows[i]["speedup"]))
    path, best = [], -np.inf
    for i in order:
        if rows[i]["speedup"] > best + 1e-9:
            path.append(i)
            best = rows[i]["speedup"]
    return path


def select_family(fam: str, fit: Traces, cert: Traces, cost, B: int, seed: int, key: str = "drop",
                  naive: bool = False) -> dict:
    """fit part: evaluate the whole grid, keep its frontier as a fixed path. cert part: fixed-sequence test along
    the path, stop at the first candidate that is not certified, return the fastest certified one.
    Also returns the naive choice (fastest on fit+cert pooled with empirical drop <= alpha, the old protocol)."""
    cands = grid(fam)
    fb = disjoint_batches(len(fit), B, seed)
    cb = disjoint_batches(len(cert), B, seed + 1)
    fr = [evaluate(fam, p, fit, fit, fb, cost) for p in cands]
    for r in fr:
        r["drop"] = r[key] if key != "drop" else r["drop"]
    path = pareto_path(fr)
    chosen, n_cert = None, 0
    for i in path:
        r = evaluate(fam, cands[i], fit, cert, cb, cost)
        if not certified(r, key):
            break
        n_cert += 1
        if chosen is None or r["speedup"] > chosen[1]["speedup"]:
            chosen = (i, r)
    # naive protocol: tables and selection on the same evolve data, pick the fastest with empirical drop <= alpha
    ni, nr = None, []
    if naive:
        both = Traces.concat(fit, cert)
        both.attacker = None
        nb = disjoint_batches(len(both), B, seed + 2)
        nr = [evaluate(fam, p, both, both, nb, cost) for p in cands]
        okn = [i for i, r in enumerate(nr) if r["drop"] <= ALPHA]
        ni = max(okn, key=lambda i: nr[i]["speedup"]) if okn else None
    out = {"family": fam, "B": B, "n_cands": len(cands), "path_len": len(path), "n_certified": n_cert,
           "cert_params": None if chosen is None else cands[chosen[0]],
           "cert_ev": None if chosen is None else chosen[1],
           "naive_params": None if ni is None else cands[ni], "naive_ev": None if ni is None else nr[ni]}
    return out


# ------------------------------------------------------------------------------------------ device runner
class Device:
    def __init__(self, task: str, device: str, seq_len: int = 128, n_inputs: int = 2048) -> None:
        import torch
        from simulator.exit_model import build
        self.torch, self.dev, self.T = torch, device, seq_len
        torch.manual_seed(0)
        cfg = dict(kind="text", backbone="roberta-base", exit_layers=LAYERS, num_labels=NUM_LABELS[task])
        self.model = build(cfg, load_backbone_weights=False).to(device).float().eval()   # fp32: fp16 is 4x slower on GTX 16xx
        ids = torch.randint(5, 50000, (n_inputs, seq_len))
        ids[:, 0] = 0
        lens = torch.randint(seq_len // 4, seq_len + 1, (n_inputs,))
        am = (torch.arange(seq_len)[None, :] < lens[:, None]).long()
        self.inp = {"input_ids": ids.to(device), "attention_mask": am.to(device)}
        self.host = torch.empty(4096, dtype=torch.float32, pin_memory=device == "cuda")

    def sync(self) -> None:
        if self.dev == "cuda":
            self.torch.cuda.synchronize()

    def run_batch(self, slot: np.ndarray, plan: tuple[int, ...], decide, conf: np.ndarray, pred: np.ndarray,
                  cost: Cost) -> np.ndarray:
        """Execute one batch. slot: rows of the synthetic input pool [B]; conf/pred: recorded [B, 6] for the
        trace rows this batch replays. Returns the predictions [B] the policy returned."""
        torch = self.torch
        m = self.model
        with torch.inference_mode():
            st = torch.from_numpy(slot).to(self.dev)
            h, mask = m.embed({k: v.index_select(0, st) for k, v in self.inp.items()})
            B = h.shape[0]
            ids = np.arange(B)
            out = np.empty(B, np.int64)
            layer = 0
            for j, e in enumerate(plan):
                while layer < LAYERS[e]:
                    h = m.layer(layer, h, mask)
                    layer += 1
                probs = m.head(e, h).float().softmax(-1)
                c_dev, _ = probs.max(-1)
                n = len(ids)
                ch = self.host[:n]
                ch.copy_(c_dev, non_blocking=True)
                self.sync()                                   # the one host sync per evaluated exit
                c = conf[ids, e].astype(np.float64)           # recorded confidences of the trained model
                if j == len(plan) - 1:
                    out[ids] = pred[ids, e]
                    break
                mk, stop = decide(j, c, n, B, LAYERS[-1] - LAYERS[e], cost.a, cost.b)
                leave = np.ones(n, bool) if stop else np.asarray(mk, bool)
                out[ids[leave]] = pred[ids[leave], e]
                keep = np.flatnonzero(~leave)
                if len(keep) == 0:
                    break
                if len(keep) < n:
                    k = torch.from_numpy(keep).to(self.dev)
                    h = h.index_select(0, k)
                    mask = mask.index_select(0, k)
                    ids = ids[keep]
            self.sync()
        return out

    def time_pass(self, batches, slots, plan, decide, tr: Traces, cost: Cost) -> tuple[float, np.ndarray]:
        self.sync()
        t0 = time.perf_counter()
        preds = [self.run_batch(s, plan, decide, tr.conf[r], tr.pred[r], cost) for r, s in zip(batches, slots)]
        self.sync()
        return time.perf_counter() - t0, np.stack(preds)


def tag(args) -> str:
    return f"{args.device}_T{args.seq_len}"


def cost_path(args) -> Path:
    return ROOT / f"results/local_cost_{tag(args)}.json"


def load_cost(args, kind: str | None = None) -> TCost:
    return TCost(json.loads(cost_path(args).read_text()), kind or args.cost_model)


# ------------------------------------------------------------------------------------------ subcommands
def cmd_profile(args) -> None:
    import torch
    dv = Device("mnli", args.device, args.seq_len, 512)
    zero = Cost("none", 0.0, 0.0, 0.0)
    conf, pred = np.zeros((256, 6)), np.zeros((256, 6), np.int64)
    Bs = [1, 2, 4, 8, 16, 32, 64] + ([128] if args.device == "cuda" else [])
    rows = []
    for B in Bs:
        reps = max(3, min(20, int(args.budget / ((0.02 if args.device == "cuda" else 0.08) * B * args.seq_len / 128))))
        slot = np.arange(B)

        def med(plan):
            dv.run_batch(slot, plan, never, conf[:B], pred[:B], zero)
            ts = []
            for _ in range(reps):
                dv.sync()
                t0 = time.perf_counter()
                dv.run_batch(slot, plan, never, conf[:B], pred[:B], zero)
                dv.sync()
                ts.append(time.perf_counter() - t0)
            return 1000 * float(np.median(ts))
        f1, h1, f2, h2 = med((5,)), med(ALL), med((5,)), med(ALL)      # interleaved
        rows.append({"B": B, "full_ms": (f1 + f2) / 2, "heads_ms": (h1 + h2) / 2, "reps": reps})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    b, a12 = np.polyfit(d.B, d.full_ms, 1, w=1 / d.full_ms)      # relative-error fit: small batches count too
    head = float(np.median((d.heads_ms - d.full_ms) / 5))
    fit = a12 + b * d.B
    out = {"device": args.device, "seq_len": args.seq_len, "dtype": "fp32", "a": float(max(a12, 0) / 12),
           "b": float(b / 12), "head": max(head, 0.0), "head_raw": head,
           "rho8": float(a12 / b / 8), "rho64": float(a12 / b / 64),
           "affine_max_rel_err": float(np.max(np.abs(fit - d.full_ms) / d.full_ms)), "points": rows,
           "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
           "threads": torch.get_num_threads()}
    cost_path(args).write_text(json.dumps(out, indent=1))
    print({k: v for k, v in out.items() if k != "points"})


def _sel_job(a):
    fam, B = a
    return select_family(fam, G["fit"], G["cert"], G["cost"], B, seed=B, key=G["key"])


def split_evolve(ev: Traces, seed: int, frac_fit: float = 0.4) -> tuple[Traces, Traces]:
    perm = np.random.default_rng(seed).permutation(len(ev))
    k = int(frac_fit * len(ev))
    return ev.subset(perm[:k]), ev.subset(perm[k:])


def cmd_select(args) -> None:
    ev = load_traces(ROOT / f"traces/{args.task}_val.parquet", "evolve")
    G["fit"], G["cert"] = split_evolve(ev, 0)
    G["cost"], G["key"] = load_cost(args), "drop"
    jobs = [(f, B) for B in args.batch_sizes for f in FAMILIES]
    with Pool(args.workers) as pool:
        res = pool.map(_sel_job, jobs, chunksize=1)
    out = ROOT / f"results/local_select_{args.task}_{tag(args)}.json"
    out.write_text(json.dumps(res, indent=1))
    for r in res:
        c = r["cert_ev"]
        print(f"B={r['B']:4d} {r['family']:18s} cert {('%.2fx drop %.2f' % (c['speedup'], 100 * c['drop'])) if c else 'none':22s}"
              f" {r['n_certified']}/{r['path_len']}  {r['cert_params']}", flush=True)


def cmd_time(args) -> None:
    cost = load_cost(args)
    ev = load_traces(ROOT / f"traces/{args.task}_val.parquet", "evolve")
    ho = load_traces(ROOT / f"traces/{args.task}_val.parquet", "heldout")
    fit, _ = split_evolve(ev, 0)
    sel = json.loads((ROOT / f"results/local_select_{args.task}_{tag(args)}.json").read_text())
    dv = Device(args.task, args.device, args.seq_len)
    rows = []
    for B in args.batch_sizes:
        batches = disjoint_batches(len(ho), B, 100 + B, limit=args.n // B)
        rng = np.random.default_rng(B)
        slots = [rng.choice(2048, B, replace=False) for _ in batches]
        pols = {"full": ((5,), never, {}), "heads_only": (ALL, never, {})}
        for r in sel:
            if r["B"] == B and r["cert_params"] is not None:
                pols[r["family"]] = (*make(r["family"], r["cert_params"], fit, cost), r["cert_params"])
        # held-out accuracy on the whole held-out split (exact, from traces), same policy objects
        hb = disjoint_batches(len(ho), B, 100 + B)
        times = {k: [] for k in pols}
        preds = {}
        dv.time_pass(batches[:2], slots[:2], ALL, never, ho, cost)           # warm-up
        for rd in range(args.rounds):
            for k, (plan, decide, _) in pols.items():
                t, p = dv.time_pass(batches, slots, plan, decide, ho, cost)
                times[k].append(t)
                preds[k] = p
            print(f"B={B} round {rd}: " + " ".join(f"{k}={times[k][-1]:.2f}s" for k in pols), flush=True)
        rb = np.stack(batches)
        full_t = np.array(times["full"])
        for k, (plan, decide, params) in pols.items():
            lat, pr, nseg = replay(decide, sub_traces(ho, plan), batches, cost, with_n=True)
            aff = load_cost(args, "affine")
            pl = [LAYERS[e] for e in plan]
            aff_sp = full_ms(aff, B) * len(batches) / latency_of(nseg, pl, aff)
            assert (pr == preds[k]).all(), f"device and replay decisions differ for {k}"
            latH, prH = replay(decide, sub_traces(ho, plan), hb, cost)
            rH = np.stack(hb)
            sp = full_t / np.array(times[k])                                   # paired by round
            rows.append({"task": args.task, "device": tag(args), "B": B, "family": k, "params": json.dumps(params),
                         "real_speedup": float(np.median(sp)), "real_lo": float(sp.min()), "real_hi": float(sp.max()),
                         "model_speedup": float(full_ms(cost, B) * len(batches) / lat.sum()) if k != "full" else 1.0,
                         "affine_speedup": float(aff_sp) if k != "full" else 1.0,
                         "ho_drop_pts": 100 * float((ho.pred[rH, -1] == ho.label[rH]).mean() - (prH == ho.label[rH]).mean()),
                         "timed_drop_pts": 100 * float((ho.pred[rb, -1] == ho.label[rb]).mean() - (preds[k] == ho.label[rb]).mean()),
                         "wall_s": float(np.median(times[k])), "rounds": args.rounds, "n_timed": int(rb.size)})
        pd.DataFrame(rows).to_csv(ROOT / f"results/local_time_{args.task}_{tag(args)}.csv", index=False)
    df = pd.DataFrame(rows)
    print(df[["B", "family", "real_speedup", "real_lo", "real_hi", "model_speedup", "ho_drop_pts"]].round(3).to_string(index=False))


def _resplit_job(a):
    seed, B, fam = a
    tr, cost = G["all"], G["cost"]
    perm = np.random.default_rng(1000 + seed).permutation(len(tr))
    n = len(tr)
    fit, cert, test = tr.subset(perm[:n // 5]), tr.subset(perm[n // 5:n // 2]), tr.subset(perm[n // 2:])
    r = select_family(fam, fit, cert, cost, B, seed=seed, key="drop", naive=True)
    tb = disjoint_batches(len(test), B, seed + 7)
    out = {"seed": seed, "B": B, "family": fam}
    both = Traces.concat(fit, cert)
    for tag, cal_tr in (("cert", fit), ("naive", both)):
        p = r[f"{tag}_params"]
        if p is None:
            out.update({f"{tag}_drop": 0.0, f"{tag}_speedup": 1.0})
            continue
        e = evaluate(fam, p, cal_tr, test, tb, cost)
        out.update({f"{tag}_drop": e["drop"], f"{tag}_speedup": e["speedup"]})
    return out


def cmd_resplit(args) -> None:
    path = ROOT / f"traces/{args.task}_val.parquet"
    G["all"] = Traces.concat(load_traces(path, "evolve"), load_traces(path, "heldout"))
    G["all"].attacker = None
    G["cost"] = load_cost(args)
    jobs = [(s, B, f) for s in range(args.seeds) for B in args.batch_sizes for f in args.families]
    with Pool(args.workers) as pool:
        res = pool.map(_resplit_job, jobs, chunksize=1)
    df = pd.DataFrame(res)
    df.to_csv(ROOT / f"results/local_resplit_{args.task}_{tag(args)}.csv", index=False)
    se = lambda d: np.sqrt(0.03 / (len(G["all"]) / 2))
    g = df.groupby(["B", "family"]).agg(
        naive_viol=("naive_drop", lambda d: float((d > ALPHA).mean())),
        cert_viol=("cert_drop", lambda d: float((d > ALPHA).mean())),
        naive_drop=("naive_drop", "mean"), cert_drop=("cert_drop", "mean"),
        naive_speedup=("naive_speedup", "mean"), cert_speedup=("cert_speedup", "mean"))
    print(g.round(4).to_string())


def _front_job(a):
    """Accuracy-speed front of one family at one batch size. Candidates and their order come from the fit part,
    the certified choice at each target alpha from the cert part; each choice is then scored once on held-out."""
    fam, B = a
    fit, cert, ho, cost = G["fit"], G["cert"], G["ho"], G["cost"]
    cands = grid(fam)
    fb = disjoint_batches(len(fit), B, B)
    cb = disjoint_batches(len(cert), B, B + 1)
    hb = disjoint_batches(len(ho), B, 100 + B)
    fr = [evaluate(fam, p, fit, fit, fb, cost) for p in cands]
    path = pareto_path(fr)
    cr = [evaluate(fam, cands[i], fit, cert, cb, cost) for i in path]
    rows_h = np.stack(hb)
    hard = G["hard"][rows_h]
    full_ok = ho.pred[rows_h, -1] == ho.label[rows_h]
    out, seen = [], {}
    for alpha in FRONT_ALPHAS:
        best = None
        for i, r in zip(path, cr):
            if not certified(r, "drop", alpha):
                break
            if best is None or r["speedup"] > best[1]["speedup"]:
                best = (i, r)
        if best is None:
            continue
        i = best[0]
        if i not in seen:
            plan, decide = make(fam, cands[i], fit, cost)
            lat, pred = replay(decide, sub_traces(ho, plan), hb, cost)
            ok = pred == ho.label[rows_h]
            seen[i] = {"ho_drop_pts": 100 * float(full_ok.mean() - ok.mean()),
                       "ho_speedup": float(full_ms(cost, B) / lat.mean()),
                       "hard_drop_pts": 100 * float(full_ok[hard].mean() - ok[hard].mean()),
                       "easy_drop_pts": 100 * float(full_ok[~hard].mean() - ok[~hard].mean())}
        out.append({"task": G["task"], "device": G["tag"], "B": B, "family": fam, "alpha_pts": 100 * alpha,
                    "n_cands": len(cands), "params": json.dumps(cands[i]), **seen[i]})
    return out


def cmd_front(args) -> None:
    """Speed at matched held-out accuracy: certified selection at a range of accuracy targets, scored on held-out.
    Speedups are from the measured cost table (CPU replay), not timed."""
    ev = load_traces(ROOT / f"traces/{args.task}_val.parquet", "evolve")
    G["ho"] = load_traces(ROOT / f"traces/{args.task}_val.parquet", "heldout")
    G["fit"], G["cert"] = split_evolve(ev, 0)
    G["cost"], G["task"], G["tag"] = load_cost(args), args.task, tag(args)
    m = G["ho"].conf[:, :-1].mean(1)                       # hard inputs: lowest quartile of mean intermediate confidence
    G["hard"] = m <= np.quantile(m, 0.25)
    jobs = [(f, B) for B in args.batch_sizes for f in FAMILIES]
    with Pool(args.workers) as pool:
        res = pool.map(_front_job, jobs, chunksize=1)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.to_csv(ROOT / f"results/local_front_{args.task}_{tag(args)}.csv", index=False)
    print(df[df.alpha_pts == 1.0][["B", "family", "ho_speedup", "ho_drop_pts", "hard_drop_pts"]].round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("profile", "select", "time", "resplit", "front"):
        s = sub.add_parser(name)
        s.add_argument("--device", default="cuda")
        s.add_argument("--task", default="qqp")
        s.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
        s.add_argument("--workers", type=int, default=6)
        s.add_argument("--seq-len", type=int, default=128)
        s.add_argument("--budget", type=float, default=20.0, help="profile: seconds per batch size and plan")
        s.add_argument("--rounds", type=int, default=3)
        s.add_argument("--n", type=int, default=768, help="time: held-out inputs executed per pass")
        s.add_argument("--seeds", type=int, default=40)
        s.add_argument("--cost-model", default="table", choices=["table", "affine"])
        s.add_argument("--families", nargs="+", default=["threshold", "majority", "theory"])
    args = ap.parse_args()
    {"profile": cmd_profile, "select": cmd_select, "time": cmd_time, "resplit": cmd_resplit, "front": cmd_front}[args.cmd](args)


if __name__ == "__main__":
    main()
