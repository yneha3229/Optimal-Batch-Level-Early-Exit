"""Trace-replay simulator. Replays batches drawn from logged exit traces under a policy and charges latency with
the measured per-layer cost c(n) = a + b*n, plus a per-exit overhead for every exit point evaluated:

    T_batch = sum_l ( a * 1[n_l > 0] + b * n_l ) + head * (#exit points evaluated)

Policy interface (what FunSearch/OpenEvolve rewrites, what the bandit tunes):
    decide(exit_idx, active_confs, n_active, batch_size, layers_left, a, b) -> (exit_mask, stop_batch)
exit_mask[j] = True sends active sample j out at this exit with its own prediction; stop_batch = True ends the
batch for every remaining sample (their prediction is the current exit's, or a rescue combination of the exits they
passed; see simulator/rescue.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

Decide = Callable[..., tuple]


@dataclass(frozen=True)
class Cost:
    name: str
    a: float          # ms, fixed per-layer cost while any sample is active
    b: float          # ms per active sample per layer
    head: float = 0.0  # ms per exit point evaluated (head + decision + sync; roughly flat in n)
    skip: float = 0.0  # ms per layer skipped by a collective stop (decoder: KV propagation for later tokens)
    exit_pass: float = 0.0  # ms for producing predictions when only *some* rows leave at a non-final exit
                            # (decoder: an extra 152k-vocab lm_head pass; classifiers: already in `head`)

    def rho(self, n: int) -> float:
        return self.a / (self.b * n)


@dataclass
class Traces:
    conf: np.ndarray   # [N, E] max softmax probability at each exit
    pred: np.ndarray   # [N, E]
    label: np.ndarray  # [N]
    exit_layers: list[int]
    probs: np.ndarray | None = None   # [N, E, C] full probability vectors (needed by rescue)
    ids: np.ndarray | None = None     # original validation index of each row
    attacker: np.ndarray | None = None  # [N] bool, rows that are slowdown-attack inputs

    @property
    def correct(self) -> np.ndarray:
        return self.pred == self.label[:, None]

    def __len__(self) -> int:
        return len(self.label)

    def subset(self, rows: np.ndarray) -> "Traces":
        pick = lambda x: None if x is None else x[rows]
        return Traces(self.conf[rows], self.pred[rows], self.label[rows], self.exit_layers, pick(self.probs),
                      pick(self.ids), pick(self.attacker))

    @staticmethod
    def concat(a: "Traces", b: "Traces") -> "Traces":
        cat = lambda x, y: None if x is None or y is None else np.concatenate([x, y])
        att = np.concatenate([a.attacker if a.attacker is not None else np.zeros(len(a), bool),
                              b.attacker if b.attacker is not None else np.zeros(len(b), bool)])
        return Traces(np.concatenate([a.conf, b.conf]), np.concatenate([a.pred, b.pred]),
                      np.concatenate([a.label, b.label]), a.exit_layers, cat(a.probs, b.probs), cat(a.ids, b.ids), att)


def load_traces(path: str | Path, split: str | None = None, exit_layers: list[int] | None = None,
                with_probs: bool = False) -> Traces:
    df = pd.read_parquet(path)
    if split is not None:
        df = df[df.split == split]
    E = sum(c.startswith("conf_") for c in df.columns)
    probs = None
    if with_probs:
        C = sum(c.startswith("p_0_") for c in df.columns)
        probs = np.stack([df[[f"p_{e}_{c}" for c in range(C)]].to_numpy(np.float32) for e in range(E)], 1)
    return Traces(
        conf=df[[f"conf_{e}" for e in range(E)]].to_numpy(np.float64),
        pred=df[[f"pred_{e}" for e in range(E)]].to_numpy(np.int64),
        label=df["label"].to_numpy(np.int64),
        exit_layers=exit_layers or [2, 4, 6, 8, 10, 12],
        probs=probs,
        ids=df["idx"].to_numpy() if "idx" in df.columns else None,
        attacker=df["attacker"].to_numpy(bool) if "attacker" in df.columns else None,
    )


def full_latency(exit_layers: list[int], cost: Cost, batch_size: int) -> float:
    """No-exit baseline: every layer at full batch, one head at the end."""
    return exit_layers[-1] * (cost.a + cost.b * batch_size) + cost.head


@dataclass
class BatchOut:
    latency: float            # ms, whole batch (= every sample's latency under synchronous return)
    exit_of: np.ndarray       # [B] exit index each sample left at
    by_stop: np.ndarray       # [B] True if the sample was cut off by a collective stop
    pred: np.ndarray          # [B] returned prediction
    ready: np.ndarray         # [B] ms at which the sample's answer was available (asynchronous return)
    sample_layers: int        # sum over layers of active samples (FLOPs proxy)
    trajectory: list = field(default_factory=list)  # (exit_idx, n_active_before_decision, stop) per decision


def run_batch(decide: Decide, tr: Traces, rows: np.ndarray, cost: Cost, compact: bool = True,
              rescue: Callable | None = None, record: bool = False) -> BatchOut:
    """Run one batch (rows of tr) under a policy.
    compact=True: exited samples leave the batch, so cost uses n_active.
    compact=False: exited samples are masked but still computed until the batch ends (most research code)."""
    B = len(rows)
    E = tr.conf.shape[1]
    L = tr.exit_layers
    active = np.ones(B, bool)
    exit_of = np.full(B, E - 1)
    by_stop = np.zeros(B, bool)
    ready = np.zeros(B)
    t, prev, sl = 0.0, 0, 0
    traj = []
    for e, layer in enumerate(L):
        n = int(active.sum())
        n_cost = n if compact else B
        t += (layer - prev) * (cost.a + cost.b * n_cost) + cost.head
        sl += (layer - prev) * n
        prev = layer
        if e == E - 1:
            ready[active] = t
            break
        ids = np.flatnonzero(active)
        if getattr(decide, "history", False):     # trajectory-aware policies get confs at exits 0..e, [n, e+1]
            c_in = tr.conf[rows[ids], :e + 1].copy()
        else:
            c_in = tr.conf[rows[ids], e].copy()
        exit_mask, stop = decide(e, c_in, n, B, L[-1] - layer, cost.a, cost.b)
        exit_mask = np.asarray(exit_mask, bool).reshape(-1)
        if exit_mask.shape != (n,):
            raise ValueError(f"invalid mask shape {exit_mask.shape}, expected ({n},)")
        stop = bool(stop)
        if record:
            traj.append((e, n, stop))
        leaving = ids if stop else ids[exit_mask]
        if stop:
            by_stop[ids[~exit_mask]] = True
            t += (L[-1] - layer) * cost.skip
        elif exit_mask.any():
            t += cost.exit_pass
        exit_of[leaving] = e
        ready[leaving] = t
        active[leaving] = False
        if not active.any():
            break
    pred = tr.pred[rows, exit_of].copy()
    if rescue is not None and by_stop.any():
        cut = np.flatnonzero(by_stop)
        pred[cut] = rescue(tr, rows[cut], exit_of[cut])
    return BatchOut(t, exit_of, by_stop, pred, ready, sl, traj)


def draw_batches(n_rows: int, batch_size: int, n_batches: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.choice(n_rows, batch_size, replace=False) for _ in range(n_batches)]


def simulate(decide: Decide, tr: Traces, cost: Cost, batch_size: int, n_batches: int, seed: int = 0,
             compact: bool = True, exits_enabled: bool = True, rescue: Callable | None = None,
             batches: list[np.ndarray] | None = None, per_sample: bool = False) -> dict:
    """Aggregate metrics over random batches. exits_enabled=False is the plain full model (no intermediate heads).
    per_sample=True adds arrays rows/correct/exit_of/by_stop/latency for each simulated sample."""
    if not exits_enabled:
        r = simulate(lambda *a: (np.zeros(a[2], bool), False), tr, Cost(cost.name, cost.a, cost.b, 0.0),
                     batch_size, n_batches, seed, batches=batches, per_sample=per_sample)
        return {**r, "rel_latency": 1.0, "speedup": 1.0}
    batches = batches if batches is not None else draw_batches(len(tr), batch_size, n_batches, seed)
    full = full_latency(tr.exit_layers, cost, batch_size)
    outs = [run_batch(decide, tr, rows, cost, compact, rescue) for rows in batches]
    rows = np.concatenate(batches)
    correct = np.concatenate([o.pred for o in outs]) == tr.label[rows]
    lat = np.array([o.latency for o in outs])
    rel = lat.mean() / full
    r = {"accuracy": float(correct.mean()), "rel_latency": float(rel), "speedup": float(1.0 / rel),
         "latency_ms": float(lat.mean()), "full_ms": float(full),
         "flops_frac": float(sum(o.sample_layers for o in outs) / (len(rows) * tr.exit_layers[-1])),
         "frac_cut": float(np.mean(np.concatenate([o.by_stop for o in outs])))}
    if per_sample:
        r.update(rows=rows, correct=correct, exit_of=np.concatenate([o.exit_of for o in outs]),
                 by_stop=np.concatenate([o.by_stop for o in outs]),
                 latency=np.repeat(lat, [len(b) for b in batches]),
                 ready=np.concatenate([o.ready for o in outs]))
    return r


def simulate_lanes(decide: Decide, tr: Traces, cost: Cost, batch_size: int, n_batches: int, lanes: int = 4,
                   seed: int = 0) -> dict:
    """Difficulty-aware lanes (Recall/DART-style, simplified): `lanes` arrival batches run their first segment
    and exit-0 decision; the survivors of the whole pool are sorted by exit-0 confidence (the difficulty
    predictor) and repacked into lanes of up to batch_size, and each lane continues under `decide`.
    Regrouping is charged nothing and the pool's queueing delay is ignored, both of which favour this baseline."""
    rng = np.random.default_rng(seed)
    L = tr.exit_layers
    full = full_latency(L, cost, batch_size)
    sub = Traces(tr.conf[:, 1:], tr.pred[:, 1:], tr.label, L[1:])
    shifted = lambda e, c, n, B, left, a, b: decide(e + 1, c, n, batch_size, left, a, b)
    seg0 = L[0] * (cost.a + cost.b * batch_size) + cost.head
    tot_lat, correct, sl, n_s = 0.0, 0, 0, 0
    n_pools = max(1, n_batches // lanes)
    for _ in range(n_pools):
        pool = rng.choice(len(tr), lanes * batch_size, replace=False)
        survivors = []
        for k in range(lanes):                                   # arrival batches: first segment + exit-0 decision
            rows = pool[k * batch_size:(k + 1) * batch_size]
            tot_lat += seg0
            sl += L[0] * batch_size
            mask, stop = decide(0, tr.conf[rows, 0].copy(), batch_size, batch_size, L[-1] - L[0], cost.a, cost.b)
            leave = np.ones(batch_size, bool) if stop else np.asarray(mask, bool)
            correct += int((tr.pred[rows[leave], 0] == tr.label[rows[leave]]).sum())
            survivors.append(rows[~leave])
        surv = np.concatenate(survivors)
        order = surv[np.argsort(-tr.conf[surv, 0])]
        for k in range(0, len(order), batch_size):               # lanes, hardest last
            rows = order[k:k + batch_size]
            o = run_batch(shifted, sub, rows, cost)
            n0 = len(rows)
            tot_lat += o.latency - L[0] * (cost.a + cost.b * n0)   # run_batch charged layers 0..L[0] again: remove
            sl += o.sample_layers - L[0] * n0
            correct += int((o.pred == tr.label[rows]).sum())
        n_s += lanes * batch_size
    rel = tot_lat / (n_pools * lanes * full)
    return {"accuracy": correct / n_s, "rel_latency": rel, "speedup": 1 / rel, "latency_ms": tot_lat / (n_pools * lanes),
            "full_ms": full, "flops_frac": sl / (n_s * L[-1]), "frac_cut": float("nan")}
