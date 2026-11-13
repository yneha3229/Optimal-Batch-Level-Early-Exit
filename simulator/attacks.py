"""Slowdown attacks on synchronous early-exit batches (simulation side).

In a synchronous batch every input gets its answer when the batch finishes, so one input that never becomes
confident (a DeepSloth/SAME-style slowdown input, or simply a very hard input) holds every co-batched user to
full depth under per-sample exits. We mix k attacker inputs into batches of B and measure what the *other*
B - k inputs (the victims) pay. Policies never see which input is the attacker.

Attacker pools:
  hardest_pool  label-free stand-in: the inputs whose confidence stays lowest across all exits (text tasks)
  attack traces from attacks/deepsloth_vit.py (real white-box slowdown inputs for ViT / CIFAR-100)
"""
from __future__ import annotations

import numpy as np

from simulator.simulator import Cost, Decide, Traces, full_latency, run_batch


def hardest_pool(tr: Traces, frac: float = 0.02) -> np.ndarray:
    """Rows whose mean confidence over all exits is in the bottom `frac` (no labels used)."""
    score = tr.conf.mean(1)
    k = max(1, int(round(frac * len(tr))))
    return np.argsort(score)[:k]


def victim_experiment(decide: Decide, clean: Traces, attack: Traces, cost: Cost, batch_size: int = 64,
                      ks: tuple[int, ...] = (0, 1, 2, 4), n_batches: int = 300, seed: int = 0,
                      rescue=None, compact: bool = True) -> list[dict]:
    """clean and attack are separate Traces (same exits). For each k, batches = (B - k) clean + k attack rows.
    Returns one row per k with victim latency (sync and async return), victim and attacker accuracy."""
    both = Traces.concat(clean, attack)
    n_clean = len(clean)
    full = full_latency(clean.exit_layers, cost, batch_size)
    out = []
    for k in ks:
        rng = np.random.default_rng(seed)          # same clean draws for every k
        lat, lat_async, v_ok, v_n, a_ok, a_n, deep = [], [], 0, 0, 0, 0, []
        for _ in range(n_batches):
            v = rng.choice(n_clean, batch_size - k, replace=False)
            a = n_clean + rng.choice(len(attack), k, replace=len(attack) < k)
            rows = np.concatenate([v, a])
            rng.shuffle(rows)
            o = run_batch(decide, both, rows, cost, compact=compact, rescue=rescue)
            is_v = rows < n_clean
            ok = o.pred == both.label[rows]
            lat.append(o.latency)
            lat_async.append(o.ready[is_v].mean())
            v_ok += int(ok[is_v].sum()); v_n += int(is_v.sum())
            a_ok += int(ok[~is_v].sum()); a_n += int((~is_v).sum())
            deep.append(clean.exit_layers[o.exit_of.max()])
        out.append({"k": k, "victim_ms": float(np.mean(lat)), "victim_ms_async": float(np.mean(lat_async)),
                    "victim_rel_full": float(np.mean(lat) / full), "victim_acc": v_ok / max(v_n, 1),
                    "attacker_acc": a_ok / a_n if a_n else float("nan"), "mean_depth": float(np.mean(deep))})
    base = out[0]["victim_ms"]
    for r in out:
        r["victim_slowdown"] = r["victim_ms"] / base
    return out
