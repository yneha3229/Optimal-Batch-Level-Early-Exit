"""Checks for GraphDevice (--exec graph) before the full CUDA-graph run (docs/PROMPT_CUDA_GRAPH_DEVICE.md).

1. Same predictions as Device.run_batch for the selected policies of results/local_select_<task>_<name>_T<T>.json
   (decisions come from the recorded traces, so they must match exactly), on held-out batches.
2. Device confidences at every exit: graph replay vs an eager forward of the same model on the same inputs,
   allclose(atol=1e-4) in fp32; padding rows never leak into the live rows.
3. No CUDA-graph capture inside time_pass after prepare().
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python experiments/check_graph_device.py --task mnli
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import local_hw as lh  # noqa: E402
from simulator.simulator import Cost, load_traces  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    ap.add_argument("--name", default="a6000")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
    ap.add_argument("--n-batches", type=int, default=4)
    a = ap.parse_args()
    lh.set_task(a.task)
    import torch
    args = argparse.Namespace(name=a.name, device="cuda", seq_len=a.seq_len, task=a.task, cost_model="table",
                              exec="eager")
    cost = lh.load_cost(args)                                     # eager cost: only used by decide(), same for both
    path = ROOT / f"traces/{a.task}_val.parquet"
    ev = load_traces(path, "evolve", exit_layers=list(lh.LAYERS))
    ho = load_traces(path, "heldout", exit_layers=list(lh.LAYERS))
    fit, _ = lh.split_evolve(ev, 0)
    sel = json.loads((ROOT / f"results/local_select_{a.task}_{a.name}_T{a.seq_len}.json").read_text())
    eager = lh.Device(a.task, "cuda", a.seq_len)
    graph = lh.GraphDevice(a.task, "cuda", a.seq_len)
    ok = True

    # 1. identical predictions under the selected policies
    n_cmp = 0
    for B in a.batch_sizes:
        pols = {"full": (lh.FULL, lh.never), "heads_only": (lh.ALL, lh.never)}
        for r in sel:
            if r["B"] == B and r["cert_params"] is not None:
                pols[r["family"]] = lh.make(r["family"], r["cert_params"], fit, cost)
        graph.prepare([B], [p for p, _ in pols.values()])
        batches = lh.disjoint_batches(len(ho), B, 100 + B, limit=a.n_batches)
        rng = np.random.default_rng(B)
        slots = [rng.choice(2048, B, replace=False) for _ in batches]
        for k, (plan, decide) in pols.items():
            for rows, s in zip(batches, slots):
                pe = eager.run_batch(s, plan, decide, ho.conf[rows], ho.pred[rows], cost)
                pg = graph.run_batch(s, plan, decide, ho.conf[rows], ho.pred[rows], cost)
                n_cmp += 1
                if not (pe == pg).all():
                    ok = False
                    print(f"MISMATCH B={B} {k}: {int((pe != pg).sum())} rows differ")
    print(f"1. predictions: {n_cmp} (batch, policy) pairs compared, {'all equal' if ok else 'MISMATCHES'}")

    # 2. confidences at every exit vs an eager forward (B not a bucket size, so padding rows are present)
    worst = 0.0
    zero = Cost("none", 0.0, 0.0, 0.0)
    for B in (5, 13, 64):
        slot = np.random.default_rng(7 + B).choice(2048, B, replace=False)
        graph.prepare([B], [lh.ALL])
        graph.record = []
        graph.run_batch(slot, lh.ALL, lh.never, np.zeros((B, len(lh.LAYERS))), np.zeros((B, len(lh.LAYERS)), np.int64), zero)
        rec, graph.record = graph.record, None
        with torch.inference_mode():
            st = torch.from_numpy(slot).cuda()
            logits = eager.model({k: v.index_select(0, st) for k, v in eager.inp.items()})     # [B, E, C]
            ce = logits.float().softmax(-1).max(-1)[0].cpu().numpy()                          # [B, E]
        for e, c in rec:
            worst = max(worst, float(np.abs(c - ce[:, e]).max()))
    ok2 = worst <= 1e-4
    ok &= ok2
    print(f"2. confidences vs eager forward: max |diff| = {worst:.2e} ({'ok' if ok2 else 'FAIL'}, atol 1e-4)")

    # 3. no capture inside a timed pass
    B = a.batch_sizes[0]
    batches = lh.disjoint_batches(len(ho), B, 100 + B, limit=a.n_batches)
    slots = [np.arange(B) for _ in batches]
    n0 = graph.captures
    graph.time_pass(batches, slots, lh.ALL, lh.never, ho, cost)
    ok3 = graph.captures == n0
    ok &= ok3
    print(f"3. captures during time_pass: {graph.captures - n0} ({'ok' if ok3 else 'FAIL'}); {graph.captures} graphs in total")
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
