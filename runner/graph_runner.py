"""Step 2 of docs/PLAN_PAPER.md: the real-GPU runner with every layer segment captured as a CUDA graph.

Same host-side decide() code and the same one-sync-per-exit protocol as runner/gpu_runner.py, but the model code runs
as replayed CUDA graphs, so Python and kernel-launch overhead per layer is gone (what a compiled serving stack sees).
CUDA graphs need static shapes, so the active rows are padded up to the next bucket size (BUCKETS), as a serving
engine with shape profiles would do. The no-exit baseline is one graph of the whole model.

One run does: calibrate (a, b, head from full / heads-only time vs batch size) -> select policies in the simulator
under that cost on the evolve split (within 1 point) -> time them on the held-out split.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python runner/graph_runner.py --task mnli
Outputs: results/runner_cost_graph_<task>.json, results/graph_<task>_all.csv, results/graph_<task>_gpu.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runner.gpu_runner import DEV, load_inputs, never, slice_batch, time_policy  # noqa: E402
from simulator.exit_model import build  # noqa: E402
from simulator.simulator import Cost  # noqa: E402

BUCKETS = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
FAMS = ["threshold", "threshold_nocompact", "majority", "consensus", "unified", "random_collective",
        "theory_indiv_only", "theory"]


def bucket(n: int) -> int:
    return next(b for b in BUCKETS if b >= n)


class GraphRunner:
    def __init__(self, ck_path: Path, example: dict) -> None:
        ck = torch.load(ck_path, weights_only=False)
        self.cfg = ck["config"]
        self.model = build(self.cfg, load_backbone_weights=False)
        self.model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model = self.model.to(DEV).half().eval()
        self.L = self.model.exit_layers
        self.example = example                      # one row of inputs, gives static shapes and dtypes
        self.conf_host = torch.empty(4096, dtype=torch.float32, pin_memory=True)
        self.seg0, self.seg, self.full = {}, {}, {}
        with torch.inference_mode():
            h, mask = self.model.embed(example)
        self.hshape, self.mshape = tuple(h.shape[1:]), None if mask is None else tuple(mask.shape[1:])

    @staticmethod
    def _capture(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        return g, out

    def _static_inputs(self, B: int) -> dict:
        return {k: torch.zeros((B, *v.shape[1:]), dtype=v.dtype, device=DEV) + (1 if k == "attention_mask" else 0)
                for k, v in self.example.items()}

    @torch.inference_mode()
    def prepare(self, B: int) -> None:
        """Capture every graph a batch of size B can need."""
        m, L = self.model, self.L
        if B not in self.seg0:
            inp = self._static_inputs(B)

            def first():
                h, mask = m.embed(inp)
                for l in range(L[0]):
                    h = m.layer(l, h, mask)
                conf, pred = m.head(0, h).float().softmax(-1).max(-1)
                return h, mask, conf, pred
            self.seg0[B] = (inp, *self._capture(first))
            inp2 = self._static_inputs(B)

            def whole():
                h, mask = m.embed(inp2)
                for l in range(L[-1]):
                    h = m.layer(l, h, mask)
                return m.head(len(L) - 1, h).argmax(-1)
            self.full[B] = (inp2, *self._capture(whole))
        for nb in [b for b in BUCKETS if b <= B]:
            for e in range(1, len(L)):
                if (e, nb) in self.seg:
                    continue
                h_in = torch.zeros((nb, *self.hshape), dtype=torch.half, device=DEV)
                m_in = None if self.mshape is None else torch.zeros((nb, *self.mshape), dtype=torch.half, device=DEV)

                def step(e=e, h_in=h_in, m_in=m_in):
                    h = h_in
                    for l in range(L[e - 1], L[e]):
                        h = m.layer(l, h, m_in)
                    conf, pred = m.head(e, h).float().softmax(-1).max(-1)
                    return h, conf, pred
                self.seg[(e, nb)] = (h_in, m_in, *self._capture(step))

    @torch.inference_mode()
    def run_batch(self, batch: dict, decide, cost: Cost, compact: bool = True, exits_enabled: bool = True,
                  rescue: str = "last") -> torch.Tensor:
        L, E = self.L, len(self.L)
        B = next(iter(batch.values())).shape[0]
        if not exits_enabled:
            inp, g, out = self.full[B]
            for k, v in batch.items():
                inp[k].copy_(v)
            g.replay()
            return out.clone()
        inp, g, (h, mask, conf, pred) = self.seg0[B]
        for k, v in batch.items():
            inp[k].copy_(v)
        g.replay()
        rows = np.arange(B)                       # host: sample index of each live row (first len(rows) rows of h)
        rows_t = torch.arange(B, device=DEV)
        live = np.ones(B, bool)                   # only used when compact=False
        preds = torch.full((B,), -1, dtype=torch.long, device=DEV)
        for e in range(E):
            n = len(rows)
            if e == E - 1:
                if compact:
                    preds[rows_t] = pred[:n]
                else:
                    idx = torch.from_numpy(np.flatnonzero(live)).to(DEV)
                    preds[idx] = pred[idx]
                break
            ch = self.conf_host[:n]
            ch.copy_(conf[:n], non_blocking=True)
            torch.cuda.current_stream().synchronize()          # the one sync per exit
            c = ch.numpy().astype(np.float64)
            act = np.flatnonzero(live[rows]) if not compact else np.arange(n)
            mask_e, stop = decide(e, c[act], len(act), B, L[-1] - L[e], cost.a, cost.b)
            leave = np.zeros(n, bool)
            leave[act] = True if stop else np.asarray(mask_e, bool)
            if leave.any():
                li = torch.from_numpy(np.flatnonzero(leave)).to(DEV)
                preds[rows_t[li]] = pred[li]
            if compact:
                keep = np.flatnonzero(~leave)
                if len(keep) == 0:
                    break
            else:
                live[rows[leave]] = False
                if not live.any():
                    break
                keep = np.arange(n)
            nb = bucket(len(keep))
            h_in, m_in, g, (h2, conf2, pred2) = self.seg[(e + 1, nb)]
            if len(keep) == n and nb == h.shape[0]:
                h_in.copy_(h)
                if m_in is not None:
                    m_in.copy_(mask)
            else:
                pad = np.concatenate([keep, np.full(nb - len(keep), keep[0])])   # pad rows repeat a live row
                k = torch.from_numpy(pad).to(DEV)
                h_in.copy_(h.index_select(0, k))
                if m_in is not None:
                    m_in.copy_(mask.index_select(0, k))
                rows_t = rows_t[k[:len(keep)]]
                rows = rows[keep]
            g.replay()
            h, mask, conf, pred = h2, m_in, conf2, pred2
        return preds


def select(task: str, cost: Cost, Bs: list[int], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate every candidate of FAMS under `cost`; pick per (family, B) the fastest within 1 point on evolve."""
    from experiments import run_main as RM
    RM.init_worker(task)
    RM.G["costs"][cost.name] = cost
    jobs = [(cost.name, B, f) for B in Bs for f in ["full"] + FAMS]
    with Pool(workers) as pool:                    # fork: workers inherit RM.G
        rows = [r for rs in pool.imap_unordered(RM.job, jobs, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    picks = []
    for B, g in df[df.split == "evolve"].groupby("B"):
        full_ev = g[g.family == "full"].accuracy.iloc[0]
        for fam, gf in g.groupby("family"):
            ok = gf[gf.accuracy >= full_ev - RM.ACC_TOL]
            if ok.empty:
                continue
            best = ok.loc[ok.speedup.idxmax()]
            p, decide, kw = RM.candidates(fam, cost, int(B))[int(best.param_id)]
            ho = RM.sim(decide, RM.G["ho"], cost, int(B), 1000 * int(B) + 1, kw)
            picks.append({"B": int(B), "family": fam, "param_id": int(best.param_id), "params": best.params,
                          "evolve_speedup": best.speedup, "sim_speedup": ho["speedup"], "sim_acc": ho["accuracy"]})
    return df, pd.DataFrame(picks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[8, 16, 64, 128])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    task = args.task
    tr = pd.read_parquet(ROOT / f"traces/{task}_val.parquet", columns=["idx", "split", "pred_5"])
    ho = tr[tr.split == "heldout"].idx.to_numpy()
    cfg = torch.load(ROOT / f"checkpoints/{task}.pt", weights_only=False)["config"]
    inp, labels = load_inputs(task, cfg, ho)
    run = GraphRunner(ROOT / f"checkpoints/{task}.pt", slice_batch(inp, 0, 1))
    zero = Cost("none", 0.0, 0.0, 0.0)
    cal_B = [1, 2, 4, 8, 16, 32, 64, 128]
    for B in sorted(set(cal_B + args.batch_sizes)):
        run.prepare(B)
    print(f"graphs captured: {len(run.seg) + 2 * len(run.seg0)}, GPU memory {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    ref = tr.set_index("idx").loc[ho[:256], "pred_5"].to_numpy()
    p_full = torch.cat([run.run_batch(slice_batch(inp, i, i + 64), never, zero, exits_enabled=False) for i in range(0, 256, 64)]).cpu().numpy()
    p_seg = torch.cat([run.run_batch(slice_batch(inp, i, i + 64), never, zero) for i in range(0, 256, 64)]).cpu().numpy()
    print(f"sanity: agreement with trace final exit: full graph {(p_full == ref).mean():.4f}, segment graphs {(p_seg == ref).mean():.4f}", flush=True)

    # 1. calibrate
    rows = []
    for B in cal_B:
        n = min(len(labels) // B * B, 2048 if B >= 16 else 256 * B)
        t_full, _ = time_policy(run, inp, n, B, never, zero, args.repeats, exits_enabled=False)
        t_heads, _ = time_policy(run, inp, n, B, never, zero, args.repeats)
        rows.append({"B": B, "full_ms_per_batch": 1000 * t_full / (n // B), "heads_ms_per_batch": 1000 * t_heads / (n // B)})
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    Lf = run.L[-1]
    b, a12 = np.polyfit(d.B, d.full_ms_per_batch, 1)
    head = float(np.median((d.heads_ms_per_batch - d.full_ms_per_batch) / (len(run.L) - 1)))
    cj = {"task": task, "mode": "cuda_graph", "a": float(a12 / Lf), "b": float(b / Lf), "head": head,
          "rho8": float(a12 / b / 8), "rho64": float(a12 / b / 64), "points": rows}
    (ROOT / f"results/runner_cost_graph_{task}.json").write_text(json.dumps(cj, indent=1))
    print(f"graph cost: a={cj['a']:.4f} ms b={cj['b']:.5f} ms/sample head={head:.4f} ms rho8={cj['rho8']:.2f} rho64={cj['rho64']:.3f}", flush=True)
    cost = Cost(f"graph_{task}", cj["a"], cj["b"], max(head, 0.0))

    # 2. select in the simulator under the graph cost
    df, sel = select(task, cost, args.batch_sizes, args.workers)
    df.to_csv(ROOT / f"results/graph_{task}_all.csv", index=False)
    print("selection done", flush=True)

    # 3. time on the GPU
    from experiments import run_main as RM
    out = []
    for B in args.batch_sizes:
        n = len(labels) // B * B
        for fam in ["full", "heads_only"] + FAMS:
            s = sel[(sel.B == B) & (sel.family == (fam if fam != "heads_only" else "full"))]
            if s.empty:
                continue
            if fam in ("full", "heads_only"):
                params, decide, kw = {}, never, {"exits_enabled": fam == "heads_only"}
            else:
                params, decide, kw = RM.candidates(fam, cost, B)[int(s.param_id.iloc[0])]
                kw = {k: v for k, v in kw.items() if k == "compact"}
            wall, preds = time_policy(run, inp, n, B, decide, cost, args.repeats, **kw)
            out.append({"B": B, "family": fam, "params": json.dumps(params), "wall_s": wall,
                        "accuracy": float((preds == labels[:n]).mean()),
                        "sim_speedup": float(s.sim_speedup.iloc[0]) if fam != "heads_only" else np.nan,
                        "sim_acc": float(s.sim_acc.iloc[0])})
            print(f"B={B:4d} {fam:22s} {wall:.3f}s acc {out[-1]['accuracy']:.4f}", flush=True)
    g = pd.DataFrame(out)
    ft, fa = g[g.family == "full"].set_index("B").wall_s, g[g.family == "full"].set_index("B").accuracy
    g["real_speedup"] = g.apply(lambda r: ft[r.B] / r.wall_s, axis=1)
    g["acc_drop_pts"] = g.apply(lambda r: 100 * (fa[r.B] - r.accuracy), axis=1)
    g.to_csv(ROOT / f"results/graph_{task}_gpu.csv", index=False)
    print(g[["B", "family", "accuracy", "acc_drop_pts", "real_speedup", "sim_speedup"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
