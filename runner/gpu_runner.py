"""Real batched early-exit inference with collective termination, for wall-clock numbers.

The runner executes the *same* numpy decide() objects the simulator scored (rebuilt from the selected param_id
via experiments/run_main.candidates), so the GPU run and the simulation use identical policy code, including the
bandit snapshot and FunSearch programs. Per exit point the host syncs exactly once: it copies the n confidences
(n floats), decides on the CPU (microseconds for n <= 128), and sends the surviving row indices back with a
non-blocking copy. Compaction then uses index_select, which needs no further sync.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python runner/gpu_runner.py --task mnli --calibrate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python runner/gpu_runner.py --task mnli --regime real_mnli
--calibrate measures this code path's own cost model end to end (a, b from full-model time vs batch size; per-exit
overhead from heads_only - full) and writes results/runner_cost_<task>.json; run_main.py then uses it as the
`real_<task>` regime, so the simulator is calibrated against exactly what is timed here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.exit_model import EarlyExitModel, build  # noqa: E402
from simulator.simulator import Cost  # noqa: E402

DEV = "cuda"


def never(e, c, n, B, left, a, b):
    return np.zeros(n, bool), False


class Runner:
    def __init__(self, ck_path: Path) -> None:
        ck = torch.load(ck_path, weights_only=False)
        self.cfg = ck["config"]
        self.model: EarlyExitModel = build(self.cfg, load_backbone_weights=False)
        self.model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model = self.model.to(DEV).half().eval()
        self.L = self.model.exit_layers
        self.conf_host = torch.empty(4096, dtype=torch.float32, pin_memory=True)

    @torch.inference_mode()
    def run_batch(self, batch: dict, decide, cost: Cost, compact: bool = True, exits_enabled: bool = True,
                  rescue: str = "last") -> torch.Tensor:
        """Returns predictions [B] on the GPU. compact=False keeps exited rows in the batch (wait-for-all)."""
        m, L, E = self.model, self.L, len(self.L)
        h, mask = m.embed(batch)
        B = h.shape[0]
        rows = np.arange(B)                       # host copy: row of h -> sample index
        rows_t = torch.arange(B, device=DEV)
        live = np.ones(B, bool)                   # only used when compact=False
        preds = torch.full((B,), -1, dtype=torch.long, device=DEV)
        acc = None
        layer = 0
        hist = np.zeros((B, E))                   # host copy of every sample's confidence at each exit passed
        want_hist = getattr(decide, "history", False)
        for e, stop_at in enumerate(L):
            while layer < stop_at:
                h = m.layer(layer, h, mask)
                layer += 1
            if not exits_enabled and e < E - 1:
                continue
            probs = m.head(e, h).float().softmax(-1)
            conf, pred = probs.max(-1)
            if rescue == "conf":
                acc = probs * conf[:, None] if acc is None else acc + probs * conf[:, None]
            if e == E - 1:
                if compact:
                    preds[rows_t] = pred
                else:
                    idx = torch.from_numpy(np.flatnonzero(live[rows])).to(DEV, non_blocking=True)
                    preds[rows_t[idx]] = pred[idx]
                break
            n_rows = len(rows)
            ch = self.conf_host[:n_rows]
            ch.copy_(conf, non_blocking=True)
            torch.cuda.current_stream().synchronize()          # the one sync per exit
            c = ch.numpy().astype(np.float64)
            hist[rows, e] = c
            act = np.flatnonzero(live[rows]) if not compact else np.arange(n_rows)
            c_in = hist[rows[act], :e + 1] if want_hist else c[act]
            mask_e, stop = decide(e, c_in, len(act), B, L[-1] - stop_at, cost.a, cost.b)
            mask_e = np.asarray(mask_e, bool)
            leave = np.zeros(n_rows, bool)
            leave[act] = True if stop else mask_e
            by_stop = np.zeros(n_rows, bool)
            if stop:
                by_stop[act[~mask_e]] = True
            if leave.any():
                li = torch.from_numpy(np.flatnonzero(leave)).to(DEV, non_blocking=True)
                out = pred
                if rescue == "conf" and by_stop.any():
                    bs = torch.from_numpy(by_stop).to(DEV, non_blocking=True)
                    out = torch.where(bs, acc.argmax(-1), pred)
                preds[rows_t[li]] = out[li]
            if compact:
                keep = np.flatnonzero(~leave)
                if len(keep) == 0:
                    break
                if len(keep) < n_rows:
                    k = torch.from_numpy(keep).to(DEV, non_blocking=True)
                    h, rows_t = h.index_select(0, k), rows_t.index_select(0, k)
                    mask = None if mask is None else mask.index_select(0, k)
                    acc = None if acc is None else acc.index_select(0, k)
                    rows = rows[keep]
            else:
                live[rows[leave]] = False
                if not live.any():
                    break
        return preds


def load_inputs(task: str, cfg: dict, rows: np.ndarray) -> tuple[dict, np.ndarray]:
    if cfg["kind"] == "text":
        from datasets import load_dataset
        from transformers import AutoTokenizer
        f1, f2, split = {"mnli": ("premise", "hypothesis", "validation_matched"),
                         "qqp": ("question1", "question2", "validation")}[task]
        va = load_dataset("nyu-mll/glue", task)[split].select(rows.tolist())
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        enc = tok(list(va[f1]), list(va[f2]), truncation=True, max_length=cfg["max_len"], padding="max_length",
                  return_tensors="pt")
        return {"input_ids": enc["input_ids"].to(DEV), "attention_mask": enc["attention_mask"].to(DEV)}, np.array(va["label"])
    import torch.nn.functional as F
    import torchvision
    va = torchvision.datasets.CIFAR100(ROOT / "data", train=False, download=False)
    x = torch.from_numpy(va.data[rows]).permute(0, 3, 1, 2).float().div(255)
    x = F.interpolate(x, size=cfg["image_size"], mode="bilinear", align_corners=False)
    return {"pixel_values": ((x - 0.5) / 0.5).half().to(DEV)}, np.array(va.targets)[rows]


def slice_batch(inp: dict, i: int, j: int) -> dict:
    return {k: v[i:j] for k, v in inp.items()}


def time_policy(run: Runner, inp: dict, n: int, B: int, decide, cost: Cost, repeats: int, **kw):
    run.run_batch(slice_batch(inp, 0, B), decide, cost, **kw)                 # warm-up
    times, preds = [], None
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        preds = torch.cat([run.run_batch(slice_batch(inp, i, i + B), decide, cost, **kw) for i in range(0, n, B)])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), preds.cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--regime", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out-suffix", default="", help="appended to the output CSV name (e.g. _B8)")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    run = Runner(ROOT / f"checkpoints/{args.task}.pt")
    tr = pd.read_parquet(ROOT / f"traces/{args.task}_val.parquet", columns=["idx", "split", "pred_5"])
    ho = tr[tr.split == "heldout"].idx.to_numpy()
    inp, labels = load_inputs(args.task, run.cfg, ho)
    zero = Cost("none", 0.0, 0.0, 0.0)

    p = run.run_batch(slice_batch(inp, 0, 256), never, zero, exits_enabled=False).cpu().numpy()
    agree = float((p == tr.set_index("idx").loc[ho[:256], "pred_5"].to_numpy()).mean())
    print(f"sanity: runner vs trace final-exit agreement {agree:.4f}", flush=True)

    if args.calibrate:
        rows = []
        for B in [1, 2, 4, 8, 16, 32, 64, 128]:
            n = min(len(labels) // B * B, 2048 if B >= 16 else 256 * B)
            t_full, _ = time_policy(run, inp, n, B, never, zero, args.repeats, exits_enabled=False)
            t_heads, _ = time_policy(run, inp, n, B, never, zero, args.repeats)
            nb = n // B
            rows.append({"B": B, "full_ms_per_batch": 1000 * t_full / nb, "heads_ms_per_batch": 1000 * t_heads / nb})
            print(rows[-1], flush=True)
        d = pd.DataFrame(rows)
        Lf = run.L[-1]
        b, a12 = np.polyfit(d.B, d.full_ms_per_batch, 1)
        head = float(np.median((d.heads_ms_per_batch - d.full_ms_per_batch) / (len(run.L) - 1)))
        out = {"task": args.task, "a": float(a12 / Lf), "b": float(b / Lf), "head": head,
               "rho64": float(a12 / b / 64), "points": rows}
        (ROOT / f"results/runner_cost_{args.task}.json").write_text(json.dumps(out, indent=1))
        print(f"end-to-end fit: a={out['a']:.4f} ms b={out['b']:.5f} ms/sample head={head:.4f} ms rho64={out['rho64']:.2f}")
        return

    from experiments import run_main as RM
    RM.init_worker(args.task)
    reg = args.regime or f"real_{args.task}"
    cost = RM.G["costs"][reg]
    sel = pd.read_csv(ROOT / f"results/main_{args.task}_selected.csv")
    sel = sel[sel.regime == reg]
    out_rows = []
    for B in args.batch_sizes:
        n = len(labels) // B * B
        fams = ["full", "heads_only"] + [f for f in sel[sel.B == B].family.unique() if f not in ("full", "lanes")]
        for fam in fams:
            s = sel[(sel.B == B) & (sel.family == (fam if fam != "heads_only" else "full"))]
            if s.empty:
                continue
            if fam in ("full", "heads_only"):
                params, decide, kw = {}, never, {"exits_enabled": fam == "heads_only"}
            else:
                params, decide, kw = RM.candidates(fam, cost, B)[int(s.param_id.iloc[0])]
                kw = {k: v for k, v in kw.items() if k == "compact"}
                kw["rescue"] = "conf" if params.get("rescue") == "conf" else "last"   # learned rescue: sim only
            wall, preds = time_policy(run, inp, n, B, decide, cost, args.repeats, **kw)
            out_rows.append({"B": B, "family": fam, "params": json.dumps(params), "wall_s": wall,
                             "accuracy": float((preds == labels[:n]).mean()),
                             "sim_speedup": float(s.heldout_speedup.iloc[0]) if fam != "heads_only" else np.nan,
                             "sim_acc": float(s.heldout_acc.iloc[0])})
            print(f"B={B:4d} {fam:24s} {wall:.3f}s acc {out_rows[-1]['accuracy']:.4f}", flush=True)
    df = pd.DataFrame(out_rows)
    ft = df[df.family == "full"].set_index("B").wall_s
    fa = df[df.family == "full"].set_index("B").accuracy
    df["real_speedup"] = df.apply(lambda r: ft[r.B] / r.wall_s, axis=1)
    df["acc_drop_pts"] = df.apply(lambda r: 100 * (fa[r.B] - r.accuracy), axis=1)
    df.to_csv(ROOT / f"results/main_{args.task}_{reg}_gpu{args.out_suffix}.csv", index=False)
    print(df[["B", "family", "accuracy", "acc_drop_pts", "real_speedup", "sim_speedup"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
