"""Step 1 of docs/PLAN_PAPER.md: log the CLS hidden state at every exit (validation set + a training subset), so a
probe can be trained to predict "this exit is correct" from more than the confidence.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python traces/extract_hidden.py mnli [--n-train 50000]
Output: traces/hidden/<task>_{val,train}.npz with cls [N,E,H] float16, probs [N,E,C] float32, label [N]
(validation rows are in the same order as traces/<task>_val.parquet idx).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.exit_model import build  # noqa: E402
from traces.train_exits import TextData, VisionData  # noqa: E402


@torch.inference_mode()
def run(model, batches):
    cls, probs = [], []
    for xb in batches:
        with torch.autocast("cuda", dtype=torch.float16):
            h, mask = model.embed(xb)
            layer, c, p = 0, [], []
            for e, stop in enumerate(model.exit_layers):
                while layer < stop:
                    h = model.layer(layer, h, mask)
                    layer += 1
                c.append(h[:, 0])
                p.append(model.head(e, h).float().softmax(-1))
        cls.append(torch.stack(c, 1).half().cpu())
        probs.append(torch.stack(p, 1).cpu())
    return torch.cat(cls).numpy(), torch.cat(probs).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("--n-train", type=int, default=50000)
    args = ap.parse_args()
    ck = torch.load(ROOT / f"checkpoints/{args.task}.pt", weights_only=False)
    cfg = ck["config"]
    model = build(cfg, load_backbone_weights=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model = model.cuda().eval()
    data = TextData(cfg, False) if cfg["kind"] == "text" else VisionData(cfg, False)
    out = ROOT / "traces/hidden"
    out.mkdir(exist_ok=True)
    bs = cfg.get("eval_batch", 256)
    cls, probs = run(model, data.eval_batches(bs))
    np.savez(out / f"{args.task}_val.npz", cls=cls, probs=probs, label=data.y_va)
    print(args.task, "val", cls.shape, "final acc", (probs[:, -1].argmax(-1) == data.y_va).mean(), flush=True)
    n = min(args.n_train, data.n_train())
    if cfg["kind"] == "text":
        tb = (data._collate(data.x_tr[i:min(i + bs, n)]) for i in range(0, n, bs))
    else:
        tb = (data._prep(data.x_tr[i:min(i + bs, n)], False) for i in range(0, n, bs))
    cls, probs = run(model, tb)
    np.savez(out / f"{args.task}_train.npz", cls=cls, probs=probs, label=data.y_tr[:n])
    print(args.task, "train", cls.shape, "final acc", (probs[:, -1].argmax(-1) == data.y_tr[:n]).mean(), flush=True)


if __name__ == "__main__":
    main()
