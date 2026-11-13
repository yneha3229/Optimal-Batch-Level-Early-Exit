"""DeepSloth-style slowdown attack on the early-exit ViT (Hong et al., ICLR 2021, re-implemented, not their code).

PGD in the 32x32 image space (before the resize to 224), L-inf budget eps, pushes every internal exit towards the
uniform distribution (loss = sum_e CE(logits_e, uniform)), so per-sample exits never fire and the input runs to
full depth. Attack images are taken from the *evolve* split, so they never coincide with the held-out victims.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python attacks/deepsloth_vit.py --n 500
Output: traces/cifar100_attack.parquet (same schema as the clean traces, plus attacker=True)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.exit_model import build  # noqa: E402


def to_input(x32: torch.Tensor, res: int) -> dict:
    x = F.interpolate(x32, size=res, mode="bilinear", align_corners=False)
    return {"pixel_values": (x - 0.5) / 0.5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--eps", type=float, default=8 / 255)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=50)
    args = ap.parse_args()
    import torchvision
    ck = torch.load(ROOT / "checkpoints/cifar100.pt", weights_only=False)
    cfg = ck["config"]
    model = build(cfg, load_backbone_weights=False)
    model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    model = model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tr = pd.read_parquet(ROOT / "traces/cifar100_val.parquet", columns=["idx", "split"])
    rows = tr[tr.split == "evolve"].idx.to_numpy()[:args.n]
    va = torchvision.datasets.CIFAR100(ROOT / "data", train=False, download=False)
    X = torch.from_numpy(va.data[rows]).permute(0, 3, 1, 2).float().div(255).cuda()
    Y = np.array(va.targets)[rows]
    alpha = 2.5 * args.eps / args.steps
    probs_all = []
    for i in range(0, len(X), args.bs):
        x0 = X[i:i + args.bs]
        x = (x0 + torch.empty_like(x0).uniform_(-args.eps, args.eps)).clamp(0, 1)
        for _ in range(args.steps):
            x.requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(to_input(x, cfg["image_size"])).float()       # [b, E, C]
            C = logits.shape[-1]
            uni = torch.full_like(logits[:, 0], 1.0 / C)
            loss = sum(F.cross_entropy(logits[:, e], uni) for e in range(logits.shape[1] - 1))
            g, = torch.autograd.grad(loss, x)
            with torch.no_grad():
                x = x - alpha * g.sign()                                         # minimise CE to uniform
                x = (x0 + (x - x0).clamp(-args.eps, args.eps)).clamp(0, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            probs_all.append(model(to_input(x, cfg["image_size"])).float().softmax(-1).cpu())
        print(f"{i + len(x0)}/{len(X)}", flush=True)
    P = torch.cat(probs_all).numpy()
    N, E, C = P.shape
    cols = {"idx": rows, "label": Y, "split": "attack", "attacker": np.ones(N, bool)}
    for e in range(E):
        cols[f"conf_{e}"], cols[f"pred_{e}"] = P[:, e].max(-1), P[:, e].argmax(-1)
    for e in range(E):
        for c in range(C):
            cols[f"p_{e}_{c}"] = P[:, e, c].astype(np.float32)
    df = pd.DataFrame(cols)
    df.to_parquet(ROOT / "traces/cifar100_attack.parquet", index=False)
    clean = pd.read_parquet(ROOT / "traces/cifar100_val.parquet")
    clean = clean[clean.idx.isin(rows)]
    for e in range(E):
        print(f"exit {e}: mean conf clean {clean[f'conf_{e}'].mean():.3f} -> attacked {df[f'conf_{e}'].mean():.3f}; "
              f"acc clean {(clean[f'pred_{e}'] == clean.label).mean():.3f} -> attacked {(df[f'pred_{e}'] == df.label).mean():.3f}")


if __name__ == "__main__":
    main()
