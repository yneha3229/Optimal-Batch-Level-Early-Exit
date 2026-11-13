"""P2a: train exit heads on a frozen MNLI-fine-tuned RoBERTa-base (DeeBERT stage-2 style) and log traces.

Heads see only the CLS hidden state at their exit layer, so we cache CLS states once and train heads on them.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python traces/train_exit_heads.py configs/p2_mnli.yaml

Outputs:
  traces/heads_mnli.pt          state dicts of the exit heads
  traces/mnli_val.parquet       per validation sample: label, split (evolve/heldout), conf/pred/probs at every exit
  results/p2_exit_accuracy.csv  accuracy per exit on each split
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.exit_model import ExitHead  # noqa: E402


@torch.inference_mode()
def cls_features(backbone, tok, premises, hypotheses, exits, max_len, bs) -> torch.Tensor:
    """CLS hidden state at each exit layer: [N, E, H] float16 on CPU."""
    out = []
    for i in range(0, len(premises), bs):
        enc = tok(premises[i:i + bs], hypotheses[i:i + bs], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to("cuda")
        hs = backbone(**enc, output_hidden_states=True).hidden_states  # hs[l] = output of layer l
        out.append(torch.stack([hs[l][:, 0] for l in exits], 1).cpu())
    return torch.cat(out)


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    exits = cfg["exit_layers"]

    tok = AutoTokenizer.from_pretrained(cfg["backbone"])
    full = AutoModelForSequenceClassification.from_pretrained(cfg["backbone"], torch_dtype=torch.float16).cuda().eval()
    backbone = full.roberta
    ds = load_dataset("nyu-mll/glue", "mnli")

    train = ds["train"].shuffle(seed=cfg["seed"]).select(range(cfg["train_samples"]))
    val = ds["validation_matched"]
    t0 = time.time()
    x_tr = cls_features(backbone, tok, train["premise"], train["hypothesis"], exits, cfg["max_len"], cfg["feat_batch"])
    x_va = cls_features(backbone, tok, val["premise"], val["hypothesis"], exits, cfg["max_len"], cfg["feat_batch"])
    y_tr, y_va = torch.tensor(train["label"]), torch.tensor(val["label"])
    print(f"features: train {tuple(x_tr.shape)} val {tuple(x_va.shape)} in {time.time() - t0:.0f}s", flush=True)

    H, C = x_tr.shape[-1], cfg["num_labels"]
    heads = nn.ModuleList(ExitHead(H, C, dropout=cfg["dropout"]) for _ in exits).cuda()
    opt = torch.optim.AdamW(heads.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * (len(x_tr) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    x_tr_gpu, y_tr_gpu = x_tr.cuda().float(), y_tr.cuda()
    loss_fn = nn.CrossEntropyLoss()
    heads.train()
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(len(x_tr_gpu), device="cuda")
        tot = 0.0
        for i in range(0, len(perm) - cfg["batch_size"] + 1, cfg["batch_size"]):
            idx = perm[i:i + cfg["batch_size"]]
            xb, yb = x_tr_gpu[idx], y_tr_gpu[idx]
            loss = sum(loss_fn(h(xb[:, e]), yb) for e, h in enumerate(heads))  # equal weight per exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
        print(f"epoch {epoch} loss {tot / (len(perm) // cfg['batch_size']):.4f}", flush=True)

    heads.eval()
    with torch.inference_mode():
        probs = torch.stack([h(x_va.cuda().float()[:, e]).softmax(-1) for e, h in enumerate(heads)], 1).cpu().numpy()
    conf, pred = probs.max(-1), probs.argmax(-1)
    split = np.where(rng.permutation(len(y_va)) < len(y_va) // 2, "evolve", "heldout")
    rows = {"idx": np.arange(len(y_va)), "label": y_va.numpy(), "split": split}
    for e in range(len(exits)):
        rows[f"conf_{e}"], rows[f"pred_{e}"] = conf[:, e], pred[:, e]
        for c in range(C):
            rows[f"p_{e}_{c}"] = probs[:, e, c]
    df = pd.DataFrame(rows)
    df.to_parquet(ROOT / "traces/mnli_val.parquet", index=False)
    torch.save({"exit_layers": exits, "hidden": H, "num_labels": C, "state_dict": heads.state_dict(),
                "backbone": cfg["backbone"], "max_len": cfg["max_len"]}, ROOT / "traces/heads_mnli.pt")

    acc = [{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
            "mean_conf": float(g[f"conf_{e}"].mean())}
           for e, l in enumerate(exits) for s, g in [("all", df), *df.groupby("split")]]
    acc_df = pd.DataFrame(acc)
    acc_df.to_csv(ROOT / "results/p2_exit_accuracy.csv", index=False)
    print(acc_df[acc_df.split == "all"].round(4).to_string(index=False))
    # oracle: fraction of samples whose earliest correct-and-stays-correct exit is e (shows exit potential)
    correct = np.stack([df[f"pred_{e}"] == df.label for e in range(len(exits))], 1)
    print(f"oracle accuracy (any exit correct): {correct.any(1).mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs/p2_mnli.yaml"))
