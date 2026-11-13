"""Jointly fine-tune a backbone with all exit heads, then log traces (full probability vectors at every exit).

Joint training (backbone + heads, loss = layer-weighted sum of per-exit CE, as in DeeBERT/PABEE) replaces the
earlier frozen-backbone heads (traces/train_exit_heads.py), whose early exits were too weak (layer 6: 57%).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python traces/train_exits.py configs/train_mnli.yaml
  ... --traces-only        reload checkpoints/<task>.pt and only re-log traces
  ... --smoke              200 train samples, 512 eval samples, no files written over real outputs

Outputs (task = mnli | qqp | cifar100):
  checkpoints/<task>.pt                 fp16 state dict + config
  traces/<task>_val.parquet             idx, label, split (evolve/heldout), conf_e, pred_e, p_e_c for every exit e
  results/train_<task>_exit_accuracy.csv
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator.exit_model import EarlyExitModel, build  # noqa: E402

TEXT_FIELDS = {"mnli": ("premise", "hypothesis", "validation_matched"),
               "qqp": ("question1", "question2", "validation")}


# ----------------------------------------------------------------------------------------------- data
class TextData:
    def __init__(self, cfg: dict, smoke: bool) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer
        f1, f2, val_split = TEXT_FIELDS[cfg["task"]]
        ds = load_dataset("nyu-mll/glue", cfg["task"])
        tr, va = ds["train"].shuffle(seed=cfg["seed"]), ds[val_split]
        if cfg.get("train_samples"):
            tr = tr.select(range(min(cfg["train_samples"], len(tr))))
        if smoke:
            tr, va = tr.select(range(200)), va.select(range(512))
        self.tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
        self.max_len = cfg["max_len"]
        enc = lambda d: self.tok(list(d[f1]), list(d[f2]), truncation=True, max_length=self.max_len)["input_ids"]
        self.x_tr, self.y_tr = enc(tr), np.array(tr["label"])
        self.x_va, self.y_va = enc(va), np.array(va["label"])
        self.pad = self.tok.pad_token_id

    def _collate(self, seqs: list[list[int]]) -> dict:
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), self.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
        return {"input_ids": ids.cuda(non_blocking=True), "attention_mask": (ids != self.pad).long().cuda(non_blocking=True)}

    def train_batches(self, bs: int, rng: np.random.Generator):
        perm = rng.permutation(len(self.y_tr))
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            yield self._collate([self.x_tr[j] for j in idx]), torch.from_numpy(self.y_tr[idx]).cuda()

    def eval_batches(self, bs: int):
        for i in range(0, len(self.y_va), bs):
            yield self._collate(self.x_va[i:i + bs])

    def n_train(self) -> int:
        return len(self.y_tr)


class VisionData:
    """CIFAR-100 held on the GPU as uint8; augmentation (pad-crop, flip) and resize to 224 happen on the GPU."""
    MEAN, STD = 0.5, 0.5  # google/vit-base-patch16-224-in21k preprocessing

    def __init__(self, cfg: dict, smoke: bool) -> None:
        import torchvision
        root = ROOT / "data"
        tr = torchvision.datasets.CIFAR100(root, train=True, download=True)
        va = torchvision.datasets.CIFAR100(root, train=False, download=True)
        to_t = lambda d: torch.from_numpy(d.data).permute(0, 3, 1, 2).contiguous()  # [N,3,32,32] uint8
        self.x_tr, self.y_tr = to_t(tr).cuda(), np.array(tr.targets)
        self.x_va, self.y_va = to_t(va).cuda(), np.array(va.targets)
        if smoke:
            self.x_tr, self.y_tr, self.x_va, self.y_va = self.x_tr[:200], self.y_tr[:200], self.x_va[:512], self.y_va[:512]
        self.res = cfg["image_size"]

    def _prep(self, x: torch.Tensor, train: bool) -> dict:
        x = x.float() / 255.0
        if train:
            x = F.pad(x, (4, 4, 4, 4), mode="reflect")
            i, j = np.random.randint(0, 9, 2)
            x = x[:, :, i:i + 32, j:j + 32]
            flip = torch.rand(x.shape[0], device=x.device) < 0.5
            x = torch.where(flip[:, None, None, None], x.flip(3), x)
        x = F.interpolate(x, size=self.res, mode="bilinear", align_corners=False)
        return {"pixel_values": (x - self.MEAN) / self.STD}

    def train_batches(self, bs: int, rng: np.random.Generator):
        perm = rng.permutation(len(self.y_tr))
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i:i + bs]
            yield self._prep(self.x_tr[torch.from_numpy(idx).cuda()], True), torch.from_numpy(self.y_tr[idx]).cuda()

    def eval_batches(self, bs: int):
        for i in range(0, len(self.y_va), bs):
            yield self._prep(self.x_va[i:i + bs], False)

    def n_train(self) -> int:
        return len(self.y_tr)


# ----------------------------------------------------------------------------------------------- train / eval
def exit_weights(exit_layers: list[int]) -> torch.Tensor:
    w = torch.tensor(exit_layers, dtype=torch.float32)
    return (w / w.sum()).cuda()


def train(model: EarlyExitModel, data, cfg: dict) -> None:
    rng = np.random.default_rng(cfg["seed"])
    head_params = list(model.heads.parameters())
    head_ids = {id(p) for p in head_params}
    bb_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": bb_params, "lr": cfg["lr"]},
                             {"params": head_params, "lr": cfg["head_lr"]}], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * (data.n_train() // cfg["batch_size"])
    warm = max(1, int(cfg.get("warmup_frac", 0.06) * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / max(1, steps - warm)))
    w = exit_weights(model.exit_layers)
    ls = cfg.get("label_smoothing", 0.0)
    model.train()
    step, t0 = 0, time.time()
    for epoch in range(cfg["epochs"]):
        run = torch.zeros(len(w), device="cuda")
        nb = 0
        for xb, yb in data.train_batches(cfg["batch_size"], rng):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xb)                                     # [B, E, C]
            losses = torch.stack([F.cross_entropy(logits[:, e].float(), yb, label_smoothing=ls)
                                  for e in range(logits.shape[1])])
            loss = (w * losses).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += losses.detach()
            nb += 1
            step += 1
            if step % cfg.get("log_every", 200) == 0:
                el = time.time() - t0
                print(f"ep {epoch} step {step}/{steps} loss/exit {np.round((run / nb).tolist(), 3)} "
                      f"{el / 60:.1f} min, eta {(steps - step) * el / step / 60:.1f} min", flush=True)
        print(f"epoch {epoch} done: mean loss per exit {np.round((run / max(nb, 1)).tolist(), 4)}", flush=True)


@torch.inference_mode()
def log_traces(model: EarlyExitModel, data, cfg: dict, task: str, out_dir: Path) -> pd.DataFrame:
    model.eval()
    probs = []
    for xb in data.eval_batches(cfg.get("eval_batch", 256)):
        with torch.autocast("cuda", dtype=torch.float16):
            probs.append(model(xb).float().softmax(-1).cpu())
    probs = torch.cat(probs).numpy()                                   # [N, E, C]
    y = data.y_va
    N, E, C = probs.shape
    conf, pred = probs.max(-1), probs.argmax(-1)
    rng = np.random.default_rng(cfg["seed"])
    split = np.where(rng.permutation(N) < N // 2, "evolve", "heldout")
    cols = {"idx": np.arange(N), "label": y, "split": split}
    for e in range(E):
        cols[f"conf_{e}"], cols[f"pred_{e}"] = conf[:, e], pred[:, e]
    for e in range(E):
        for c in range(C):
            cols[f"p_{e}_{c}"] = probs[:, e, c].astype(np.float32)
    df = pd.DataFrame(cols)
    df.to_parquet(out_dir / f"traces/{task}_val.parquet", index=False)
    acc = pd.DataFrame([{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
                         "mean_conf": float(g[f"conf_{e}"].mean())}
                        for e, l in enumerate(model.exit_layers) for s, g in [("all", df), *df.groupby("split")]])
    acc.to_csv(out_dir / f"results/train_{task}_exit_accuracy.csv", index=False)
    print(acc[acc.split == "all"].round(4).to_string(index=False))
    correct = pred == y[:, None]
    print(f"oracle (any exit correct): {correct.any(1).mean():.4f}; final exit: {correct[:, -1].mean():.4f}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--traces-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    task = cfg["task"] + ("_smoke" if args.smoke else "")
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    out = ROOT / ("logs/smoke" if args.smoke else "")
    for d in ("traces", "results", "checkpoints"):
        (out / d).mkdir(parents=True, exist_ok=True)

    data = TextData(cfg, args.smoke) if cfg["kind"] == "text" else VisionData(cfg, args.smoke)
    ck = out / f"checkpoints/{task}.pt"
    if args.traces_only:
        model = build(cfg, load_backbone_weights=False)
        model.load_state_dict(torch.load(ck, weights_only=False)["state_dict"])
        model = model.cuda()
    else:
        model = build(cfg).cuda()
        if args.smoke:
            cfg = {**cfg, "epochs": 1, "log_every": 1}
        t0 = time.time()
        train(model, data, cfg)
        print(f"training took {(time.time() - t0) / 60:.1f} min", flush=True)
        torch.save({"config": cfg, "state_dict": {k: v.half() for k, v in model.state_dict().items()}}, ck)
    log_traces(model, data, cfg, task, out)


if __name__ == "__main__":
    main()
