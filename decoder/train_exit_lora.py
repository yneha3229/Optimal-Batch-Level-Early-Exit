"""Decoder exits trained *with* the backbone (LoRA + exit heads + self-distillation).

Why this script exists. The two decoder attempts so far (decoder/train_decoder_exits.py: frozen backbone + heads;
decoder/layerskip_exits.py: public LayerSkip checkpoints) left the 75%-depth exit about 5 points below the full
model, so no batch-level stop was usable. The encoders had the same problem (MNLI layer-8 exit 70.6% on a frozen
backbone) and joint training fixed it (87.4%). That fix was never applied to a decoder. This script applies it.

Model.   Base causal LM, float32 master weights, LoRA (rank r) on the attention and MLP projections of every layer.
         Exit e at layer l_e:  logits_e = lm_head(norm_e(h + W_e h)),  conf_e = sigmoid(c_e . features)
         (same ExitHeads as decoder/train_decoder_exits.py, so decoder/decode_runner.py runs the result unchanged).
Loss.    anchor   KL(original model || fine-tuned final layer)   keeps the full model what it was (--no-anchor: CE)
         exits    sum_e [ 0.5 KL(stopgrad final || exit_e) + 0.5 CE(next token) ]  on a random subset of positions
         conf     BCE(conf_e, exit_e token == final token) on detached features
--rank 0 trains the heads only on the frozen backbone: the baseline, in the same code and on the same data.

Outputs (TAG = --tag; task name is wikitext_TAG everywhere else in the repository):
  checkpoints/TAG_merged/                    backbone with LoRA merged in (rank > 0 only), HF format
  checkpoints/TAG_exits.pt                   exit heads; its "model" field points at the merged backbone
  traces/wikitext_TAG_val.parquet            per-token traces, same schema as the other decoder traces
  traces/wikitext_TAG_val.meta.json          exit layers (read by experiments/run_main.py)
  data/wikitext_TAG_windows.pt               evaluation windows and their evolve/held-out split
  results/train_wikitext_TAG_exit_accuracy.csv   accuracy and agreement per exit, plus the original model's accuracy

Usage (RTX A6000):
  python decoder/train_exit_lora.py --tag qlora --rank 16 --steps 6000
  python decoder/train_exit_lora.py --tag qfrozen --rank 0 --steps 6000          # baseline, same budget
Functional test on a small GPU:
  python decoder/train_exit_lora.py --tag smoke --model Qwen/Qwen2.5-0.5B --exits 8 12 16 20 24 --amp none \
      --steps 20 --bs 2 --win 64 --ctx 32 --train-lines 2000 --eval-windows 16
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decoder.train_decoder_exits import ExitHeads  # noqa: E402

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    """y = base(x) + scale * B A x. `enabled` is a shared one-element list so the teacher pass can switch it off."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, enabled: list) -> None:
        super().__init__()
        self.base, self.scale, self.enabled = base, alpha / r, enabled
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if not self.enabled[0]:
            return y
        return y + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight += (self.B @ self.A).to(self.base.weight.dtype) * self.scale
        return self.base


def add_lora(lm: nn.Module, r: int, alpha: float, enabled: list) -> list[nn.Parameter]:
    params = []
    for layer in lm.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                lin = getattr(parent, name, None)
                if isinstance(lin, nn.Linear):
                    w = LoRALinear(lin, r, alpha, enabled)
                    setattr(parent, name, w)
                    params += [w.A, w.B]
    return params


def merge_lora(lm: nn.Module) -> None:
    for layer in lm.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                w = getattr(parent, name, None)
                if isinstance(w, LoRALinear):
                    setattr(parent, name, w.merged())


def windows(tok, texts: list[str], win: int) -> torch.Tensor:
    ids = tok("\n\n".join(t for t in texts if t.strip()), return_tensors="pt")["input_ids"][0]
    n = len(ids) // win
    return ids[: n * win].view(n, win)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--exits", type=int, nargs="+", default=[8, 12, 16, 20, 24, 28], help="last one = model depth")
    ap.add_argument("--rank", type=int, default=16, help="0 = frozen backbone (heads only)")
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--positions", type=int, default=64, help="token positions per sequence used for the exit losses")
    ap.add_argument("--no-anchor", action="store_true", help="train the final layer with CE instead of KL to the original model")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "none"])
    ap.add_argument("--train-lines", type=int, default=400_000)
    ap.add_argument("--eval-windows", type=int, default=0, help="0 = all validation + test windows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    EX, E, WIN, CTX = args.exits, len(args.exits), args.win, args.ctx
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).cuda().eval()
    assert len(lm.model.layers) == EX[-1], f"last exit must be the model depth {len(lm.model.layers)}"
    for p in lm.parameters():
        p.requires_grad_(False)
    enabled = [True]
    lora = add_lora(lm, args.rank, args.alpha, enabled) if args.rank > 0 else []
    heads = ExitHeads(lm.config.hidden_size, lm.model.norm, E - 1).cuda()
    groups = [{"params": heads.parameters(), "lr": args.head_lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=0.05)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp == "bf16")

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    tr = windows(tok, ds["train"]["text"][: args.train_lines], WIN)
    va = torch.cat([windows(tok, ds["validation"]["text"], WIN), windows(tok, ds["test"]["text"], WIN)])
    if args.eval_windows:
        va = va[: args.eval_windows]
    print(f"train windows {len(tr)}, eval windows {len(va)}, LoRA tensors {len(lora)}, "
          f"trainable {sum(p.numel() for g in groups for p in g['params']) / 1e6:.1f} M", flush=True)

    def teacher(x):
        """Final-layer logits of the original model (LoRA switched off)."""
        enabled[0] = False
        with torch.no_grad(), amp():
            lg = lm(input_ids=x).logits
        enabled[0] = True
        return lg

    t0 = time.time()
    for step in range(args.steps):
        x = tr[rng.choice(len(tr), args.bs, replace=False)].cuda()
        y = x[:, 1:]
        pos = torch.from_numpy(np.sort(rng.choice(WIN - 1, min(args.positions, WIN - 1), replace=False))).cuda()
        with amp():
            out = lm(input_ids=x, output_hidden_states=True)
        final = out.logits[:, pos].float()                                  # [bs, P, V]
        yp = y[:, pos]
        if lora:
            if args.no_anchor:
                loss = F.cross_entropy(final.reshape(-1, final.shape[-1]), yp.reshape(-1))
            else:
                pt = teacher(x)[:, pos].float().softmax(-1)
                loss = (pt * (pt.clamp_min(1e-9).log() - final.log_softmax(-1))).sum(-1).mean()
            anchor = loss.item()
        else:
            loss, anchor = 0.0, 0.0
        p_full = final.detach().softmax(-1)
        top_full = final.detach().argmax(-1)
        log = []
        for e in range(E - 1):
            h = out.hidden_states[EX[e]][:, pos]
            if not lora:
                h = h.detach()
            with amp():
                feat = heads.features(e, h)
                lg = lm.lm_head(feat).float()
            lp = lg.log_softmax(-1)
            kl = (p_full * (p_full.clamp_min(1e-9).log() - lp)).sum(-1).mean()
            ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), yp.reshape(-1))
            match = (lg.argmax(-1) == top_full).float()
            bce = F.binary_cross_entropy_with_logits(heads.conf[e](feat.detach().float()).squeeze(-1), match)
            loss = loss + 0.5 * kl + 0.5 * ce + bce
            log.append(round(match.mean().item(), 3))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step}/{args.steps} anchor {anchor:.3f} agreement per exit {log} {el / 60:.1f} min "
                  f"eta {(args.steps - step - 1) * el / (step + 1) / 60:.1f} min", flush=True)

    # ---------------------------------------------------------------- original-model accuracy, merge, save
    heads.eval()
    sl = slice(CTX - 1, WIN - 1)
    orig_ok, n_tok = 0, 0
    with torch.no_grad():
        for i in range(0, len(va), 8):
            x = va[i:i + 8].cuda()
            orig_ok += int((teacher(x)[:, sl].argmax(-1) == x[:, CTX:]).sum())
            n_tok += x[:, CTX:].numel()
    model_ref = args.model
    for d in ("checkpoints", "traces", "data", "results"):
        (ROOT / d).mkdir(exist_ok=True)
    if lora:
        merge_lora(lm)
        model_ref = str(ROOT / f"checkpoints/{args.tag}_merged")
        lm.to(torch.bfloat16).save_pretrained(model_ref)
        tok.save_pretrained(model_ref)
        lm.float()

    # ---------------------------------------------------------------- traces on positions CTX..WIN-1
    parts = []
    with torch.no_grad():
        for i in range(0, len(va), 8):
            x = va[i:i + 8].cuda()
            with amp():
                out = lm(input_ids=x, output_hidden_states=True)
            y = x[:, CTX:]
            full = out.logits[:, sl].float()
            d = {"window": np.repeat(np.arange(i, i + x.shape[0]), y.shape[1]),
                 "pos": np.tile(np.arange(CTX, WIN), x.shape[0]), "label": y.reshape(-1).cpu().numpy()}
            for e in range(E - 1):
                with amp():
                    feat = heads.features(e, out.hidden_states[EX[e]][:, sl])
                    pr = lm.lm_head(feat).float().argmax(-1)
                d[f"conf_{e}"] = heads.confidence(e, feat).reshape(-1).cpu().numpy()
                d[f"pred_{e}"] = pr.reshape(-1).cpu().numpy()
            d[f"conf_{E - 1}"] = full.softmax(-1).max(-1).values.reshape(-1).cpu().numpy()
            d[f"pred_{E - 1}"] = full.argmax(-1).reshape(-1).cpu().numpy()
            parts.append(pd.DataFrame(d))
    df = pd.concat(parts, ignore_index=True)
    df["idx"] = np.arange(len(df))
    wsplit = np.where(np.random.default_rng(0).permutation(len(va)) < len(va) // 2, "evolve", "heldout")
    df["split"] = wsplit[df.window.to_numpy()]
    task = f"wikitext_{args.tag}"
    df.to_parquet(ROOT / f"traces/{task}_val.parquet", index=False)
    (ROOT / f"traces/{task}_val.meta.json").write_text(json.dumps({"exit_layers": EX, "decoder": True}))
    torch.save({"model": model_ref, "exits": EX, "state_dict": heads.state_dict(), "ctx": CTX, "win": WIN,
                "args": vars(args)}, ROOT / f"checkpoints/{args.tag}_exits.pt")
    torch.save({"windows": va, "split": wsplit}, ROOT / f"data/{task}_windows.pt")
    acc = pd.DataFrame([{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
                         "fidelity": float((g[f"pred_{e}"] == g[f"pred_{E - 1}"]).mean()),
                         "mean_conf": float(g[f"conf_{e}"].mean())}
                        for e, l in enumerate(EX) for s, g in [("all", df), *df.groupby("split")]])
    acc["original_model_accuracy"] = orig_ok / n_tok
    acc.to_csv(ROOT / f"results/train_{task}_exit_accuracy.csv", index=False)
    a = acc[acc.split == "all"]
    print(a.round(4).to_string(index=False))
    full_acc = float(a.accuracy.iloc[-1])
    print(f"original model accuracy {orig_ok / n_tok:.4f}; fine-tuned full model {full_acc:.4f}")
    print("points below the full model per exit:", [round(100 * (full_acc - v), 2) for v in a.accuracy.tolist()])


if __name__ == "__main__":
    main()
