"""Early-exit decoder for the real large-rho experiment: Qwen2.5-1.5B (frozen) + exit heads at layers 8..24.

Why a decoder: in token-by-token decoding each sequence contributes one token per step, so a layer's cost is almost
all fixed (weights + launch), rho = a/(b n) is large, and the theory predicts that only collective stopping pays.

Exit e (layer l_e < 28):  logits_e = lm_head(norm_e(h + W_e h))      W_e: d x d, zero-init; norm_e: copy of final norm
                          conf_e   = sigmoid(c_e . norm_e(...))       tiny head: P(exit token == full-model token)
The lm_head is the frozen, shared output layer. Exit 28 is the unmodified model. Loss: 0.5 KL(full || exit) +
0.5 CE(next token) for the exit heads; BCE for the confidence heads (on detached features).
The confidence head exists because the 152k-vocab lm_head costs as much as several layers in memory-bound decoding,
so a policy cannot afford to evaluate it at every exit; the runner evaluates it only where tokens leave.

Data: WikiText-103 (raw). Windows of 256 tokens; traces cover positions 128..255 of validation+test windows, split
into evolve/held-out *by window*.

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python decoder/train_decoder_exits.py [--smoke]
Outputs: checkpoints/qwen_exits.pt, traces/wikitext_val.parquet, data/wikitext_windows.pt,
         results/train_wikitext_exit_accuracy.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen2.5-1.5B"
EXITS = [8, 12, 16, 20, 24, 28]
WIN, CTX = 256, 128


class ExitHeads(nn.Module):
    def __init__(self, d: int, final_norm: nn.Module, n: int) -> None:
        super().__init__()
        self.W = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(n))
        for w in self.W:
            nn.init.zeros_(w.weight)
        self.norm = nn.ModuleList(type(final_norm)(d, eps=final_norm.variance_epsilon) for _ in range(n))
        for nm in self.norm:
            nm.weight.data.copy_(final_norm.weight.data)
        self.conf = nn.ModuleList(nn.Linear(d, 1) for _ in range(n))

    def features(self, e: int, h: torch.Tensor) -> torch.Tensor:
        return self.norm[e](h + self.W[e](h))

    def confidence(self, e: int, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conf[e](feat.to(self.conf[e].weight.dtype)).float().squeeze(-1))


def windows(tok, split_texts: list[str]) -> torch.Tensor:
    ids = tok("\n\n".join(t for t in split_texts if t.strip()), return_tensors="pt")["input_ids"][0]
    n = len(ids) // WIN
    return ids[: n * WIN].view(n, WIN)


def main() -> None:
    global EXITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--exits", type=int, nargs="+", default=EXITS)
    ap.add_argument("--tag", default="", help="'' = Qwen run (task wikitext); e.g. llama -> task wikitext_llama")
    args = ap.parse_args()
    EXITS = args.exits
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    n_tr = 2000 if args.smoke else 400_000
    tr = windows(tok, ds["train"]["text"][:n_tr])
    va = torch.cat([windows(tok, ds["validation"]["text"]), windows(tok, ds["test"]["text"])])
    if args.smoke:
        va = va[:16]
        args.steps = 5
    print(f"train windows {len(tr)}, eval windows {len(va)}", flush=True)
    d = lm.config.hidden_size
    E = len(EXITS)
    heads = ExitHeads(d, lm.model.norm, E - 1).cuda()
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.05)
    lm_head = lm.lm_head
    rng = np.random.default_rng(0)
    t0 = time.time()
    for step in range(args.steps):
        x = tr[rng.choice(len(tr), args.bs, replace=False)].cuda()
        with torch.no_grad():
            out = lm(input_ids=x, output_hidden_states=True)
            full = out.logits[:, :-1].float()
            p_full = full.softmax(-1)
            top_full = full.argmax(-1)
            hs = [out.hidden_states[l][:, :-1] for l in EXITS[:-1]]
        y = x[:, 1:]
        tot = []
        opt.zero_grad(set_to_none=True)
        for e in range(E - 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                feat = heads.features(e, hs[e])
                logits = lm_head(feat).float()
            lp = logits.log_softmax(-1)
            kl = (p_full * (p_full.clamp_min(1e-9).log() - lp)).sum(-1).mean()
            ce = F.nll_loss(lp.reshape(-1, lp.shape[-1]), y.reshape(-1))
            match = (logits.argmax(-1) == top_full).float()
            bce = F.binary_cross_entropy_with_logits(heads.conf[e](feat.detach().float()).squeeze(-1), match)
            loss = 0.5 * kl + 0.5 * ce + bce
            loss.backward()
            tot.append((kl.item(), match.mean().item()))
            del logits, lp
        opt.step()
        sched.step()
        if step % 100 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"step {step}/{args.steps} kl/agree per exit {[(round(a, 2), round(b, 3)) for a, b in tot]} "
                  f"{el / 60:.1f} min eta {(args.steps - step - 1) * el / (step + 1) / 60:.1f} min", flush=True)

    # ---- traces on positions CTX..WIN-1 of every eval window
    heads.eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(va), 8):
            x = va[i:i + 8].cuda()
            out = lm(input_ids=x, output_hidden_states=True)
            sl = slice(CTX - 1, WIN - 1)                     # predict tokens CTX..WIN-1
            y = x[:, CTX:]
            full = out.logits[:, sl].float()
            confs, preds = [], []
            for e in range(E - 1):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    feat = heads.features(e, out.hidden_states[EXITS[e]][:, sl])
                    preds.append(lm_head(feat).float().argmax(-1))
                confs.append(heads.confidence(e, feat))
            preds.append(full.argmax(-1))
            confs.append(full.softmax(-1).max(-1).values)     # final exit: its own max prob (never used to decide)
            for b in range(x.shape[0]):
                for t in range(y.shape[1]):
                    r = {"window": i + b, "pos": CTX + t, "label": int(y[b, t])}
                    for e in range(E):
                        r[f"conf_{e}"] = float(confs[e][b, t])
                        r[f"pred_{e}"] = int(preds[e][b, t])
                    rows.append(r)
    df = pd.DataFrame(rows)
    df["idx"] = np.arange(len(df))
    wsplit = np.where(np.random.default_rng(0).permutation(len(va)) < len(va) // 2, "evolve", "heldout")
    df["split"] = wsplit[df.window.to_numpy()]
    suffix = ("_" + args.tag if args.tag else "") + ("_smoke" if args.smoke else "")
    df.to_parquet(ROOT / f"traces/wikitext{suffix}_val.parquet", index=False)
    torch.save({"model": args.model, "exits": EXITS, "state_dict": heads.state_dict(), "ctx": CTX, "win": WIN},
               ROOT / f"checkpoints/{'qwen' if not args.tag else args.tag}_exits{'_smoke' if args.smoke else ''}.pt")
    torch.save({"windows": va, "split": wsplit}, ROOT / f"data/wikitext{suffix}_windows.pt" if args.tag
               else ROOT / f"data/wikitext_windows{suffix}.pt")
    acc = pd.DataFrame([{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
                         "fidelity": float((g[f"pred_{e}"] == g[f"pred_{E - 1}"]).mean()),
                         "mean_conf": float(g[f"conf_{e}"].mean())}
                        for e, l in enumerate(EXITS) for s, g in [("all", df), *df.groupby("split")]])
    acc.to_csv(ROOT / f"results/train_wikitext{suffix}_exit_accuracy.csv", index=False)
    print(acc[acc.split == "all"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
