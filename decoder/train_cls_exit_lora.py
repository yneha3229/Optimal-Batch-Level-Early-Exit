"""An LLM used as a batched classifier, with early exits trained jointly with the backbone (LoRA + exit heads).

Workload. The prompt is run once (prefill) and the label is read from the hidden state of the last prompt token by a
small head after each exit layer. Nothing is decoded and there is no KV cache to repair when an input leaves early.
This is how LLMs are served as guard models, rerankers, routers and judges. It is the same problem as the encoder
track (traces/train_exits.py) with a decoder-only LLM as the backbone, so the traces use the same schema and every
encoder script (simulator, experiments/run_main.py, analysis/adaptivity_vs_batch.py) runs on them unchanged.

Model.   Causal LM, frozen base weights (bf16), LoRA (rank r) on the attention and MLP projections of every layer.
Heads.   Exit e: own copy of the final norm + Linear(d, classes) on the last token's hidden state after layer EX[e].
         The last exit is the full model.
Loss.    Layer-weighted sum of per-exit cross-entropy (as in traces/train_exits.py, DeeBERT / PABEE).

Outputs (task = mnli | qqp, TAG e.g. qwen):
  checkpoints/<task>_<TAG>_cls.pt              heads + LoRA tensors + config (base weights are not copied)
  traces/<task>_<TAG>_val.parquet, .meta.json  idx, label, split, n_tokens, conf_e, pred_e, p_e_c for every exit e
  results/train_<task>_<TAG>_exit_accuracy.csv

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python decoder/train_cls_exit_lora.py --task mnli --tag qwen
  ... --model meta-llama/Llama-3.1-8B-Instruct --exits 8 12 16 20 24 28 32 --tag llama --bs 16
  ... --rank 0          frozen backbone, heads only (baseline)
  ... --smoke           200 steps, 512 eval examples, tag forced to smoke
"""
from __future__ import annotations

import argparse
import copy
import json
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
from decoder.train_exit_lora import LoRALinear, add_lora  # noqa: E402

TASKS = {"mnli": ("premise", "hypothesis", "validation_matched", 3, "Premise: {a}\nHypothesis: {b}",
                  "\nRelation of the hypothesis to the premise (entailment, neutral or contradiction):"),
         "qqp": ("question1", "question2", "validation", 2, "Question 1: {a}\nQuestion 2: {b}",
                 "\nDo the two questions ask the same thing (yes or no)?")}


class ClsHeads(nn.Module):
    def __init__(self, d: int, norm: nn.Module, E: int, C: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList([copy.deepcopy(norm).float() for _ in range(E)])
        self.fcs = nn.ModuleList([nn.Linear(d, C) for _ in range(E)])

    def forward(self, e: int, h: torch.Tensor) -> torch.Tensor:
        return self.fcs[e](self.norms[e](h.float()))


def encode(tok, a: list[str], b: list[str], body: str, suffix: str, max_len: int) -> list[list[int]]:
    """Prompt ids; the body is truncated so that the suffix (and its last token, where we read the label) is kept."""
    suf = tok(suffix, add_special_tokens=False)["input_ids"]
    ids = tok([body.format(a=x, b=y) for x, y in zip(a, b)])["input_ids"]
    return [s[: max_len - len(suf)] + suf for s in ids]


def collate(seqs: list[list[int]], pad: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    T = max(map(len, seqs))
    x = torch.full((len(seqs), T), pad, dtype=torch.long)
    m = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
        m[i, : len(s)] = 1
    return x.cuda(), m.cuda(), (m.sum(1) - 1).cuda()          # right padding: causal attention never sees it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli", choices=list(TASKS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--tag", default="qwen")
    ap.add_argument("--exits", type=int, nargs="+", default=[4, 8, 12, 16, 20, 24, 28])
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lr", type=float, default=2e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--eval-bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if args.smoke:
        args.steps, args.tag = 200, "smoke"
    torch.manual_seed(args.seed)
    f1, f2, val_split, C, body, suffix = TASKS[args.task]
    EX, E = args.exits, len(args.exits)
    name = f"{args.task}_{args.tag}"
    tok = AutoTokenizer.from_pretrained(args.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    assert len(lm.model.layers) == EX[-1], f"last exit must be the model depth {len(lm.model.layers)}"
    for p in lm.parameters():
        p.requires_grad_(False)
    lm.lm_head = nn.Identity()                                  # not used
    lora = add_lora(lm, args.rank, args.alpha, [True]) if args.rank > 0 else []
    heads = ClsHeads(lm.config.hidden_size, lm.model.norm, E, C).cuda()
    groups = [{"params": list(heads.parameters()), "lr": args.head_lr}]
    if lora:
        groups.append({"params": lora, "lr": args.lr})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=args.steps,
                                                pct_start=max(0.05, 2 / args.steps))
    w = torch.tensor(EX, dtype=torch.float32, device="cuda")
    w = w / w.sum()

    ds = load_dataset("nyu-mll/glue", args.task)
    tr = ds["train"].shuffle(seed=args.seed).select(range(min(args.steps * args.bs, len(ds["train"]))))
    va = ds[val_split].select(range(512)) if args.smoke else ds[val_split]
    x_tr, y_tr = encode(tok, tr[f1], tr[f2], body, suffix, args.max_len), torch.tensor(tr["label"])
    x_va, y_va = encode(tok, va[f1], va[f2], body, suffix, args.max_len), np.array(va["label"])
    print(f"{name}: train {len(x_tr)}, eval {len(x_va)}, mean prompt length {np.mean([len(s) for s in x_va]):.1f} tokens, "
          f"exits {EX}, trainable {sum(p.numel() for g in groups for p in g['params']) / 1e6:.1f} M", flush=True)

    def forward(seqs):
        x, m, last = collate(seqs, pad)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hs = lm.model(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
        ar = torch.arange(len(seqs), device="cuda")
        return [heads(e, hs[EX[e]][ar, last]) for e in range(E)]

    t0, run = time.time(), np.zeros(E)
    for step in range(args.steps):
        i = [k % len(x_tr) for k in range(step * args.bs, (step + 1) * args.bs)]
        y = y_tr[i].cuda()
        logits = forward([x_tr[k] for k in i])
        loss = sum(w[e] * F.cross_entropy(logits[e], y) for e in range(E))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step(), sched.step()
        run += [float((lg.argmax(-1) == y).float().mean()) for lg in logits]
        if (step + 1) % 100 == 0:
            el = time.time() - t0
            print(f"step {step + 1}/{args.steps} loss {loss.item():.3f} train acc per exit {np.round(run / 100, 3).tolist()} "
                  f"{el / 60:.1f} min eta {el / (step + 1) * (args.steps - step - 1) / 60:.1f} min", flush=True)
            run[:] = 0

    # ---------------------------------------------------------------------------------- traces
    N = len(x_va)
    order = np.argsort([len(s) for s in x_va])
    P = np.zeros((E, N, C), dtype=np.float32)
    with torch.inference_mode():
        for k in range(0, N, args.eval_bs):
            j = order[k:k + args.eval_bs]
            for e, lg in enumerate(forward([x_va[q] for q in j])):
                P[e, j] = lg.softmax(-1).float().cpu().numpy()
    rng = np.random.default_rng(args.seed)
    cols = {"idx": np.arange(N), "label": y_va, "split": np.where(rng.permutation(N) < N // 2, "evolve", "heldout"),
            "n_tokens": [len(s) for s in x_va]}
    for e in range(E):
        cols[f"conf_{e}"], cols[f"pred_{e}"] = P[e].max(-1), P[e].argmax(-1)
    for e in range(E):
        for c in range(C):
            cols[f"p_{e}_{c}"] = P[e][:, c]
    df = pd.DataFrame(cols)
    df.to_parquet(ROOT / f"traces/{name}_val.parquet", index=False)
    (ROOT / f"traces/{name}_val.meta.json").write_text(json.dumps(
        {"exit_layers": EX, "decoder": False, "model": args.model, "task": args.task, "rank": args.rank}))
    acc = pd.DataFrame([{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
                         "agree_full": float((g[f"pred_{e}"] == g[f"pred_{E - 1}"]).mean()),
                         "mean_conf": float(g[f"conf_{e}"].mean())}
                        for e, l in enumerate(EX) for s, g in [("all", df), *df.groupby("split")]])
    acc.to_csv(ROOT / f"results/train_{name}_exit_accuracy.csv", index=False)
    print(acc[acc.split == "all"].round(4).to_string(index=False), flush=True)
    torch.save({"model": args.model, "task": args.task, "exits": EX, "rank": args.rank, "alpha": args.alpha,
                "max_len": args.max_len, "heads": heads.state_dict(),
                "lora": [p.detach().cpu() for p in lora]}, ROOT / f"checkpoints/{name}_cls.pt")
    print("wrote", f"traces/{name}_val.parquet", f"checkpoints/{name}_cls.pt", flush=True)


if __name__ == "__main__":
    main()
