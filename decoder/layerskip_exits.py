"""Step 3 of docs/PLAN_PAPER.md: how good are the exits of a backbone trained for early exit (LayerSkip)?

LayerSkip exits need no extra heads: exit at layer l = lm_head(final_norm(h_l)). This logs next-token accuracy and
agreement with the full model at every layer on WikiText-103 (same windows protocol as train_decoder_exits.py), and
writes traces for the simulator (confidence = max softmax probability of the exit).

Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 ~/venv/bin/python decoder/layerskip_exits.py \
           --model facebook/layerskip-llama3.2-1B --tag ls1b --exits 6 8 10 12 14 16
Outputs: results/layerskip_<tag>_layer_accuracy.csv (every layer), traces/wikitext_<tag>_val.parquet,
         data/wikitext_<tag>_windows.pt, results/train_wikitext_<tag>_exit_accuracy.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
WIN, CTX = 256, 128


def windows(tok, texts):
    ids = tok("\n\n".join(t for t in texts if t.strip()), return_tensors="pt")["input_ids"][0]
    n = len(ids) // WIN
    return ids[: n * WIN].view(n, WIN)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exits", type=int, nargs="+", required=True)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    nL = lm.config.num_hidden_layers
    assert args.exits[-1] == nL
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    va = torch.cat([windows(tok, ds["validation"]["text"]), windows(tok, ds["test"]["text"])])
    if args.max_windows:
        va = va[:args.max_windows]
    print(f"{args.model}: {nL} layers, eval windows {len(va)}", flush=True)
    E = len(args.exits)
    hit, agree, n = np.zeros(nL + 1), np.zeros(nL + 1), 0
    conf, pred, lab, win, pos = [], [], [], [], []
    sl = slice(CTX - 1, WIN - 1)
    for i in range(0, len(va), args.bs):
        x = va[i:i + args.bs].cuda()
        out = lm(input_ids=x, output_hidden_states=True)
        y = x[:, CTX:]
        top = {}
        for l in range(1, nL + 1):
            h = out.hidden_states[l][:, sl]
            logits = lm.lm_head(h if l == nL else lm.model.norm(h)).float()   # the last hidden state is already normed
            p = logits.softmax(-1).max(-1)
            top[l] = (p.values, p.indices)
        full = top[nL][1]
        for l in range(1, nL + 1):
            hit[l] += (top[l][1] == y).sum().item()
            agree[l] += (top[l][1] == full).sum().item()
        n += y.numel()
        conf.append(torch.stack([top[l][0] for l in args.exits], -1).reshape(-1, E).cpu())
        pred.append(torch.stack([top[l][1] for l in args.exits], -1).reshape(-1, E).cpu())
        lab.append(y.reshape(-1).cpu())
        b, t = y.shape
        win.append(torch.arange(i, i + b).repeat_interleave(t))
        pos.append(torch.arange(CTX, WIN).repeat(b))
    full_logits_ok = (out.logits[:, sl].float().argmax(-1) == full).float().mean().item()
    print(f"sanity: last-layer exit equals model logits on {full_logits_ok:.4f} of tokens", flush=True)
    lay = pd.DataFrame({"layer": range(1, nL + 1), "depth_frac": np.arange(1, nL + 1) / nL,
                        "accuracy": hit[1:] / n, "fidelity": agree[1:] / n})
    lay["drop_pts"] = 100 * (lay.accuracy.iloc[-1] - lay.accuracy)
    lay.to_csv(ROOT / f"results/layerskip_{args.tag}_layer_accuracy.csv", index=False)
    print(lay.round(4).to_string(index=False), flush=True)
    conf, pred = torch.cat(conf).numpy(), torch.cat(pred).numpy()
    df = pd.DataFrame({"window": torch.cat(win).numpy(), "pos": torch.cat(pos).numpy(), "label": torch.cat(lab).numpy()})
    for e in range(E):
        df[f"conf_{e}"], df[f"pred_{e}"] = conf[:, e], pred[:, e]
    df["idx"] = np.arange(len(df))
    wsplit = np.where(np.random.default_rng(0).permutation(len(va)) < len(va) // 2, "evolve", "heldout")
    df["split"] = wsplit[df.window.to_numpy()]
    df.to_parquet(ROOT / f"traces/wikitext_{args.tag}_val.parquet", index=False)
    torch.save({"windows": va, "split": wsplit}, ROOT / f"data/wikitext_{args.tag}_windows.pt")
    acc = pd.DataFrame([{"exit": e, "layer": l, "split": s, "accuracy": float((g[f"pred_{e}"] == g.label).mean()),
                         "fidelity": float((g[f"pred_{e}"] == g[f"pred_{E - 1}"]).mean()),
                         "mean_conf": float(g[f"conf_{e}"].mean())}
                        for e, l in enumerate(args.exits) for s, g in [("all", df), *df.groupby("split")]])
    acc.to_csv(ROOT / f"results/train_wikitext_{args.tag}_exit_accuracy.csv", index=False)


if __name__ == "__main__":
    main()
