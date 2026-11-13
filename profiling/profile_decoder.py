"""P1b: decoder decode-step cost per layer, c(n) for n sequences each generating 1 token with a KV cache.

Weights are random (from_config): timing does not depend on them, and this avoids a 3 GB download.
Per-layer cost = (decode step through all decoder layers) / num_layers, eager mode.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python profiling/profile_decoder.py \
      --model Qwen/Qwen2.5-1.5B --gpu-label A6000 --contexts 256 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache

ROOT = Path(__file__).resolve().parents[1]


@torch.inference_mode()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--gpu-label", default="A6000")
    p.add_argument("--contexts", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()
    torch.manual_seed(0)

    cfg = AutoConfig.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float16, attn_implementation="sdpa").cuda().eval()
    body, L = model.model, cfg.num_hidden_layers
    rows = []
    for ctx in args.contexts:
        for n in args.batch_sizes:
            ids = torch.randint(0, cfg.vocab_size, (n, ctx), device="cuda")
            cache = DynamicCache()
            body(input_ids=ids, past_key_values=cache, use_cache=True)
            step = torch.randint(0, cfg.vocab_size, (n, 1), device="cuda")
            pos = torch.full((n, 1), ctx, device="cuda")
            ts = []
            for i in range(args.repeats + 5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                body(input_ids=step, past_key_values=cache, position_ids=pos, use_cache=True)
                end.record()
                torch.cuda.synchronize()
                cache.crop(ctx)  # keep context length fixed across repeats
                if i >= 5:
                    ts.append(start.elapsed_time(end))
            step_ms = float(np.median(ts))
            rows.append({"gpu": args.gpu_label, "model": args.model, "context": ctx, "n": n,
                         "step_ms": step_ms, "layer_ms": step_ms / L})
            print(f"ctx={ctx} n={n:4d} step {step_ms:.3f} ms  per-layer {step_ms / L:.4f} ms", flush=True)
            del cache, ids
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    fits = []
    for ctx, g in df.groupby("context"):
        b, a = np.polyfit(g.n.to_numpy(float), g.layer_ms.to_numpy(), 1)
        pred = a + b * g.n
        r2 = 1 - ((g.layer_ms - pred) ** 2).sum() / ((g.layer_ms - g.layer_ms.mean()) ** 2).sum()
        fits.append({"context": int(ctx), "a_ms": float(a), "b_ms": float(b), "r2": float(r2),
                     "n_star": float(a / b) if b > 0 else float("inf"), "rho_at_64": float(a / (b * 64)) if b > 0 else float("inf")})
    tag = f"{args.model.split('/')[-1]}_{args.gpu_label}"
    df.to_csv(ROOT / f"results/p1_decoder_timing_{tag}.csv", index=False)
    (ROOT / f"results/p1_decoder_fit_{tag}.json").write_text(json.dumps({"device": torch.cuda.get_device_name(), "fits": fits}, indent=2))
    print(pd.DataFrame(fits).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
