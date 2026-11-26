"""Does early exit in decoding get better with longer prompts (contexts)?

Two independent measurements:
  acc   exit quality by token position: agreement with the full model and accuracy of every exit, in position buckets
        up to --win tokens, on WikiText-103 validation+test windows (teacher-forced, HF forward, bf16).
        The exit heads were trained on 256-token windows, so positions beyond 256 also test length generalisation.
  cost  time of one decode step through all layers as a function of context length T and batch size B, with our own
        KV-cache layer loop (decoder/decode_runner.py). Fits c(n) = a + b n per T: the per-sequence cost b grows with
        T because attention reads the KV cache, so rho = a / (b n) falls with longer prompts.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 ~/venv/bin/python experiments/context_scaling.py acc --ckpt qlora
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 ~/venv/bin/python experiments/context_scaling.py cost --ckpt qwen
Outputs: results/context_acc_<ckpt>.csv, results/context_cost_<ckpt>_<eager|graph>_<gpu>.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decoder.train_decoder_exits import ExitHeads  # noqa: E402

BUCKETS = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def acc(args) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = torch.load(ROOT / f"checkpoints/{args.ckpt}_exits.pt", weights_only=False)
    EX = ck["exits"]
    E = len(EX)
    tok = AutoTokenizer.from_pretrained(ck["model"])
    lm = AutoModelForCausalLM.from_pretrained(ck["model"], dtype=torch.bfloat16).cuda().eval()
    heads = ExitHeads(lm.config.hidden_size, lm.model.norm, E - 1)
    heads.load_state_dict(ck["state_dict"])
    heads = heads.cuda().to(torch.bfloat16).eval()
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    text = "\n\n".join(t for t in list(ds["validation"]["text"]) + list(ds["test"]["text"]) if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    n = min(len(ids) // args.win, args.windows)
    X = ids[: n * args.win].view(n, args.win)
    print(f"{n} windows of {args.win} tokens, exits {EX}", flush=True)
    rows = []
    with torch.inference_mode():
        for i in range(n):
            x = X[i:i + 1].cuda()
            out = lm.model(input_ids=x, output_hidden_states=True)
            y = x[0, 1:]
            hs = out.hidden_states
            def top(feat):                                  # argmax token and its probability, 512 positions at a time
                a, c = [], []
                for k in range(0, feat.shape[0], 512):
                    lg = lm.lm_head(feat[k:k + 512]).float()
                    a.append(lg.argmax(-1)), c.append(lg.softmax(-1).max(-1).values)
                return torch.cat(a), torch.cat(c)
            pf, conf_f = top(out.last_hidden_state[0, :-1])   # last_hidden_state is already normed
            d = {"window": i, "label": y.cpu().numpy(), f"pred_{E - 1}": pf.cpu().numpy(),
                 f"conf_{E - 1}": conf_f.cpu().numpy(), "pos": np.arange(1, args.win), "label_ok_full": (pf == y).cpu().numpy(), "conf_full": conf_f.cpu().numpy()}
            for e in range(E - 1):
                feat = heads.features(e, hs[EX[e]][0, :-1])
                pe = top(feat)[0]
                d[f"pred_{e}"] = pe.cpu().numpy()
                d[f"agree_{EX[e]}"] = (pe == pf).cpu().numpy()
                d[f"ok_{EX[e]}"] = (pe == y).cpu().numpy()
                d[f"conf_{EX[e]}"] = heads.confidence(e, feat).float().cpu().numpy()
            rows.append(pd.DataFrame(d))
            if i % 16 == 0:
                print(f"window {i}/{n}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    if args.save_traces:      # simulator schema, one file per position range: traces/wikitext_<ckpt>ctx<lo>_val.parquet
        import json
        wsplit = np.where(np.random.default_rng(0).permutation(n) < n // 2, "evolve", "heldout")
        for lo, hi in zip(args.save_traces[::2], args.save_traces[1::2]):
            g = df[(df.pos >= lo) & (df.pos < hi)]
            t = pd.DataFrame({"window": g.window, "pos": g.pos, "label": g.label,
                              **{f"conf_{e}": g[f"conf_{EX[e]}"] if e < E - 1 else g[f"conf_{E - 1}"] for e in range(E)},
                              **{f"pred_{e}": g[f"pred_{e}"] for e in range(E)}}).reset_index(drop=True)
            t["idx"] = np.arange(len(t))
            t["split"] = wsplit[t.window.to_numpy()]
            task = f"wikitext_{args.ckpt}ctx{lo}"
            t.to_parquet(ROOT / f"traces/{task}_val.parquet", index=False)
            (ROOT / f"traces/{task}_val.meta.json").write_text(json.dumps({"exit_layers": EX, "decoder": True, "pos": [lo, hi]}))
            print("wrote", task, len(t), flush=True)
    df["bucket"] = pd.cut(df.pos, [b for b in BUCKETS if b < args.win] + [args.win], right=False)
    agg = {"label_ok_full": "mean", "conf_full": "mean", "pos": "size"}
    for l in EX[:-1]:
        agg.update({f"agree_{l}": "mean", f"ok_{l}": "mean", f"conf_{l}": "mean"})
    t = df.groupby("bucket", observed=True).agg(agg).rename(columns={"pos": "tokens", "label_ok_full": "acc_full"})
    for l in EX[:-1]:
        t[f"drop_pts_{l}"] = 100 * (t.acc_full - t[f"ok_{l}"])
    t.to_csv(ROOT / f"results/context_acc_{args.ckpt}.csv")
    pd.set_option("display.width", 250)
    cols = ["tokens", "acc_full"] + [f"agree_{l}" for l in EX[-3:-1]] + [f"drop_pts_{l}" for l in EX[-3:-1]]
    print(t[cols].round(3).to_string())


def _time(fn, reps: int = 20, rounds: int = 5) -> float:
    for _ in range(5):
        fn()
    ts = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - s) / reps * 1000)
    return float(np.median(ts))


def _graph(fn):
    """Capture fn in a CUDA graph and return its replay (GPU work without Python's kernel-launch overhead)."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def cost(args) -> None:
    """Per decode step at context T and batch B, eager and (--graph) CUDA-graph: all layers, one exit check
    (confidence head + copy to host + sync), one lm_head pass, KV propagation for all layers."""
    from decoder.decode_runner import EEDecoder
    dec = EEDecoder("", ckpt=f"{args.ckpt}_exits")
    mode = "graph" if args.graph else "eager"
    rows = []
    for T in args.ctx:
        for B in args.batch_sizes:
            dec.win = T + 1
            try:
                cache = dec.new_cache(B)
                cache[0].normal_(0, 0.02), cache[1].normal_(0, 0.02)
                t0 = T - 1
                x = torch.randint(0, 1000, (B, 1), device="cuda")
                with torch.inference_mode():
                    h0 = dec.m.embed_tokens(x)
                    cos, sin = dec.m.rotary_emb(h0, torch.full((B, 1), t0, device="cuda"))
                    host = dec.conf_host[:B]
                    buf = {}

                    def layers():
                        h = h0
                        for j in range(dec.L):
                            h = dec.layer(j, h, cache, t0, cos, sin)

                    def head_gpu():
                        buf["c"] = dec.heads.confidence(2, dec.heads.features(2, h0[:, 0]))

                    def lmh():
                        dec.lm.lm_head(dec.m.norm(h0[:, 0])).argmax(-1)

                    def prop():
                        dec.propagate_from(0, h0[:, 0], cache, t0, cos, sin)
                    if args.graph:
                        layers, head_gpu, lmh, prop = _graph(layers), _graph(head_gpu), _graph(lmh), _graph(prop)

                    def head():
                        head_gpu()
                        host.copy_(buf["c"].float(), non_blocking=True)
                        torch.cuda.current_stream().synchronize()
                    r = {"T": T, "B": B, "mode": mode, "per_layer_ms": _time(layers, 10) / dec.L, "head_ms": _time(head, 50),
                         "exit_pass_ms": _time(lmh, 50), "skip_ms": _time(prop, 20) / dec.L}
            except torch.OutOfMemoryError:
                print(f"T={T} B={B}: OOM", flush=True)
                r = None
            if r:
                rows.append(r)
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            cache = layers = head_gpu = lmh = prop = None
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    gpu = torch.cuda.get_device_name().replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
    df["gpu"] = gpu
    df.to_csv(ROOT / f"results/context_cost_{args.ckpt}_{mode}_{gpu}.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.pivot_table(index="T", columns="B", values="per_layer_ms").round(3).to_string())
    c1 = df[df.B == df.B.min()].set_index("T").per_layer_ms
    df["per_seq_share"] = 1 - df["T"].map(c1) / df.per_layer_ms          # share of a layer's time that scales with n
    print("share of per-layer time that is per-sequence cost (1 - c(1)/c(B)):")
    print(df.pivot_table(index="T", columns="B", values="per_seq_share").round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["acc", "cost"])
    ap.add_argument("--ckpt", default="qlora", help="checkpoints/<ckpt>_exits.pt")
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=64)
    ap.add_argument("--save-traces", type=int, nargs="+", default=None, help="lo hi [lo hi ...]: position ranges to save as traces")
    ap.add_argument("--ctx", type=int, nargs="+", default=[256, 1024, 2048, 4096, 8192])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--graph", action="store_true", help="cost mode: time CUDA-graph replays instead of eager calls")
    args = ap.parse_args()
    acc(args) if args.mode == "acc" else cost(args)
