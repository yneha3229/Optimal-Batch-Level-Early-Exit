"""Real batched early-exit *decoding* on the GPU (Qwen2.5-1.5B + exit heads): the large-rho experiment.

B held-out WikiText windows decode together, one token each per step, teacher-forced (the next input is always the
true token, so every policy sees the same inputs and accuracy is next-token accuracy). Per step:
  - layers run on the whole batch; after each exit layer the cheap confidence head scores every row and the
    *same numpy decide()* as the simulator decides (one host sync per exit point);
  - per-sample exits: rows leave with the exit's token (one extra lm_head pass at that exit), but stay in the batch
    (no rebatching), which is what per-sample exit costs in batched decoding;
  - collective stop: every remaining row takes the exit's token, and the K/V of all skipped layers are filled from
    the exit hidden state (CALM-style state propagation: input norm + k/v projections + rotary, ~2% of a layer
    for this GQA model), so later tokens can still attend to them.
Own KV cache (preallocated), own layer loop with SDPA; prefill of the first CTX-1 tokens runs the full model.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python decoder/decode_runner.py --calibrate
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python decoder/decode_runner.py --regime real_wikitext
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from decoder.train_decoder_exits import ExitHeads  # noqa: E402
from simulator.simulator import Cost  # noqa: E402

DEV = "cuda"
DT = torch.bfloat16


def never(e, c, n, B, left, a, b):
    return np.zeros(n, bool), False


class EEDecoder:
    def __init__(self, suffix: str = "", ckpt: str = "qwen_exits") -> None:
        from transformers import AutoModelForCausalLM
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        ck = torch.load(ROOT / f"checkpoints/{ckpt}{suffix}.pt", weights_only=False)
        self.lm = AutoModelForCausalLM.from_pretrained(ck["model"], dtype=DT).to(DEV).eval()
        self.m = self.lm.model
        self.exits = ck["exits"]                    # [8, 12, 16, 20, 24, 28]
        self.ctx, self.win = ck["ctx"], ck["win"]
        cfg = self.lm.config
        self.heads = ExitHeads(cfg.hidden_size, self.m.norm, len(self.exits) - 1)
        self.heads.load_state_dict(ck["state_dict"])
        self.heads = self.heads.to(DEV).to(DT).eval()
        self.nh, self.nkv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self.rope = apply_rotary_pos_emb
        self.conf_host = torch.empty(1024, dtype=torch.float32, pin_memory=True)
        self.L = len(self.m.layers)
        # Fused state propagation: fold each layer's input-RMSNorm weight into its k/v projection and stack all
        # layers, so the K/V of every skipped layer come from one batched matmul + one cache write.
        lys = self.m.layers
        self.eps = lys[0].input_layernorm.variance_epsilon
        self.Wk = torch.stack([l.self_attn.k_proj.weight * l.input_layernorm.weight[None, :] for l in lys])  # [L, kv, d]
        self.Wv = torch.stack([l.self_attn.v_proj.weight * l.input_layernorm.weight[None, :] for l in lys])
        zb = lambda lin: lin.bias if lin.bias is not None else torch.zeros(lin.out_features, device=DEV, dtype=DT)
        self.bk = torch.stack([zb(l.self_attn.k_proj) for l in lys])                                          # [L, kv]
        self.bv = torch.stack([zb(l.self_attn.v_proj) for l in lys])

    # ---------------------------------------------------------------- layer pieces
    def _qkv(self, j: int, x: torch.Tensor, cos, sin):
        at = self.m.layers[j].self_attn
        B, T, _ = x.shape
        q = at.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = at.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = at.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        return q, k, v

    def layer(self, j: int, h: torch.Tensor, cache, t0: int, cos, sin) -> torch.Tensor:
        ly = self.m.layers[j]
        T = h.shape[1]
        q, k, v = self._qkv(j, ly.input_layernorm(h), cos, sin)
        K, V = cache[0][j], cache[1][j]
        K[:, :, t0:t0 + T], V[:, :, t0:t0 + T] = k, v
        o = F.scaled_dot_product_attention(q, K[:, :, :t0 + T], V[:, :, :t0 + T], is_causal=T > 1, enable_gqa=True)
        h = h + ly.self_attn.o_proj(o.transpose(1, 2).reshape(h.shape[0], T, -1))
        return h + ly.mlp(ly.post_attention_layernorm(h))

    def propagate_from(self, j0: int, h: torch.Tensor, cache, t0: int, cos, sin) -> None:
        """K/V of all skipped layers j0..L-1 from the exit hidden state h [B, d] (fused state propagation)."""
        if j0 >= self.L:
            return
        B = h.shape[0]
        xh = h * torch.rsqrt(h.float().pow(2).mean(-1, keepdim=True) + self.eps).to(h.dtype)    # RMSNorm, no weight
        k = torch.einsum("bd,ljd->lbj", xh, self.Wk[j0:]) + self.bk[j0:, None]                 # [Ls, B, kv]
        v = torch.einsum("bd,ljd->lbj", xh, self.Wv[j0:]) + self.bv[j0:, None]
        k = k.view(-1, B, self.nkv, 1, self.hd)
        v = v.view(-1, B, self.nkv, 1, self.hd)
        c, s_ = cos[:, None], sin[:, None]                                                       # [B, 1, 1, hd]
        k = k * c + self._rot(k) * s_
        cache[0][j0:, :, :, t0:t0 + 1] = k
        cache[1][j0:, :, :, t0:t0 + 1] = v

    @staticmethod
    def _rot(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def new_cache(self, B: int):
        shape = (self.L, B, self.nkv, self.win, self.hd)
        return (torch.empty(shape, device=DEV, dtype=DT), torch.empty(shape, device=DEV, dtype=DT))

    # ---------------------------------------------------------------- prefill + decode
    @torch.inference_mode()
    def prefill(self, x: torch.Tensor, cache) -> None:
        T = self.ctx - 1
        h = self.m.embed_tokens(x[:, :T])
        pos = torch.arange(T, device=DEV)[None].expand(x.shape[0], T)
        cos, sin = self.m.rotary_emb(h, pos)
        for j in range(self.L):
            h = self.layer(j, h, cache, 0, cos, sin)

    @torch.inference_mode()
    def decode(self, x: torch.Tensor, decide, cost: Cost, exits_enabled=True, stats=None) -> torch.Tensor:
        """Teacher-forced decode of positions ctx..win-1 for the batch x [B, win]. Returns predictions [B, steps]."""
        B = x.shape[0]
        cache = self.new_cache(B)
        self.prefill(x, cache)
        steps = self.win - self.ctx
        preds = torch.empty(B, steps, dtype=torch.long, device=DEV)
        E = len(self.exits)
        want_hist = getattr(decide, "history", False)
        for s in range(steps):
            t0 = self.ctx - 1 + s
            h = self.m.embed_tokens(x[:, t0:t0 + 1])
            pos = torch.full((B, 1), t0, device=DEV)
            cos, sin = self.m.rotary_emb(h, pos)
            out = torch.full((B,), -1, dtype=torch.long, device=DEV)
            left = np.ones(B, bool)                       # rows without a token yet
            hist = np.zeros((B, E))                       # host copy of each row's confidence at every exit
            j, e = 0, 0
            stopped = False
            while j < self.L:
                h = self.layer(j, h, cache, t0, cos, sin)
                j += 1
                if not exits_enabled or e >= E - 1 or j != self.exits[e]:
                    continue
                feat = self.heads.features(e, h[:, 0])
                conf = self.heads.confidence(e, feat)
                ch = self.conf_host[:B]
                ch.copy_(conf.float(), non_blocking=True)
                torch.cuda.current_stream().synchronize()
                act = np.flatnonzero(left)
                hist[:, e] = ch.numpy()
                c_in = hist[act, :e + 1] if want_hist else hist[act, e]
                mask, stop = decide(e, c_in, len(act), B, self.L - j, cost.a, cost.b)
                mask = np.asarray(mask, bool)
                leave = act if stop else act[mask]
                if len(leave):
                    tok = self.lm.lm_head(feat).argmax(-1)          # one lm_head pass at this exit
                    li = torch.from_numpy(leave).to(DEV, non_blocking=True)
                    out[li] = tok[li]
                    left[leave] = False
                if stop or not left.any():
                    self.propagate_from(j, h[:, 0], cache, t0, cos, sin)
                    stopped = True
                    if stats is not None:
                        stats["layers"] += j * B
                    break
                e += 1
            if not stopped:
                if stats is not None:
                    stats["layers"] += self.L * B
                if left.any():
                    tok = self.lm.lm_head(self.m.norm(h[:, 0])).argmax(-1)
                    li = torch.from_numpy(np.flatnonzero(left)).to(DEV, non_blocking=True)
                    out[li] = tok[li]
            preds[:, s] = out
        return preds


def timed(dec: EEDecoder, X: torch.Tensor, B: int, decide, cost: Cost, repeats: int, **kw):
    dec.decode(X[:B], decide, cost, **kw)                                   # warm-up
    times, P = [], None
    for _ in range(repeats):
        stats = {"layers": 0}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        P = torch.cat([dec.decode(X[i:i + B], decide, cost, stats=stats, **kw) for i in range(0, len(X), B)])
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), P.cpu().numpy(), stats


@torch.inference_mode()
def calibrate(dec: EEDecoder, X: torch.Tensor, repeats: int) -> dict:
    """Per-layer a, b from decode-step time vs B; head (confidence head + sync), exit_pass (lm_head), skip (KV prop)."""
    rows = []
    for B in [1, 2, 4, 8, 16, 32, 64, 128]:
        x = X[:B] if B <= len(X) else X.repeat(B // len(X) + 1, 1)[:B]
        cache = dec.new_cache(B)
        dec.prefill(x, cache)
        t0 = dec.ctx - 1
        h0 = dec.m.embed_tokens(x[:, t0:t0 + 1])
        pos = torch.full((B, 1), t0, device=DEV)
        cos, sin = dec.m.rotary_emb(h0, pos)

        def run_layers():
            h = h0
            for j in range(dec.L):
                h = dec.layer(j, h, cache, t0, cos, sin)
            return h

        def head():
            f = dec.heads.features(2, h0[:, 0])
            c = dec.heads.confidence(2, f)
            dec.conf_host[:B].copy_(c.float(), non_blocking=True)
            torch.cuda.current_stream().synchronize()

        def lmh():
            dec.lm.lm_head(dec.m.norm(h0[:, 0])).argmax(-1)

        def prop():
            dec.propagate_from(0, h0[:, 0], cache, t0, cos, sin)

        r = {"B": B}
        for name, fn, reps in [("layers28", run_layers, 20), ("head", head, 50), ("lm_head", lmh, 50), ("prop28", prop, 20)]:
            for _ in range(3):
                fn()
            ts = []
            for _ in range(repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(reps):
                    fn()
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t) / reps * 1000)
            r[name] = float(np.median(ts))
        rows.append(r)
        print(r, flush=True)
    d = pd.DataFrame(rows)
    b28, a28 = np.polyfit(d.B, d.layers28, 1)
    return {"task": "wikitext", "a": float(a28 / dec.L), "b": float(max(b28, 1e-6) / dec.L),
            "head": float(d["head"].median()), "exit_pass": float(d.lm_head.median()),
            "skip": float(d.prop28.median() / dec.L), "rho64": float(a28 / max(b28, 1e-6) / 64), "points": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", default=None)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--max-windows", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="use the smoke checkpoint/traces (functional test only)")
    ap.add_argument("--task", default="wikitext", help="wikitext (Qwen2.5-1.5B) or wikitext_llama (Llama-3.1-8B)")
    ap.add_argument("--out-suffix", default="", help="appended to the output CSV name (parallel runs per batch size)")
    args = ap.parse_args()
    tag = args.task.replace("wikitext", "").lstrip("_")
    sfx = "_smoke" if args.smoke and not tag else ""   # tagged tasks (train_exit_lora.py) carry the tag in every file name
    dec = EEDecoder(sfx, ckpt=f"{tag or 'qwen'}_exits")
    wpath = ROOT / (f"data/{args.task}{sfx}_windows.pt" if tag else f"data/wikitext_windows{sfx}.pt")
    W = torch.load(wpath, weights_only=False)
    ho = np.flatnonzero(W["split"] == "heldout")[: args.max_windows]
    X = W["windows"][ho].to(DEV)
    Y = X[:, dec.ctx:].cpu().numpy()

    # sanity: our layer loop vs the traced (HF forward) final-exit predictions
    tr = pd.read_parquet(ROOT / f"traces/{args.task}{sfx}_val.parquet", columns=["window", "pos", "pred_5"])
    ref = tr.set_index(["window", "pos"]).pred_5
    p = dec.decode(X[:8], never, Cost("z", 0, 0), exits_enabled=False).cpu().numpy()
    r = np.array([[ref[(int(ho[i]), dec.ctx + s)] for s in range(p.shape[1])] for i in range(8)])
    print(f"sanity: own decode loop vs HF forward, final-exit token agreement {(p == r).mean():.4f}", flush=True)

    if args.smoke:   # functional check of every code path: decide with stops, per-sample exits, propagation
        import itertools
        cnt = itertools.count()
        mixed = lambda e, c, n, B, left, a, b: (c > 0.3, next(cnt) % 3 == 0)
        P = dec.decode(X[:4], mixed, Cost("z", 0, 0))
        print("smoke decode ok, preds", tuple(P.shape), "acc", float((P.cpu().numpy() == Y[:4]).mean()))
        print(calibrate(dec, X[:4], repeats=1)["rho64"])
        return
    if args.calibrate:
        out = calibrate(dec, X[:16], repeats=3)
        out["task"] = args.task
        (ROOT / f"results/runner_cost_{args.task}.json").write_text(json.dumps(out, indent=1))
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items() if k != "points"})
        return

    args.regime = args.regime or f"real_{args.task}"
    from experiments import run_main as RM
    RM.init_worker(args.task)
    cost = RM.G["costs"][args.regime]
    sel = pd.read_csv(ROOT / f"results/main_{args.task}_selected.csv")
    sel = sel[sel.regime == args.regime]
    rows = []
    full_pred = {}
    for B in args.batch_sizes:
        n = len(X) // B * B
        fams = ["full", "heads_only"] + [f for f in sel[sel.B == B].family.unique()
                                          if f not in ("full", "lanes") and not f.startswith("funsearch")]
        for fam in fams:
            s = sel[(sel.B == B) & (sel.family == (fam if fam != "heads_only" else "full"))]
            if s.empty:
                continue
            if fam in ("full", "heads_only"):
                params, decide, kw = {}, never, {"exits_enabled": fam == "heads_only"}
            else:
                params, decide, _ = RM.candidates(fam, cost, B)[int(s.param_id.iloc[0])]
                kw = {}
            wall, P, st = timed(dec, X[:n], B, decide, cost, args.repeats, **kw)
            if fam == "full":
                full_pred[B] = P
            rows.append({"B": B, "family": fam, "params": json.dumps(params), "wall_s": wall,
                         "accuracy": float((P == Y[:n]).mean()),
                         "fidelity": float((P == full_pred[B]).mean()),
                         "layers_frac": st["layers"] / (n * Y.shape[1] * dec.L),
                         "sim_speedup": float(s.heldout_speedup.iloc[0]) if fam != "heads_only" else np.nan,
                         "sim_acc": float(s.heldout_acc.iloc[0])})
            print(f"B={B:3d} {fam:22s} {wall:7.2f}s acc {rows[-1]['accuracy']:.4f} fid {rows[-1]['fidelity']:.4f} "
                  f"layers {rows[-1]['layers_frac']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    ft = df[df.family == "full"].set_index("B").wall_s
    fa = df[df.family == "full"].set_index("B").accuracy
    df["real_speedup"] = df.apply(lambda r: ft[r.B] / r.wall_s, axis=1)
    df["acc_drop_pts"] = df.apply(lambda r: 100 * (fa[r.B] - r.accuracy), axis=1)
    df.to_csv(ROOT / f"results/main_{args.task}_{args.regime}_gpu{args.out_suffix}.csv", index=False)
    print(df[["B", "family", "accuracy", "acc_drop_pts", "fidelity", "real_speedup", "sim_speedup"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
