"""P1: time one transformer layer, the exit head, and the full encoder at n active samples; fit c(n) = a + b*n.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python profiling/profile_layer.py configs/p1_profile.yaml

Outputs:
  results/p1_timing_<gpu>.csv   one row per (dtype, seq_len, component, n)
  results/p1_fit_<gpu>.json     a, b, R^2, crossover n* = a/b per (dtype, seq_len, component)
  figures/p1_cost_fit_<gpu>.png measured layer latency vs n with fitted line
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from transformers import AutoConfig, AutoModel

ROOT = Path(__file__).resolve().parents[1]
DTYPES = {"float32": torch.float32, "float16": torch.float16}


class ExitHead(nn.Module):
    """Same shape as RobertaClassificationHead: CLS token -> dense -> tanh -> out_proj."""

    def __init__(self, hidden: int, num_labels: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, num_labels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.out_proj(torch.tanh(self.dense(h[:, 0])))
        return logits.softmax(-1).max(-1).values  # confidence, as the exit rule would use it


@torch.inference_mode()
def time_ms(fn, warmup: int, repeats: int, cuda_graph: bool = False) -> np.ndarray:
    """Per-call latency. With cuda_graph=True the call is captured once and replayed,
    which removes Python and kernel-launch overhead (what an optimized server would see)."""
    if cuda_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        fn = graph.replay
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = np.empty(repeats)
    for i in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out[i] = start.elapsed_time(end)
    return out


def fit_line(n: np.ndarray, t: np.ndarray) -> dict[str, float]:
    b, a = np.polyfit(n, t, 1)
    pred = a + b * n
    r2 = 1.0 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)
    return {
        "a_ms": float(a),
        "b_ms": float(b),
        "r2": float(r2),
        "max_abs_resid_pct": float(np.max(np.abs(t - pred) / t) * 100),
        "n_star": float(a / b) if b > 0 else float("inf"),  # rho = a/(b*n) = 1 here; memory-bound for n < n_star
    }


def fit_hinge(n: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """c(n) = a + b * max(0, n - k): flat until the GPU saturates at knee k, then linear.
    k is grid-searched over the measured n; a, b by least squares."""
    best = None
    for k in n[:-2]:
        x = np.maximum(0.0, n - k)
        b, a = np.polyfit(x, t, 1)
        sse = float(np.sum((t - (a + b * x)) ** 2))
        if best is None or sse < best[0]:
            best = (sse, a, b, k)
    sse, a, b, k = best
    return {
        "hinge_a_ms": float(a),
        "hinge_b_ms": float(b),
        "hinge_knee_n": float(k),
        "hinge_r2": float(1.0 - sse / np.sum((t - t.mean()) ** 2)),
        "hinge_max_abs_resid_pct": float(np.max(np.abs(t - (a + b * np.maximum(0.0, n - k))) / t) * 100),
    }


def main(cfg_path: str) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    torch.manual_seed(cfg["seed"])
    gpu = cfg["gpu_label"]
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}  torch {torch.__version__}", flush=True)

    hf_cfg = AutoConfig.from_pretrained(cfg["model"])
    rows = []
    for dtype_name in cfg["dtypes"]:
        dtype = DTYPES[dtype_name]
        model = AutoModel.from_config(hf_cfg, attn_implementation="sdpa").to(device, dtype).eval()
        layer = model.encoder.layer[0]
        head = ExitHead(hf_cfg.hidden_size, cfg["num_labels"]).to(device, dtype).eval()
        for seq_len in cfg["seq_lens"]:
            for n in cfg["batch_sizes"]:
                h = torch.randn(n, seq_len, hf_cfg.hidden_size, device=device, dtype=dtype)
                components = {
                    "layer": lambda: layer(h),
                    "exit_head": lambda: head(h),
                    "encoder_full": lambda: model.encoder(h),
                }
                for mode in cfg["modes"]:
                    for comp, fn in components.items():
                        t = time_ms(fn, cfg["warmup"], cfg["repeats"], cuda_graph=(mode == "cuda_graph"))
                        rows.append({
                            "gpu": gpu, "mode": mode, "dtype": dtype_name, "seq_len": seq_len, "component": comp,
                            "n": n, "median_ms": float(np.median(t)), "mean_ms": float(t.mean()),
                            "std_ms": float(t.std()), "p10_ms": float(np.percentile(t, 10)),
                            "p90_ms": float(np.percentile(t, 90)),
                        })
                    last = rows[-3]
                    print(f"{mode:10s} {dtype_name} L={seq_len} n={n:4d} layer {last['median_ms']:.4f} ms", flush=True)
        del model, layer, head
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    num_layers = hf_cfg.num_hidden_layers
    fits = []
    for (mode, dtype_name, seq_len, comp), g in df.groupby(["mode", "dtype", "seq_len", "component"]):
        g = g.sort_values("n")
        n_arr, t_arr = g["n"].to_numpy(float), g["median_ms"].to_numpy()
        fits.append({"gpu": gpu, "mode": mode, "dtype": dtype_name, "seq_len": int(seq_len), "component": comp,
                     **fit_line(n_arr, t_arr), **fit_hinge(n_arr, t_arr)})
    # Simulator assumption check: full encoder ~= num_layers x one layer
    piv = df.pivot_table(index=["mode", "dtype", "seq_len", "n"], columns="component", values="median_ms")
    ratio = piv["encoder_full"] / (num_layers * piv["layer"])
    additivity = {mode: r.describe()[["min", "mean", "max"]].to_dict() for mode, r in ratio.groupby(level="mode")}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    df.to_csv(ROOT / f"results/p1_timing_{gpu}.csv", index=False)
    meta = {
        "device": torch.cuda.get_device_name(device), "torch": torch.__version__, "config": cfg,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "encoder_over_layers_ratio": additivity, "fits": fits,
    }
    (ROOT / f"results/p1_fit_{gpu}.json").write_text(json.dumps(meta, indent=2))

    layer_df = df[df.component == "layer"]
    grid = [(m, d) for m in cfg["modes"] for d in cfg["dtypes"]]
    fig, axes = plt.subplots(1, len(grid), figsize=(5.5 * len(grid), 4.5), squeeze=False)
    for ax, (mode, dtype_name) in zip(axes[0], grid):
        for seq_len in cfg["seq_lens"]:
            g = layer_df[(layer_df["mode"] == mode) & (layer_df.dtype == dtype_name) & (layer_df.seq_len == seq_len)].sort_values("n")
            f = next(x for x in fits if (x["mode"], x["dtype"], x["seq_len"], x["component"]) == (mode, dtype_name, seq_len, "layer"))
            pts = ax.plot(g.n, g.median_ms, "o", label=f"L={seq_len}: a={f['a_ms']:.3f}, b={f['b_ms']:.4f}, R²={f['r2']:.3f}")
            ax.plot(g.n, f["a_ms"] + f["b_ms"] * g.n, "-", color=pts[0].get_color(), alpha=0.6)
            ax.plot(g.n, f["hinge_a_ms"] + f["hinge_b_ms"] * np.maximum(0, g.n - f["hinge_knee_n"]), ":", color=pts[0].get_color())
        ax.set(title=f"one layer, {mode}, {dtype_name}", xlabel="active samples n", ylabel="median latency (ms)")
        ax.legend(fontsize=7, title="solid: a+b·n   dotted: hinge", title_fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{cfg['model']} on {gpu}")
    fig.tight_layout()
    fig.savefig(ROOT / f"figures/p1_cost_fit_{gpu}.png", dpi=150)

    cols = ["mode", "dtype", "seq_len", "a_ms", "b_ms", "r2", "n_star", "hinge_a_ms", "hinge_b_ms", "hinge_knee_n", "hinge_r2"]
    print("\nlayer fits (linear c(n) = a + b*n; hinge c(n) = a + b*max(0, n-k)):")
    print(pd.DataFrame([f for f in fits if f["component"] == "layer"])[cols].round(4).to_string(index=False))
    print("\nexit head median ms at n=64:")
    print(df[(df.component == "exit_head") & (df.n == 64)][["mode", "dtype", "seq_len", "median_ms"]].round(4).to_string(index=False))
    print(f"\nencoder_full / ({num_layers} x layer): {json.dumps(additivity, indent=1)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "configs/p1_profile.yaml"))
