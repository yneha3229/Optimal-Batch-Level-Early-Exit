"""P6 (real GPU): run batched early-exit inference on the held-out MNLI split and measure wall-clock time.
Validates the simulator: same policies, same parameters as selected in simulation (regime enc_eager).

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ~/venv/bin/python simulator/wallclock.py --batch-sizes 16 64 128
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
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import policies as P  # noqa: E402
from simulator.exit_model import ExitHead  # noqa: E402
from simulator.run_sim import load_costs  # noqa: E402
from simulator.simulator import load_traces  # noqa: E402


class EarlyExitRoberta:
    def __init__(self, heads_path: Path) -> None:
        ck = torch.load(heads_path, weights_only=False)
        self.exit_layers = ck["exit_layers"]
        full = AutoModelForSequenceClassification.from_pretrained(ck["backbone"], dtype=torch.float16)
        self.rb = full.roberta.cuda().eval()
        self.heads = torch.nn.ModuleList(ExitHead(ck["hidden"], ck["num_labels"]) for _ in self.exit_layers)
        self.heads.load_state_dict(ck["state_dict"])
        self.heads = self.heads.cuda().half().eval()
        self.min = torch.finfo(torch.float16).min

    @torch.inference_mode()
    def run_batch(self, ids: torch.Tensor, attn: torch.Tensor, decide, compact: bool,
                  exits_enabled: bool = True) -> np.ndarray:
        """Returns predictions [B]. Exited samples leave the batch if compact, else stay (masked) until it ends.
        exits_enabled=False is the plain full model: only the final head runs."""
        B = ids.shape[0]
        h = self.rb.embeddings(input_ids=ids)
        mask = ((1.0 - attn[:, None, None, :].half()) * self.min)
        alive = torch.arange(B, device="cuda")            # rows of h -> original sample index
        done = torch.zeros(B, dtype=torch.bool, device="cuda")
        preds = torch.full((B,), -1, dtype=torch.long, device="cuda")
        E, layer = len(self.exit_layers), 0
        for e, stop_at in enumerate(self.exit_layers):
            while layer < stop_at:
                h = self.rb.encoder.layer[layer](h, attention_mask=mask)
                layer += 1
            if not exits_enabled and e < E - 1:
                continue
            rows = torch.nonzero(~done[alive]).squeeze(1) if not compact else None
            hc = h[:, 0] if compact else h[rows, 0]
            probs = self.heads[e](hc).float().softmax(-1)
            conf, pred = probs.max(-1)
            idx = alive if compact else alive[rows]
            if e == E - 1:
                preds[idx] = pred
                break
            exit_mask, stop = decide(e, conf.cpu().numpy().astype(np.float64), len(idx), B,
                                     self.exit_layers[-1] - stop_at, 0.0, 0.0)
            leave = torch.ones_like(conf, dtype=torch.bool) if stop else torch.from_numpy(np.asarray(exit_mask)).cuda()
            preds[idx[leave]] = pred[leave]
            done[idx[leave]] = True
            if bool(done.all()):
                break
            if compact:
                keep = ~leave
                h, mask, alive = h[keep], mask[keep], alive[keep]
        return preds.cpu().numpy()


def make_decider(cost, cal, exit_layers, fam: str, params: dict):
    """Real a, b come from the cost regime (the rule needs them); the loop passes 0s, so close over them."""
    if fam == "full":
        base = P.never_exit()
    elif fam.startswith("threshold") and "lam" not in params:
        base = P.threshold(params["tau"])
    elif fam in ("consensus", "majority", "greedy"):
        base = getattr(P, fam)(params["tau"])
    elif fam == "random_collective":
        base = P.param_collective(**params)
    elif fam == "theory":
        base = P.theory_rule(params["lam"], cal, exit_layers, cost)
    elif fam == "theory_indiv_only":
        base = P.theory_rule(params["lam"], cal, exit_layers, cost, collective=False)
    else:
        raise ValueError(fam)
    return lambda e, c, n, B, left, a, b: base(e, c, n, B, left, cost.a, cost.b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64, 128])
    ap.add_argument("--regime", default="enc_eager")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    tr_df = pd.read_parquet(ROOT / "traces/mnli_val.parquet")
    ho = tr_df[tr_df.split == "heldout"].idx.to_numpy()
    val = load_dataset("nyu-mll/glue", "mnli")["validation_matched"].select(ho.tolist())
    model = EarlyExitRoberta(ROOT / "traces/heads_mnli.pt")
    tok = AutoTokenizer.from_pretrained("textattack/roberta-base-MNLI")
    enc = tok(list(val["premise"]), list(val["hypothesis"]), truncation=True, max_length=128, padding="max_length", return_tensors="pt")
    ids_all, attn_all, labels = enc["input_ids"].cuda(), enc["attention_mask"].cuda(), np.array(val["label"])

    cost = next(c for c in load_costs() if c.name == args.regime)
    cal = P.Calibration(load_traces(ROOT / "traces/mnli_val.parquet", "evolve"))
    sel = pd.read_csv(ROOT / "results/p6_sim_selected.csv")
    fams = ["full", "heads_only", "threshold", "threshold_nocompact", "consensus", "majority", "random_collective",
            "theory_indiv_only", "theory"]

    # Sanity: layer-by-layer full run must reproduce the traced final-exit predictions
    sanity = model.run_batch(ids_all[:256], attn_all[:256], P.never_exit(), True, exits_enabled=False)
    traced = tr_df.set_index("idx").loc[ho[:256], "pred_5"].to_numpy()
    print(f"sanity: layer loop vs trace agreement on final exit = {(sanity == traced).mean():.4f}", flush=True)

    rows = []
    for B in args.batch_sizes:
        N = (len(labels) // B) * B
        for fam in fams:
            if fam == "heads_only":  # evaluate every exit head + decision, never exit: isolates exit overhead
                s = sel[(sel.regime == args.regime) & (sel.B == B) & (sel.family == "full")]
                params, decide = {}, P.never_exit()
            else:
                s = sel[(sel.regime == args.regime) & (sel.B == B) & (sel.family == fam)]
                if s.empty:
                    continue
                params = json.loads(s.params.iloc[0])
                decide = make_decider(cost, cal, model.exit_layers, fam, params)
            compact = fam != "threshold_nocompact"
            model.run_batch(ids_all[:B], attn_all[:B], decide, compact, fam != "full")  # warmup
            times, preds = [], None
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                preds = np.concatenate([model.run_batch(ids_all[i:i + B], attn_all[i:i + B], decide, compact, fam != "full")
                                        for i in range(0, N, B)])
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            rows.append({"B": B, "family": fam, "params": json.dumps(params), "wall_s": float(np.median(times)),
                         "accuracy": float((preds == labels[:N]).mean()),
                         "sim_speedup": float(s.heldout_speedup.iloc[0]), "sim_acc": float(s.heldout_acc.iloc[0])})
            print(f"B={B:4d} {fam:20s} {rows[-1]['wall_s']:.3f}s acc {rows[-1]['accuracy']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    full_t = df[df.family == "full"].set_index("B").wall_s
    full_a = df[df.family == "full"].set_index("B").accuracy
    df["real_speedup"] = df.apply(lambda r: full_t[r.B] / r.wall_s, axis=1)
    df["acc_drop_pct"] = df.apply(lambda r: 100 * (full_a[r.B] - r.accuracy), axis=1)
    df.to_csv(ROOT / f"results/p6_wallclock_{args.regime}.csv", index=False)
    # Per-exit overhead (head + GPU->CPU sync + host decision), ms per exit point per batch
    E1 = len(model.exit_layers) - 1
    overhead = {int(B): 1000 * (g.set_index("family").wall_s["heads_only"] - g.set_index("family").wall_s["full"])
                / ((len(labels) // B) * E1) for B, g in df.groupby("B")}
    (ROOT / f"results/p6_exit_overhead_{args.regime}.json").write_text(json.dumps(overhead, indent=2))
    print(f"per-exit overhead ms (by B): {overhead}")
    print(df[["B", "family", "accuracy", "acc_drop_pct", "real_speedup", "sim_speedup"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
