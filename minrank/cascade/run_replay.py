"""Replay stopping policies over evaluation traces (Option D).

Usage:
  python -m minrank.cascade.run_replay --traces minrank/evolve/runs/traces/evals.jsonl --batch-size 8 \
      --policies all --lambdas 0.001 0.003 0.01 0.03 0.1 --out minrank/results/cascade/
  python -m minrank.cascade.run_replay --synthetic 400 --out minrank/results/cascade_synthetic/

Outputs in --out: summary.csv (one row per policy), per_batch.csv (anytime data), anytime.png (if matplotlib is
installed), and synthetic_traces.jsonl when --synthetic is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .policies import make_policies
from .simulator import SimConfig, simulate
from .traces import load_traces, save_traces, synthetic_traces, to_batches

ROOT = Path(__file__).resolve().parents[1]


def plot_anytime(per_batch: pd.DataFrame, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, g in per_batch.groupby("policy", sort=False):
        ax.step(g["cum_seconds"], g["incumbent"], where="post", label=label, linewidth=1.4)
    ax.set_xlabel("evaluation seconds (cumulative)")
    ax.set_ylabel("incumbent fitness")
    ax.set_title("Incumbent vs evaluation compute")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--traces", type=Path)
    src.add_argument("--synthetic", type=int, metavar="N_CANDIDATES")
    ap.add_argument("--synthetic-graphs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--policies", nargs="+", default=["all"])
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.001, 0.003, 0.01, 0.03, 0.1])
    ap.add_argument("--checkpoints", nargs="*", type=int, default=None, help="graph counts; default 1,2,4,...,L")
    ap.add_argument("--a", type=float, default=0.05, help="fixed cost per evaluated graph (seconds)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--explore", type=int, default=1)
    ap.add_argument("--audit-every", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results/cascade")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        df = synthetic_traces(a.synthetic, a.synthetic_graphs, seed=a.seed)
        save_traces(df, a.out / "synthetic_traces.jsonl")
    else:
        df = load_traces(a.traces)
    batches = to_batches(df, a.batch_size)
    L = batches[0].n_graphs
    print(f"{len(batches)} batches, {sum(b.size for b in batches)} candidates, {L} graphs", flush=True)
    cfg = SimConfig(a.checkpoints or None, a.a, a.warmup, a.explore, a.audit_every, a.seed)
    rows, per = [], []
    for pol in make_policies(a.policies, a.lambdas, a.alpha, a.delta):
        r = simulate(batches, pol, cfg)
        rows.append(r.summary)
        per += r.per_batch
        s = r.summary
        print(f"{s['policy']:28s} savings {s['savings']:6.1%}  speedup {s['speedup']:5.2f}x  "
              f"final {s['final_incumbent']:+.4f} (full {s['full_final_incumbent']:+.4f})  "
              f"lost {s['lost_winners']}/{s['batches_with_winner']}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "summary.csv", index=False)
    pb = pd.DataFrame(per)
    pb.to_csv(a.out / "per_batch.csv", index=False)
    if not a.no_plot and plot_anytime(pb, a.out / "anytime.png"):
        print(f"wrote {a.out / 'anytime.png'}")
    print(f"wrote {a.out / 'summary.csv'} and {a.out / 'per_batch.csv'}")


if __name__ == "__main__":
    main()
