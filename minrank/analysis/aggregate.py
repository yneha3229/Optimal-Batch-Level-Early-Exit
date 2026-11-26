"""Aggregate every result into two tables and figures.

Inputs (all optional except the benchmark index; missing files are skipped with a note):
  minrank/results/baselines.csv      sweep rows (run_baselines.py)
  minrank/results/evolved.csv        evolved programs (evolve/eval_heldout.py), method "evolved:<id>"; failed runs
                                     have an empty rank and verified=False and only count in failure_rate
  minrank/results/lower_bounds.csv   certified lower bounds (run_baselines.py --lower-only)
  minrank/results/exact.csv          Option B driver output (proven optima)
  minrank/certificates/<graph>.json  best matrices; every one is re-verified here

Outputs:
  minrank/results/summary_graphs.csv   one row per graph: lb, best upper bound, gap, best method, proven optimal,
                                       certificate verified, best clique cover (TabuCol) and the gain over it
  minrank/results/summary_methods.csv  one row per (method, budget, family group): mean rank / best-known ratio,
                                       share of graphs where the method matches the best known, mean seconds
  minrank/results/figures/*.png        ratio-vs-budget per family group (only if matplotlib is installed)

Usage:
  python -m minrank.analysis.aggregate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..bench.instances import load_index
from ..core.graph import Graph
from ..core.verify import load_certificate, verify

ROOT = Path(__file__).resolve().parents[1]
COVER_METHODS = ("dsatur", "random_greedy", "tabucol")


def family_group(family: str) -> str:
    """Coarse groups for reporting: random, planted, structured."""
    if family in ("gnp", "dnp"):
        return "random"
    if family.startswith("planted") or family == "orthogonality":
        return "planted"
    return "structured"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        print(f"note: {path} not found, skipped")
        return None
    return pd.read_csv(path)


def verified_certificates(cert_dir: Path, graph_dir: Path, index: pd.DataFrame) -> pd.DataFrame:
    """Re-verify every certificate against the current graph file. Rows: graph, cert_rank, cert_method, cert_ok."""
    rows = []
    for _, r in index.iterrows():
        p = cert_dir / f"{r['name']}.json"
        if not p.exists():
            continue
        g = Graph.load(graph_dir / r["path"])
        c = load_certificate(p)
        ok, rank = False, None
        if c.get("digest") == g.digest():
            try:
                rank = verify(g, c["rows"])
                ok = True
            except ValueError:
                ok = False
        rows.append({"graph": r["name"], "cert_rank": rank, "cert_method": c.get("method"), "cert_ok": ok})
    return pd.DataFrame(rows, columns=["graph", "cert_rank", "cert_method", "cert_ok"])


def summarize(results_dir: Path, graph_dir: Path, cert_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DataFrame(load_index(graph_dir))
    index["n"] = index["n"].astype(int)
    graphs = index.rename(columns={"name": "graph"})[["graph", "split", "family", "n"]].copy()
    graphs["group"] = graphs["family"].map(family_group)

    parts = [d for d in (_read(results_dir / "baselines.csv"), _read(results_dir / "evolved.csv")) if d is not None]
    base = pd.concat(parts, ignore_index=True) if parts else None
    if base is not None:
        # eval_heldout writes failed runs (matrix did not fit / crash) with an empty rank and verified=False.
        ok = base["rank"].notna() & base["verified"].astype(str).str.lower().eq("true")
        n_fail = int((~ok).sum())
        if n_fail:
            print(f"note: {n_fail} unverified/failed runs excluded from ranks (counted in failure_rate)")
        base["failed"] = ~ok
    lbs = _read(results_dir / "lower_bounds.csv")
    exact = _read(results_dir / "exact.csv")
    certs = verified_certificates(cert_dir, graph_dir, index)

    if lbs is not None:
        graphs = graphs.merge(lbs[["graph", "lb", "proven_max"]].rename(columns={"proven_max": "lb_proven"}),
                              on="graph", how="left")
    else:
        graphs["lb"], graphs["lb_proven"] = pd.NA, pd.NA

    fails = None
    if base is not None and len(base):
        fails = base.groupby(["method", "budget"])["failed"].mean().rename("failure_rate").reset_index()
        base = base[~base["failed"]].copy()
        base["rank"] = base["rank"].astype(int)
    if base is not None and len(base):
        best = base.sort_values(["rank", "seconds"]).groupby("graph").first()[["rank", "method", "budget"]]
        best = best.rename(columns={"rank": "best_run_rank", "method": "best_method", "budget": "best_budget"})
        graphs = graphs.merge(best, on="graph", how="left")
        cover = base[base["method"].isin(COVER_METHODS)].groupby("graph")["rank"].min().rename("cover_rank")
        graphs = graphs.merge(cover, on="graph", how="left")
    else:
        graphs["best_run_rank"], graphs["best_method"], graphs["best_budget"], graphs["cover_rank"] = (pd.NA,) * 4

    graphs = graphs.merge(certs, on="graph", how="left")
    # Best known upper bound: a verified certificate if present, else the best sweep row (also verified at run time).
    graphs["best_ub"] = graphs["cert_rank"].where(graphs["cert_ok"].fillna(False).astype(bool), graphs["best_run_rank"])
    if exact is not None and len(exact):
        ex = exact.groupby("graph").agg(exact_rank=("best_rank", "min"), exact_proven=("proven_optimal", "max"))
        graphs = graphs.merge(ex, on="graph", how="left")
        graphs["best_ub"] = graphs[["best_ub", "exact_rank"]].min(axis=1)
    else:
        graphs["exact_proven"] = False
    graphs["gap"] = graphs["best_ub"] - graphs["lb"]
    graphs["gain_over_cover"] = graphs["cover_rank"] - graphs["best_ub"]
    graphs["proven_optimal"] = (graphs["gap"] == 0) | graphs["exact_proven"].fillna(False).astype(bool)

    if base is not None and len(base):
        m = base.merge(graphs[["graph", "best_ub", "group"]], on="graph", how="left")
        m["ratio"] = m["rank"] / m["best_ub"]
        m["matches_best"] = m["rank"] <= m["best_ub"]
        methods = (m.groupby(["method", "budget", "group"])
                   .agg(graphs=("graph", "nunique"), runs=("rank", "size"), mean_rank=("rank", "mean"),
                        mean_ratio_to_best=("ratio", "mean"), share_matches_best=("matches_best", "mean"),
                        mean_seconds=("seconds", "mean"))
                   .reset_index().sort_values(["group", "budget", "mean_ratio_to_best"]))
        methods = methods.merge(fails, on=["method", "budget"], how="left")
    else:
        methods = pd.DataFrame()
    return graphs, methods


def figures(methods: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Mean rank / best-known vs budget, one panel per family group. Skipped if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not installed, figures skipped")
        return []
    if methods.empty:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for group, d in methods[methods["budget"] > 0].groupby("group"):
        fig, ax = plt.subplots(figsize=(6, 4))
        for method, dm in d.groupby("method"):
            dm = dm.sort_values("budget")
            ax.plot(dm["budget"], dm["mean_ratio_to_best"], marker="o", label=method)
        ax.set_xscale("log")
        ax.set_xlabel("budget per graph (s)")
        ax.set_ylabel("mean rank / best known (lower is better)")
        ax.set_title(f"{group} graphs")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"ratio_vs_budget_{group}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--graphs", type=Path, default=ROOT / "bench/graphs")
    ap.add_argument("--certs", type=Path, default=ROOT / "certificates")
    a = ap.parse_args()
    graphs, methods = summarize(a.results, a.graphs, a.certs)
    a.results.mkdir(parents=True, exist_ok=True)
    graphs.to_csv(a.results / "summary_graphs.csv", index=False)
    methods.to_csv(a.results / "summary_methods.csv", index=False)
    figs = figures(methods, a.results / "figures")
    n_bad = int((graphs["cert_ok"] == False).sum())  # noqa: E712  (NaN = no certificate, not a failure)
    print(f"wrote {a.results / 'summary_graphs.csv'} ({len(graphs)} graphs, {int(graphs['proven_optimal'].sum())} "
          f"proven optimal, {n_bad} certificates FAILED verification)")
    print(f"wrote {a.results / 'summary_methods.csv'} ({len(methods)} rows); figures: {len(figs)}")
    if n_bad:
        raise SystemExit("some certificates do not verify against the current graphs; see cert_ok in summary_graphs.csv")


if __name__ == "__main__":
    main()
