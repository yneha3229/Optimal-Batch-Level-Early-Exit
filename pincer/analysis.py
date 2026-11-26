"""Summarise a benchmark file:  python -m pincer.analysis pincer/results/bench_test_b30.jsonl [--md OUT.md]"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ORDER = ["classic", "satdescend", "roundrobin", "index", "llm"]
FAMILY_GROUP = {"dnp": "random digraph", "gnp": "random undirected", "planted_d": "planted digraph",
                "planted": "planted undirected", "clustered": "clustered digraph (many SCCs)"}


def group(r: dict) -> str:
    return FAMILY_GROUP.get(r["family"], "structured")


def load(path: str) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    seen = {}
    for r in rows:
        seen[(r["graph"], r["method"])] = r
    return list(seen.values())


def table(rows: list[dict]) -> str:
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    graphs = sorted({r["graph"] for r in rows})
    by = {(r["graph"], r["method"]): r for r in rows}
    common = [g for g in graphs if all((g, m) in by for m in methods)]
    out = [f"{len(common)} graphs with every method; budget {rows[0]['budget']:.0f} s per graph\n",
           "| method | mean minrank gap | mean beta gap | minrank closed | beta closed | both closed | mean seconds used |",
           "|---|---|---|---|---|---|---|"]
    for m in methods:
        rs = [by[(g, m)] for g in common]
        both = sum(r["closed_mr_all"] and r["closed_beta_all"] for r in rs)
        out.append(f"| {m} | {sum(r['gap_mr'] for r in rs) / len(rs):.2f} | "
                   f"{sum(r['gap_beta'] for r in rs) / len(rs):.3f} | {sum(r['closed_mr_all'] for r in rs)} | "
                   f"{sum(r['closed_beta_all'] for r in rs)} | {both} | {sum(r['seconds'] for r in rs) / len(rs):.1f} |")
    out.append("\nPer family, mean minrank gap / mean beta gap:\n")
    fams = sorted({group(by[(g, methods[0])]) for g in common})
    out.append("| family | graphs | " + " | ".join(methods) + " |")
    out.append("|---|---|" + "---|" * len(methods))
    for f in fams:
        gs = [g for g in common if group(by[(g, methods[0])]) == f]
        cells = []
        for m in methods:
            rs = [by[(g, m)] for g in gs]
            cells.append(f"{sum(r['gap_mr'] for r in rs) / len(rs):.2f} / {sum(r['gap_beta'] for r in rs) / len(rs):.2f}")
        out.append(f"| {f} | {len(gs)} | " + " | ".join(cells) + " |")
    out.append("\nPaired comparison on total gap (minrank gap + beta gap), wins / ties / losses of the row method:\n")
    out.append("| | " + " | ".join(methods) + " |")
    out.append("|---|" + "---|" * len(methods))
    for a in methods:
        cells = []
        for b in methods:
            if a == b:
                cells.append("-")
                continue
            w = t = l = 0
            for g in common:
                ga = by[(g, a)]["gap_mr"] + by[(g, a)]["gap_beta"]
                gb = by[(g, b)]["gap_mr"] + by[(g, b)]["gap_beta"]
                if ga < gb - 1e-6:
                    w += 1
                elif ga > gb + 1e-6:
                    l += 1
                else:
                    t += 1
            cells.append(f"{w}/{t}/{l}")
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    base = [m for m in ("classic", "satdescend") if m in methods]
    pin = [m for m in ("index", "roundrobin") if m in methods]
    if base and pin:
        out.append("\nTime to match: seconds until the Pincer run's total gap first drops to the baseline's FINAL total gap "
                   "(graphs where it does), against the seconds the baseline used:\n")
        out.append("| Pincer policy | baseline | graphs matched | median Pincer s | median baseline s | median speed-up |")
        out.append("|---|---|---|---|---|---|")
        for m in pin:
            for b in base:
                ratios, tp, tb = [], [], []
                for g in common:
                    t = time_to_gap(by[(g, m)], by[(g, b)]["gap_mr"] + by[(g, b)]["gap_beta"])
                    if t is not None:
                        tp.append(t)
                        tb.append(by[(g, b)]["seconds"])
                        ratios.append(by[(g, b)]["seconds"] / max(t, 0.05))
                if ratios:
                    out.append(f"| {m} | {b} | {len(ratios)}/{len(common)} | {_median(tp):.2f} | {_median(tb):.1f} | "
                               f"{_median(ratios):.0f}x |")
    if pin:
        out.append("\nWhich tools moved the bounds (sum of gap reduction over all graphs, excluding the quick pass):\n")
        for m in pin:
            acc: dict[str, float] = defaultdict(float)
            for g in common:
                for k, v in by[(g, m)].get("tool_gain", {}).items():
                    acc[k] += v
            out.append(f"- {m}: " + ", ".join(f"{k} {v:.1f}" for k, v in sorted(acc.items(), key=lambda t: -t[1])))
    return "\n".join(out)


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    k = len(xs)
    return (xs[(k - 1) // 2] + xs[k // 2]) / 2


def time_to_gap(r: dict, target: float) -> float | None:
    """First time the run's total gap was <= target, reconstructed from the step log (gap before step k = final
    gap + gains of steps k.. end; time = cumulative tool seconds)."""
    final = r["gap_mr"] + r["gap_beta"]
    if final > target + 1e-9:
        return None
    steps = r.get("step_log", [])
    gains = [s[3] for s in steps]
    t = 0.0
    remaining = sum(gains)
    for k, s in enumerate(steps):
        t += s[2]
        remaining -= gains[k]
        if k == 0 and s[0] == "quick":
            continue  # the quick pass starts from the identity matrix; judge from its end on
        if final + remaining <= target + 1e-9:
            return t
    return t


def per_graph(rows: list[dict]) -> str:
    methods = [m for m in ORDER if any(r["method"] == m for r in rows)]
    by = {(r["graph"], r["method"]): r for r in rows}
    graphs = sorted({r["graph"] for r in rows})
    out = ["| graph | n | " + " | ".join(methods) + " |", "|---|---|" + "---|" * len(methods)]
    for g in graphs:
        cells = []
        n = next(by[(g, m)]["n"] for m in methods if (g, m) in by)
        for m in methods:
            r = by.get((g, m))
            if r is None:
                cells.append("")
                continue
            from fractions import Fraction

            bl, bh = Fraction(r["b_lo"]), Fraction(r["b_hi"])
            beta = f"{bl}" if bl == bh else f"[{float(bl):.2f},{float(bh):.2f}]"
            mr = f"{r['mr_lo']}" if r["mr_lo"] == r["mr_hi"] else f"[{r['mr_lo']},{r['mr_hi']}]"
            cells.append(f"{mr} ; {beta} ; {r['seconds']:.0f}s")
        out.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--md", default=None)
    ap.add_argument("--per-graph", action="store_true")
    a = ap.parse_args()
    rows = [r for f in a.files for r in load(f)]
    text = table(rows)
    if a.per_graph:
        text += "\n\nPer graph (minrank ; beta ; seconds used):\n\n" + per_graph(rows)
    print(text)
    if a.md:
        Path(a.md).write_text(text + "\n")


if __name__ == "__main__":
    main()
