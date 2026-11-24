"""Compare methods on the suite: accuracy against the true optima, and time.
    python beta-agent/bench/analyze.py      -> bench/RESULTS.md, bench/summary.json (used by the demo page)
True optimum of each bound = the value any method PROVED (all proofs must agree, which is checked)."""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {"greedy": "Greedy heuristics", "plain_exact": "Plain exact search", "cpsat": "CP-SAT (OR-Tools)",
         "pincer": "Pincer agent"}
ORDER = ["greedy", "plain_exact", "cpsat", "pincer"]


def load():
    rows = []
    for f in ("results_js.jsonl", "results_cpsat.jsonl"):
        for line in (HERE / f).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    rows = [r for r in load() if "error" not in r]
    suite = json.loads((HERE / "suite.json").read_text())
    opt = {}
    for g in suite:
        rs = [r for r in rows if r["graph"] == g["name"]]
        lo = {r["L"] for r in rs if r["lower_optimal_proven"]}
        hi = {r["U"] for r in rs if r["upper_optimal_proven"]}
        assert len(lo) <= 1 and len(hi) <= 1, f"proofs disagree on {g['name']}: {lo} {hi}"
        opt[g["name"]] = (lo.pop() if lo else None, hi.pop() if hi else None)
    by = {(r["graph"], r["method"]): r for r in rows}
    groups = {"all graphs": [g["name"] for g in suite],
              "small (n <= 20)": [g["name"] for g in suite if g["n"] <= 20],
              "medium (21-40)": [g["name"] for g in suite if 20 < g["n"] <= 40],
              "large (41-60)": [g["name"] for g in suite if g["n"] > 40]}
    summary = {"graphs": len(suite), "limit_s": 30, "groups": {}}
    lines = ["# Benchmark: Pincer vs standard methods", "",
             f"{len(suite)} graphs (`suite.json`: random networks n = 10-60, caching groups, textbook cases), 30 s limit per "
             "graph for every method, one CPU core each. Accuracy is measured against the true optimum of each bound "
             "(proved by at least one method; all proofs agree). β exact = the two bounds met, so β is known.", ""]
    for gname, names in groups.items():
        lines += [f"## {gname} ({len(names)} graphs)", "",
                  "| method | lower bound optimal | upper bound optimal | β pinned exactly | missing transmissions (sum of gaps to optimum) | median time to final answer | max time to final answer | median time incl. proofs | max time incl. proofs |",
                  "|---|---|---|---|---|---|---|---|---|"]
        summary["groups"][gname] = {}
        for m in ORDER:
            rs = [by[(n, m)] for n in names if (n, m) in by]
            lo_ok = sum(1 for r in rs if opt[r["graph"]][0] is not None and r["L"] == opt[r["graph"]][0])
            hi_ok = sum(1 for r in rs if opt[r["graph"]][1] is not None and r["U"] == opt[r["graph"]][1])
            miss = sum((opt[r["graph"]][0] - r["L"]) + (r["U"] - opt[r["graph"]][1]) for r in rs
                       if None not in opt[r["graph"]])
            exact = sum(1 for r in rs if r["beta_exact"])
            ts = sorted(r["seconds"] for r in rs)
            # time to the final answer: Pincer logs when its bounds last changed; the other methods only report
            # their total time (their answer arrives when they stop)
            ta = sorted(r.get("seconds_to_final_values", r["seconds"]) for r in rs)
            p90 = ts[int(0.9 * (len(ts) - 1))]
            lines.append(f"| {NAMES[m]} | {lo_ok}/{len(rs)} | {hi_ok}/{len(rs)} | {exact}/{len(rs)} | {miss} | "
                         f"{st.median(ta):.3f} s | {max(ta):.2f} s | {st.median(ts):.3f} s | {max(ts):.2f} s |")
            summary["groups"][gname][m] = {"name": NAMES[m], "n": len(rs), "lower_opt": lo_ok, "upper_opt": hi_ok,
                                           "beta_exact": exact, "missing": miss, "median_s": round(st.median(ts), 4),
                                           "median_answer_s": round(st.median(ta), 4), "max_answer_s": round(max(ta), 3),
                                           "p90_s": round(p90, 3), "max_s": round(max(ts), 3)}
        lines.append("")
    # per-graph table
    lines += ["## Per graph", "", "| graph | n | true MAIS | true cover | " + " | ".join(NAMES[m] for m in ORDER) + " |",
              "|---|---|---|---|" + "---|" * len(ORDER)]
    per = []
    for g in suite:
        o = opt[g["name"]]
        cells = []
        for m in ORDER:
            r = by.get((g["name"], m))
            cells.append("" if r is None else f"[{r['L']}, {r['U']}] {r['seconds']:.2f}s")
        lines.append(f"| {g['name']} | {g['n']} | {o[0]} | {o[1]} | " + " | ".join(cells) + " |")
        per.append({"graph": g["name"], "n": g["n"], "opt": o,
                    "methods": {m: {"L": by[(g['name'], m)]["L"], "U": by[(g['name'], m)]["U"],
                                    "s": by[(g['name'], m)]["seconds"]} for m in ORDER if (g["name"], m) in by}})
    summary["per_graph"] = per
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines[: 6 + 4 * 7]))


if __name__ == "__main__":
    main()
