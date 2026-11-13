"""After an OpenEvolve run: score-vs-programs curve, best program (re-checked by the static guard), and the theory
rule's score on the same judge for reference. Also re-scores the winner on the held-out split.

Usage: ~/venv/bin/python evolve/extract.py --task mnli
Outputs: results/evolve_curve_<task>.csv, evolve/best/<task>_best.py, results/evolve_reference_<task>.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evolve.evaluator import evaluate, static_check  # noqa: E402

PAT = re.compile(r"Evaluated program (\S+) in [\d.]+s: (.*)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="mnli")
    args = ap.parse_args()
    run = ROOT / f"evolve/runs/{args.task}"
    rows = []
    for log in sorted((run / "logs").glob("*.log")):
        for line in log.read_text().splitlines():
            m = PAT.search(line)
            if m:
                sc = re.search(r"combined_score=([\d.]+)", m.group(2))
                rows.append({"program": m.group(1), "score": float(sc.group(1)) if sc else 0.0,
                             "error": "error=" in m.group(2)})
    df = pd.DataFrame(rows).drop_duplicates("program").reset_index(drop=True)
    df["i"] = np.arange(1, len(df) + 1)
    df["best_so_far"] = df.score.cummax()
    df.to_csv(ROOT / f"results/evolve_curve_{args.task}.csv", index=False)
    print(f"{len(df)} programs, {df.error.mean():.0%} crashed/invalid, best {df.score.max():.4f}")

    best = run / "best/best_program.py"
    if not best.exists():
        cands = sorted(run.glob("checkpoints/*/best_program.py"), key=lambda p: int(p.parent.name.split("_")[-1]))
        best = cands[-1] if cands else None
    ref = {}
    if best is not None:
        code = best.read_text()
        err = static_check(code)
        if err:
            print("best program failed the static guard:", err)
        else:
            out = ROOT / f"evolve/best/{args.task}_best.py"
            out.parent.mkdir(exist_ok=True)
            out.write_text(code)
            ref["funsearch_evolve"] = evaluate(str(out), "evolve")["combined_score"]
            ref["funsearch_heldout"] = evaluate(str(out), "heldout")["combined_score"]
            print("wrote", out)
    # theory rule on the same judge: best lambda on evolve, then held-out
    from evolve.score import costs, score_decide, traces
    from simulator.policies import Calibration, theory_rule
    from simulator.simulator import Cost
    ev = traces("evolve")
    cal = Calibration(ev)
    # the judge mixes two regimes; give the rule the right cost per regime
    def theory(lam):
        rules = {c.name: theory_rule(lam, cal, ev.exit_layers, c) for c in costs().values()}
        by_ab = {(c.a, c.b): rules[c.name] for c in costs().values()}
        return lambda e, cf, n, B, left, a, b: by_ab[(a, b)](e, cf, n, B, left, a, b)
    grid = np.logspace(-3.5, 1.5, 40)
    scores = [score_decide(theory(l))["combined_score"] for l in grid]
    lam = float(grid[int(np.argmax(scores))])
    ref.update(theory=float(max(scores)), theory_lam=lam, theory_heldout=score_decide(theory(lam), "heldout")["combined_score"])
    seed = score_decide(__import__("simulator.policies", fromlist=["threshold"]).threshold(0.9))["combined_score"]
    ref["seed"] = seed
    (ROOT / f"results/evolve_reference_{args.task}.json").write_text(json.dumps(ref, indent=1))
    print(json.dumps(ref, indent=1))


if __name__ == "__main__":
    main()
