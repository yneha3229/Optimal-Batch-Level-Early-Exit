"""Does an evolved priority still help once SA runs after it (the way the controller uses it)?

    python -m pincer.funsearch.deploy_check pincer/funsearch/runs/gp42_s3/best.py --budgets 0.5,2

For every graph of the FunSearch test split: greedy(seed priority) + SA(b) vs greedy(evolved) + SA(b), same SA seed,
3 SA seeds each. Reports mean |acyclic set| / reference and wins/ties/losses per budget."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..mais import default_priority, greedy_fvs, sa_order
from .evaluate import load_priority, reference, split_graphs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("program")
    ap.add_argument("--budgets", default="0.5,2")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    evo = load_priority(Path(a.program).read_text())
    gs = split_graphs("test")
    out = {}
    for b in [float(x) for x in a.budgets.split(",")]:
        w = t = l = 0
        sd = se = 0.0
        for g in gs:
            ref = reference(g)
            s0, s1 = greedy_fvs(g, default_priority), greedy_fvs(g, evo)
            for k in range(a.seeds):
                x = len(sa_order(g, b, random.Random(k), init=s0)) / ref
                y = len(sa_order(g, b, random.Random(k), init=s1)) / ref
                sd += x
                se += y
                w, t, l = w + (y > x), t + (y == x), l + (y < x)
        n = len(gs) * a.seeds
        out[b] = {"seed+SA": round(sd / n, 4), "evolved+SA": round(se / n, 4), "wins/ties/losses": [w, t, l]}
        print(b, out[b], flush=True)
    Path(a.program).with_name("deploy_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
