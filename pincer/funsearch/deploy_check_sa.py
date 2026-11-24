"""Deployment check for target "sa": library SA (pincer.mais.sa_order, one random candidate per move) vs the
sandboxed tournament SA with a given score program, same greedy start, same wall time per graph.

    python -m pincer.funsearch.deploy_check_sa [PROGRAM.py] --budgets 0.5,2 --seeds 2
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from ..mais import greedy_fvs, sa_order
from .evaluate import SA_RUNNER, SEED_SA, reference, split_graphs


def run_program(code: str, gs, seconds: float, seed: int) -> list[int]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    job = json.dumps({"graphs": [{"n": g.n, "out": list(g.out), "init": greedy_fvs(g)} for g in gs],
                      "seconds": seconds, "seed": seed})
    p = subprocess.run([sys.executable, "-I", str(SA_RUNNER), path], input=job, capture_output=True, text=True,
                       timeout=60 + 3 * seconds * len(gs))
    Path(path).unlink(missing_ok=True)
    out = json.loads(p.stdout.strip().splitlines()[-1])
    if "error" in out:
        raise RuntimeError(out)
    return [len(s) for s in out["sets"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?")
    ap.add_argument("--budgets", default="0.5,2")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    code = Path(a.program).read_text() if a.program else SEED_SA
    gs = split_graphs(a.split)
    refs = [reference(g) for g in gs]
    res = {}
    for b in [float(x) for x in a.budgets.split(",")]:
        lib, evo = [], []
        for k in range(a.seeds):
            lib += [len(sa_order(g, b, random.Random(1000 * k + i), init=greedy_fvs(g))) / r
                    for i, (g, r) in enumerate(zip(gs, refs))]
            evo += [x / r for x, r in zip(run_program(code, gs, b, k), refs)]
        w = sum(e > l for e, l in zip(evo, lib))
        t = sum(e == l for e, l in zip(evo, lib))
        res[b] = {"library_sa": round(sum(lib) / len(lib), 4), "program_sa": round(sum(evo) / len(evo), 4),
                  "wins/ties/losses": [w, t, len(lib) - w - t]}
        print(b, res[b], flush=True)
    if a.program:
        Path(a.program).with_name("deploy_check_sa.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
