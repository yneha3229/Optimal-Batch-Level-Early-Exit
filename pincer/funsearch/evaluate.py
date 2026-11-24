"""Score a candidate MAIS priority function (the FunSearch "evaluate" step).

Pipeline per candidate: static check -> sandboxed run of the fixed skeleton on every graph of the split (one
subprocess, wall-clock and memory limits) -> parent re-checks every returned set with `is_acyclic_induced` ->
score = mean over graphs of |set| / ref(graph), where ref is the best acyclic set from a long reference run
(SA, cached in reference.json). Any crash, timeout, cyclic or malformed answer gives score None plus the error and
traceback, which the sampler feeds back to the LLM.

Splits (fixed; selection never touches val/test):
  train: random digraphs n 40-100 (seeds 0, 1, 10-13), clustered and planted digraphs  (fitness for evolution)
  train_small: the same with seeds 0-1 only (14 graphs; the first GP runs, which over-fitted)
  val:   the same families, seeds 2-3                                             (pick the final program)
  test:  seeds 5-6 plus held-out sizes n = 200, 400 and families never trained on  (report only)
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from minrank.bench import generators as gen
from minrank.core.graph import Graph
from minrank.exact.lower import is_acyclic_induced

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "sandbox_runner.py"
SA_RUNNER = HERE / "sandbox_sa.py"
SA_MOVES_PER_VERTEX = 40  # target "sa": SA move budget per graph = 40 n (deterministic, load-independent)
REF_FILE = HERE / "reference.json"

FORBIDDEN = [r"\bopen\s*\(", r"\bos\b", r"\bsys\b", r"subprocess", r"__", r"\beval\s*\(", r"\bexec\s*\(",
             r"importlib", r"pickle", r"socket", r"globals\s*\(", r"locals\s*\(", r"builtins", r"getattr",
             r"setattr", r"\bctx\.(out|inn|alive|kept|deleted|step|n)\s*=[^=]"]
ALLOWED_IMPORTS = {"math"}


def static_check(code: str, target: str = "greedy") -> str | None:
    for pat in FORBIDDEN:
        if re.search(pat, code):
            return f"forbidden pattern {pat!r}"
    for m in re.finditer(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", code, re.M):
        mod = m.group(1) or m.group(2)
        if mod not in ALLOWED_IMPORTS:
            return f"import not allowed: {mod} (only math)"
    if target == "sa":
        if not re.search(r"^def score\s*\(\s*f\s*\)\s*:", code, re.M):
            return "no top-level `def score(f):`"
    elif not re.search(r"^def priority\s*\(\s*v\s*,\s*ctx\s*\)\s*:", code, re.M):
        return "no top-level `def priority(v, ctx):`"
    if len(code) > 6000:
        return "program longer than 6000 characters"
    try:
        compile(code, "candidate.py", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    return None


def split_graphs(split: str) -> list[Graph]:
    from ..bench import clustered

    if split in ("train_sa", "val_sa"):  # harder graphs: SA saturates the n <= 100 ones in a fraction of a second
        gs = []
        for s in ((0, 10) if split == "train_sa" else (2, 3)):
            for n, p in [(150, 0.05), (150, 0.15), (250, 0.03), (250, 0.1)]:
                gs.append(gen.gnp(n, p, s, True))
            gs.append(clustered(10, 15, 0.5, 0.1, 0.001, s))
            gs.append(gen.planted(120, 14, 0.03, s, True))
        return gs
    if split in ("train", "train_small", "val"):
        seeds = {"train": (0, 1, 10, 11, 12, 13), "train_small": (0, 1), "val": (2, 3)}[split]
        gs = []
        for s in seeds:
            for n, p in [(40, 0.3), (60, 0.15), (60, 0.4), (100, 0.1), (100, 0.25)]:
                gs.append(gen.gnp(n, p, s, True))
            gs.append(clustered(6, 12, 0.6, 0.15, 0.002, s))
            gs.append(gen.planted(60, 10, 0.05, s, True))
        return gs
    if split == "test":
        gs = []
        for s in (5, 6):
            for n, p in [(60, 0.15), (100, 0.25), (200, 0.05), (200, 0.15), (400, 0.03)]:
                gs.append(gen.gnp(n, p, s, True))
            gs.append(clustered(10, 15, 0.5, 0.1, 0.001, s))
            gs.append(gen.planted(120, 14, 0.03, s, True))
            gs.append(gen.gnp(100, 0.3, s, False))  # undirected: MAIS = independence number (never trained on)
        return gs
    raise ValueError(split)


def reference(g: Graph, seconds: float = 20.0) -> int:
    """Best acyclic set size from a long SA run (cached by graph digest)."""
    from ..mais import greedy_fvs, sa_order

    cache = json.loads(REF_FILE.read_text()) if REF_FILE.exists() else {}
    key = g.digest()
    if key not in cache:
        best = 0
        for seed in range(2):
            s = sa_order(g, seconds / 2, random.Random(seed), init=greedy_fvs(g))
            assert is_acyclic_induced(g, s)
            best = max(best, len(s))
        cache = json.loads(REF_FILE.read_text()) if REF_FILE.exists() else {}
        cache[key] = {"name": g.name, "ref": best, "seconds": seconds}
        REF_FILE.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return int(cache[key]["ref"])


@dataclass
class EvalResult:
    score: float | None
    per_graph: list[float] = field(default_factory=list)
    error: str | None = None
    traceback: str | None = None
    seconds: float = 0.0


_CACHE: dict[tuple[str, str], EvalResult] = {}


def code_key(code: str) -> str:
    norm = "\n".join(line.rstrip() for line in code.strip().splitlines() if line.strip())
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def evaluate(code: str, split: str = "train", timeout: float | None = None, graphs: list[Graph] | None = None,
             refs: list[int] | None = None) -> EvalResult:
    target = "sa" if re.search(r"^def score\s*\(\s*f\s*\)\s*:", code, re.M) else "greedy"
    key = (code_key(code), split, target)
    if graphs is None and key in _CACHE:
        return _CACHE[key]
    t0 = time.perf_counter()
    err = static_check(code, target)
    if err:
        return EvalResult(None, error=err)
    gs = graphs if graphs is not None else split_graphs(split)
    rs = refs if refs is not None else [reference(g) for g in gs]
    if target == "sa":
        from ..mais import greedy_fvs

        runner = SA_RUNNER
        job = json.dumps({"graphs": [{"n": g.n, "out": list(g.out), "init": greedy_fvs(g)} for g in gs],
                          "moves_per_vertex": SA_MOVES_PER_VERTEX, "seed": 0, "mem_mb": 2048})
        timeout = timeout or 60.0 + 8.0 * len(gs)
    else:
        runner = RUNNER
        job = json.dumps({"graphs": [{"n": g.n, "out": list(g.out)} for g in gs], "mem_mb": 2048})
        timeout = timeout or 60.0
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        p = subprocess.run([sys.executable, "-I", str(runner), path], input=job, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return EvalResult(None, error=f"timeout after {timeout:.0f} s (priority too slow?)",
                          seconds=time.perf_counter() - t0)
    finally:
        Path(path).unlink(missing_ok=True)
    lines = p.stdout.strip().splitlines()
    if not lines:
        return EvalResult(None, error=f"sandbox crashed: {p.stderr.strip()[-300:]}", seconds=time.perf_counter() - t0)
    out = json.loads(lines[-1])
    if "error" in out:
        return EvalResult(None, error=out["error"], traceback=out.get("traceback"), seconds=time.perf_counter() - t0)
    per = []
    for g, s, r in zip(gs, out["sets"], rs):
        if not is_acyclic_induced(g, [int(v) for v in s]):
            return EvalResult(None, error=f"returned a cyclic set on {g.name}", seconds=time.perf_counter() - t0)
        per.append(len(s) / r)
    res = EvalResult(sum(per) / len(per), per, seconds=time.perf_counter() - t0)
    if graphs is None:
        _CACHE[key] = res
    return res


SEED_PROGRAM = '''import math


def priority(v, ctx):
    """Vertex to delete next (highest first) when breaking cycles. Classic rule: in-degree x out-degree."""
    return ctx.in_deg(v) * ctx.out_deg(v)
'''


SEED_SA = '''import math


def score(f):
    """Which of 5 sampled outside vertices SA tries to insert (highest first). f = (delta, conf_in, conf_out,
    in_kept, out_kept, in_deg, out_deg, kept, n, temp, step); delta = vertices lost - 1 at the best slot.
    Seed: fewest conflicts first (= pincer.mais.sa_order with tournament=5)."""
    return -f[0]
'''


def load_priority(code: str):
    """In-process loader for a *reviewed* program (deployment as a controller tool). Evaluation always uses the
    sandbox; this only runs code that passed static_check and was committed to pincer/funsearch/best/."""
    err = static_check(code)
    if err:
        raise ValueError(err)
    ns: dict = {}
    exec(compile(code, "evolved_priority.py", "exec"), ns)
    return ns["priority"]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("program", nargs="?")
    ap.add_argument("--split", default="train")
    a = ap.parse_args()
    code = Path(a.program).read_text() if a.program else SEED_PROGRAM
    r = evaluate(code, a.split)
    print(json.dumps({"score": r.score, "error": r.error, "seconds": round(r.seconds, 2),
                      "per_graph": [round(x, 3) for x in r.per_graph]}))
