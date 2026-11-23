"""OpenEvolve evaluator for evolved minrank heuristics (Option A).

OpenEvolve API: evaluate(program_path), evaluate_stage1(program_path) (cascade: the 4 smallest training graphs),
evaluate_stage2(program_path) (all training graphs; stage-1 results are reused, not re-run).

Contract
--------
- A candidate is a Python file defining solve(n, out, budget_s, seed) -> list[int] (see seed_program.py). A static
  check rejects file/OS/interpreter access and imports outside ALLOWED_IMPORTS before anything runs.
- Each (candidate, graph) runs in a separate `python -I sandbox_runner.py` process, killed at budget + grace.
  The answer is verified HERE (minrank.core.verify); the child is never trusted.
- Score per graph: (cover_rank - rank) / cover_rank if the answer fits, else -1.0 (status invalid/timeout/error).
  cover_rank and lb come from reference.json (minrank.evolve.reference), computed on the fly if missing.
- combined_score = mean score over the evaluated graphs. Other metrics: mean_rank_gap_to_lb (ok graphs with a
  bound), n_improved (rank < cover_rank), n_invalid (status != ok), frac_ok, mean_seconds, score_<family>.
- Training graphs: split "train", families MINRANK_EVO_FAMILIES (default planted,planted_d,gnp), n <=
  MINRANK_EVO_MAX_N (default 50), in the fixed ORDER (n ascending, then name). `order` in the trace is the position
  in that order; Option D replays traces in this order.
- One trace record per (candidate, graph) is appended to $MINRANK_TRACE_DIR/evals.jsonl under an exclusive file
  lock. Fields: run, ts, generation, candidate, graph, order, n, split, family, budget, seconds, status, rank,
  cover_rank, lb, score (plus "stage").

Environment: MINRANK_EVO_BUDGET (s per graph, default 2.0), MINRANK_EVO_WORKERS (parallel graphs, default 4),
MINRANK_EVO_SEED (default 0), MINRANK_TRACE_DIR, MINRANK_RUN, MINRANK_GENERATION, MINRANK_GRAPHS, MINRANK_REFERENCE.

Re-targeted evolution (opt-in, MINRANK_EVO_TARGET=sat; unset or "cover" = everything above, unchanged)
-------------------------------------------------------------------------------------------------------
- Graphs: hard_graphs() = the graphs above (MINRANK_EVO_FAMILIES, MINRANK_EVO_MAX_N) with n >= MINRANK_EVO_MIN_N
  (default 0), filtered by MINRANK_EVO_HARD: "auto" (default) keeps a graph iff the SAT baseline at the evaluation
  budget neither proves optimality (UNSAT) nor reaches the certified lower bound; "all" keeps every graph; anything
  else is a comma-separated list of graph names. Same fixed ORDER; stage 1 = the STAGE1_GRAPHS smallest of them.
- Score per graph: (sat_rank - rank) / sat_rank if the answer fits, else -1.0, where sat_rank is the rank SAT descent
  (minrank.exact.sat.sat_descend, seed MINRANK_EVO_SAT_SEED, default 0) reaches with the SAME budget per graph,
  cached in MINRANK_SAT_REFERENCE (default minrank/evolve/reference_sat.json; minrank.evolve.reference_sat).
  0 = as good as SAT at equal time, > 0 = better. n_improved counts rank < sat_rank; extra metrics n_tied_ref and
  n_below_cover; trace records carry two extra fields, ref_rank (= sat_rank) and target ("sat").
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:  # OpenEvolve loads this file by path, not as part of the package
    sys.path.insert(0, str(ROOT))

from minrank.bench.instances import load_index  # noqa: E402
from minrank.core.f2 import rank_f2  # noqa: E402
from minrank.core.graph import Graph  # noqa: E402
from minrank.core.verify import fit_error  # noqa: E402
from minrank.evolve.reference import ensure_reference  # noqa: E402
from minrank.evolve.reference_sat import ensure_sat_reference, sat_entry  # noqa: E402

RUNNER = ROOT / "minrank/evolve/sandbox_runner.py"
FORBIDDEN = [r"\bopen\s*\(", r"\bos\b", r"\bsys\b", r"subprocess", r"socket", r"\beval\s*\(", r"\bexec\s*\(",
             r"__import__", r"importlib", r"pickle", r"builtins", r"globals\s*\(", r"\bcompile\s*\(", r"ctypes"]
ALLOWED_IMPORTS = {"numpy", "math", "random", "itertools", "collections", "heapq", "time"}
TRACE_FIELDS = ["run", "ts", "generation", "candidate", "graph", "order", "n", "split", "family", "budget",
                "seconds", "status", "rank", "cover_rank", "lb", "score"]
STAGE1_GRAPHS = 4


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def graph_dir() -> Path:
    return Path(os.environ.get("MINRANK_GRAPHS", ROOT / "minrank/bench/graphs"))


def reference_path() -> Path:
    return Path(os.environ.get("MINRANK_REFERENCE", ROOT / "minrank/evolve/reference.json"))


def trace_path() -> Path:
    return Path(os.environ.get("MINRANK_TRACE_DIR", ROOT / "minrank/evolve/runs/traces")) / "evals.jsonl"


def target() -> str:
    """What a program is scored against: "cover" (default, the clique cover) or "sat" (SAT at the same budget)."""
    t = os.environ.get("MINRANK_EVO_TARGET", "cover") or "cover"
    if t not in ("cover", "sat"):
        raise ValueError(f"MINRANK_EVO_TARGET must be 'cover' or 'sat', got {t!r}")
    return t


def sat_reference_path() -> Path:
    return Path(os.environ.get("MINRANK_SAT_REFERENCE", ROOT / "minrank/evolve/reference_sat.json"))


def sat_seed() -> int:
    return int(os.environ.get("MINRANK_EVO_SAT_SEED", "0"))


# ---- static check ----------------------------------------------------------------------------------------------------
def static_check(code: str) -> str | None:
    """None if the program may run, else the reason it is rejected."""
    for pat in FORBIDDEN:
        if re.search(pat, code):
            return f"forbidden pattern: {pat}"
    for m in re.finditer(r"^\s*(?:from\s+([\w.]+)|import\s+([\w., ]+))", code, re.M):
        mods = [m.group(1)] if m.group(1) else [x.strip().split(" ")[0] for x in m.group(2).split(",")]
        for mod in mods:
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                return f"import not allowed: {mod}"
    if not re.search(r"^def\s+solve\s*\(", code, re.M):
        return "no top-level solve() function"
    return None


def candidate_id(code: str) -> str:
    return hashlib.sha1(code.encode()).hexdigest()[:12]


# ---- graph sets ------------------------------------------------------------------------------------------------------
def evolve_graphs(split: str = "train", families: list[str] | None = None, max_n: int | None = None) -> list[dict]:
    """Index rows of the graphs used for evolution, in the fixed evaluation ORDER (n ascending, then name)."""
    fams = families if families is not None else os.environ.get("MINRANK_EVO_FAMILIES", "planted,planted_d,gnp").split(",")
    mx = max_n if max_n is not None else int(os.environ.get("MINRANK_EVO_MAX_N", "50"))
    rows = [r for r in load_index(graph_dir()) if r["split"] == split and r["family"] in fams and int(r["n"]) <= mx]
    return sorted(rows, key=lambda r: (int(r["n"]), r["name"]))


def hard_graphs(split: str = "train", budget: float | None = None, workers: int | None = None) -> list[dict]:
    """Index rows of the re-targeted evolution graphs (MINRANK_EVO_TARGET=sat), in ORDER; see the module docstring.

    With MINRANK_EVO_HARD=auto this computes (once, cached) the clique-cover/lower-bound and SAT references of every
    candidate graph at `budget` (default MINRANK_EVO_BUDGET).
    """
    min_n = int(os.environ.get("MINRANK_EVO_MIN_N", "0"))
    rows = [r for r in evolve_graphs(split) if int(r["n"]) >= min_n]
    mode = os.environ.get("MINRANK_EVO_HARD", "auto") or "auto"
    if mode == "all":
        return rows
    if mode != "auto":
        names = {x.strip() for x in mode.split(",") if x.strip()}
        return [r for r in rows if r["name"] in names]
    budget = _env_float("MINRANK_EVO_BUDGET", 2.0) if budget is None else float(budget)
    workers = int(os.environ.get("MINRANK_EVO_WORKERS", "4")) if workers is None else int(workers)
    ref = ensure_reference(rows, graph_dir(), reference_path())
    sref = ensure_sat_reference(rows, budget, sat_seed(), graph_dir(), sat_reference_path(), workers)
    keep = []
    for r in rows:
        e = sat_entry(sref, r["name"], budget, sat_seed())
        if not e["sat_proven"] and e["sat_rank"] > int(ref[r["name"]]["lb"]):
            keep.append(r)
    return keep


def training_graphs(split: str = "train", budget: float | None = None) -> list[dict]:
    """evolve_graphs(split) by default; hard_graphs(split) when MINRANK_EVO_TARGET=sat."""
    return hard_graphs(split, budget) if target() == "sat" else evolve_graphs(split)


_GRAPHS: dict[str, Graph] = {}
_GRAPHS_LOCK = threading.Lock()


def _graph(row: dict) -> Graph:
    with _GRAPHS_LOCK:
        g = _GRAPHS.get(row["name"])
        if g is None:
            g = Graph.load(graph_dir() / row["path"])
            _GRAPHS[row["name"]] = g
        return g


# ---- running one candidate on one graph -----------------------------------------------------------------------------
def run_one(program_path: str, graph_path: str, budget: float, seed: int, grace: float | None = None) -> dict:
    """Run the sandbox; returns {"status": ok|timeout|error, "rows": list[int]|None, "seconds": float, "error": str}.

    Status "ok" here only means the child returned rows; fitting is checked by the caller.
    """
    grace = max(1.5, 0.5 * budget) if grace is None else grace
    cmd = [sys.executable, "-I", str(RUNNER), str(program_path), str(graph_path), repr(float(budget)), str(int(seed))]
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=budget + grace)  # kills the child on timeout
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "rows": None, "seconds": time.perf_counter() - t, "error": "timeout"}
    wall = time.perf_counter() - t
    lines = [ln for ln in p.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        err = (p.stderr.strip().splitlines() or ["no output"])[-1][:300]
        return {"status": "error", "rows": None, "seconds": wall, "error": err}
    try:
        d = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return {"status": "error", "rows": None, "seconds": wall, "error": f"bad JSON: {e}"}
    if "error" in d:
        return {"status": "error", "rows": None, "seconds": wall, "error": str(d["error"])}
    try:
        rows = [int(x, 16) for x in d["rows"]]
    except (KeyError, TypeError, ValueError) as e:
        return {"status": "error", "rows": None, "seconds": wall, "error": f"bad rows: {e}"}
    return {"status": "ok", "rows": rows, "seconds": float(d.get("seconds", wall)), "error": ""}


def score_record(g: Graph, ref: dict, res: dict, ref_rank: int | None = None) -> tuple[str, int | None, float, str]:
    """(status, rank, score, error) after verifying the child's rows against g.

    The score is relative to ref["cover_rank"], or to `ref_rank` when given (the SAT rank in the re-targeted mode).
    """
    if res["status"] != "ok":
        return res["status"], None, -1.0, res["error"]
    err = fit_error(g, res["rows"])
    if err:
        return "invalid", None, -1.0, err
    rank = rank_f2(res["rows"])
    cover = int(ref["cover_rank"]) if ref_rank is None else int(ref_rank)
    return "ok", rank, (cover - rank) / cover, ""


# ---- evaluation ------------------------------------------------------------------------------------------------------
@dataclass
class EvalReport:
    metrics: dict
    records: list[dict] = field(default_factory=list)
    error: str = ""
    details: list[str] = field(default_factory=list)


_CACHE: dict[tuple, dict] = {}  # (candidate, graph, budget, seed[, target]) -> trace record
_CACHE_LOCK = threading.Lock()


def _append_trace(records: list[dict]) -> None:
    if not records:
        return
    path = trace_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps({k: r[k] for k in TRACE_FIELDS} | {"stage": r.get("stage", "")}
                                 | {k: r[k] for k in ("ref_rank", "target") if k in r}) + "\n"
                      for r in records)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(payload)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def summarize(records: list[dict]) -> dict:
    """Numeric metrics over trace records (see module docstring)."""
    if not records:
        return {"combined_score": -1.0, "n_graphs": 0}
    scores = [r["score"] for r in records]
    ok = [r for r in records if r["status"] == "ok"]
    gaps = [r["rank"] - r["lb"] for r in ok if r["lb"] is not None]
    m = {"combined_score": sum(scores) / len(scores), "n_graphs": len(records),
         "n_improved": sum(1 for r in ok if r["rank"] < r.get("ref_rank", r["cover_rank"])),
         "n_invalid": len(records) - len(ok),
         "frac_ok": len(ok) / len(records), "mean_seconds": sum(r["seconds"] for r in records) / len(records),
         "mean_rank_gap_to_lb": (sum(gaps) / len(gaps)) if gaps else -1.0}
    if any("ref_rank" in r for r in records):  # re-targeted mode only
        m["n_tied_ref"] = sum(1 for r in ok if r["rank"] == r["ref_rank"])
        m["n_below_cover"] = sum(1 for r in ok if r["rank"] < r["cover_rank"])
    for fam in sorted({r["family"] for r in records}):
        s = [r["score"] for r in records if r["family"] == fam]
        m[f"score_{fam}"] = sum(s) / len(s)
    return m


def run_evaluation(program_path: str, graphs: list[dict] | None = None, budget: float | None = None,
                   workers: int | None = None, seed: int | None = None, trace: bool = True, stage: str = "full",
                   order_base: list[dict] | None = None) -> EvalReport:
    """Evaluate a candidate on `graphs` (index rows; default: training graphs in ORDER).

    `order_base` is the list whose positions define the trace field `order` (default: `graphs`).
    With MINRANK_EVO_TARGET=sat the default graphs are hard_graphs() and scores are relative to SAT at `budget`.
    """
    code = Path(program_path).read_text()
    graphs = training_graphs("train", budget) if graphs is None else graphs
    order_base = graphs if order_base is None else order_base
    pos = {r["name"]: k for k, r in enumerate(order_base)}
    budget = _env_float("MINRANK_EVO_BUDGET", 2.0) if budget is None else float(budget)
    workers = int(os.environ.get("MINRANK_EVO_WORKERS", "4")) if workers is None else int(workers)
    seed = int(os.environ.get("MINRANK_EVO_SEED", "0")) if seed is None else int(seed)
    err = static_check(code)
    if err:
        return EvalReport({"combined_score": -1.0, "n_graphs": len(graphs), "n_invalid": len(graphs),
                           "frac_ok": 0.0}, [], err)
    cid = candidate_id(code)
    ref = ensure_reference(graphs, graph_dir(), reference_path())
    sat = target() == "sat"
    sref = ensure_sat_reference(graphs, budget, sat_seed(), graph_dir(), sat_reference_path(), workers) if sat else {}
    run = os.environ.get("MINRANK_RUN", "default")
    gen = int(os.environ.get("MINRANK_GENERATION", "-1"))

    def one(row: dict) -> tuple[dict, bool]:
        key = (cid, row["name"], budget, seed) + (("sat", sat_seed()) if sat else ())
        with _CACHE_LOCK:
            hit = _CACHE.get(key)
        if hit is not None:
            return hit, False
        g = _graph(row)
        res = run_one(program_path, str(graph_dir() / row["path"]), budget, seed)
        e = ref[row["name"]]
        ref_rank = int(sat_entry(sref, row["name"], budget, sat_seed())["sat_rank"]) if sat else None
        status, rank, score, why = score_record(g, e, res, ref_rank)
        rec = {"run": run, "ts": time.time(), "generation": gen, "candidate": cid, "graph": row["name"],
               "order": pos.get(row["name"], -1), "n": g.n, "split": row["split"], "family": row["family"],
               "budget": budget, "seconds": round(res["seconds"], 4), "status": status, "rank": rank,
               "cover_rank": int(e["cover_rank"]), "lb": e.get("lb"), "score": score, "stage": stage,
               "_error": why}
        if sat:
            rec |= {"ref_rank": ref_rank, "target": "sat"}
        with _CACHE_LOCK:
            _CACHE[key] = rec
        return rec, True

    with ThreadPoolExecutor(max(1, workers)) as ex:
        results = list(ex.map(one, graphs))
    records = [r for r, _ in results]
    if trace:
        _append_trace([r for r, new in results if new])
    details = [f"{r['graph']}: {r['status']} rank={r['rank']} cover={r['cover_rank']} lb={r['lb']}"
               + (f" sat={r['ref_rank']}" if "ref_rank" in r else "")
               + (f" ({r['_error']})" if r["_error"] else "") for r in records]
    return EvalReport(summarize(records), records, "", details)


def _to_openevolve(rep: EvalReport):
    """EvaluationResult with metrics + text artifacts if OpenEvolve is installed, else the metrics dict."""
    try:
        from openevolve.evaluation_result import EvaluationResult
    except ImportError:
        return rep.metrics
    arts = {"per_graph": "\n".join(rep.details)}
    if rep.error:
        arts["error"] = rep.error
    return EvaluationResult(metrics=rep.metrics, artifacts=arts)


def evaluate_stage1(program_path: str):
    """Cascade stage 1: the STAGE1_GRAPHS smallest training graphs (cheap filter)."""
    base = training_graphs("train")
    return _to_openevolve(run_evaluation(program_path, base[:STAGE1_GRAPHS], stage="stage1", order_base=base))


def evaluate_stage2(program_path: str):
    """Cascade stage 2: every training graph (stage-1 graphs come from the cache)."""
    base = training_graphs("train")
    return _to_openevolve(run_evaluation(program_path, base, stage="stage2", order_base=base))


def evaluate(program_path: str):
    """Direct (non-cascade) evaluation on every training graph."""
    base = training_graphs("train")
    return _to_openevolve(run_evaluation(program_path, base, stage="full", order_base=base))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate one program on the training graphs and print metrics.")
    ap.add_argument("program")
    ap.add_argument("--split", default="train")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    rep = run_evaluation(a.program, training_graphs(a.split, a.budget), a.budget, trace=not a.no_trace)
    print("\n".join(rep.details))
    print(json.dumps(rep.metrics, indent=1))
    if rep.error:
        print("rejected:", rep.error)
