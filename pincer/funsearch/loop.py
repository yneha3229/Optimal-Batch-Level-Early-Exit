"""FunSearch for the MAIS priority function (lower bound side of Pincer).

    python -m pincer.funsearch.loop --proposer gp --iterations 400 --out pincer/funsearch/runs/gp1
    python -m pincer.funsearch.loop --proposer anthropic --iterations 300 --parallel 4 --out pincer/funsearch/runs/claude1
    python -m pincer.funsearch.loop --proposer local:http://127.0.0.1:8000/v1 --iterations 300 --out ...

Loop: sample an island and k=2 programs (pincer/funsearch/database.py) -> proposer writes a new `priority`
(LLM with a best-shot prompt, or the GP baseline) -> sandboxed evaluation on the train split -> register.
The proposer sees the last error of its island (crash, timeout, cyclic answer) with the traceback.
At the end: the top programs by train score are re-scored on val, the val winner on test (with the seed).
Everything is logged to OUT/log.jsonl; the winner goes to OUT/best.py and OUT/summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .database import ProgramDatabase
from .evaluate import SEED_PROGRAM, evaluate, split_graphs, reference
from .gp import GPProposer

PROBLEM = '''"""Find a large acyclic induced subgraph (MAIS) of a directed graph: a lower bound on the broadcast rate of
an index-coding problem. The skeleton repeatedly (1) moves every undecided vertex that has no undecided in-arc or
no undecided out-arc into the acyclic set, (2) otherwise DELETES the undecided vertex with the HIGHEST priority,
then re-inserts deleted vertices greedily (most recently deleted first) if they do not close a cycle.
Score = size of the acyclic set relative to a strong reference, averaged over random, clustered and planted
digraphs with 40-100 vertices. Larger is better.

ctx fields (read only; bitmask ints, bit u of ctx.out[v] means arc v -> u):
  ctx.n, ctx.out[v], ctx.inn[v], ctx.alive (undecided), ctx.kept (in the acyclic set), ctx.deleted, ctx.step
  ctx.in_deg(v), ctx.out_deg(v), ctx.mutual_deg(v)   # counts among undecided vertices
Use int.bit_count() for popcounts. Only `math` may be imported. priority is called O(n^2) times: keep it cheap
(an O(degree) loop is fine, anything quadratic per call is too slow).
"""
import math
'''

TRAIN = "train"  # fitness split; --train-split train_small reproduces the first GP runs

SYSTEM = ("You improve heuristics for combinatorial optimisation by writing Python. Reply with exactly one "
          "function definition in a ```python block, named as requested, and nothing else.")


def strip_meta(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if not line.startswith("# expr:") and
                     not line.startswith("import math")).strip()


def build_prompt(parents, last_error: str | None) -> str:
    parts = [PROBLEM]
    for k, p in enumerate(parents):
        body = strip_meta(p.code).replace("def priority(", f"def priority_v{k}(", 1)
        parts.append(f"# score {p.score:.4f}\n{body}\n")
    k = len(parents)
    parts.append(f"def priority_v{k}(v, ctx):\n    \"\"\"Improved version of `priority_v{k - 1}`.\"\"\"\n")
    if last_error:
        parts.append(f"# Note: the previous attempt on this island failed with:\n# "
                     + last_error.replace("\n", "\n# ")[:1200] + "\n")
    parts.append(f"Write `priority_v{k}`.")
    return "\n".join(parts)


def extract(text: str) -> str | None:
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    src = m.group(1) if m else text
    m = re.search(r"^def priority_v\d+\s*\(", src, re.M)
    if not m:
        return None
    src = src[m.start():]
    src = re.sub(r"^def priority_v\d+\s*\(", "def priority(", src, count=1)
    # keep the def and everything indented below it (drop trailing prose)
    out = []
    for i, line in enumerate(src.splitlines()):
        if i > 0 and line and not line[0].isspace():
            break
        out.append(line)
    return "import math\n\n\n" + "\n".join(out).rstrip() + "\n"


class LLMProposer:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.name = backend.name

    def propose(self, parents, last_error=None) -> str | None:
        text = self.backend.complete(SYSTEM, build_prompt(parents, last_error))
        return extract(text)


def run(proposer, iterations: int, out: Path, parallel: int = 1, seed: int = 0, top_val: int = 8) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")
    db = ProgramDatabase(seed=seed)
    seed_res = evaluate(SEED_PROGRAM, TRAIN)
    if seed_res.score is None:
        raise RuntimeError(f"seed program fails: {seed_res.error}")
    db.register(SEED_PROGRAM, seed_res.score, seed_res.per_graph, None, 0, meta={"seed": True})
    lock = threading.Lock()
    last_error: dict[int, str | None] = {}
    stats = {"evaluated": 0, "failed": 0, "duplicates": 0, "start": time.time()}
    seen: set[str] = set()

    def one(it: int) -> None:
        with lock:
            island, parents = db.sample()
            err = last_error.get(island)
        t0 = time.perf_counter()
        try:
            if isinstance(proposer, GPProposer):
                code = proposer.propose(parents)
            else:
                code = proposer.propose(parents, err)
        except Exception as e:  # LLM/API failure: log and move on
            code, gen_err = None, f"proposer error: {e}"
        else:
            gen_err = None if code else "no `def priority_vK` found in the reply"
        gen_s = time.perf_counter() - t0
        rec = {"it": it, "island": island, "parents": [round(p.score, 4) for p in parents], "gen_s": round(gen_s, 2)}
        if code is None:
            rec.update({"score": None, "error": gen_err})
            with lock:
                stats["failed"] += 1
                last_error[island] = gen_err
                log.write(json.dumps(rec) + "\n")
                log.flush()
            return
        from .evaluate import code_key

        key = code_key(code)
        r = evaluate(code, TRAIN)
        rec.update({"score": r.score, "error": r.error, "eval_s": round(r.seconds, 2), "key": key})
        with lock:
            stats["evaluated"] += 1
            if key in seen:
                stats["duplicates"] += 1
            seen.add(key)
            if r.score is None:
                stats["failed"] += 1
                last_error[island] = (r.error or "") + ("\n" + r.traceback if r.traceback else "")
            else:
                last_error[island] = None
                db.register(code, r.score, r.per_graph, island, it)
            rec["best"] = round(db.best().score, 4)
            rec["code"] = code
            log.write(json.dumps(rec) + "\n")
            log.flush()

    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(one, range(1, iterations + 1)))
    summary = select_and_report(db, proposer.name, iterations, stats, seed_res.score, out, top_val)
    if hasattr(proposer, "backend"):
        summary["llm_usage"] = proposer.backend.usage.as_dict()
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log.close()
    return summary


def select_and_report(db: ProgramDatabase, name: str, iterations: int, stats: dict, train_seed: float, out: Path,
                      top_val: int = 8) -> dict:
    """Pick the final program on val among the top programs by train score AND the seed (so "no improvement" can
    win), then report it and the seed on test."""
    uniq: dict[str, object] = {}
    for p in sorted(db.all, key=lambda p: -p.score):
        uniq.setdefault(p.code, p)
    cands = [p.code for p in list(uniq.values())[:top_val]]
    if SEED_PROGRAM not in cands:
        cands.append(SEED_PROGRAM)
    val = [(evaluate(c, "val").score or 0.0, c) for c in cands]
    best_val, best_code = max(val, key=lambda t: t[0])
    train_best = max(p.score for p in db.all)
    test_best = evaluate(best_code, "test")
    test_seed = evaluate(SEED_PROGRAM, "test")
    (out / "best.py").write_text(best_code)
    summary = {"proposer": name, "iterations": iterations, **stats,
               "wall_s": round(time.time() - stats.get("start", time.time())),
               "train_seed": train_seed, "train_best": train_best,
               "train_selected": evaluate(best_code, TRAIN).score, "train_split": TRAIN,
               "val_seed": evaluate(SEED_PROGRAM, "val").score, "val_selected": best_val,
               "selected_is_seed": best_code == SEED_PROGRAM,
               "val_top_by_train": [round(v, 4) for v, _ in val[:-1]],
               "test_selected": test_best.score, "test_seed": test_seed.score,
               "test_per_graph_selected": test_best.per_graph, "test_per_graph_seed": test_seed.per_graph,
               "test_graphs": [g.name for g in split_graphs("test")]}
    return summary


def reselect(run_dir: Path, top_val: int = 8) -> dict:
    """Rebuild the database from OUT/log.jsonl and redo the selection (no new proposals)."""
    db = ProgramDatabase()
    seed = evaluate(SEED_PROGRAM, TRAIN)
    db.register(SEED_PROGRAM, seed.score, seed.per_graph, None, 0)
    n = 0
    import gzip

    lp = run_dir / "log.jsonl"
    text = lp.read_text() if lp.exists() else gzip.decompress((run_dir / "log.jsonl.gz").read_bytes()).decode()
    for line in text.splitlines():
        d = json.loads(line)
        if d.get("score") is not None:
            db.register(d["code"], d["score"], [d["score"]], 0, d["it"])
            n += 1
    old = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    stats = {k: old[k] for k in ("evaluated", "failed", "duplicates") if k in old}
    s = select_and_report(db, old.get("proposer", "?"), old.get("iterations", n), stats, seed.score, run_dir, top_val)
    s["wall_s"] = old.get("wall_s")
    if "llm_usage" in old:
        s["llm_usage"] = old["llm_usage"]
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1))
    return s


def main(argv: list[str] | None = None) -> None:
    from ..llm import make_backend

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proposer", default="gp", help="gp | anthropic[:model[:effort]] | local[:url|model]")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs-only", action="store_true", help="compute the reference values and exit")
    ap.add_argument("--train-split", default="train", choices=["train", "train_small"])
    ap.add_argument("--reselect", action="store_true", help="redo val selection / test report of --out and exit")
    a = ap.parse_args(argv)
    if a.refs_only:
        for sp in ("train", "val", "test"):
            for g in split_graphs(sp):
                print(sp, g.name, reference(g), flush=True)
        return
    global TRAIN
    TRAIN = a.train_split
    if a.reselect:
        s = reselect(Path(a.out))
        print(json.dumps({k: v for k, v in s.items() if not k.startswith("test_per") and k != "test_graphs"}, indent=1))
        return
    prop = GPProposer(a.seed) if a.proposer == "gp" else LLMProposer(make_backend(a.proposer))
    s = run(prop, a.iterations, Path(a.out), a.parallel, a.seed)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("test_per")}, indent=1))


if __name__ == "__main__":
    main()
