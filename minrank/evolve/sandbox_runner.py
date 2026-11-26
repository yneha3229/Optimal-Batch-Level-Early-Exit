"""Run one candidate program on one graph in an isolated interpreter.

Usage (the evaluator calls this; the parent enforces the wall-clock timeout and verifies the answer):
  python -I minrank/evolve/sandbox_runner.py PROGRAM_PATH GRAPH_JSON BUDGET_S SEED

Prints one JSON line: {"rows": [hex, ...], "seconds": float}. On any exception prints {"error": "..."} and exits 2.

Isolation: `-I` ignores PYTHON* environment variables and the user site directory and does not put the script's
directory on sys.path, so the candidate cannot import repository code (such as the trace or label files). An
address-space limit (env MINRANK_EVO_MEM_MB, default 4096) is set before the candidate is loaded. The graph is read
here, with plain json, and handed to solve() as bitmasks; the candidate never sees a file path.
"""
import importlib.util
import json
import resource
import sys
import time


def load_graph(path: str) -> tuple[int, list[int]]:
    """(n, out masks) from a minrank Graph JSON file (edges of an undirected graph are listed once)."""
    with open(path) as f:
        d = json.load(f)
    n = int(d["n"])
    out = [0] * n
    for i, j in d["edges"]:
        out[i] |= 1 << j
        if not d["directed"]:
            out[j] |= 1 << i
    return n, out


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(json.dumps({"error": "usage: sandbox_runner.py PROGRAM GRAPH BUDGET SEED"}))
        return 2
    prog, graph, budget, seed = argv[1], argv[2], float(argv[3]), int(argv[4])
    try:
        n, out = load_graph(graph)
        mem_mb = 4096
        try:
            import os
            mem_mb = int(os.environ.get("MINRANK_EVO_MEM_MB", "4096"))
        except ValueError:
            pass
        lim = mem_mb * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ValueError, OSError):
            pass  # some platforms do not allow lowering RLIMIT_AS; the parent timeout still applies
        spec = importlib.util.spec_from_file_location("candidate", prog)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {prog}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        t = time.perf_counter()
        rows = mod.solve(n, list(out), budget, seed)
        sec = time.perf_counter() - t
        rows = [int(x) for x in rows]
        if any(x < 0 for x in rows):
            raise ValueError("negative row")
        print(json.dumps({"rows": [f"{x:x}" for x in rows], "seconds": sec}))
        return 0
    except BaseException as e:  # report everything (including MemoryError) as a structured error
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:500]}))
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
