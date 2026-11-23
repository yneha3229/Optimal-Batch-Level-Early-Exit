"""Run one candidate `priority(v, ctx)` inside the fixed greedy-FVS skeleton, in an isolated interpreter.

Usage (called by pincer/funsearch/evaluate.py, never by hand):  python -I sandbox_runner.py PROGRAM.py < graphs.json
stdin: {"graphs": [{"n": n, "out": [mask, ...]}, ...], "mem_mb": 2048}
stdout (one JSON line): {"sets": [[v, ...], ...], "seconds": [...]} or {"error": "...", "traceback": "..."}

This file imports nothing from the repository (python -I keeps the repository off sys.path), so a candidate can
only see the graphs it is given. The skeleton is a copy of pincer.mais.greedy_fvs; tests check they agree. The
candidate's exceptions are NOT caught inside the skeleton: a crash is reported with its traceback and scores as a
failure (the first OpenEvolve run lost two thirds of its budget to crashes hidden by a fallback).
"""
import json
import resource
import sys
import time
import traceback


class Ctx:
    """What priority(v, ctx) may read. Bitmask ints: bit u of out[v] = arc v -> u (v knows message u).
    alive: undecided vertices, kept: vertices already in the acyclic set, deleted: removed vertices."""

    __slots__ = ("n", "out", "inn", "alive", "kept", "deleted", "step")

    def __init__(self, n, out, inn):
        self.n, self.out, self.inn = n, out, inn
        self.alive = (1 << n) - 1
        self.kept = 0
        self.deleted = 0
        self.step = 0

    def out_deg(self, v):
        return (self.out[v] & self.alive).bit_count()

    def in_deg(self, v):
        return (self.inn[v] & self.alive).bit_count()

    def mutual_deg(self, v):
        return (self.out[v] & self.inn[v] & self.alive).bit_count()


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _reach(start, allowed, out):
    seen = start & allowed
    fr = seen
    while fr:
        nxt = 0
        for u in _bits(fr):
            nxt |= out[u]
        nxt &= allowed & ~seen
        seen |= nxt
        fr = nxt
    return seen


def _can_add(v, kept, out, inn):
    if out[v] & inn[v] & kept:
        return False
    return not (_reach(out[v] & kept, kept, out) & inn[v])


def greedy(n, out, priority):
    inn = [0] * n
    for i in range(n):
        for j in _bits(out[i]):
            inn[j] |= 1 << i
    ctx = Ctx(n, out, inn)
    deleted = []
    while ctx.alive:
        changed = True
        while changed and ctx.alive:
            changed = False
            for v in list(_bits(ctx.alive)):
                if not (out[v] & ctx.alive & ~(1 << v)) or not (inn[v] & ctx.alive & ~(1 << v)):
                    ctx.alive &= ~(1 << v)
                    ctx.kept |= 1 << v
                    changed = True
        if not ctx.alive:
            break
        best, best_p = -1, None
        for v in _bits(ctx.alive):
            p = float(priority(v, ctx))
            if p != p:
                raise ValueError("priority returned NaN")
            if best_p is None or p > best_p:
                best, best_p = v, p
        ctx.alive &= ~(1 << best)
        ctx.deleted |= 1 << best
        deleted.append(best)
        ctx.step += 1
    kept = ctx.kept
    for v in reversed(deleted):
        if _can_add(v, kept, out, inn):
            kept |= 1 << v
    return list(_bits(kept))


def main():
    try:
        job = json.loads(sys.stdin.read())
        lim = int(job.get("mem_mb", 2048)) * 1024 * 1024
        try:
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (ValueError, OSError):
            pass
        ns = {"__name__": "candidate"}
        with open(sys.argv[1]) as f:
            code = f.read()
        exec(compile(code, "candidate.py", "exec"), ns)
        pr = ns["priority"]
        sets, secs = [], []
        for gd in job["graphs"]:
            t = time.perf_counter()
            sets.append(greedy(int(gd["n"]), [int(x) for x in gd["out"]], pr))
            secs.append(time.perf_counter() - t)
        print(json.dumps({"sets": sets, "seconds": secs}))
    except BaseException as e:
        tb = traceback.format_exc().splitlines()
        tb = [line for line in tb if "candidate.py" in line or not line.startswith("  File")][-8:]
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:400], "traceback": "\n".join(tb)[:1500]}))


if __name__ == "__main__":
    main()
