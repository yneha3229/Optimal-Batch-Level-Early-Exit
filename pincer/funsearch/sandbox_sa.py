"""Run one candidate `score(f)` inside a fixed simulated-annealing skeleton for MAIS, in an isolated interpreter.

Usage (called by pincer/funsearch/evaluate.py):  python -I sandbox_sa.py PROGRAM.py < job.json
stdin: {"graphs": [{"n", "out", "init"}, ...], "seconds": s | "moves_per_vertex": m, "seed": k, "mem_mb": m}
(a move budget makes the fitness independent of machine load; the deployment check uses wall time)
stdout: {"sets": [[v, ...], ...]} or {"error": ..., "traceback": ...}

The skeleton is pincer.mais.sa_order (Galinier-style SA on a topological order of the kept set) with ONE change:
each move samples K = 5 outside vertices, computes for each the best insertion slot and its conflicts, and moves the
candidate with the highest score(f). f is a tuple of plain numbers (see FEATURES); the candidate cannot see or
change the search state. The default SA is the special case "sample one vertex".
"""
import json
import math
import random
import resource
import sys
import time
import traceback

K = 5
FEATURES = ("delta", "conf_in", "conf_out", "in_kept", "out_kept", "in_deg", "out_deg", "kept", "n", "temp", "step")


def _bits(m):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


def _topo(verts, out):
    vs = set(verts)
    mask = 0
    for v in vs:
        mask |= 1 << v
    indeg = {v: 0 for v in vs}
    for u in vs:
        for w in _bits(out[u] & mask):
            indeg[w] += 1
    q = sorted(v for v in vs if indeg[v] == 0)
    order = []
    while q:
        u = q.pop(0)
        order.append(u)
        for w in _bits(out[u] & mask):
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    return order


def sa(n, out, init, seconds, rng, score, t0=0.6, alpha=0.99, max_moves=None):
    inn = [0] * n
    for i in range(n):
        for j in _bits(out[i]):
            inn[j] |= 1 << i
    seq = _topo(init, out)
    best = list(seq)
    T = t0
    mpt = max(50, 2 * n)
    fails = 0
    step = 0
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline and (max_moves is None or step < max_moves):
        improved = False
        for _ in range(mpt):
            pos = {u: k for k, u in enumerate(seq)}
            outside = [v for v in range(n) if v not in pos]
            if not outside:
                return list(range(n))
            kept_mask = 0
            for u in seq:
                kept_mask |= 1 << u
            cands = []
            for v in (rng.sample(outside, K) if len(outside) > K else outside):
                ins = [pos[u] for u in _bits(inn[v] & kept_mask)]
                outs = [pos[u] for u in _bits(out[v] & kept_mask)]
                p_minus = (max(ins) + 1) if ins else 0
                p_plus = min(outs) if outs else len(seq)
                bm = None
                for p in {p_minus, p_plus}:
                    ci = [u for u in _bits(inn[v] & kept_mask) if pos[u] >= p]
                    co = [u for u in _bits(out[v] & kept_mask) if pos[u] < p]
                    conf = set(ci) | set(co)
                    d = len(conf) - 1
                    if bm is None or d < bm[0]:
                        bm = (d, p, conf, len(ci), len(co))
                f = (bm[0], bm[3], bm[4], len(ins), len(outs), inn[v].bit_count(), out[v].bit_count(), len(seq), n,
                     T, step)
                s = float(score(f))
                if s != s:
                    raise ValueError("score returned NaN")
                cands.append((s, rng.random(), v, bm))
            _, _, v, (delta, p, conf, _, _) = max(cands)
            step += 1
            if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-9)):
                seq = [u for u in seq[:p] if u not in conf] + [v] + [u for u in seq[p:] if u not in conf]
                if len(seq) > len(best):
                    best = list(seq)
                    improved = True
            if time.perf_counter() >= deadline or (max_moves is not None and step >= max_moves):
                break
        T *= alpha
        fails = 0 if improved else fails + 1
        if fails > 50:
            T, fails = t0, 0
    return sorted(best)


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
        score = ns["score"]
        sets = []
        for k, gd in enumerate(job["graphs"]):
            rng = random.Random(int(job.get("seed", 0)) * 1000 + k)
            mm = int(job["moves_per_vertex"]) * int(gd["n"]) if job.get("moves_per_vertex") else None
            sets.append(sa(int(gd["n"]), [int(x) for x in gd["out"]], gd["init"], float(job.get("seconds", 3600)), rng,
                           score, max_moves=mm))
        print(json.dumps({"sets": sets}))
    except BaseException as e:
        tb = traceback.format_exc().splitlines()
        tb = [line for line in tb if "candidate.py" in line or not line.startswith("  File")][-8:]
        print(json.dumps({"error": f"{type(e).__name__}: {e}"[:400], "traceback": "\n".join(tb)[:1500]}))


if __name__ == "__main__":
    main()
