"""The parallel agent (v2): three workers run at the same time and a coordinator combines what they find.

  search worker     the learned agent (learned prior + online switching between the 7 strategies) on the graph;
                    when the coordinator says that parts of the graph are finished, it projects its current
                    solutions onto the rest and continues there only ("focus")
  structure worker  splits the graph into SCCs, shrinks each SCC with the kernel rules, then solves pieces exactly
                    from the smallest up: CP-SAT on the acyclic-set kernel, CP-SAT on each mutual component for the
                    clique cover, and the full lambda-agent (Shannon LP + vector-linear GF(2) codes) on SCCs of
                    at most 10 receivers; unproven pieces are retried with doubled time
  lambda worker     block codes over GF(2) on top of the best clique cover (lam.py): lambda_2 <= sum of block ranks

The coordinator keeps, for every SCC, the best certified lower and upper bound from any worker:
    lower  = max(acyclic set found by search or by CP-SAT, Shannon LP)            (beta >= lower)
    upper  = min(clique cover from search or CP-SAT, block codes, lambda-agent)   (lambda_2 >= beta, <= upper)
and the totals are the sums over SCCs. Every certificate is re-checked on the original graph at the end.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import time
from fractions import Fraction

import numpy as np

from .agent import LearnedBandit, Stats, gap_integral, observe
from .exact import LAMBDA_MAX_N, acyclic_exact, cover_exact, lambda_exact
from .graph import Graph, is_acyclic_set, is_clique_cover
from .lam import BlockSearch, check_block
from .strategies import STRATEGIES, Search, warm_start
from .structure import Pieces, kernel, lift_acyclic, mutual_labels

SLICE = 0.25
EXACT_MAX_KERNEL = 4000      # CP-SAT on acyclic-set kernels up to this many receivers (and sparse enough)
EXACT_MAX_ARCS = 400_000
COVER_MAX_N = 3000           # CP-SAT on mutual components up to this many receivers


def _left(t0, T):
    return T - (time.perf_counter() - t0)


# ------------------------------------------------------------------------------------------------ search worker
def _search_worker(g: Graph, model, seed, t0, T, cmd, out):
    sub = np.arange(g.n)
    s = Search(g, seed)
    s.t0 = t0
    warm_start(s)
    ctrl = LearnedBandit(model, k0=1.0)
    ctrl.reset(s.g)
    st, gap0 = Stats(), max(1, s.hi - s.lo)
    sent, last = None, 0.0

    def send(force=False):
        nonlocal sent, last
        if (s.lo, s.hi, len(sub)) != sent and (force or time.perf_counter() - last > 0.2):
            out.put((time.perf_counter(), "search", sub[s.acyclic_set()], sub.copy(), s.colour.copy()))
            sent, last = (s.lo, s.hi, len(sub)), time.perf_counter()

    send(True)
    while s.clock() < T:
        try:
            msg = cmd.get_nowait()
        except queue_mod.Empty:
            msg = None
        if msg is not None:
            if msg[0] == "stop":
                break
            if msg[0] == "focus":
                new = np.asarray(msg[1])
                pos = np.searchsorted(sub, new)
                keep_lo = np.isin(sub[s.acyclic_set()], new)
                acyc_new = np.searchsorted(new, sub[np.asarray(s.acyclic_set(), dtype=np.int64)[keep_lo]])
                colour_new = s.colour[pos]
                gg = Graph(g.adj[np.ix_(new, new)], "focus")
                s2 = Search(gg, seed)
                s2.t0 = t0
                s2.load(acyc_new, colour_new)
                s, sub = s2, new
                ctrl.reset(s.g)
                st, gap0 = Stats(), max(1, s.hi - s.lo)
                send(True)
                continue
        if s.hi <= s.lo or len(sub) == 0:
            time.sleep(0.02)
            continue
        obs = observe(s, st, T, gap0)
        a = ctrl.choose(obs, st, s)
        lo, hi, t = s.lo, s.hi, s.clock()
        s.run(STRATEGIES[a], min(SLICE, max(0.0, T - t)))
        t_end = s.clock()
        gained = (s.lo - lo) + (hi - s.hi)
        st.used[a] += SLICE
        st.recent[a] = gained / max(1e-3, t_end - t)
        st.streak = st.streak + 1 if a == st.last_action else 1
        st.last_action = a
        if gained:
            st.last_gain[a], st.fails[a] = t_end, 0
            st.gain[a] += gained
            if s.lo > lo:
                st.last_lo_gain = t_end
            if s.hi < hi:
                st.last_hi_gain = t_end
        else:
            st.fails[a] += 1
        ctrl.feedback(a, gained / gap0, t_end - t)
        send()
    send(True)


# --------------------------------------------------------------------------------------------- structure worker
def _structure_worker(g: Graph, t0, T, out):
    P = Pieces(g)
    out.put((time.perf_counter(), "scc", P.label, P.count))
    jobs = []
    for s in np.argsort(P.sizes, kind="stable"):
        vs = P.verts[s]
        if len(vs) > 1:
            jobs.append({"s": int(s), "verts": vs, "kernel": None, "lo": True, "hi": True, "limit": None})
    rnd = 0
    while jobs and _left(t0, T) > 0.1:
        nxt = []
        for j in jobs:
            if _left(t0, T) <= 0.1:
                return
            s, vs = j["s"], j["verts"]
            adj = g.adj[np.ix_(vs, vs)]
            n = len(vs)
            if rnd == 0 and n <= LAMBDA_MAX_N:
                lo, hi, code = lambda_exact(adj, budget=min(3.0, _left(t0, T)))
                out.put((time.perf_counter(), "lamx", s, lo, hi, code))
                if lo >= hi:
                    continue
            if j["kernel"] is None:
                k = kernel(adj)
                j["kernel"] = k
                out.put((time.perf_counter(), "kernel", s, n, len(k.verts), len(k.keep), len(k.drop), k.rules))
            k = j["kernel"]
            lim = j["limit"] or min(5.0, 0.1 + 0.002 * max(len(k.verts), n))
            lim = min(lim, _left(t0, T))
            if j["lo"] and len(k.verts) <= EXACT_MAX_KERNEL and len(k.arcs) <= EXACT_MAX_ARCS:
                val, kset, proven = acyclic_exact(len(k.verts), k.arcs, lim)
                if val or not k.verts:
                    S = vs[lift_acyclic(k, kset)]
                    out.put((time.perf_counter(), "lox", s, S, proven))
                j["lo"] = not proven
            else:
                j["lo"] = False
            cnt, lab = mutual_labels(adj) if j["hi"] else (0, None)
            if j["hi"] and np.bincount(lab).max() <= COVER_MAX_N and _left(t0, T) > 0.1:
                colour = np.zeros(n, dtype=np.int64)
                base, proven_all = 0, True
                for c in range(cnt):
                    idx = np.nonzero(lab == c)[0]
                    kk, col, proven = cover_exact(adj[np.ix_(idx, idx)], min(lim, _left(t0, T)) / max(1, cnt) + 0.05)
                    colour[idx] = base + col
                    base += kk
                    proven_all &= proven
                out.put((time.perf_counter(), "hix", s, vs, colour, proven_all))
                j["hi"] = not proven_all
            else:
                j["hi"] = False
            if j["lo"] or j["hi"]:
                j["limit"] = 2 * lim
                nxt.append(j)
        jobs, rnd = nxt, rnd + 1


# ------------------------------------------------------------------------------------------------ lambda worker
def _lambda_worker(g: Graph, t0, T, cmd, out, seed):
    bs = None
    while _left(t0, T) > 0:
        try:
            msg = cmd.get(timeout=0.05) if bs is None else cmd.get_nowait()
        except queue_mod.Empty:
            msg = None
        if msg is not None:
            if msg[0] == "stop":
                return
            if msg[0] == "init":
                bs = BlockSearch(g.adj, msg[1], msg[2], seed)
            elif msg[0] == "cover" and bs is not None:
                bs.rebase(msg[1])
        if bs is None:
            continue
        s = bs.cycle_step() if bs.rng.random() < 0.5 else bs.step()
        if s is not None:
            out.put((time.perf_counter(), "lam", s, int(bs.scc_cost[s]), bs.certificate(s)))


# -------------------------------------------------------------------------------------------------- coordinator
def solve(g: Graph, model, T: float = 30.0, seed: int = 0, use_structure: bool = True, use_lambda: bool = True,
          log=None) -> dict:
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    cmd_search, cmd_lam = ctx.Queue(), ctx.Queue()
    t0 = time.perf_counter()
    procs = [ctx.Process(target=_search_worker, args=(g, model, seed, t0, T, cmd_search, out), daemon=True)]
    if use_structure:
        procs.append(ctx.Process(target=_structure_worker, args=(g, t0, T, out), daemon=True))
    if use_lambda:
        procs.append(ctx.Process(target=_lambda_worker, args=(g, t0, T, cmd_lam, out, seed), daemon=True))
    for p in procs:
        p.start()

    C, label = None, None
    if not use_structure:  # the coordinator still needs the SCCs to combine bounds
        P = Pieces(g)
        C, label = P.count, P.label
    S_search, col_search, sub = None, None, None
    best_lo, best_hi = {}, {}      # scc -> (value, certificate kind, certificate)
    lp_lo, lam_hi = {}, {}         # scc -> Fraction (lambda-agent lower / upper), block-code cost
    lam_cert, lamx_code = {}, {}
    stats = {"kernel": [], "closed_by": {}, "focus": 0, "lambda_merges": 0, "lamx": 0}
    events, totals = [], None
    lo0 = hi0 = None
    last_focus, focus_size, lam_ready, last_cover_sent = 0.0, g.n, False, 0.0
    cover_dirty = False

    def per_scc_lower(s):
        a = best_lo.get(s, (0,))[0]
        return max(Fraction(a), lp_lo.get(s, Fraction(0)))

    def per_scc_upper(s, n_s):
        c = best_hi.get(s, (n_s,))[0]
        return min(Fraction(c), lam_hi.get(s, Fraction(c)))

    def take_search():
        """Per-SCC bounds from the latest search solution."""
        nonlocal cover_dirty
        present = np.unique(label[sub])
        lo_cnt = np.bincount(label[S_search], minlength=C)
        pairs = np.unique(np.stack([label[sub], col_search], 1), axis=0)
        hi_cnt = np.bincount(pairs[:, 0], minlength=C)
        order = np.argsort(label[S_search], kind="stable")
        Ssorted = S_search[order]
        cut = np.searchsorted(label[S_search][order], np.arange(C + 1))
        for s in present:
            if lo_cnt[s] > best_lo.get(s, (-1,))[0]:
                best_lo[s] = (int(lo_cnt[s]), "search", Ssorted[cut[s]:cut[s + 1]].copy())
            if hi_cnt[s] < best_hi.get(s, (1 << 60,))[0]:
                m = label[sub] == s
                best_hi[s] = (int(hi_cnt[s]), "search", (sub[m].copy(), col_search[m].copy()))
                cover_dirty = True

    def totals_now():
        if C is None:
            return Fraction(int(len(S_search))), Fraction(int(len(np.unique(col_search))))
        sizes = np.bincount(label, minlength=C)
        lo = sum((per_scc_lower(s) for s in range(C)), Fraction(0))
        hi = sum((per_scc_upper(s, int(sizes[s])) for s in range(C)), Fraction(0))
        return lo, hi

    def global_cover():
        colour = np.full(g.n, -1, dtype=np.int64)
        base = 0
        for s, (k, kind, (vs, col)) in best_hi.items():
            colour[vs] = base + np.unique(col, return_inverse=True)[1]
            base += k
        missing = colour < 0
        colour[missing] = base + np.arange(missing.sum())
        return colour

    while True:
        left = _left(t0, T)
        if left <= 0:
            break
        batch = []
        try:
            batch.append(out.get(timeout=min(0.05, left)))
            while True:
                batch.append(out.get_nowait())
        except queue_mod.Empty:
            pass
        new_search = False
        stamp = None
        for msg in batch:
            stamp, kind = msg[0], msg[1]
            msg = msg[1:]
            if kind == "search":
                S_search, sub, col_search = msg[1], msg[2], msg[3]
                new_search = True
                if lo0 is None:
                    lo0, hi0 = len(S_search), len(np.unique(col_search))
            elif kind == "scc":
                label, C = msg[1], msg[2]
                new_search = True
            elif kind == "kernel":
                stats["kernel"].append(msg[1:])
            elif kind == "lox":
                s, S, proven = msg[1], msg[2], msg[3]
                if len(S) > best_lo.get(s, (-1,))[0]:
                    best_lo[s] = (len(S), "cpsat" + ("*" if proven else ""), S)
            elif kind == "hix":
                s, vs, col, proven = msg[1], msg[2], msg[3], msg[4]
                k = len(np.unique(col))
                if k < best_hi.get(s, (1 << 60,))[0]:
                    best_hi[s] = (k, "cpsat" + ("*" if proven else ""), (vs, col))
                    cover_dirty = True
            elif kind == "lamx":
                s, lo, hi, code = msg[1:]
                stats["lamx"] += 1
                lp_lo[s] = max(lp_lo.get(s, Fraction(0)), lo)
                if hi < lam_hi.get(s, Fraction(1 << 60)):
                    lam_hi[s], lamx_code[s] = hi, code
            elif kind == "lam":
                s, cost, cert = msg[1:]
                stats["lambda_merges"] += 1
                if Fraction(cost) < lam_hi.get(s, Fraction(1 << 60)):
                    lam_hi[s], lam_cert[s] = Fraction(cost), cert
                    lamx_code.pop(s, None)
            if kind == "search" and C is not None:   # per-SCC bounds of every search report, at its own time
                take_search()
                new_search = False
        if S_search is None or not batch:
            continue
        if C is not None and new_search:
            take_search()
        lo, hi = totals_now()
        if totals != (lo, hi):
            t = (stamp if stamp is not None else time.perf_counter()) - t0
            if totals is None or lo != totals[0]:
                events.append((t, "lo", float(lo), "total"))
            if totals is None or hi != totals[1]:
                events.append((t, "hi", float(hi), "total"))
            totals = (lo, hi)
            if log:
                log(f"{t:6.2f}s  {float(lo):.1f} <= beta <= lambda_2 <= {float(hi):.1f}")
        if C is None:
            continue
        # tell the lambda worker about better clique covers
        if use_lambda and (not lam_ready or (cover_dirty and time.perf_counter() - last_cover_sent > 0.5)):
            cmd_lam.put(("init", label, global_cover()) if not lam_ready else ("cover", global_cover()))
            lam_ready, cover_dirty, last_cover_sent = True, False, time.perf_counter()
        # focus the search on the SCCs that are still open
        if use_structure and time.perf_counter() - last_focus > 1.0:
            sizes = np.bincount(label, minlength=C)
            open_s = np.array([per_scc_lower(s) < per_scc_upper(s, int(sizes[s])) for s in range(C)])
            if not open_s.any():
                break
            openv = np.nonzero(open_s[label])[0]
            if len(openv) <= 0.7 * focus_size:
                cmd_search.put(("focus", openv))
                stats["focus"] += 1
                focus_size, last_focus = len(openv), time.perf_counter()
    for q in (cmd_search, cmd_lam):
        q.put(("stop",))
    for p in procs:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
    seconds = time.perf_counter() - t0

    # ------------------------------------------------------------------ re-check every certificate on the original graph
    sizes = np.bincount(label, minlength=C)
    S = np.concatenate([v[2] for v in best_lo.values()]) if best_lo else np.zeros(0, dtype=np.int64)
    ok_lo = is_acyclic_set(g, S) and len(S) == sum(v[0] for v in best_lo.values())
    ok_hi = is_clique_cover(g, global_cover())
    ok_lam = all(check_block(g.adj, np.asarray(vs), rows, c) for s, cert in lam_cert.items()
                 if lam_hi.get(s) == Fraction(sum(c for _, c, _ in cert)) for vs, c, rows in cert)
    lo, hi = totals_now()
    closed = [s for s in range(C) if per_scc_lower(s) >= per_scc_upper(s, int(sizes[s]))]
    gap0 = max(1, (hi0 or 1) - (lo0 or 0))
    upper_from = {"clique cover": 0, "block codes": 0, "lambda-agent codes": 0}
    for s in range(C):
        c = Fraction(best_hi.get(s, (int(sizes[s]),))[0])
        if lam_hi.get(s, c) < c:
            upper_from["lambda-agent codes" if s in lamx_code else "block codes"] += 1
        else:
            upper_from["clique cover"] += 1
    return {"graph": g.name, "n": g.n, "T": T, "lo0": lo0, "hi0": hi0, "gap0": gap0,
            "lo": float(lo), "hi": float(hi), "lo_exact": str(lo), "hi_exact": str(hi), "gap": float(hi - lo),
            "integral": gap_integral(events, T, lo0, hi0, gap0), "events": events, "seconds": round(seconds, 2),
            "sccs": C, "sccs_closed": len(closed), "receivers_in_closed": int(sizes[closed].sum()) if closed else 0,
            "kernel": stats["kernel"][-5:], "kernel_total": [sum(k[1] for k in stats["kernel"]),
                                                             sum(k[2] for k in stats["kernel"])],
            "focus": stats["focus"], "lambda_merges": stats["lambda_merges"], "lambda_agent_pieces": stats["lamx"],
            "upper_from": upper_from, "lower_by_lp": sum(1 for s in lp_lo if lp_lo[s] > best_lo.get(s, (0,))[0]),
            "verified": bool(ok_lo and ok_hi and ok_lam)}
