"""The parallel agent (v2): three workers run at the same time and a coordinator combines what they find.

  worker 0  search     the learned agent (learned prior + online switching between the 7 strategies); when the
                       coordinator reports that parts of the graph are finished, it projects its solutions onto the
                       rest and continues there only ("focus")
  worker 1  structure  splits the graph into SCCs, shrinks each with the kernel rules, then solves pieces exactly
                       from the smallest up: CP-SAT on the acyclic-set kernel, CP-SAT on each mutual component for the
                       clique cover, the full lambda-agent (Shannon LP + vector-linear GF(2) codes) on SCCs of at most
                       10 receivers; unproven pieces are retried with doubled time (three rounds at most)
  worker 2  lambda     GF(2) block codes on top of the best clique cover (lam.py): merges of cliques found by SAT,
                       cycles of cliques, and cycles carved out of the cover with the rest re-covered exactly;
                       it picks among the three moves by their success so far

Re-planning: a worker that runs out of useful work (the structure worker once every piece is proven, too large or
given up; the lambda worker after a stretch without any saving) tells the coordinator, which hands it the open part
of the graph and the best solutions so far: it becomes another search worker with its own seed.

The coordinator keeps, for every SCC, the best certified bounds from any worker:
    lower = max(acyclic set from any search or CP-SAT, Shannon LP)               (beta >= lower)
    upper = min(clique cover from any search or CP-SAT, block codes, lambda-agent) (beta <= lambda_2 <= upper)
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
EXACT_MAX_KERNEL = 4000
EXACT_MAX_ARCS = 400_000
COVER_MAX_N = 3000


def _left(t0, T):
    return T - (time.perf_counter() - t0)


def _put(out, *msg):
    out.put((time.perf_counter(),) + msg)


# ------------------------------------------------------------------------------------------------- search loop
def _search_loop(g: Graph, model, seed, t0, T, cmd, out, wid, sub, init=None):
    """Learned agent on the receivers `sub` (sorted original ids). init = (acyclic set, colour) in local ids."""
    def build(sub, init):
        gg = g if len(sub) == g.n else Graph(g.adj[np.ix_(sub, sub)], "sub")
        s = Search(gg, seed)
        s.t0 = t0
        if init is None:
            warm_start(s)
        else:
            s.load(*init)
        return s

    s = build(sub, init)
    ctrl = LearnedBandit(model, k0=1.0)
    ctrl.reset(s.g)
    st, gap0 = Stats(), max(1, s.hi - s.lo)
    sent, last = None, 0.0

    def send(force=False):
        nonlocal sent, last
        if (s.lo, s.hi, len(sub)) != sent and (force or time.perf_counter() - last > 0.2):
            _put(out, "search", wid, sub[s.acyclic_set()], sub.copy(), s.colour.copy())
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
                acyc = sub[np.asarray(s.acyclic_set(), dtype=np.int64)]
                acyc = np.searchsorted(new, acyc[np.isin(acyc, new)])
                colour = s.colour[np.searchsorted(sub, new)]
                sub = new
                s = build(sub, (acyc, colour))
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


def _become_search(g, model, seed, t0, T, cmd, out, wid):
    """Ask the coordinator for work; then run as a search worker."""
    _put(out, "idle", wid)
    while _left(t0, T) > 0.2:
        try:
            msg = cmd.get(timeout=0.1)
        except queue_mod.Empty:
            continue
        if msg[0] == "stop":
            return
        if msg[0] == "search":
            _search_loop(g, model, seed, t0, T, cmd, out, wid, np.asarray(msg[1]), (msg[2], msg[3]))
            return


def _search_worker(g, model, seed, t0, T, cmd, out, wid):
    _search_loop(g, model, seed, t0, T, cmd, out, wid, np.arange(g.n))


# --------------------------------------------------------------------------------------------- structure worker
def _structure_worker(g: Graph, model, seed, t0, T, cmd, out, wid):
    P = Pieces(g)
    _put(out, "scc", P.label, P.count)
    jobs = [{"s": int(s), "verts": P.verts[s], "kernel": None, "lo": True, "hi": True, "limit": None, "rnd": 0}
            for s in np.argsort(P.sizes, kind="stable") if P.sizes[s] > 1]
    while jobs and _left(t0, T) > 0.1:
        nxt = []
        for j in jobs:
            if _left(t0, T) <= 0.1:
                return
            try:
                if cmd.get_nowait()[0] == "stop":
                    return
            except queue_mod.Empty:
                pass
            s, vs = j["s"], j["verts"]
            adj = g.adj[np.ix_(vs, vs)]
            n = len(vs)
            if j["rnd"] == 0 and n <= LAMBDA_MAX_N:
                lo, hi, code = lambda_exact(adj, budget=min(3.0, _left(t0, T)))
                _put(out, "lamx", s, lo, hi, code)
                if lo >= hi:
                    continue
            if j["kernel"] is None:
                k = kernel(adj)
                j["kernel"] = k
                _put(out, "kernel", s, n, len(k.verts), len(k.keep), len(k.drop), k.rules)
            k = j["kernel"]
            lim = min(j["limit"] or min(5.0, 0.1 + 0.002 * max(len(k.verts), n)), _left(t0, T))
            if j["lo"] and k.arcs is not None and len(k.verts) <= EXACT_MAX_KERNEL and len(k.arcs) <= EXACT_MAX_ARCS:
                val, kset, proven = acyclic_exact(len(k.verts), k.arcs, lim)
                if val or not k.verts:
                    _put(out, "lox", s, vs[lift_acyclic(k, kset)], proven)
                j["lo"] = not proven
            else:
                j["lo"] = False
            if j["hi"]:
                cnt, lab = mutual_labels(adj)
                if np.bincount(lab).max() <= COVER_MAX_N and _left(t0, T) > 0.1:
                    colour = np.zeros(n, dtype=np.int64)
                    base, proven_all = 0, True
                    for c in range(cnt):
                        idx = np.nonzero(lab == c)[0]
                        kk, col, proven = cover_exact(adj[np.ix_(idx, idx)],
                                                      min(lim, _left(t0, T)) / max(1, cnt) + 0.05)
                        colour[idx] = base + col
                        base += kk
                        proven_all &= proven
                    _put(out, "hix", s, vs, colour, proven_all)
                    j["hi"] = not proven_all
                else:
                    j["hi"] = False
            if j["lo"] or j["hi"]:
                j["limit"], j["rnd"] = 2 * lim, j["rnd"] + 1
                if j["rnd"] <= 3:          # give up on a piece after three rounds: search does better there
                    nxt.append(j)
        jobs = nxt
    _become_search(g, model, seed, t0, T, cmd, out, wid)


# ------------------------------------------------------------------------------------------------ lambda worker
def _lambda_worker(g: Graph, model, seed, t0, T, cmd, out, wid):
    bs, moves = None, None
    patience = max(3.0, 0.15 * T)
    last_gain = None
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
                moves = {"sat merge": [bs.step, 1, 1], "cycle of cliques": [bs.cycle_step, 1, 1],
                         "cycle + re-cover": [bs.cycle_cover_step, 1, 1]}
                last_gain = time.perf_counter()
            elif msg[0] == "cover" and bs is not None:
                bs.rebase(msg[1])
        if bs is None:
            continue
        # pick a move by its success rate so far (optimistic start: 1 success in 1 try)
        name = max(moves, key=lambda m: moves[m][1] / moves[m][2] + 0.3 * np.sqrt(1.0 / moves[m][2])
                   + 1e-3 * bs.rng.random())
        s = moves[name][0]()
        moves[name][2] += 1
        if s is not None:
            moves[name][1] += 1
            last_gain = time.perf_counter()
            _put(out, "lam", s, int(bs.scc_cost[s]), bs.certificate(s), name)
        if time.perf_counter() - last_gain > patience:
            _become_search(g, model, seed, t0, T, cmd, out, wid)
            return


# -------------------------------------------------------------------------------------------------- coordinator
def solve(g: Graph, model, T: float = 30.0, seed: int = 0, use_structure: bool = True, use_lambda: bool = True,
          log=None) -> dict:
    ctx = mp.get_context("fork")
    out = ctx.Queue()
    t0 = time.perf_counter()
    roles = [_search_worker] + ([_structure_worker] if use_structure else []) + ([_lambda_worker] if use_lambda else [])
    cmds = [ctx.Queue() for _ in roles]
    procs = [ctx.Process(target=f, args=(g, model, seed + w, t0, T, cmds[w], out, w), daemon=True)
             for w, f in enumerate(roles)]
    for p in procs:
        p.start()
    lam_w = len(roles) - 1 if use_lambda else None
    searching = {0}

    C, label = None, None
    if not use_structure:
        P = Pieces(g)
        C, label = P.count, P.label
    last_search = {}                       # worker -> (S, sub, colour)
    best_lo, best_hi = {}, {}              # scc -> (value, source, certificate)
    lp_lo, lam_hi, lam_cert, lamx_code = {}, {}, {}, {}
    stats = {"kernel": [], "focus": 0, "lambda_merges": {}, "lamx": 0, "became_search": []}
    events, totals, lo0, hi0 = [], None, None, None
    last_focus, focus_size, lam_ready, last_cover_sent, cover_dirty = 0.0, g.n, False, 0.0, False

    def lower(s):
        return max(Fraction(best_lo.get(s, (0,))[0]), lp_lo.get(s, Fraction(0)))

    def upper(s, n_s):
        c = Fraction(best_hi.get(s, (n_s,))[0])
        return min(c, lam_hi.get(s, c))

    def take_search(S, sub, col):
        nonlocal cover_dirty
        present = np.unique(label[sub])
        lo_cnt = np.bincount(label[S], minlength=C)
        pairs = np.unique(np.stack([label[sub], col], 1), axis=0)
        hi_cnt = np.bincount(pairs[:, 0], minlength=C)
        order = np.argsort(label[S], kind="stable")
        Ss = S[order]
        cut = np.searchsorted(label[S][order], np.arange(C + 1))
        lab_sub = label[sub]
        for s in present:
            if lo_cnt[s] > best_lo.get(s, (-1,))[0]:
                best_lo[s] = (int(lo_cnt[s]), "search", Ss[cut[s]:cut[s + 1]].copy())
            if hi_cnt[s] < best_hi.get(s, (1 << 60,))[0]:
                m = lab_sub == s
                best_hi[s] = (int(hi_cnt[s]), "search", (sub[m].copy(), col[m].copy()))
                cover_dirty = True

    def totals_now():
        if C is None:
            S, sub, col = last_search[0]
            return Fraction(len(S)), Fraction(len(np.unique(col)))
        sizes = np.bincount(label, minlength=C)
        return (sum((lower(s) for s in range(C)), Fraction(0)),
                sum((upper(s, int(sizes[s])) for s in range(C)), Fraction(0)))

    def global_cover():
        colour = np.full(g.n, -1, dtype=np.int64)
        base = 0
        for s, (k, _, (vs, col)) in best_hi.items():
            colour[vs] = base + np.unique(col, return_inverse=True)[1]
            base += k
        missing = colour < 0
        colour[missing] = base + np.arange(missing.sum())
        return colour

    def open_vertices():
        sizes = np.bincount(label, minlength=C)
        open_s = np.array([lower(s) < upper(s, int(sizes[s])) for s in range(C)])
        return np.nonzero(open_s[label])[0]

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
        stamp = None
        for msg in batch:
            stamp, kind, body = msg[0], msg[1], msg[2:]
            if kind == "search":
                w, S, sub, col = body
                last_search[w] = (S, sub, col)
                if lo0 is None:
                    lo0, hi0 = len(S), len(np.unique(col))
                if C is not None:
                    take_search(S, sub, col)
            elif kind == "scc":
                label, C = body
                for w, (S, sub, col) in last_search.items():
                    take_search(S, sub, col)
            elif kind == "kernel":
                stats["kernel"].append(body)
            elif kind == "lox":
                s, S, proven = body
                if len(S) > best_lo.get(s, (-1,))[0]:
                    best_lo[s] = (len(S), "cpsat" + ("*" if proven else ""), S)
            elif kind == "hix":
                s, vs, col, proven = body
                k = len(np.unique(col))
                if k < best_hi.get(s, (1 << 60,))[0]:
                    best_hi[s] = (k, "cpsat" + ("*" if proven else ""), (vs, col))
                    cover_dirty = True
            elif kind == "lamx":
                s, lo, hi, code = body
                stats["lamx"] += 1
                lp_lo[s] = max(lp_lo.get(s, Fraction(0)), lo)
                if hi < lam_hi.get(s, Fraction(1 << 60)):
                    lam_hi[s], lamx_code[s] = hi, code
            elif kind == "lam":
                s, cost, cert, move = body
                stats["lambda_merges"][move] = stats["lambda_merges"].get(move, 0) + 1
                if Fraction(cost) < lam_hi.get(s, Fraction(1 << 60)):
                    lam_hi[s], lam_cert[s] = Fraction(cost), cert
                    lamx_code.pop(s, None)
            elif kind == "idle":
                (w,) = body
                if C is not None:
                    openv = open_vertices()
                    if len(openv):
                        acyc = np.concatenate([best_lo[s][2] for s in best_lo]) if best_lo else np.zeros(0, int)
                        acyc = np.searchsorted(openv, acyc[np.isin(acyc, openv)])
                        cmds[w].put(("search", openv, acyc, global_cover()[openv]))
                        searching.add(w)
                        stats["became_search"].append((w, round(stamp - t0, 2)))
        if not last_search:
            continue
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
        if lam_w is not None and lam_w not in searching and (
                not lam_ready or (cover_dirty and time.perf_counter() - last_cover_sent > 0.5)):
            cmds[lam_w].put(("init", label, global_cover()) if not lam_ready else ("cover", global_cover()))
            lam_ready, cover_dirty, last_cover_sent = True, False, time.perf_counter()
        if time.perf_counter() - last_focus > 1.0:
            openv = open_vertices()
            if not len(openv):
                break
            if len(openv) <= 0.7 * focus_size:
                for w in searching:
                    cmds[w].put(("focus", openv))
                stats["focus"] += 1
                focus_size, last_focus = len(openv), time.perf_counter()
    for q in cmds:
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
    ok_lam = True
    for s, cert in lam_cert.items():
        if lam_hi.get(s) != Fraction(sum(c for _, c, _ in cert)):
            continue
        part = sorted(int(v) for vs, _, _ in cert for v in vs) == sorted(np.nonzero(label == s)[0].tolist())
        ok_lam &= part and all(check_block(g.adj, np.asarray(vs), rows, c) for vs, c, rows in cert)
    lo, hi = totals_now()
    closed = [s for s in range(C) if lower(s) >= upper(s, int(sizes[s]))]
    gap0 = max(1, (hi0 or 1) - (lo0 or 0))
    upper_from = {"clique cover": 0, "block codes": 0, "lambda-agent codes": 0}
    saved = 0.0
    for s in range(C):
        c = Fraction(best_hi.get(s, (int(sizes[s]),))[0])
        if lam_hi.get(s, c) < c:
            upper_from["lambda-agent codes" if s in lamx_code else "block codes"] += 1
            saved += float(c - lam_hi[s])
        else:
            upper_from["clique cover"] += 1
    return {"graph": g.name, "n": g.n, "T": T, "lo0": lo0, "hi0": hi0, "gap0": gap0,
            "lo": float(lo), "hi": float(hi), "lo_exact": str(lo), "hi_exact": str(hi), "gap": float(hi - lo),
            "integral": gap_integral(events, T, lo0, hi0, gap0), "events": events, "seconds": round(seconds, 2),
            "sccs": C, "sccs_closed": len(closed), "receivers_in_closed": int(sizes[closed].sum()) if closed else 0,
            "kernel_total": [sum(k[1] for k in stats["kernel"]), sum(k[2] for k in stats["kernel"])],
            "focus": stats["focus"], "lambda_merges": stats["lambda_merges"], "lambda_agent_pieces": stats["lamx"],
            "became_search": stats["became_search"], "upper_from": upper_from, "saved_by_codes_vs_cover": saved,
            "lower_by_lp": sum(1 for s in lp_lo if lp_lo[s] > best_lo.get(s, (0,))[0]),
            "verified": bool(ok_lo and ok_hi and ok_lam)}
