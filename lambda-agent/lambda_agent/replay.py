"""Move logs and the replay simulator.

log_graph(g): for every strongly connected component, run every candidate move ALONE (cap seconds each) and
record its events (time, kind, value). Certificates are checked here, once. Candidate SAT questions are all
(t, r) with t = 1..4 and rate r/t between the largest acyclic set and the best cover found by the other moves.

simulate(log, controller, slots): replay a controller over the logs as if the moves ran in `slots` parallel worker
processes: a move started at time s reports its logged event e at s + e.time. The controller is called at the start,
whenever an event arrives or a move ends, and when it asks to be woken; its decisions can take `latency` seconds
(the time an LLM needs to decide), during which the running moves keep going. Returns the time all components are
pinned (lo == hi), the final bounds, and the decision trace. Moves never interact except through slots, so the
replay matches a live run up to CPU noise (live.py checks this).
"""
from __future__ import annotations

import heapq
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

from .graph import Digraph, blowup, induced, scc
from .moves import available, context, max_t_for, run_move
from .verify import fitting_rank, frac_cover_value, is_acyclic, is_clique_cover

MAX_T = 4


def features(g: Digraph) -> dict:
    n = g.n
    arcs = len(g.arcs())
    two = sum(1 for i, j in g.arcs() if i < j and g.knows(j, i))
    return {"n": n, "density": round(arcs / max(1, n * (n - 1)), 3), "two_way": round(2 * two / max(1, arcs), 3)}


def _run_alone(name: str, g: Digraph, fix: list[int], cap: float) -> dict:
    ctx = context()
    q = ctx.Queue()
    p = ctx.Process(target=run_move, args=(name, g.to_json(), fix, cap, q), daemon=True)
    t0 = time.perf_counter()
    p.start()
    events, ended, timeout = [], None, False
    while True:
        left = cap + 0.5 - (time.perf_counter() - t0)
        if left <= 0:
            p.kill()
            timeout, ended = True, cap
            break
        try:
            ev = q.get(timeout=min(left, 0.5))
        except Exception:
            if not p.is_alive() and q.empty():
                ended = round(time.perf_counter() - t0, 4)
                break
            continue
        events.append(ev)
        if ev[1] in ("done", "error"):
            ended = ev[0]
            break
    p.join(timeout=1)
    if p.is_alive():
        p.kill()
    return {"events": [list(e[:3]) + [e[3]] for e in events], "ended": ended, "timeout": timeout}


def _check(g: Digraph, name: str, ev) -> None:
    kind, val, cert = ev[1], ev[2], ev[3]
    if kind == "lower" and name == "acyclic":
        assert is_acyclic(g, cert) and len(cert) == val
    if kind == "upper" and name == "cover":
        assert is_clique_cover(g, cert) and len(cert) == val
    if kind == "upper" and name == "frac":
        assert frac_cover_value(g, cert["cliques"], [Fraction(w) for w in cert["weights"]]) == Fraction(val)
    if kind == "upper" and name.startswith("code:"):
        t = cert["t"]
        assert Fraction(fitting_rank(blowup(g, t), cert["rows"]), t) <= Fraction(val)


def log_component(g: Digraph, cap: float) -> dict:
    """Every move alone on component g. SAT questions: rates from the largest acyclic set up to (not including) the
    best clique cover, block lengths 1..max_t_for(n)."""
    moves = {}
    for name in ("acyclic", "cover", "lp", "frac"):
        if available(name, g.n):
            moves[name] = _run_alone(name, g, [], cap)
    acyc = [e for e in moves.get("acyclic", {"events": []})["events"] if e[1] == "lower"]
    fix = acyc[-1][3] if acyc else [0]
    lo = Fraction(acyc[-1][2]) if acyc else Fraction(1)
    his = [Fraction(e[2]) for e in moves.get("cover", {"events": []})["events"] if e[1] == "upper"]
    hi = min(his) if his else Fraction(g.n)
    if g.n > 1:
        for t in range(1, max_t_for(g.n) + 1):
            for r in range(max(1, math.ceil(lo * t)), math.ceil(hi * t)):
                moves[f"code:{t}:{r}"] = _run_alone(f"code:{t}:{r}", g, fix, cap)
    for name, m in moves.items():
        for ev in m["events"]:
            _check(g, name, ev)
            if name == "acyclic" and ev[1] == "lower":
                pass
        for ev in m["events"]:  # certificates are bulky: keep only what the replay and the features need
            if not (name == "acyclic" and ev is m["events"][-1]):
                ev[3] = None
    return {"features": features(g), "fix": fix, "moves": moves}


def log_graph(d: dict, cap: float = 10.0) -> dict:
    g = Digraph.from_json(d)
    comps = []
    for c in scc(g):
        h = induced(g, c)
        comp = {"verts": c} if h.n > 1 else {"verts": c, "single": True}
        if h.n > 1:
            comp.update(log_component(h, cap))
        comps.append(comp)
    return {"graph": g.name, "n": g.n, "cap": cap, "comps": comps}


def log_suite(suite_path: Path, out_dir: Path, cap: float = 10.0, workers: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suite = json.loads(Path(suite_path).read_text())
    todo = [d for d in suite if not (out_dir / f"{d['name']}.json").exists()]
    with ProcessPoolExecutor(workers) as ex:
        for d, res in zip(todo, ex.map(log_graph, todo, [cap] * len(todo))):
            (out_dir / f"{d['name']}.json").write_text(json.dumps(res))
            print(d["name"], sum(len(c.get("moves", {})) for c in res["comps"]), "moves logged", flush=True)


# ------------------------------------------------------------------------------------------- simulator
class View:
    """What a controller sees: per component bounds and features, running moves, free slots, the clock."""

    def __init__(self, sim):
        self.sim = sim

    @property
    def time(self):
        return self.sim.now

    @property
    def free(self):
        return self.sim.slots - len(self.sim.running)

    def open_comps(self):
        return [k for k, c in enumerate(self.sim.comps) if c["lo"] < c["hi"]]

    def comp(self, k):
        return self.sim.comps[k]

    def running(self):
        return [(k, name, self.sim.now - s) for (k, name), s in self.sim.running.items()]

    def tried(self, k, name):
        return (k, name) in self.sim.tried

    def stalled(self, k, name):
        """Seconds since this running move last improved a bound (or since it started)."""
        return self.sim.now - self.sim.progress.get((k, name), self.sim.running.get((k, name), self.sim.now))

    def candidates(self, k):
        """Moves worth starting now on component k (not running, not finished, and able to move a bound)."""
        c = self.sim.comps[k]
        out = []
        for name in c["log"]["moves"]:
            if (k, name) in self.sim.running or (k, name) in self.sim.tried:
                continue
            if name.startswith("code:"):
                _, t, r = name.split(":")
                rate = Fraction(int(r), int(t))
                if not (c["lo"] <= rate < c["hi"]):
                    continue
                # a smaller block length already asked for the same rate makes this one redundant
                if c["asked"].get(rate):
                    continue
            out.append(name)
        return out


class Sim:
    def __init__(self, log: dict, controller, slots: int = 3, budget: float = 60.0, latency: float = 0.0):
        self.log, self.ctrl, self.slots, self.budget, self.latency = log, controller, slots, budget, latency
        self.now = 0.0
        self.comps = []
        for c in log["comps"]:
            if c.get("single"):
                self.comps.append({"lo": Fraction(1), "hi": Fraction(1), "log": {"moves": {}}, "n": 1, "asked": {}})
            else:
                self.comps.append({"lo": Fraction(0), "hi": Fraction(len(c["verts"])), "log": c, "n": len(c["verts"]),
                                   "asked": {}, "features": c["features"]})
        self.running: dict[tuple[int, str], float] = {}
        self.progress: dict[tuple[int, str], float] = {}
        self.tried: set[tuple[int, str]] = set()
        self.queue: list = []
        self.trace: list = []
        self.pinned_at = None

    def start(self, k, name):
        if (k, name) in self.running or len(self.running) >= self.slots:
            return False
        self.running[(k, name)] = self.now
        mv = self.comps[k]["log"]["moves"][name]
        for ev in mv["events"]:
            heapq.heappush(self.queue, (self.now + ev[0], "ev", k, name, ev))
        heapq.heappush(self.queue, (self.now + (mv["ended"] or self.log["cap"]), "end", k, name, None))
        if name.startswith("code:"):
            _, t, r = name.split(":")
            self.comps[k]["asked"][Fraction(int(r), int(t))] = True
        self.trace.append((round(self.now, 3), "start", k, name))
        return True

    def stop(self, k, name):
        if (k, name) in self.running:
            del self.running[(k, name)]
            self.tried.add((k, name))
            self.trace.append((round(self.now, 3), "stop", k, name))

    def _apply(self, actions):
        for a in actions:
            if a[0] == "start":
                self.start(a[1], a[2])
            elif a[0] == "stop":
                self.stop(a[1], a[2])

    def run(self) -> dict:
        view = View(self)
        self.decisions = 0

        def decide():
            acts = self.ctrl.decide(view)
            self.decisions += 1
            self._apply(acts)
            # one-action-at-a-time controllers are asked again (after their latency) while slots are free
            if acts and view.free > 0 and any(view.candidates(k) for k in view.open_comps()):
                return self.now + self.latency
            return None

        pending_decision = decide()
        while True:
            if all(c["lo"] >= c["hi"] for c in self.comps):
                self.pinned_at = self.now
                break
            nxt = self.queue[0][0] if self.queue else math.inf
            if getattr(self.ctrl, "tick", None) and self.running:
                nxt_tick = self.now + self.ctrl.tick
                if nxt_tick < nxt and (pending_decision is None or nxt_tick < pending_decision):
                    pending_decision = nxt_tick
            if pending_decision is not None and pending_decision <= nxt and pending_decision <= self.budget:
                self.now = pending_decision
                pending_decision = decide()
                continue
            if nxt == math.inf or nxt > self.budget:
                self.now = min(self.budget, nxt)
                if not self.running and pending_decision is None and nxt == math.inf:
                    before = len(self.trace)
                    pending_decision = decide()
                    if len(self.trace) == before:
                        break
                    continue
                break
            t, kind, k, name, ev = heapq.heappop(self.queue)
            if (k, name) not in self.running:
                continue  # stopped earlier: ignore its future events
            self.now = t
            c = self.comps[k]
            if kind == "ev":
                if ev[1] == "lower" and Fraction(ev[2]) > c["lo"]:
                    c["lo"] = Fraction(ev[2])
                    self.progress[(k, name)] = self.now
                elif ev[1] == "upper" and Fraction(ev[2]) < c["hi"]:
                    c["hi"] = Fraction(ev[2])
                    self.progress[(k, name)] = self.now
            else:
                del self.running[(k, name)]
                self.tried.add((k, name))
            # a pinned component: free its slots at once
            if c["lo"] >= c["hi"]:
                for key in [key for key in self.running if key[0] == k]:
                    self.stop(*key)
            if pending_decision is None:
                pending_decision = self.now + self.latency
        lo = sum((c["lo"] for c in self.comps), Fraction(0))
        hi = sum((c["hi"] for c in self.comps), Fraction(0))
        return {"pinned": self.pinned_at is not None, "time": self.pinned_at if self.pinned_at is not None else self.budget,
                "lo": str(lo), "hi": str(hi), "gap": float(hi - lo), "trace": self.trace, "decisions": self.decisions}
