"""Live runtime: real worker processes, any graph, any controller.

    from lambda_agent.live import Live
    res = Live(graph, RuleController(), slots=3, budget=60).run()

The runtime decomposes the graph into strongly connected components, offers the controller the same View as the
replay simulator (bounds per component, running moves with elapsed time, candidate moves), starts and kills worker
processes on its decisions, and re-checks every certificate a move reports before it can change a bound:
acyclic sets (no cycle), clique covers, fractional covers (exact coverage), SAT codes (fit + GF(2) rank of the
blown-up matrix). The Shannon LP value is computed by lower.shannon's exact weak-duality certificate in the worker.
Candidate SAT questions are generated as in the logs: block lengths 1..max_t_for(n), rates between the current lower
and upper bound. A component is pinned when its bounds meet; its moves are then killed at once.
"""
from __future__ import annotations

import math
import queue as queue_mod
import time
from fractions import Fraction

from .graph import Digraph, blowup, induced, scc
from .moves import available, context, max_t_for, run_move
from .verify import fitting_rank, frac_cover_value, is_acyclic, is_clique_cover


class LiveView:
    def __init__(self, live):
        self.live = live

    @property
    def time(self):
        return time.perf_counter() - self.live.t0

    @property
    def free(self):
        return self.live.slots - len(self.live.running)

    def open_comps(self):
        return [k for k, c in enumerate(self.live.comps) if c["lo"] < c["hi"]]

    def comp(self, k):
        return self.live.comps[k]

    def running(self):
        now = time.perf_counter()
        return [(k, name, now - p["start"]) for (k, name), p in self.live.running.items()]

    def candidates(self, k):
        c = self.live.comps[k]
        if c["n"] == 1:
            return []
        out = [m for m in ("acyclic", "cover", "lp", "frac") if available(m, c["n"])]
        acyc = c.get("acyclic") or []
        if acyc:  # SAT questions need a fixed acyclic set (symmetry breaking)
            for t in range(1, max_t_for(c["n"]) + 1):
                for r in range(max(1, math.ceil(c["lo"] * t)), math.ceil(c["hi"] * t)):
                    rate = Fraction(r, t)
                    if c["lo"] <= rate < c["hi"] and not c["asked"].get(rate):
                        out.append(f"code:{t}:{r}")
        return [m for m in out if (k, m) not in self.live.running and (k, m) not in self.live.tried]


class Live:
    def __init__(self, g: Digraph, controller, slots: int = 3, budget: float = 60.0, on_event=None):
        self.g, self.ctrl, self.slots, self.budget, self.on_event = g, controller, slots, budget, on_event
        self.comps = []
        for c in scc(g):
            h = induced(g, c)
            arcs = len(h.arcs())
            two = sum(1 for i, j in h.arcs() if i < j and h.knows(j, i))
            self.comps.append({"verts": c, "g": h, "n": h.n, "lo": Fraction(1) if h.n == 1 else Fraction(0),
                               "hi": Fraction(h.n), "asked": {}, "acyclic": [0] if h.n == 1 else None,
                               "lo_by": "", "hi_by": "", "certs": {},
                               "features": {"n": h.n, "density": round(arcs / max(1, h.n * (h.n - 1)), 3),
                                            "two_way": round(2 * two / max(1, arcs), 3)}})
        self.running: dict = {}
        self.tried: set = set()
        self.timeline: list = []
        self.t0 = 0.0

    def _start(self, k, name):
        if (k, name) in self.running or len(self.running) >= self.slots:
            return
        c = self.comps[k]
        ctx = context()
        q = ctx.Queue()
        p = ctx.Process(target=run_move, args=(name, c["g"].to_json(), c.get("acyclic") or [], self.budget, q),
                        daemon=True)
        p.start()
        self.running[(k, name)] = {"proc": p, "q": q, "start": time.perf_counter()}
        if name.startswith("code:"):
            _, t, r = name.split(":")
            c["asked"][Fraction(int(r), int(t))] = True
        self._log(k, "start", name)

    def _stop(self, k, name, why="stop"):
        p = self.running.pop((k, name), None)
        if p:
            if p["proc"].is_alive():
                p["proc"].kill()
            p["proc"].join(timeout=1)
            self.tried.add((k, name))
            self._log(k, why, name)

    def _log(self, k, what, name, detail=""):
        c = self.comps[k]
        row = (round(time.perf_counter() - self.t0, 3), k, what, name, str(c["lo"]), str(c["hi"]), detail)
        self.timeline.append(row)
        if self.on_event:
            self.on_event(row)

    def _event(self, k, name, ev):
        c = self.comps[k]
        _, kind, val, cert = ev
        if kind == "lower":
            v = Fraction(val)
            if name == "acyclic":
                if not is_acyclic(c["g"], cert) or len(cert) != v:
                    raise RuntimeError("bad acyclic certificate")
                if c["acyclic"] is None or len(cert) > len(c["acyclic"]):
                    c["acyclic"] = cert
            if v > c["lo"]:
                c["lo"], c["lo_by"] = v, name
                self._log(k, "lower", name, str(v))
        elif kind == "upper":
            v = Fraction(val)
            if name == "cover" and not (is_clique_cover(c["g"], cert) and len(cert) == v):
                raise RuntimeError("bad cover")
            if name == "frac" and frac_cover_value(c["g"], cert["cliques"], [Fraction(w) for w in cert["weights"]]) != v:
                raise RuntimeError("bad fractional cover")
            if name.startswith("code:") and Fraction(fitting_rank(blowup(c["g"], cert["t"]), cert["rows"]), cert["t"]) > v:
                raise RuntimeError("bad code")
            if v < c["hi"]:
                c["hi"], c["hi_by"] = v, name
                c["certs"]["upper"] = (name, cert)
                self._log(k, "upper", name, str(v))
        elif kind == "no":
            self._log(k, "no", name)
        elif kind in ("done", "error"):
            self._stop(k, name, "done" if kind == "done" else "error")

    def run(self) -> dict:
        self.t0 = time.perf_counter()
        view = LiveView(self)
        decision_due = True
        while True:
            if all(c["lo"] >= c["hi"] for c in self.comps):
                break
            if time.perf_counter() - self.t0 > self.budget:
                break
            if decision_due:
                acts = self.ctrl.decide(view)
                for a in acts:
                    (self._start if a[0] == "start" else self._stop)(a[1], a[2])
                decision_due = bool(acts) and view.free > 0 and any(view.candidates(k) for k in view.open_comps())
                if not acts and not self.running:
                    break
            got = False
            for (k, name), p in list(self.running.items()):
                try:
                    while True:
                        self._event(k, name, p["q"].get_nowait())
                        got = True
                except queue_mod.Empty:
                    pass
                if (k, name) in self.running and not p["proc"].is_alive() and p["q"].empty():
                    self._stop(k, name, "done")
                    got = True
            for k, c in enumerate(self.comps):
                if c["lo"] >= c["hi"]:
                    for key in [key for key in self.running if key[0] == k]:
                        self._stop(*key, why="pinned")
            if got:
                decision_due = True
            else:
                time.sleep(0.005)
        for key in list(self.running):
            self._stop(*key, why="budget")
        lo = sum((c["lo"] for c in self.comps), Fraction(0))
        hi = sum((c["hi"] for c in self.comps), Fraction(0))
        return {"graph": self.g.name, "n": self.g.n, "pinned": lo == hi, "lo": str(lo), "hi": str(hi),
                "seconds": round(time.perf_counter() - self.t0, 3), "timeline": self.timeline,
                "components": [{"n": c["n"], "lo": str(c["lo"]), "hi": str(c["hi"]), "lo_by": c["lo_by"],
                                "hi_by": c["hi_by"]} for c in self.comps]}
