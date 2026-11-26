"""The orchestrator: one graph, one run. Deterministic Python; owns the board, the worker pool, the time budget and the
trace. LLM agents are consulted at the start (Solver <-> Critic, at most 3 rounds), at checkpoints (stall,
contradiction, new phase) and at the end (Verifier, in verify/stage.py). LLM calls run in a background thread, so the
tools never wait for them.

modes: full | no_critic | no_llm | small_model (= full with another model; the caller passes the LLM)
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import agents as AG
from .board import Board, Trace
from .graphs import load_graph
from .scheduler import Pool
from .tools.codes import integer_spectrum, primes_upto, spectral_candidates
from .tools.describe import summary_text
from .workers import Task

ROOT = Path(__file__).resolve().parents[1]
SPEC_PRIORITY = {"critic_check": 300, "near_miss": 250, "plan": 100, "checkpoint": 120, "default": 10}


class Run:
    def __init__(self, graph_path: str | Path, mode: str = "no_llm", llm=None, budget: float = 300.0,
                 cores: list[int] | None = None, trace_path: Path | None = None, seed: int = 0,
                 cert_dir: Path | None = None, max_spec: int | None = None) -> None:
        from .warmup import warmup

        t_w = time.perf_counter()
        warmup()                                   # one-time JIT load, before the run clock starts
        self.warmup_seconds = round(time.perf_counter() - t_w, 2)
        self.graph_path = Path(graph_path)
        self.gid = self.graph_path.stem
        self.mode, self.llm, self.budget, self.seed = mode, llm, budget, seed
        self.cores = cores or list(range(8))
        self.t_start = time.perf_counter()
        self.trace = Trace(trace_path, self.t_start)
        self.cert_dir = cert_dir
        self.A = load_graph(self.graph_path)
        self.n = self.A.shape[0]
        self.board = Board(self.A, self.trace)
        self.pool = Pool(self.A, self.trace, self.cores)
        big = self.n > 3000
        self.spec_cores = 2 if big else 1
        self.max_spec = max_spec or max(1, (len(self.cores) - 3) // self.spec_cores)
        self.describe: dict | None = None
        self.summary = ""
        self.cands: list[dict] = []
        self.spec_queue: list[dict] = []
        self.spec_deferred: list[dict] = []
        self.spec_tried: dict[tuple[int, int], dict] = {}
        self.spec_running: dict[str, dict] = {}
        self.spec_results: list[dict] = []
        self.plan = None
        self.plan_source = None
        self.hypothesis = None
        self.predictions: dict = {}
        self.contradicted: set[str] = set()
        self.history: list[str] = []
        self.critic_rounds = 0
        self.unresolved: list[str] = []
        self.dialogue = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.settings = {"indep_set": {"restarts": 4}, "clique_cover": {"tabu_iters": 20000}}
        self.cover_floor = None
        self.t_plan = None
        self.t_describe = None
        self.last_progress = 0.0
        self.checkpoints = 0
        self.check_tasks: dict[str, dict] = {}
        self.latest_cover = None
        self.task_reports: dict[str, list] = {}
        self.stopped_reason = None
        self.events: list[dict] = []
        self.blowup_done = set()
        self.sat_asked: set[int] = set()

    # ------------------------------------------------------------------------------------------------ helpers
    def now(self) -> float:
        return time.perf_counter() - self.t_start

    def left(self) -> float:
        return self.budget - self.now()

    def uses_llm(self) -> bool:
        return self.mode in ("full", "no_critic", "small_model") and self.llm is not None

    def submit(self, tool: str, args: dict | None = None, cores: int = 1, priority: float = 50.0,
               source: str = "scheduler", time_limit: float | None = None) -> Task:
        t = Task(self.pool.new_id(tool), tool, dict(args or {}), cores, priority, source, time_limit)
        return self.pool.submit(t)

    def ensure(self, tool: str, args: dict | None = None, priority: float = 50.0, source: str = "scheduler"):
        if not self.pool.running_of(tool) and not any(t.tool == tool for t in self.pool.pending):
            self.submit(tool, args, priority=priority, source=source)

    # ------------------------------------------------------------------------------------------------ main loop
    def run(self) -> dict:
        tr = self.trace
        tr.log("run_start", graph=self.gid, n=self.n, mode=self.mode, budget=self.budget, cores=self.cores,
               model=getattr(self.llm, "model", None))
        self.submit("describe", cores=min(4, len(self.cores)), priority=1000)
        self.submit("indep_set", self.settings["indep_set"] | {"seed": self.seed}, priority=900)
        self.submit("clique_cover", self.settings["clique_cover"] | {"seed": self.seed}, priority=800)
        if self.n <= 300:
            self.submit("indep_cpsat", cores=1, priority=700, time_limit=60)
            self.submit("clique_cover_cpsat", cores=1, priority=600, time_limit=60)
        try:
            while True:
                self.pool.pump()
                self._pump_spectral()
                for msg in self.pool.poll(0.05):
                    self._on_message(msg)
                self._step_dialogue()
                if self.board.closed():
                    self.stopped_reason = "L == U"
                    break
                if self.left() <= 0:
                    self.stopped_reason = "budget"
                    tr.log("event", name="BUDGET", detail={"budget": self.budget})
                    break
                self._check_stall_and_phase()
        finally:
            self.pool.shutdown()
            self.executor.shutdown(wait=False, cancel_futures=True)
        res = self.result()
        tr.log("run_end", **{k: res[k] for k in ("L", "U", "proven", "wall", "stopped")})
        return res

    # ------------------------------------------------------------------------------------------------ messages
    def _on_message(self, msg) -> None:
        typ, tid = msg[0], msg[1]
        tool = tid.split("#")[0]
        if typ == "done":
            self.trace.log("tool_done", task=tid, tool=tool, summary=msg[2])
            self._task_finished(tid, msg[2])
            return
        if typ == "error":
            self.trace.log("tool_error", task=tid, tool=tool, error=msg[2])
            self._task_finished(tid, {"error": msg[2][-300:]})
            return
        kind, payload = msg[2], msg[3]
        self.task_reports.setdefault(tid, []).append((kind, payload))
        if kind == "describe":
            self._on_describe(payload)
        elif kind in ("indep", "cover", "frac", "spectral", "sat_rows", "sat_unsat", "blowup"):
            if kind == "frac":
                payload = dict(payload)
                payload["weights"] = [Fraction(w) for w in payload["weights"]]
            if kind == "cover":
                self.latest_cover = payload["cliques"]
            ev = self.board.offer(kind, tool, payload)
            for name, det in ev:
                self._on_event(name, det)
        elif kind == "spectral_result":
            self._on_spectral_result(tid, payload)
        elif kind == "alpha_info":
            self.board.info.update({k: v for k, v in payload.items() if v is not None})
            self.trace.log("info", tool=tool, **payload)
            self._check_contradiction()
        elif kind in ("sat_result", "blowup_result"):
            self.trace.log("tool_report", task=tid, tool=tool, **payload)

    def _task_finished(self, tid: str, summary) -> None:
        if tid in self.spec_running:
            self.spec_running.pop(tid)
        if tid in self.check_tasks:
            self.check_tasks[tid]["done"] = True
            self.check_tasks[tid]["summary"] = summary

    # ------------------------------------------------------------------------------------------------ describe
    def _on_describe(self, d: dict) -> None:
        self.describe = d
        self.summary = summary_text(d)
        self.t_describe = self.now()
        self.trace.log("describe", summary=self.summary, seconds=d["seconds"])
        sp = d.get("spectrum")
        if sp is not None:
            ints = integer_spectrum(sp)
            self.int_spec = ints
            self.cands = spectral_candidates(self.n, ints, primes_upto(max(2, self.n)), lower=self._lower_hint())
            # Hoffman ratio bounds (numerical, information only)
            if d["regular"] and len(sp["eigenvalues"]) >= 2:
                lmin, lmax = sp["lambda_min"], sp["lambda_max"]
                ah = self.n * (-lmin) / (lmax - lmin) if lmax > lmin else None
                key = "mult" if sp["multiplicities_exact"] else "mult_est"
                second = max(float(e["value"]) for e in sp["eigenvalues"] if abs(float(e["value"]) - lmax) > 1e-6) \
                    if any(abs(float(e["value"]) - lmax) > 1e-6 for e in sp["eigenvalues"]) else None
                if ah is not None:
                    self.board.info["alpha_hoffman_ratio_bound"] = round(ah, 4)
                if second is not None:
                    mu_max, mu_min = self.n - 1 - lmax, -1 - second
                    if mu_max > mu_min:
                        om = self.n * (-mu_min) / (mu_max - mu_min)
                        self.board.info["omega_hoffman_bound"] = round(om, 4)
                        self.cover_floor = math.ceil(self.n / math.floor(om + 1e-6) - 1e-9)
                        self.board.info["clique_cover_floor"] = self.cover_floor
                        self.board.info["fractional_cover_floor"] = round(self.n / om, 4)
                _ = key
        else:
            self.int_spec, self.cands = [], []
        self.trace.log("candidates", count=len(self.cands),
                       top=[{k: c[k] for k in ("p", "theta", "c", "rank_pred")} for c in self.cands[:8]])
        if self.uses_llm():
            self.dialogue = {"state": "need_solver", "round": 1, "reason": "initial plan", "future": None}
        else:
            self.apply_plan(self.default_plan(), "default")

    # ------------------------------------------------------------------------------------------------ plans
    def _lower_hint(self) -> int:
        """Largest independent-set size seen (board or describe's greedy sets): only orders the candidates."""
        g = (self.describe or {}).get("greedy_independent_set_size", 0)
        return max(self.board.L, int(g))

    def default_plan(self) -> dict:
        plan = [{"tool": "code_rules", "args": {"rule": "spectral"}, "why": "default: all primes <= n by rank_pred"},
                {"tool": "frac_cover", "args": {}, "why": "default"}]
        if self.n <= 150:
            plan.append({"tool": "sat_minrank", "args": {}, "why": "default (n <= 150)"})
        if 2 * self.n <= 150:
            plan.append({"tool": "code_rules", "args": {"rule": "blowup", "t": 2}, "why": "default (2n <= 150)"})
        if self.n <= 300:
            plan.append({"tool": "alpha_bound", "args": {}, "why": "default (n <= 300)"})
        return {"family_hypothesis": {"name": "none", "params": {}, "confidence": 0.0, "evidence": []},
                "plan": plan, "settings": {}, "stop_when": "L == U or budget", "notes": "scheduler default"}

    def apply_plan(self, plan: dict, source: str) -> None:
        self.plan, self.plan_source = plan, source
        if self.t_plan is None:
            self.t_plan = self.now()
        fh = plan.get("family_hypothesis") or {}
        self.hypothesis = fh
        self.predictions = {k: v for k, v in (plan.get("predictions") or {}).items() if isinstance(v, (int, float))}
        self.trace.log("plan_accepted", source=source, hypothesis=fh, predictions=self.predictions,
                       plan=plan.get("plan"), settings=plan.get("settings"), stop_when=plan.get("stop_when"),
                       unresolved_objections=self.unresolved)
        self._apply_settings(plan.get("settings") or {})
        if self.cands:
            self.cands = spectral_candidates(self.n, self.int_spec, primes_upto(max(2, self.n)),
                                             lower=self._lower_hint())
        ordered: list[dict] = []
        for it in plan.get("plan", []):
            tool, args = it.get("tool"), it.get("args") or {}
            if tool == "code_rules" and args.get("rule", "spectral") == "spectral":
                ordered += self._pairs_from_args(args)
                if not args.get("fields") and not args.get("pairs"):
                    ordered += [c for c in self.cands]
            elif tool == "code_rules" and args.get("rule") == "blowup":
                self._queue_blowup(int(args.get("t", 2)), source)
            elif tool == "frac_cover":
                self.ensure("frac_cover", {"init_cliques": self.latest_cover}, priority=60, source=source)
            elif tool == "sat_minrank":
                self._queue_sat(args.get("r"), source)
            elif tool == "alpha_bound":
                if self.n <= 300:
                    self.ensure("alpha_bound", {}, priority=55, source=source)
            elif tool in ("indep_set", "clique_cover"):
                self.ensure(tool, self.settings.get(tool, {}), priority=70, source=source)
        # the plan's pairs first, then every other candidate in the default order: a wrong plan costs time only
        seen = set()
        q = []
        for c in ordered + self.cands:
            key = (c["p"], c["c"])
            if key in seen or key in self.spec_tried:
                continue
            seen.add(key)
            q.append(dict(c, source=("plan" if c in ordered else "default")))
        self.spec_queue = q
        self.spec_deferred = []
        self._prune_spectral("plan applied")
        self.trace.log("spectral_queue", size=len(self.spec_queue),
                       head=[{k: c[k] for k in ("p", "theta", "rank_pred", "source")} for c in self.spec_queue[:6]])
        self.last_progress = self.now()

    def _pairs_from_args(self, args: dict) -> list[dict]:
        out = []
        primes = set(primes_upto(max(2, self.n)))
        for pr in args.get("pairs") or []:
            try:
                p, th = int(pr[0]), int(pr[1])
            except (TypeError, ValueError, IndexError):
                continue
            if p in primes and th % p:
                pred = next((c["rank_pred"] for c in self.cands if c["p"] == p and c["theta"] == th), None)
                out.append({"p": p, "theta": th, "c": (-pow(th % p, -1, p)) % p,
                            "rank_pred": pred if pred is not None else self.n,
                            "plausible": pred is not None and pred >= self.board.L})
        for f in args.get("fields") or []:
            try:
                p = int(f)
            except (TypeError, ValueError):
                continue
            if p not in primes:
                self.trace.log("plan_item_ignored", item=f, reason="not a prime <= n")
                continue
            out += [c for c in self.cands if c["p"] == p]
        return out

    def _apply_settings(self, st: dict) -> None:
        for tool in ("indep_set", "clique_cover"):
            new = st.get(tool)
            if not isinstance(new, dict):
                continue
            clean = {}
            for k, v in new.items():
                if k in ("restarts", "tabu_iters") and isinstance(v, (int, float)) and 1 <= v <= 10 ** 6:
                    clean[k] = int(v)
            if clean and clean != {k: self.settings[tool].get(k) for k in clean}:
                self.settings[tool].update(clean)
                self.trace.log("settings_change", tool=tool, settings=self.settings[tool])
                for tid in self.pool.running_of(tool):
                    self.pool.stop(tid, "restart with new settings")
                self.submit(tool, self.settings[tool] | {"seed": self.seed + 1}, priority=900, source="plan")

    def _queue_blowup(self, t: int, source: str) -> None:
        if self.n * t > 150:
            self.trace.log("plan_item_ignored", item=f"blowup t={t}", reason=f"n*t = {self.n * t} > 150")
            return
        r = max(1, math.ceil(self.board.L * t))
        if (t, r) not in self.blowup_done:
            self.blowup_done.add((t, r))
            I = self.board.certs.get(self.board.L_cert, {}).get("vertices", [])
            self.submit("blowup", {"t": t, "r": r, "indep": I, "limit": 30}, priority=40, source=source,
                        time_limit=40)

    def _queue_sat(self, r, source: str) -> None:
        if self.n > 150:
            self.trace.log("plan_item_ignored", item="sat_minrank", reason=f"n = {self.n} > 150")
            return
        r = int(r) if isinstance(r, (int, float)) else self.board.L
        if r < 1 or r in self.sat_asked or r >= self.board.U:
            return
        self.sat_asked.add(r)
        I = self.board.certs.get(self.board.L_cert, {}).get("vertices", [])
        cnf = None if self.cert_dir is None else str(self.cert_dir / f"unsat_r{r}.cnf.gz")
        self.submit("sat_minrank", {"r": r, "indep": I, "limit": 60, "cnf_path": cnf}, priority=45, source=source,
                    time_limit=70)

    # ------------------------------------------------------------------------------------------------ spectral
    def _pump_spectral(self) -> None:
        if self.board.closed():
            return
        while (len(self.spec_running) < self.max_spec and len(self.pool.free) >= self.spec_cores
               and (self.spec_queue or (not self.spec_running and self.spec_deferred))):
            if not self.spec_queue:
                self.spec_queue, self.spec_deferred = self.spec_deferred, []
                self.trace.log("settings_change", what="spectral queue empty: running the deferred pairs",
                               count=len(self.spec_queue))
            c = self.spec_queue.pop(0)
            key = (c["p"], c["c"])
            if key in self.spec_tried:
                continue
            self.spec_tried[key] = c
            src = c.get("source", "default")
            t = self.submit("spectral", {"p": c["p"], "c": c["c"], "theta": c["theta"], "rank_pred": c["rank_pred"],
                                         "seed": self.seed}, cores=self.spec_cores,
                            priority=SPEC_PRIORITY.get(src, 10), source=src,
                            time_limit=c.get("time_limit"))
            self.spec_running[t.id] = c
            self.pool.pump()

    def _prune_spectral(self, why: str) -> None:
        U = self.board.U
        keep, drop = [], []
        for c in self.spec_queue:
            (drop if c["plausible"] and c["rank_pred"] >= U and c.get("source") not in ("near_miss", "critic_check")
             else keep).append(c)
        if drop:
            self.spec_queue = keep
            self.spec_deferred += drop
            self.trace.log("settings_change", what=f"deferred {len(drop)} (p, theta) pairs with rank_pred >= U = {U}",
                           why=why)

    def _on_spectral_result(self, tid: str, r: dict) -> None:
        self.spec_results.append(dict(r, t=round(self.now(), 2)))
        self.trace.log("tool_report", task=tid, tool="spectral", **r)
        L = self.board.L
        rank = r.get("rank")
        if rank is not None and L <= rank <= L + 2 and rank > L and not self.board.closed():
            self._on_event("NEAR_MISS", {"p": r["p"], "theta": r["theta"], "c": r["c"], "rank": rank, "L": L})

    # ------------------------------------------------------------------------------------------------ events
    def _on_event(self, name: str, det: dict) -> None:
        self.trace.log("event", name=name, detail=det)
        self.events.append({"t": round(self.now(), 2), "name": name, **{k: str(v) for k, v in det.items()}})
        if name in ("NEW_LOWER", "NEW_UPPER"):
            self.last_progress = self.now()
            self.pool.shared["L"].value = self.board.L
            self.pool.shared["U"].value = float(self.board.U)
            self._prune_spectral(name)
            self._check_contradiction()
        if name == "NEW_LOWER" and self.n <= 150 and self.plan is not None:
            self._queue_sat(self.board.L, "event")         # ask "minrk <= L ?" first (Pincer's sat_up)
        if name == "NEW_UPPER" and self.cover_floor is not None and self.board.U <= self.cover_floor:
            for tool in ("clique_cover", "frac_cover", "clique_cover_cpsat"):
                for t in self.pool.running_of(tool):
                    self.pool.stop(t, f"cannot beat U = {self.board.U}: any clique cover >= {self.cover_floor}")
        if name == "CLOSED":
            self.pool.stop_all("L == U")
        if name == "NEAR_MISS":
            p, th = det["p"], det["theta"]
            extra = [dict(c, source="near_miss") for c in self.cands
                     if c["p"] == p and (c["p"], c["c"]) not in self.spec_tried]
            for dc in (-1, 1, 2):
                cc = (det["c"] + dc) % p
                if cc and (p, cc) not in self.spec_tried:
                    extra.append({"p": p, "theta": None, "c": cc, "rank_pred": None, "plausible": False,
                                  "source": "near_miss"})
            self.spec_queue = extra + [c for c in self.spec_queue if (c["p"], c["c"]) not in
                                       {(e["p"], e["c"]) for e in extra}]
            self.trace.log("settings_change", what=f"near miss at p = {p} (rank {det['rank']} vs L = {det['L']}): "
                                                   f"{len(extra)} more (theta, c) for p = {p} moved to the front")
        if name == "REJECTED":
            self.trace.log("warning", what="a tool result failed the board's entry check", **det)

    def _check_contradiction(self) -> None:
        pr = self.predictions
        if not pr:
            return
        issues = []
        a = pr.get("alpha")
        b = pr.get("beta")
        if a is not None and self.board.L > a:
            issues.append(f"an independent set of size {self.board.L} exceeds the predicted alpha = {a}")
        ah = self.board.info.get("alpha_upper_cpsat")
        if a is not None and ah is not None and ah < a:
            issues.append(f"CP-SAT proves alpha <= {ah} < predicted alpha = {a}")
        hb = self.board.info.get("alpha_hoffman_ratio_bound")
        if a is not None and hb is not None and hb + 1e-6 < a:
            issues.append(f"the Hoffman ratio bound {hb} is below the predicted alpha = {a}")
        if b is not None and (self.board.L > b or self.board.U < b):
            issues.append(f"the certified bracket [{self.board.L}, {self.board.U}] excludes the predicted beta = {b}")
        new = [i for i in issues if i not in self.contradicted]
        if new:
            self.contradicted |= set(new)
            self._on_checkpoint("CONTRADICTION", "; ".join(new))

    def _check_stall_and_phase(self) -> None:
        if self.plan is None or self.board.closed():
            return
        if self.now() - self.last_progress > 0.25 * self.budget:
            self.last_progress = self.now()
            self._on_checkpoint("STALL", f"no bound change for {0.25 * self.budget:.0f} s")
        if (not self.spec_queue and not self.spec_running and not self.spec_deferred
                and not getattr(self, "_phase_done", False)):
            self._phase_done = True
            self._on_checkpoint("NEW_PHASE", "every queued (p, theta) pair has been tried")

    def _on_checkpoint(self, name: str, detail: str) -> None:
        self.checkpoints += 1
        self.trace.log("event", name=name, detail={"detail": detail})
        self.events.append({"t": round(self.now(), 2), "name": name, "detail": detail})
        if not self.uses_llm() or self.dialogue is not None and self.dialogue["state"] != "done":
            self.trace.log("checkpoint_default", name=name, action="continue with the current queue")
            return
        if self.llm.left() <= 2:                                   # keep two calls for the Verifier
            self.trace.log("checkpoint_default", name=name, action="LLM call budget reserved for the Verifier")
            return
        first = "need_critic_checkpoint" if name == "CONTRADICTION" and self.mode != "no_critic" else "need_solver"
        self.dialogue = {"state": first, "round": 1, "reason": f"checkpoint {name}: {detail}", "future": None,
                         "checkpoint": f"{name}: {detail}"}

    # ------------------------------------------------------------------------------------------------ dialogue
    def _failures_text(self) -> str:
        rs = self.spec_results[-12:]
        if not rs:
            return "no spectral codes tried yet"
        return "; ".join(f"p={r['p']} theta={r['theta']} c={r['c']}: " +
                         (f"rank {r['rank']}" if r["rank"] is not None else f"rank > {r['cap']}") for r in rs)

    def tools_text(self) -> str:
        fin = {}
        for tid, f in self.pool.finished.items():
            fin.setdefault(f["tool"], []).append(f["seconds"])
        spec = [r["seconds"] for r in self.spec_results if r.get("seconds")]
        lines = [f"n = {self.n}; budget left {self.left():.0f} s of {self.budget:.0f} s; "
                 f"{len(self.cores)} cores; LLM calls left {self.llm.left() if self.llm else 0}",
                 "indep_set (anytime local search), clique_cover (DSatur + tabu, anytime): already running",
                 "frac_cover: column generation, anytime" + (" (exact pricing, n <= 300)" if self.n <= 300 else
                                                             " (heuristic pricing)"),
                 f"code_rules/spectral: one exact rank per (p, theta); measured so far: "
                 f"{'%.1f s median' % float(np.median(spec)) if spec else 'none yet'}; "
                 f"{len(self.cands)} candidate pairs over {len(set(c['p'] for c in self.cands))} primes <= n",
                 "top candidates by rank_pred: " + ", ".join(
                     f"(p={c['p']}, theta={c['theta']}, pred {c['rank_pred']})" for c in self.cands[:6]),
                 f"sat_minrank: {'available' if self.n <= 150 else 'not available (n > 150)'}; "
                 f"blowup: {'available' if 2 * self.n <= 150 else 'not available (2n > 150)'}"]
        return "\n".join(lines)

    def _step_dialogue(self) -> None:
        d = self.dialogue
        if d is None or d["state"] == "done":
            return
        st = d["state"]
        if st == "need_solver":
            prompt = AG.solver_prompt(self.summary, self.board.text(), self.tools_text(), self.history,
                                      d.get("checkpoint") and (d["checkpoint"] + "\nRecent spectral results: " +
                                                               self._failures_text()))
            d["future"] = self.executor.submit(self.llm.ask, "solver", AG.system("solver"), prompt,
                                               lambda o: AG.validate_solver(o, self.summary), self.trace)
            d["state"] = "await_solver"
        elif st == "await_solver" and d["future"].done():
            rep = d["future"].result()
            if rep is None:
                if self.plan is None:
                    self.apply_plan(self.default_plan(), "default (llm_fallback)")
                d["state"] = "done"
                return
            d["solver"] = rep
            if self.mode == "no_critic" or d.get("checkpoint") or d["round"] > 3 or self.llm.left() <= 2:
                if d["round"] > 3:
                    self.unresolved = list(d.get("last_objections", []))
                self.apply_plan(rep, "solver" if not d.get("checkpoint") else "solver_checkpoint")
                d["state"] = "done"
            else:
                d["state"] = "need_critic"
        elif st in ("need_critic", "need_critic_checkpoint"):
            src = d.get("solver") or (self.plan or {})
            prompt = AG.critic_prompt(self.summary, self.board.text(), src)
            if st == "need_critic_checkpoint":
                prompt += "\n\nCHECKPOINT\n" + d["checkpoint"]
            d["future"] = self.executor.submit(self.llm.ask, "critic", AG.system("critic"), prompt,
                                               AG.validate_critic, self.trace)
            d["state"] = "await_critic"
        elif st == "await_critic" and d["future"].done():
            rep = d["future"].result()
            self.critic_rounds += 1
            if rep is None or rep.get("verdict") == "accept":
                if d.get("checkpoint"):
                    d["state"] = "need_solver"
                    d["checkpoint"] = d["checkpoint"] + (" | critic: " + "; ".join(rep.get("objections", []))
                                                         if rep else "")
                    return
                self.apply_plan(d["solver"], "solver+critic" if rep else "solver (critic fallback)")
                d["state"] = "done"
                return
            d["last_objections"] = rep.get("objections", [])
            self.history.append(f"round {d['round']} objections: " + " | ".join(d["last_objections"]))
            d["checks"] = self._run_checks(rep.get("requested_checks", [])[:3])
            d["t_checks"] = self.now()
            d["state"] = "await_checks"
        elif st == "await_checks":
            ids = d["checks"]
            if all(self.check_tasks[t]["done"] for t in ids) or self.now() - d["t_checks"] > 30:
                self.history.append(f"round {d['round']} check results: " + self._checks_text(ids))
                if d.get("checkpoint"):
                    d["state"] = "need_solver"
                    return
                d["round"] += 1
                if d["round"] > 3:
                    self.unresolved = list(d["last_objections"])
                    self.trace.log("dialogue_limit", rounds=3, unresolved=self.unresolved)
                    self.apply_plan(d["solver"], "solver (3 rounds, objections unresolved)")
                    d["state"] = "done"
                    return
                d["state"] = "need_solver"

    def _run_checks(self, checks: list[dict]) -> list[str]:
        ids = []
        for c in checks:
            tool, args = c.get("tool"), c.get("args") or {}
            if tool == "code_rules":
                pairs = self._pairs_from_args(args)[:3]
                for pc in pairs:
                    if (pc["p"], pc["c"]) in self.spec_tried:
                        prev = next((r for r in self.spec_results if r["p"] == pc["p"] and r["c"] == pc["c"]), None)
                        tid = f"cached#{len(self.check_tasks)}"
                        self.check_tasks[tid] = {"done": True, "check": c, "summary": prev or "already running"}
                        ids.append(tid)
                        continue
                    self.spec_tried[(pc["p"], pc["c"])] = pc
                    t = self.submit("spectral", {"p": pc["p"], "c": pc["c"], "theta": pc["theta"],
                                                 "rank_pred": pc["rank_pred"]}, cores=self.spec_cores,
                                    priority=SPEC_PRIORITY["critic_check"], source="critic_check", time_limit=30)
                    self.spec_running[t.id] = pc
                    self.check_tasks[t.id] = {"done": False, "check": c}
                    ids.append(t.id)
            elif tool == "alpha_bound":
                tid = f"info#{len(self.check_tasks)}"
                info = {k: v for k, v in self.board.info.items() if "alpha" in k or "omega" in k}
                if self.n <= 300:
                    t = self.submit("alpha_bound", {}, priority=290, source="critic_check", time_limit=30)
                    self.check_tasks[t.id] = {"done": False, "check": c}
                    ids.append(t.id)
                self.check_tasks[tid] = {"done": True, "check": c, "summary": info or "no alpha bound available"}
                ids.append(tid)
            elif tool in ("indep_set", "clique_cover"):
                tid = f"board#{len(self.check_tasks)}"
                val = self.board.L if tool == "indep_set" else self.board.best("upper", "beta")
                self.check_tasks[tid] = {"done": True, "check": c,
                                         "summary": f"{tool} is running continuously; current best {val}"}
                ids.append(tid)
            self.trace.log("critic_check", check=c, tasks=ids)
        return ids

    def _checks_text(self, ids: list[str]) -> str:
        out = []
        for t in ids:
            ct = self.check_tasks[t]
            rep = self.task_reports.get(t, [])
            sres = [p for k, p in rep if k == "spectral_result"]
            if sres:
                r = sres[-1]
                out.append(f"spectral p={r['p']} theta={r['theta']}: " +
                           (f"exact rank {r['rank']}" if r["rank"] is not None else f"rank > {r['cap']}"))
            else:
                s = ct.get("summary")
                out.append(f"{ct['check'].get('tool')}: {json.dumps(s, default=str)[:300]}")
        return "; ".join(out) if out else "no checks run"

    # ------------------------------------------------------------------------------------------------ result
    def result(self) -> dict:
        b = self.board
        U = b.U
        proven = b.closed()
        t_closed = next((e["t"] for e in self.events if e["name"] == "CLOSED"), None)
        ucert = b.certs.get(b.U_cert, {})
        field = ucert.get("p") if ucert.get("kind") == "spectral" else (
            2 if ucert.get("kind") in ("f2_rows", "blowup_f2") else ("any" if ucert.get("kind") in
                                                                      ("clique_cover", "fractional_cover") else None))
        return {
            "graph": self.gid, "n": self.n, "mode": self.mode, "L": b.L, "U": str(U), "U_float": float(U),
            "gap": float(U - b.L), "proven": proven, "upper_kind": ucert.get("kind"), "field": field,
            "upper_cert": b.U_cert, "lower_cert": b.L_cert,
            "t_first_bound": b.t_first, "t_proof": t_closed, "t_final_change": b.t_last_change,
            "wall": round(self.now(), 2), "stopped": self.stopped_reason,
            "t_describe": self.t_describe, "t_plan": self.t_plan,
            "llm": self.llm.usage() if self.llm else None,
            "hypothesis": self.hypothesis, "predictions": self.predictions, "plan_source": self.plan_source,
            "critic_rounds": self.critic_rounds, "unresolved_objections": self.unresolved,
            "checkpoints": self.checkpoints, "spectral_tried": len(self.spec_results),
            "spectral_results": self.spec_results, "events": self.events,
            "board": b.snapshot(), "describe": self.describe,
        }

    def save_certificates(self, out_dir: Path) -> dict:
        """Compact certificates of the final L and U (and any minrk_2 UNSAT record): lists and small npz files."""
        out_dir.mkdir(parents=True, exist_ok=True)
        b = self.board
        saved = {}
        for role, cid in (("lower", b.L_cert), ("upper", b.U_cert)):
            c = b.certs.get(cid)
            if c is None:
                continue
            if c["kind"] == "spectral":
                f = out_dir / f"{role}_spectral_p{c['p']}.npz"
                np.savez_compressed(f, S=np.asarray(c["S"], dtype=np.int32), C=c["C"], p=c["p"], c=c["c"])
                saved[role] = {"kind": "spectral", "p": c["p"], "c": c["c"], "theta": c.get("theta"),
                               "rank": c["rank"], "file": f.name}
            elif c["kind"] == "fractional_cover":
                saved[role] = {"kind": c["kind"], "cliques": c["cliques"], "weights": [str(w) for w in c["weights"]]}
            elif c["kind"] == "identity":
                saved[role] = c
            else:
                saved[role] = {k: v for k, v in c.items()}
        unsat = [c for c in b.certs.values() if c["kind"] == "f2_unsat"]
        if unsat:
            saved["minrk2_unsat"] = unsat
        best_cover = None
        for e in b.entries:
            if e.quantity == "beta" and e.bound == "upper" and b.certs[e.certificate_id]["kind"] == "clique_cover":
                best_cover = e.certificate_id
        if best_cover is not None and best_cover != b.U_cert:
            saved["best_clique_cover"] = b.certs[best_cover]
        (out_dir / "certificates.json").write_text(json.dumps(saved, default=str))
        return saved
