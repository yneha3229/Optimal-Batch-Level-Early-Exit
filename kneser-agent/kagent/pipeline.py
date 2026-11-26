"""Orchestrator: bound board + worker pool + event-driven scheduler + the three agents + verification.

    python -m kagent.pipeline "K(16,4)" --llm chat|local|none [--seed 0] [--budget 300] [--no-family-rule]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from math import comb
from pathlib import Path

import numpy as np

from . import graphs, tools, verify
from . import llm as LLM
from .agents import Agents

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict = {}


# ------------------------------------------------------------------------------------- worker side
def _A(path):
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = graphs.load(path)
    return _CACHE[path]


def w_describe(path):
    return tools.describe(_A(path))


def w_indep(path, restarts, seed):
    return tools.independent_set(_A(path), restarts=restarts, seed=seed)


def w_cover(path):
    return tools.clique_cover(_A(path))


def w_code(path, p, c, S):
    return tools.try_code(_A(path), p, c, S)


def binomial_match(N, d):
    hits = []
    for k in range(1, 40):
        n = 2 * k + 1
        while comb(n, k) <= N:
            if comb(n, k) == N and comb(n - k, k) == d:
                hits.append({"n": n, "k": k})
            n += 1
    return hits


# ------------------------------------------------------------------------------------- orchestrator
class Run:
    def __init__(self, spec, seed=0, llm="none", budget=300.0, family_rule=True, workers=3, live=True, tag=""):
        self.spec, self.seed, self.budget, self.allow_family = spec, seed, budget, family_rule
        self.path = str(graphs.make(spec, seed))
        self.name = Path(self.path).stem + (f"_{tag}" if tag else "")
        self.A = graphs.load(self.path)
        self.N = self.A.shape[0]
        self.backend = LLM.make(llm)
        self.llm_name = llm
        self.events = []
        self.live = live
        self.t0 = time.perf_counter()
        self.agents = Agents(self.backend, self.trace)
        self.pool = ProcessPoolExecutor(max_workers=workers)
        self.board = {"lower": 0, "upper": self.N, "lower_cert": None, "upper_cert": {"kind": "trivial"}}
        self.S = None
        self.cover = None
        self.tried = set()
        self.pending = {}           # future -> (kind, info)

    # ----- trace
    def trace(self, ev, **data):
        rec = {"t": round(time.perf_counter() - self.t0, 3), "ev": ev, **data}
        self.events.append(rec)
        if self.live:
            short = {k: v for k, v in data.items() if k not in ("reply", "summary", "S", "cover")}
            msg = json.dumps(short)[:160]
            if ev == "llm_reply":
                r = data["reply"]
                msg = data["agent"] + ": " + json.dumps(r)[:300]
            print(f"[{rec['t']:8.2f}s] {ev:16s} {msg}", flush=True)

    def llm_seconds(self):
        return getattr(self.backend, "seconds", 0.0) if self.backend else 0.0

    # ----- board
    def offer_lower(self, S, tool):
        if verify.check_independent(self.A, S) and len(S) > self.board["lower"]:
            self.board.update(lower=len(S), lower_cert={"kind": "independent_set", "tool": tool})
            self.S = list(S)
            self.trace("board", side="lower", value=len(S), tool=tool, bracket=[self.board["lower"], self.board["upper"]])
            return True
        return False

    def offer_upper(self, value, cert):
        if value is not None and value < self.board["upper"]:
            self.board.update(upper=value, upper_cert=cert)
            self.trace("board", side="upper", value=value, tool=cert["kind"], bracket=[self.board["lower"], value])
            return True
        return False

    def closed(self):
        return self.board["lower"] >= self.board["upper"]

    def left(self):
        return self.budget - (time.perf_counter() - self.t0)

    # ----- scheduling
    def submit(self, kind, fn, *args, **info):
        f = self.pool.submit(fn, *args)
        self.pending[f] = (kind, info)
        self.trace("start", tool=kind, **{k: v for k, v in info.items() if k != "S"})
        return f

    def submit_code(self, p, c, why=""):
        key = (int(p), int(c) % int(p))
        if key in self.tried or key[1] == 0:
            return
        self.tried.add(key)
        self.submit("code", w_code, self.path, key[0], key[1], self.S, p=key[0], c=key[1], why=why)

    def handle(self, f):
        kind, info = self.pending.pop(f)
        try:
            res = f.result()
        except Exception as e:  # noqa: BLE001
            self.trace("tool_error", tool=kind, error=str(e)[:200])
            return kind, None
        if kind == "describe":
            self.trace("done", tool="describe", seconds=res.get("seconds"))
        elif kind == "indep":
            self.offer_lower(res, "greedy independent set")
        elif kind == "cover":
            if verify.check_clique_cover(self.A, res):
                self.cover = res
                self.offer_upper(len(res), {"kind": "clique_cover", "cliques": len(res)})
        elif kind == "code":
            self.trace("done", tool="code", p=res["p"], c=res["c"], rank=res["rank"], seconds=res["seconds"])
            if res["rank"] is not None:
                self.offer_upper(res["rank"], {"kind": "linear_code", "p": res["p"], "c": res["c"],
                                               "meets_lower": res["meets_lower"]})
                lo = self.board["lower"]
                if not self.closed() and not res["meets_lower"] and lo and res["rank"] <= lo + 2:     # NEAR_MISS: other c for this p
                    self.trace("event", name="near_miss", p=res["p"], rank=res["rank"])
                    for c in range(1, min(res["p"], 8)):
                        self.submit_code(res["p"], c, why="near miss")
        return kind, res

    def drain(self, kinds=None, stop_when_closed=True):
        """Process results until no pending task of `kinds` remains (or the bracket closes / budget ends)."""
        out = []
        while self.pending and self.left() > 0:
            if kinds and not any(k in kinds for k, _ in self.pending.values()):
                break
            done, _ = wait(list(self.pending), timeout=max(0.1, self.left()), return_when=FIRST_COMPLETED)
            for f in done:
                out.append(self.handle(f))
            if stop_when_closed and self.closed():
                self.trace("event", name="bracket_closed", bracket=[self.board["lower"], self.board["upper"]])
                break
        return out

    # ----- plan execution
    def run_checks(self, checks, summary):
        results = []
        for ch in checks[:4]:
            tool, a = ch.get("tool"), ch.get("args", {}) or {}
            if tool == "binomial_match":
                results.append({"tool": tool, "result": binomial_match(self.N, summary["degree_max"])})
            elif tool == "hoffman":
                results.append({"tool": tool, "result": tools.hoffman_alpha_bound(summary)})
            elif tool == "family_rule" and self.allow_family:
                results.append({"tool": tool, "args": a, "result": tools.family_rule(int(a["n"]), int(a["k"]))})
            elif tool == "try_code":
                r = tools.try_code(self.A, int(a["p"]), int(a["c"]) % int(a["p"]), self.S)
                self.tried.add((r["p"], r["c"]))
                self.offer_upper(r["rank"], {"kind": "linear_code", "p": r["p"], "c": r["c"], "meets_lower": r["meets_lower"]}) if r["rank"] else None
                results.append({"tool": tool, "args": a, "result": r})
            else:
                results.append({"tool": tool, "result": "not available"})
        self.trace("checks", results=results)
        return results

    def execute(self, plan, summary):
        for step in plan:
            if self.closed() or self.left() <= 0:
                return
            tool, a = step.get("tool"), step.get("args", {}) or {}
            if tool == "spectral":
                for cand in tools.spectral_candidates(summary, top=int(a.get("top", 8))):
                    self.submit_code(cand["p"], cand["c"], why=f"spectral pred {cand['rank_pred']}")
            elif tool == "try_code":
                self.submit_code(int(a["p"]), int(a["c"]), why=step.get("why", ""))
            elif tool == "family_rule" and self.allow_family:
                r = tools.family_rule(int(a["n"]), int(a["k"]))
                self.trace("rule", tool="family_rule", args=a, result=r)
                if r:
                    self.submit_code(r["p"], r["c"], why="family rule")
            elif tool == "independent_set":
                self.submit("indep", w_indep, self.path, int(a.get("restarts", 4)), self.seed + 1)
            elif tool in ("hoffman", "binomial_match"):
                self.run_checks([step], summary)
            self.drain(kinds={"code", "indep"})

    # ----- main
    def run(self):
        self.trace("input", graph=self.name, receivers=self.N, llm=self.llm_name, family_rule=self.allow_family)
        fd = self.submit("describe", w_describe, self.path)
        self.submit("indep", w_indep, self.path, 1, self.seed)
        self.submit("cover", w_cover, self.path)
        while not fd.done():
            self.drain(kinds={"describe"}, stop_when_closed=False)
        summary = fd.result()
        self.drain(kinds={"indep"}, stop_when_closed=False)   # codes need the independent set S
        self.trace("summary", summary=summary)

        default = {"family_hypothesis": {"name": "unknown", "params": {}, "confidence": 0, "evidence": []},
                   "plan": [{"tool": "spectral", "args": {"top": 8}, "why": "default"}], "notes": "default plan"}
        payload = {"summary": summary, "board": self.view()}
        plan, src = self.agents.solver(payload, default)
        rounds = 0
        while self.backend is not None and rounds < 3:
            crit, _ = self.agents.critic({"summary": summary, "solver": plan, "board": self.view()})
            rounds += 1
            if crit["verdict"] == "accept":
                break
            checks = self.run_checks(crit.get("requested_checks", []), summary)
            if self.closed():
                break
            plan, _ = self.agents.solver({**payload, "board": self.view(), "critic_objections": crit["objections"],
                                          "check_results": checks, "previous_plan": plan}, plan)
        self.trace("plan_accepted", plan=plan["plan"], hypothesis=plan.get("family_hypothesis"), critic_rounds=rounds)
        if not self.closed():
            self.execute(plan["plan"], summary)
        if not self.closed() and self.left() > 0:                       # STALL checkpoint
            self.trace("event", name="checkpoint")
            tried = sorted(self.tried)
            fallback = {"plan": [{"tool": "spectral", "args": {"top": 30}, "why": "fallback"}]}
            plan2, _ = self.agents.checkpoint({"summary": summary, "board": self.view(), "tried_codes": tried}, fallback)
            self.execute(plan2.get("plan", []), summary)
        self.drain(kinds={"cover", "indep", "code"}, stop_when_closed=True)
        wall = time.perf_counter() - self.t0
        for f in list(self.pending):
            f.cancel()
        self.stop_pool()
        return self.finish(summary, wall, rounds, plan)

    def stop_pool(self):
        procs = list(getattr(self.pool, "_processes", {}).values())
        self.pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:                      # our own worker processes, by handle
            if p.is_alive():
                p.terminate()

    def view(self):
        return {"lower": self.board["lower"], "upper": self.board["upper"], "upper_by": self.board["upper_cert"]["kind"],
                "codes_tried": len(self.tried), "seconds": round(time.perf_counter() - self.t0, 2)}

    # ----- verification + report
    def finish(self, summary, wall, rounds, plan):
        A, b = self.A, self.board
        checks = {"independent_set": verify.check_independent(A, self.S) if self.S else False}
        uc = b["upper_cert"]
        if uc["kind"] == "clique_cover":
            checks["clique_cover"] = verify.check_clique_cover(A, self.cover)
        if uc["kind"] == "linear_code":
            p, c = uc["p"], uc["c"]
            checks["code_fits"] = verify.check_fit(A, p, c)
            if b["lower"] == b["upper"]:
                checks["factorization"] = verify.check_code_meets(A, p, c, self.S)
                checks["decoding"] = verify.decode_simulation(A, p, c, self.S)
        verdict = "proven" if (b["lower"] == b["upper"] and all(checks.values())) else (
            "invalid" if not all(checks.values()) else "gap")
        truth = json.loads(graphs.TRUTH.read_text()).get(Path(self.path).stem, {})
        result = {"graph": self.name, "spec": self.spec, "receivers": self.N, "llm": self.llm_name,
                  "family_rule_allowed": self.allow_family, "bracket": [b["lower"], b["upper"]], "verdict": verdict,
                  "upper_certificate": uc, "checks": checks, "wall_s": round(wall, 3),
                  "llm_s": round(self.llm_seconds(), 3), "compute_s": round(wall - self.llm_seconds(), 3),
                  "llm_calls": getattr(self.backend, "calls", 0) if self.backend else 0, "critic_rounds": rounds,
                  "hypothesis": plan.get("family_hypothesis"), "codes_tried": len(self.tried), "truth": truth}
        base = ROOT / "results" / "baselines.jsonl"
        bl = [json.loads(l) for l in base.read_text().splitlines()] if base.exists() else []
        bl = [r for r in bl if r.get("spec") == self.spec]
        vtxt, _ = self.agents.verifier({"result": result, "baselines": bl[-1:] or "not run"})
        extra = vtxt.get("checks_requested", [])
        if "decode_more" in extra and verdict == "proven":
            checks["decoding_more"] = verify.decode_simulation(A, uc["p"], uc["c"], self.S, trials=10, seed=7)
        if "own_rank" in extra and uc["kind"] == "linear_code" and self.N <= 2500:
            M = np.where(A, uc["c"], 0).astype(np.int64); M[np.diag_indices_from(M)] = 1
            checks["own_rank_equals_upper"] = verify.rank_mod_p(M, uc["p"]) == b["upper"]
        if not all(checks.values()):
            verdict = "invalid"
        result.update(verdict=verdict, checks=checks, analysis=vtxt.get("analysis"))
        self.trace("verdict", verdict=verdict, bracket=result["bracket"], checks=checks)
        (ROOT / "traces").mkdir(exist_ok=True)
        with open(ROOT / "traces" / f"{self.name}_{self.llm_name}.jsonl", "w") as fh:
            for e in self.events:
                fh.write(json.dumps(e, default=str) + "\n")
        with open(ROOT / "results" / "runs.jsonl", "a") as fh:
            fh.write(json.dumps(result, default=str) + "\n")
        if verdict == "proven" and uc["kind"] == "linear_code":
            S = sorted(int(v) for v in self.S)
            (ROOT / "results" / "codes" / f"{self.name}.json").write_text(json.dumps(
                {"graph": self.name, "field": uc["p"], "matrix": f"I + {uc['c']}*A (mod {uc['p']})",
                 "transmissions": f"for each s in S: y_s = x_s + {uc['c']} * sum of x_u over neighbours u of s",
                 "S": S, "length": len(S),
                 "decoding": "receiver v: x_v = sum_s M[v,s] y_s - sum_{u in N(v)} M[v,u] x_u"}))
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--llm", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=300)
    ap.add_argument("--no-family-rule", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    r = Run(a.spec, a.seed, a.llm, a.budget, not a.no_family_rule, a.workers, tag=a.tag).run()
    print(json.dumps({k: r[k] for k in ("graph", "bracket", "verdict", "wall_s", "llm_s", "compute_s", "checks")}))
    os._exit(0)


if __name__ == "__main__":
    main()
