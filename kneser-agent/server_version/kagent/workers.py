"""Tool tasks, each run in its own forked worker process (the adjacency matrix is shared copy-on-write).

A task reports through a queue: ("result", task_id, kind, payload) for certified bounds and tool reports,
("done", task_id, summary) at the end, ("error", task_id, text) on an exception. It checks its stop event between
steps; the scheduler kills it by PID 2 s after a stop request if it has not finished.
"""
from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np


@dataclass
class Task:
    id: str
    tool: str
    args: dict = field(default_factory=dict)
    cores: int = 1
    priority: float = 0.0
    source: str = "scheduler"          # scheduler | plan | critic_check | near_miss | checkpoint
    time_limit: float | None = None


class Ctx:
    def __init__(self, task: Task, A: np.ndarray, q, stop_ev, shared) -> None:
        self.task, self.A, self.q, self.stop_ev, self.shared = task, A, q, stop_ev, shared
        self.t0 = time.perf_counter()
        self.deadline = None if task.time_limit is None else self.t0 + task.time_limit

    def emit(self, kind: str, payload: dict) -> None:
        self.q.put(("result", self.task.id, kind, payload))

    def stop(self) -> bool:
        if self.stop_ev.is_set():
            return True
        if self.deadline is not None and time.perf_counter() > self.deadline:
            return True
        return False

    def left(self, default: float = 1e9) -> float:
        return default if self.deadline is None else max(0.0, self.deadline - time.perf_counter())

    @property
    def L(self) -> int:
        return int(self.shared["L"].value)

    @property
    def U(self) -> float:
        return float(self.shared["U"].value)


# ------------------------------------------------------------------------------------------------ task bodies
def t_describe(ctx: Ctx) -> dict:
    from .tools.describe import describe

    d = describe(ctx.A)
    ctx.emit("describe", d)
    return {"seconds": d["seconds"]}


def t_indep(ctx: Ctx) -> dict:
    from .tools.indep import indep_ils

    last = [0.0]

    def on_improve(vs):
        ctx.emit("indep", {"vertices": vs})
        last[0] = time.perf_counter()

    a = ctx.task.args
    S = indep_ils(ctx.A, ctx.left(1e6), seed=int(a.get("seed", 0)), restarts=int(a.get("restarts", 4)),
                  on_improve=on_improve, stop=lambda: ctx.stop() or ctx.L >= ctx.U)
    return {"best": len(S)}


def t_indep_cpsat(ctx: Ctx) -> dict:
    from .tools.indep import indep_cpsat

    S, opt, ub = indep_cpsat(ctx.A, min(ctx.left(60.0), 60.0), workers=ctx.task.cores,
                             on_improve=lambda vs: ctx.emit("indep", {"vertices": vs}))
    if S:
        ctx.emit("indep", {"vertices": S})
    ctx.emit("alpha_info", {"alpha_upper_cpsat": ub, "alpha_proven": opt, "value": len(S)})
    return {"best": len(S), "optimal": opt, "alpha_upper": ub}


def t_cover(ctx: Ctx) -> dict:
    from .tools.covers import clique_cover

    a = ctx.task.args
    cl = clique_cover(ctx.A, ctx.left(1e6), seed=int(a.get("seed", 0)),
                      on_improve=lambda c: ctx.emit("cover", {"cliques": c}),
                      stop=lambda: ctx.stop() or ctx.L >= ctx.U, floor=ctx.L,
                      tabu_iters=int(a.get("tabu_iters", 20000)))
    return {"best": len(cl)}


def t_cover_cpsat(ctx: Ctx) -> dict:
    from .tools.covers import clique_cover_cpsat

    cl, opt = clique_cover_cpsat(ctx.A, min(ctx.left(60.0), 60.0), workers=ctx.task.cores,
                                 hint=ctx.task.args.get("hint"),
                                 on_improve=lambda c: ctx.emit("cover", {"cliques": c}))
    if cl:
        ctx.emit("cover", {"cliques": cl})
    return {"best": len(cl), "optimal": opt}


def t_frac(ctx: Ctx) -> dict:
    from .tools.covers import frac_cover

    def on_improve(cl, ws, val):
        ctx.emit("frac", {"cliques": cl, "weights": ws, "value": val})

    cl, ws, val, proven = frac_cover(ctx.A, ctx.left(1e6), init_cliques=ctx.task.args.get("init_cliques"),
                                     on_improve=on_improve, stop=lambda: ctx.stop() or ctx.L >= ctx.U,
                                     floor=float(ctx.L))
    return {"best": str(val), "lp_optimal": proven}


def t_spectral(ctx: Ctx) -> dict:
    from .tools.codes import spectral_code

    a = ctx.task.args
    p, c, th = int(a["p"]), int(a["c"]), a.get("theta")
    U = ctx.U
    cap = int(np.ceil(U)) - 1 if a.get("cap") is None else int(a["cap"])
    cap = min(cap, int(np.ceil(U)) - 1)
    if cap < max(1, ctx.L) - 0:
        cap = max(cap, ctx.L)                     # still compute: a code of rank L closes the bracket
    if cap * (p - 1) ** 2 >= 2 ** 53:
        ctx.emit("spectral_result", {"p": p, "c": c, "theta": th, "rank": None, "cap": cap, "skipped": "p too large"})
        return {"skipped": True}
    res = spectral_code(ctx.A, p, c, cap, seed=int(a.get("seed", 0)), stop=ctx.stop)
    rep = {"p": p, "c": c, "theta": th, "rank": res["rank"], "cap": cap, "seconds": res["seconds"],
           "rank_pred": a.get("rank_pred"), "stopped": ctx.stop_ev.is_set()}
    ctx.emit("spectral_result", rep)
    if res["rank"] is not None:
        C = res["C"].astype(np.uint16 if p < 65536 else np.int64)
        ctx.emit("spectral", {"p": p, "c": c, "theta": th, "rank": res["rank"], "S": res["S"], "C": C})
    return {"rank": res["rank"], "cap": cap}


def t_sat(ctx: Ctx) -> dict:
    from .tools.satmr import sat_minrank_at

    a = ctx.task.args
    r = int(a["r"])
    I = a["indep"]
    cnf = a.get("cnf_path")
    res = sat_minrank_at(ctx.A, r, min(ctx.left(60.0), float(a.get("limit", 30.0))), I,
                         cnf_path=None if cnf is None else __import__("pathlib").Path(cnf))
    if res is None:
        ctx.emit("sat_result", {"r": r, "status": "timeout"})
        return {"status": "timeout"}
    if res is False:
        ctx.emit("sat_unsat", {"r": r, "cnf": cnf, "indep": sorted(I)[:r]})
        ctx.emit("sat_result", {"r": r, "status": "unsat"})
        return {"status": "unsat"}
    from minrank.core.f2 import rank_f2

    ctx.emit("sat_rows", {"rows": [int(x) for x in res], "rank": int(rank_f2(list(res)))})
    ctx.emit("sat_result", {"r": r, "status": "sat"})
    return {"status": "sat"}


def t_blowup(ctx: Ctx) -> dict:
    from .tools.codes import blowup_code

    a = ctx.task.args
    t, r = int(a["t"]), int(a["r"])
    res = blowup_code(ctx.A, t, r, min(ctx.left(60.0), float(a.get("limit", 30.0))), fix=a.get("indep"))
    if res is None or res is False:
        ctx.emit("blowup_result", {"t": t, "r": r, "status": "timeout_or_size" if res is None else "unsat"})
        return {"status": str(res)}
    from minrank.core.f2 import rank_f2

    ctx.emit("blowup", {"t": t, "rows": [int(x) for x in res], "rank": int(rank_f2(list(res)))})
    return {"status": "sat"}


def t_alpha_bound(ctx: Ctx) -> dict:
    from .tools.indep import alpha_lovasz

    th = alpha_lovasz(ctx.A, min(ctx.left(30.0), 30.0))
    ctx.emit("alpha_info", {"lovasz_theta": th})
    return {"lovasz_theta": th}


TASKS = {"describe": t_describe, "indep_set": t_indep, "indep_cpsat": t_indep_cpsat, "clique_cover": t_cover,
         "clique_cover_cpsat": t_cover_cpsat, "frac_cover": t_frac, "spectral": t_spectral, "sat_minrank": t_sat,
         "blowup": t_blowup, "alpha_bound": t_alpha_bound}


def worker_main(task: Task, A: np.ndarray, q, stop_ev, shared, cores: list[int]) -> None:
    try:
        os.sched_setaffinity(0, cores)
    except OSError:
        pass
    os.environ["OMP_NUM_THREADS"] = str(len(cores))
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(len(cores))
    except Exception:
        pass
    try:
        import numba

        numba.set_num_threads(max(1, min(len(cores), numba.config.NUMBA_NUM_THREADS)))
    except Exception:
        pass
    ctx = Ctx(task, A, q, stop_ev, shared)
    try:
        summary = TASKS[task.tool](ctx)
        q.put(("done", task.id, summary))
    except Exception:
        q.put(("error", task.id, traceback.format_exc()[-2000:]))


def frac_to_str(x) -> str:
    return str(Fraction(x))
