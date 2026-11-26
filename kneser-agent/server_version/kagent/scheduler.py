"""Worker pool: starts tool tasks in forked processes pinned to cores, one pipe per worker (a killed worker can only
break its own pipe), stop = event, then SIGKILL of the worker's process group (its PID) after 2 s."""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import time
from multiprocessing.connection import wait

from .workers import Task, worker_main


WORKER_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "logs", "kneser_agent_workers.log")


def _child(task, A, conn, stop_ev, shared, cores):
    os.setpgid(0, 0)                       # own process group = own PID: SAT/CP-SAT children die with it
    # a background process group must not write to the terminal (SIGTTOU would stop it): send output to a log
    fd = os.open(WORKER_LOG, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)

    class _Q:
        def put(self, msg):
            conn.send(msg)

    worker_main(task, A, _Q(), stop_ev, shared, cores)
    conn.close()


class Pool:
    def __init__(self, A, trace, cores: list[int]) -> None:
        self.A, self.trace, self.cores = A, trace, list(cores)
        self.mp = mp.get_context("fork")
        n = A.shape[0]
        self.shared = {"L": self.mp.Value("i", 0, lock=False), "U": self.mp.Value("d", float(n), lock=False)}
        self.pending: list[Task] = []
        self.running: dict[str, dict] = {}
        self.free = set(self.cores)
        self.count = 0
        self.finished: dict[str, dict] = {}

    def new_id(self, tool: str) -> str:
        self.count += 1
        return f"{tool}#{self.count}"

    def submit(self, task: Task) -> Task:
        self.pending.append(task)
        self.trace.log("task_queued", task=task.id, tool=task.tool, args=_short(task.args), priority=task.priority,
                       source=task.source)
        return task

    def busy_tools(self) -> list[str]:
        return [r["task"].tool for r in self.running.values()]

    def running_of(self, tool: str) -> list[str]:
        return [tid for tid, r in self.running.items() if r["task"].tool == tool]

    def pump(self) -> None:
        self.pending.sort(key=lambda t: -t.priority)
        i = 0
        while i < len(self.pending):
            t = self.pending[i]
            need = max(1, min(t.cores, len(self.cores)))
            if len(self.free) >= need:
                cs = sorted(self.free)[:need]
                self.free -= set(cs)
                ev = self.mp.Event()
                parent, child = self.mp.Pipe(duplex=False)
                p = self.mp.Process(target=_child, args=(t, self.A, child, ev, self.shared, cs))
                p.start()
                child.close()
                self.running[t.id] = {"proc": p, "task": t, "cores": cs, "t0": time.perf_counter(), "stop_ev": ev,
                                      "stop_t": None, "conn": parent, "done": False, "reason": None}
                self.trace.log("tool_start", task=t.id, tool=t.tool, args=_short(t.args), pid=p.pid, cores=cs,
                               source=t.source)
                self.pending.pop(i)
            else:
                i += 1

    def stop(self, tid: str, reason: str) -> None:
        r = self.running.get(tid)
        if r is not None and r["stop_t"] is None:
            r["stop_ev"].set()
            r["stop_t"] = time.perf_counter()
            r["reason"] = reason
            self.trace.log("tool_stop_request", task=tid, tool=r["task"].tool, reason=reason)

    def cancel_pending(self, pred, reason: str) -> list[Task]:
        out = [t for t in self.pending if pred(t)]
        self.pending = [t for t in self.pending if not pred(t)]
        for t in out:
            self.trace.log("task_dropped", task=t.id, tool=t.tool, reason=reason)
        return out

    def stop_all(self, reason: str) -> None:
        self.cancel_pending(lambda t: True, reason)
        for tid in list(self.running):
            self.stop(tid, reason)

    def poll(self, timeout: float) -> list[tuple]:
        """Messages from workers; also reaps finished workers and kills stopped ones after 2 s."""
        msgs = []
        conns = {r["conn"]: tid for tid, r in self.running.items()}
        if conns:
            for c in wait(list(conns), timeout=timeout):
                tid = conns[c]
                try:
                    while c.poll():
                        msgs.append(c.recv())
                except (EOFError, OSError):
                    self.running[tid]["done"] = True
        else:
            time.sleep(timeout)
        now = time.perf_counter()
        for tid, r in list(self.running.items()):
            p = r["proc"]
            if r["done"] or not p.is_alive():
                p.join(timeout=0.5)
                if p.is_alive():
                    continue
                try:
                    while r["conn"].poll():
                        msgs.append(r["conn"].recv())
                except (EOFError, OSError):
                    pass
                self._finish(tid, "exited")
            elif r["stop_t"] is not None and now - r["stop_t"] > 2.0:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.join(timeout=1.0)
                self.trace.log("tool_killed", task=tid, tool=r["task"].tool, pid=p.pid, after_s=2.0)
                self._finish(tid, "killed")
        return msgs

    def _finish(self, tid: str, how: str) -> None:
        r = self.running.pop(tid)
        r["conn"].close()
        self.free |= set(r["cores"])
        dt = time.perf_counter() - r["t0"]
        self.finished[tid] = {"tool": r["task"].tool, "seconds": dt, "how": how, "reason": r["reason"]}
        self.trace.log("tool_stop", task=tid, tool=r["task"].tool, seconds=round(dt, 2), how=how, reason=r["reason"])

    def shutdown(self, wait_s: float = 3.0) -> None:
        self.stop_all("shutdown")
        t0 = time.perf_counter()
        while self.running and time.perf_counter() - t0 < wait_s + 1:
            self.poll(0.1)
        for tid, r in list(self.running.items()):
            try:
                os.killpg(r["proc"].pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            r["proc"].join(timeout=1.0)
            self._finish(tid, "killed")


def _short(args: dict) -> dict:
    out = {}
    for k, v in args.items():
        if isinstance(v, (list, tuple)) and len(v) > 8:
            out[k] = f"<{len(v)} items>"
        else:
            out[k] = v
    return out
