"""The bound board and the trace.

Board: the only place bounds live. Entries {bound, quantity, value, certificate_id, tool, t}. Only tool results that
carry a certificate are accepted, and each is checked on entry (fit / validity; the Verifier re-checks everything
later with its own code). LLM agents can read the board but cannot write to it.

Trace: every event, LLM message, tool start/stop, settings change and board update, one JSON object per line, with
the time since the run started. The demo is built from it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np


def _jsonable(x):
    if isinstance(x, Fraction):
        return str(x) if x.denominator != 1 else x.numerator
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


class Trace:
    def __init__(self, path: Path | None, t0: float | None = None) -> None:
        self.path = path
        self.t0 = time.perf_counter() if t0 is None else t0
        self.f = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "w")
        self.listeners = []
        self.records: list[dict] = []

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def log(self, kind: str, **data) -> dict:
        rec = {"t": round(self.now(), 3), "wall": datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3],
               "kind": kind, **data}
        self.records.append(rec)
        if self.f is not None:
            self.f.write(json.dumps(rec, default=_jsonable) + "\n")
            self.f.flush()
        for fn in self.listeners:
            fn(rec)
        return rec

    def close(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


@dataclass
class Entry:
    bound: str            # "lower" | "upper"
    quantity: str         # "beta" | "minrk_<p>" | "lambda_<p>" | "alpha"
    value: Fraction
    certificate_id: str
    tool: str
    t: float


@dataclass
class Board:
    A: np.ndarray
    trace: Trace
    entries: list[Entry] = field(default_factory=list)
    certs: dict = field(default_factory=dict)
    L: int = 0
    U: Fraction = None
    L_cert: str | None = None
    U_cert: str | None = None
    info: dict = field(default_factory=dict)
    t_first: float | None = None
    t_last_change: float = 0.0
    _n_cert: int = 0

    def __post_init__(self) -> None:
        self.U = Fraction(self.A.shape[0])           # trivial: send every message (identity code)
        self.U_cert = "trivial"
        self.certs["trivial"] = {"kind": "identity", "value": self.A.shape[0]}

    # ---- entry checks (the agent's own; verify/ re-checks independently) ------------------------------------
    def _new_id(self, kind: str) -> str:
        self._n_cert += 1
        return f"{kind}-{self._n_cert:03d}"

    def _check_indep(self, I) -> int:
        I = np.asarray(sorted(set(int(v) for v in I)), dtype=np.int64)
        if len(I) and self.A[np.ix_(I, I)].any():
            raise ValueError("independent set has an edge")
        return len(I)

    def _check_cover(self, cliques) -> int:
        n = self.A.shape[0]
        seen = np.zeros(n, dtype=bool)
        for C in cliques:
            C = np.asarray(C, dtype=np.int64)
            sub = self.A[np.ix_(C, C)]
            if sub.sum() != len(C) * (len(C) - 1):
                raise ValueError("cover contains a non-clique")
            seen[C] = True
        if not seen.all():
            raise ValueError("cover misses a vertex")
        return len(cliques)

    def _check_frac(self, cliques, weights) -> Fraction:
        n = self.A.shape[0]
        cov = [Fraction(0)] * n
        for C, w in zip(cliques, weights):
            if w < 0:
                raise ValueError("negative weight")
            Ca = np.asarray(C, dtype=np.int64)
            if self.A[np.ix_(Ca, Ca)].sum() != len(C) * (len(C) - 1):
                raise ValueError("fractional cover contains a non-clique")
            for v in C:
                cov[v] += w
        if min(cov) < 1:
            raise ValueError("fractional cover leaves a vertex under-covered")
        return sum(weights, Fraction(0))

    def _check_spectral(self, p, c, S, C, rank) -> int:
        """Fit is structural for I + cA (diagonal 1, zero off the edges); spot-check M = C M[S,:] with 2 vectors."""
        n = self.A.shape[0]
        if (1 % p) == 0 or C.shape != (n, len(S)) or len(S) != rank:
            raise ValueError("bad spectral certificate shape")
        rng = np.random.default_rng(len(self.entries))
        for _ in range(2):
            x = rng.integers(0, p, n)
            Ax = np.concatenate([self.A[a:a + 1024].astype(np.float64) @ x.astype(np.float64)
                                 for a in range(0, n, 1024)])
            Mx = np.mod(x + c * np.mod(Ax, p), p).astype(np.int64)
            MSx = Mx[S]
            if not np.array_equal(np.mod(C.astype(np.int64) @ MSx, p), Mx):
                raise ValueError("spectral factorization fails a random test")
        return rank

    # ---- offers ---------------------------------------------------------------------------------------------
    def offer(self, kind: str, tool: str, payload: dict) -> list[tuple[str, dict]]:
        """Check a tool result and record it. Returns the events it causes."""
        t = self.trace.now()
        events = []
        try:
            if kind == "indep":
                v = self._check_indep(payload["vertices"])
                cid = self._new_id("indep")
                self.certs[cid] = {"kind": "independent_set", "vertices": sorted(int(x) for x in payload["vertices"])}
                events += self._record("lower", "beta", Fraction(v), cid, tool, t)
                self._record("lower", "alpha", Fraction(v), cid, tool, t)
            elif kind == "cover":
                v = self._check_cover(payload["cliques"])
                cid = self._new_id("cover")
                self.certs[cid] = {"kind": "clique_cover", "cliques": payload["cliques"]}
                events += self._record("upper", "beta", Fraction(v), cid, tool, t)
            elif kind == "frac":
                v = self._check_frac(payload["cliques"], payload["weights"])
                cid = self._new_id("frac")
                self.certs[cid] = {"kind": "fractional_cover", "cliques": payload["cliques"],
                                   "weights": payload["weights"]}
                events += self._record("upper", "beta", v, cid, tool, t)
            elif kind == "spectral":
                p, c = int(payload["p"]), int(payload["c"])
                C = payload["C"]
                v = self._check_spectral(p, c, np.asarray(payload["S"]), C, int(payload["rank"]))
                cid = self._new_id(f"spec{p}")
                self.certs[cid] = {"kind": "spectral", "p": p, "c": c, "theta": payload.get("theta"),
                                   "S": np.asarray(payload["S"]), "C": C, "rank": v}
                self._record("upper", f"minrk_{p}", Fraction(v), cid, tool, t)
                events += self._record("upper", "beta", Fraction(v), cid, tool, t)
            elif kind == "sat_rows":
                cid = self._new_id("sat2")
                self.certs[cid] = {"kind": "f2_rows", "rows": payload["rows"], "rank": int(payload["rank"])}
                self._record("upper", "minrk_2", Fraction(payload["rank"]), cid, tool, t)
                events += self._record("upper", "beta", Fraction(payload["rank"]), cid, tool, t)
            elif kind == "sat_unsat":
                cid = self._new_id("unsat2")
                self.certs[cid] = {"kind": "f2_unsat", "r": int(payload["r"]), "cnf": payload.get("cnf"),
                                   "indep": payload.get("indep")}
                self._record("lower", "minrk_2", Fraction(payload["r"] + 1), cid, tool, t)
            elif kind == "blowup":
                cid = self._new_id("blowup")
                val = Fraction(int(payload["rank"]), int(payload["t"]))
                self.certs[cid] = {"kind": "blowup_f2", "t": int(payload["t"]), "rows": payload["rows"],
                                   "rank": int(payload["rank"])}
                self._record("upper", "lambda_2", val, cid, tool, t)
                events += self._record("upper", "beta", val, cid, tool, t)
            else:
                raise ValueError(f"unknown result kind {kind}")
        except (ValueError, KeyError) as e:
            self.trace.log("board_reject", tool=tool, result=kind, reason=str(e))
            return [("REJECTED", {"tool": tool, "reason": str(e)})]
        return events

    def _record(self, bound, quantity, value, cid, tool, t) -> list[tuple[str, dict]]:
        prev = self.best(bound, quantity)
        better = prev is None or (value > prev if bound == "lower" else value < prev)
        if not better:
            return []
        self.entries.append(Entry(bound, quantity, value, cid, tool, t))
        self.trace.log("board_update", bound=bound, quantity=quantity, value=value, certificate_id=cid, tool=tool)
        ev = []
        if quantity == "beta":
            if self.t_first is None:
                self.t_first = t
            self.t_last_change = t
            if bound == "lower" and value > self.L:
                self.L, self.L_cert = int(value), cid
                ev.append(("NEW_LOWER", {"L": self.L, "tool": tool}))
            if bound == "upper" and value < self.U:
                self.U, self.U_cert = value, cid
                ev.append(("NEW_UPPER", {"U": value, "tool": tool}))
            if self.closed():
                ev.append(("CLOSED", {"L": self.L, "U": self.U}))
        return ev

    def best(self, bound: str, quantity: str):
        vals = [e.value for e in self.entries if e.bound == bound and e.quantity == quantity]
        if not vals:
            return None
        return max(vals) if bound == "lower" else min(vals)

    def closed(self) -> bool:
        return Fraction(self.L) >= self.U

    def snapshot(self) -> dict:
        qs = sorted({(e.bound, e.quantity) for e in self.entries})
        return {"beta": {"L": self.L, "U": _jsonable(self.U), "L_cert": self.L_cert, "U_cert": self.U_cert,
                         "proven": self.closed()},
                "by_quantity": {f"{b}:{q}": _jsonable(self.best(b, q)) for b, q in qs},
                "info": self.info}

    def text(self) -> str:
        """Board as short text for the LLM prompts."""
        s = self.snapshot()
        lines = [f"beta in [{s['beta']['L']}, {s['beta']['U']}] (lower from {self._tool(self.L_cert)}, upper from "
                 f"{self._tool(self.U_cert)}); proven: {s['beta']['proven']}"]
        for k, v in s["by_quantity"].items():
            if not k.endswith(":beta"):
                lines.append(f"  {k} = {v}")
        for k, v in self.info.items():
            lines.append(f"  info {k}: {v}")
        return "\n".join(lines)

    def _tool(self, cid):
        for e in self.entries:
            if e.certificate_id == cid:
                return e.tool
        return cid
