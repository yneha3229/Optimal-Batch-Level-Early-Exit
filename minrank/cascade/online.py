"""Online gate for the live evaluator (not wired in yet; the Option A evaluator owner decides where to call it).

Two entry points:

1. Batch API (an evaluator that evaluates a generation of B candidates graph by graph):

       gate = CascadeGate.from_traces("minrank/evolve/runs/traces/evals.jsonl", CostRule(0.1), checkpoints)
       ... after evaluating the first checkpoints[e] graphs for every live candidate:
       keep = gate.should_continue({cand_id: [score_1, ..., score_c], ...}, batch_id)   # {cand_id: bool}

2. Single-candidate API (OpenEvolve calls evaluate(program_path) once per candidate; the cascade stages of that
   evaluator are the checkpoints):

       if not gate.gate_single(partial_scores, e):      # partial_scores: scores on the first checkpoints[e] graphs
           return {"combined_score": mean(partial_scores), "stopped_at": e}

   gate_single applies the cost rule's *individual* test against the current incumbent, which is the best full
   fitness among complete candidates in the trace file (re-read with `refresh()`).

Calibration from a live trace uses every complete candidate; once the gate drops candidates, only complete ones are
left, which biases the table towards survivors. Keep a fraction of candidates ungated (explore) and refresh the
table only from those, as the simulator does (pass `explore_ids`).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .policies import Context, CostRule, Policy
from .traces import load_traces


class CascadeGate:
    def __init__(self, policy: Policy, calibration: Calibration, checkpoints: list[int], b: np.ndarray,
                 a: float = 0.05, incumbent: float = -math.inf, batch_size: int = 8) -> None:
        if calibration.n_checkpoints != max(len(checkpoints) - 1, 1):
            raise ValueError("calibration must have one row per decision checkpoint")
        self.policy = policy
        self.calibration = calibration
        self.checkpoints = list(checkpoints)
        self.b = np.asarray(b, dtype=float)
        self.a = a
        self.incumbent = incumbent
        self.batch_size = batch_size
        self.trace_path: Path | None = None
        self.explore_ids: set[str] | None = None

    # ---- construction from a trace file ------------------------------------------------------------------------
    @classmethod
    def from_traces(cls, path: str | Path, policy: Policy, checkpoints: list[int], a: float = 0.05,
                    batch_size: int = 8, explore_ids: set[str] | None = None) -> "CascadeGate":
        gate = cls(policy, Calibration(max(len(checkpoints) - 1, 1)), checkpoints, np.ones(checkpoints[-1]), a,
                   batch_size=batch_size)
        gate.trace_path = Path(path)
        gate.explore_ids = explore_ids
        gate.refresh()
        return gate

    def refresh(self) -> None:
        """Rebuild incumbent, cost estimates and calibration from the trace file (complete candidates only).

        Candidates are replayed in timestamp order so each calibration record uses the incumbent of its own time.
        """
        if self.trace_path is None or not self.trace_path.exists():
            return
        df = load_traces(self.trace_path).sort_values("ts").drop_duplicates(["candidate", "order"], keep="last")
        L = self.checkpoints[-1]
        orders = np.sort(df["order"].unique())[:L]
        if len(orders) < L:
            return  # not enough graphs evaluated yet
        df = df[df["order"].isin(orders)]
        sec = df.groupby("order")["seconds"].mean().reindex(orders).to_numpy()
        self.b = np.where(np.isfinite(sec), sec, np.nanmean(sec))
        piv = df.pivot(index="candidate", columns="order", values="score").reindex(columns=orders)
        complete = piv.dropna()
        first = df.groupby("candidate")["ts"].min().reindex(complete.index).sort_values()
        cal = Calibration(self.calibration.n_checkpoints, self.calibration.edges)
        inc = -math.inf
        for cand in first.index:
            s = complete.loc[cand].to_numpy(dtype=float)  # (L,)
            full = float(s.mean())
            if self.explore_ids is None or cand in self.explore_ids:
                pm = np.array([s[:c].mean() for c in self.checkpoints[:-1]]) if len(self.checkpoints) > 1 \
                    else np.array([full])
                cal.add(pm, full, inc)
            inc = max(inc, full)
        self.calibration = cal
        self.incumbent = inc

    # ---- decisions -----------------------------------------------------------------------------------------------
    def _ctx(self, batch_size: int) -> Context:
        return Context(self.checkpoints, self.incumbent, self.calibration, self.a, self.b, batch_size)

    def _checkpoint_index(self, n_done: int) -> int:
        if n_done not in self.checkpoints[:-1]:
            raise ValueError(f"{n_done} graphs evaluated is not a decision checkpoint {self.checkpoints[:-1]}")
        return self.checkpoints.index(n_done)

    def should_continue(self, candidate_partial_scores: dict[str, list[float]], batch_id: int = -1
                        ) -> dict[str, bool]:
        """All candidates must have the same number of scores (= a decision checkpoint)."""
        if not candidate_partial_scores:
            return {}
        ids = list(candidate_partial_scores)
        lens = {len(v) for v in candidate_partial_scores.values()}
        if len(lens) != 1:
            raise ValueError(f"candidates have different numbers of scores: {sorted(lens)}")
        e = self._checkpoint_index(lens.pop())
        prefix = np.array([candidate_partial_scores[i] for i in ids], dtype=float)  # (B, c)
        alive = np.ones(len(ids), dtype=bool)
        keep = self.policy.decide(self._ctx(len(ids)), e, prefix, alive)
        return {i: bool(k) for i, k in zip(ids, keep)}

    def gate_single(self, partial_scores: list[float], e: int | None = None, lam: float | None = None) -> bool:
        """True = keep evaluating this candidate. Individual cost-rule test against the current incumbent."""
        if e is None:
            e = self._checkpoint_index(len(partial_scores))
        rule = self.policy if isinstance(self.policy, CostRule) else CostRule(lam if lam is not None else 0.1)
        if lam is not None:
            rule = CostRule(lam)
        if rule.lam <= 0:
            return True
        ctx = self._ctx(self.batch_size)
        c = self.checkpoints[e]
        prefix = np.asarray(partial_scores, dtype=float)[None, :c]  # (1, c)
        g = float(rule.gains(ctx, e, prefix)[0])
        return g >= rule.lam * float(ctx.b[c:].sum()) / ctx.full_cost
