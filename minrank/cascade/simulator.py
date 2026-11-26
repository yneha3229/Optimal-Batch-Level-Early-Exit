"""Trace-replay simulator for stopping policies over candidate-evaluation batches.

Protocol (strictly causal; every policy pays for the same data collection)
---------------------------------------------------------------------------
Batches are processed in order k = 0, 1, ...
- Cost estimate b_g: mean measured seconds of graph g over cells *evaluated in earlier batches*; graphs never seen
  yet use the mean of the known ones (1.0 before any data). The fixed cost a per evaluated graph is a constant.
- Warm-up: the first `warmup` batches are fully evaluated (they seed the calibration table and the incumbent).
- Audits: after warm-up, every `audit_every`-th batch is fully evaluated. Audits alternate roles: even audits feed
  the calibration table, odd audits are certification batches for CostRuleCertified (kept apart so the table used to
  replay a certification batch never saw it).
- Exploration: in every other batch `explore` randomly chosen candidates (rng seeded by batch id, so the choice does
  not depend on the policy) are always fully evaluated; they feed the calibration table without survivor bias.
- Within a batch, graphs are evaluated in order; evaluating graph g costs a + sum of the measured seconds of the live
  candidates. At each checkpoint except the last the policy may drop candidates (forced candidates stay).
- The incumbent is the best full fitness among *fully evaluated* candidates; a dropped candidate never updates it.
- A *lost winner*: the batch's best candidate (full fitness known to the simulator only, for reporting) beats the
  incumbent at the batch start but is not fully evaluated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .calibration import Calibration
from .policies import Context, CostRule, Full, Policy
from .traces import Batch


def default_checkpoints(L: int) -> list[int]:
    """1, 2, 4, 8, ... below L, then L."""
    out, c = [], 1
    while c < L:
        out.append(c)
        c *= 2
    return out + [L]


@dataclass
class SimConfig:
    checkpoints: list[int] | None = None
    a: float = 0.05
    warmup: int = 3
    explore: int = 1
    audit_every: int = 5
    seed: int = 0
    edges: np.ndarray | None = None


@dataclass
class BatchRun:
    keep_history: list[np.ndarray]  # keep mask after each decision checkpoint
    fully: np.ndarray  # (B,) bool, evaluated on every graph
    seconds: float
    evaluated_cells: np.ndarray  # (B, L) bool


@dataclass
class SimResult:
    label: str
    summary: dict
    per_batch: list[dict] = field(default_factory=list)
    decisions: list[list[np.ndarray]] = field(default_factory=list)


def run_batch(policy: Policy, batch: Batch, ctx: Context, forced: np.ndarray) -> BatchRun:
    """Evaluate one batch under `policy`. Only prefixes of the score matrix are shown to the policy."""
    B, L = batch.scores.shape
    if ctx.L != L or len(forced) != B:
        raise ValueError("context / forced mask do not match the batch")
    alive = np.ones(B, dtype=bool)
    evaluated = np.zeros((B, L), dtype=bool)
    spent, prev = 0.0, 0
    hist: list[np.ndarray] = []
    for e, c in enumerate(ctx.checkpoints):
        for g in range(prev, c):
            if alive.any():
                spent += ctx.a + float(batch.seconds[alive, g].sum())
                evaluated[alive, g] = True
        prev = c
        if c == L:
            break
        if not alive.any():
            hist.append(alive.copy())
            continue
        prefix = batch.scores[:, :c].copy()  # (B, c): the policy never sees graphs >= c
        keep = np.asarray(policy.decide(ctx, e, prefix, alive.copy()), dtype=bool)
        if keep.shape != (B,):
            raise ValueError(f"{policy.label()} returned shape {keep.shape}, expected ({B},)")
        if (keep & ~alive).any():
            raise ValueError(f"{policy.label()} revived a dropped candidate")
        alive = keep | (forced & alive)
        hist.append(alive.copy())
    return BatchRun(hist, alive.copy(), spent, evaluated)


def lost_winner(scores: np.ndarray, fully: np.ndarray, incumbent: float) -> bool:
    full = scores.mean(axis=1)
    best = int(np.argmax(full))
    return bool(full[best] > incumbent and not fully[best])


class _CostEstimator:
    def __init__(self, L: int) -> None:
        self.sum = np.zeros(L)
        self.cnt = np.zeros(L)

    def b(self) -> np.ndarray:
        known = self.cnt > 0
        out = np.ones(len(self.sum))
        if known.any():
            out[:] = (self.sum[known] / self.cnt[known]).mean()
            out[known] = self.sum[known] / self.cnt[known]
        return out

    def observe(self, seconds: np.ndarray, evaluated: np.ndarray) -> None:
        self.sum += (seconds * evaluated).sum(axis=0)
        self.cnt += evaluated.sum(axis=0)


def simulate(batches: list[Batch], policy: Policy, cfg: SimConfig | None = None) -> SimResult:
    if not batches:
        raise ValueError("no batches")
    cfg = cfg or SimConfig()
    L = batches[0].n_graphs
    if any(b.n_graphs != L for b in batches):
        raise ValueError("all batches must share the evaluation order")
    cps = list(cfg.checkpoints) if cfg.checkpoints else default_checkpoints(L)
    if cps[-1] != L:
        cps = [c for c in cps if c < L] + [L]
    E = len(cps) - 1  # decision checkpoints
    calib = Calibration(max(E, 1), **({"edges": cfg.edges} if cfg.edges is not None else {}))
    costs = _CostEstimator(L)
    incumbent = -math.inf
    total = 0.0
    full_total = 0.0
    lost = n_win = 0
    audits_cert: list[tuple[np.ndarray, float]] = []
    n_audit = 0
    res = SimResult(policy.label(), {})
    for k, batch in enumerate(batches):
        B = batch.size
        ctx = Context(cps, incumbent, calib, cfg.a, costs.b(), B)
        is_warm = k < cfg.warmup
        is_audit = (not is_warm) and cfg.audit_every > 0 and (k - cfg.warmup) % cfg.audit_every == 0
        forced = np.zeros(B, dtype=bool)
        if is_warm or is_audit:
            forced[:] = True
        elif cfg.explore > 0:
            rng = np.random.default_rng((cfg.seed, k))
            forced[rng.choice(B, size=min(cfg.explore, B), replace=False)] = True
        run = run_batch(policy, batch, ctx, forced)
        total += run.seconds
        full_total += sum(cfg.a + float(batch.seconds[:, g].sum()) for g in range(L))  # same order as run_batch
        win = bool(batch.full_fitness.max() > incumbent)
        is_lost = lost_winner(batch.scores, run.fully, incumbent)
        n_win += win
        lost += is_lost
        # data collection (all of it observed in this batch; used only from the next batch on)
        full = batch.full_fitness
        prefix_means = np.stack([batch.scores[:, :c].mean(axis=1) for c in cps[:-1]], axis=1) if E else None
        role = "calib"
        if is_audit:
            role = "calib" if n_audit % 2 == 0 else "cert"
            n_audit += 1
        if E:
            feed = forced if role == "calib" else np.zeros(B, dtype=bool)
            for j in np.flatnonzero(feed):
                calib.add(prefix_means[j], float(full[j]), incumbent)
        if role == "cert":
            audits_cert.append((batch.scores.copy(), incumbent))
        costs.observe(batch.seconds, run.evaluated_cells)
        inc_before = incumbent
        if run.fully.any():
            incumbent = max(incumbent, float(full[run.fully].max()))
        if role == "cert" and getattr(policy, "needs_audit", False):
            ctx_next = Context(cps, incumbent, calib, cfg.a, costs.b(), B)

            def replay(rule: CostRule, scores: np.ndarray, inc: float, _ctx=ctx_next) -> bool:
                c2 = Context(cps, inc, _ctx.calibration, cfg.a, _ctx.b, scores.shape[0])
                fake = Batch(-1, [str(i) for i in range(scores.shape[0])], scores, np.zeros_like(scores),
                             batch.graphs)
                r = run_batch(rule, fake, c2, np.zeros(scores.shape[0], dtype=bool))
                return lost_winner(scores, r.fully, inc)

            policy.recertify(audits_cert, replay, step=k)
        res.decisions.append(run.keep_history)
        res.per_batch.append({"policy": policy.label(), "batch": k, "seconds": round(run.seconds, 6),
                              "cum_seconds": round(total, 6), "incumbent": incumbent, "incumbent_before": inc_before,
                              "had_winner": win, "lost_winner": is_lost, "n_full": int(run.fully.sum()),
                              "role": "warmup" if is_warm else ("audit-" + role if is_audit else "normal"),
                              "lam": getattr(policy, "lam", float("nan"))})
    best_possible = max(float(b.full_fitness.max()) for b in batches)
    res.summary = {"policy": policy.label(), "batches": len(batches), "eval_seconds": total,
                   "full_seconds": full_total, "savings": 1.0 - total / full_total if full_total > 0 else 0.0,
                   "speedup": full_total / total if total > 0 else math.inf, "final_incumbent": incumbent,
                   "full_final_incumbent": best_possible, "lost_winners": lost, "batches_with_winner": n_win,
                   "lost_rate": lost / n_win if n_win else 0.0}
    return res


def simulate_all(batches: list[Batch], policies: list[Policy], cfg: SimConfig | None = None) -> list[SimResult]:
    return [simulate(batches, p, cfg) for p in policies]


__all__ = ["SimConfig", "SimResult", "simulate", "simulate_all", "run_batch", "default_checkpoints", "Full"]
