"""Evaluation traces of LLM-evolved candidates: loading, validation, batching, and a synthetic generator.

Trace contract (written by the Option A evaluator; JSON Lines, one record per (candidate, graph)):
    {"run": str, "ts": float, "generation": int, "candidate": str, "graph": str, "order": int, "n": int,
     "split": str, "family": str, "budget": float, "seconds": float,
     "status": "ok" | "invalid" | "timeout" | "error", "rank": int | null, "cover_rank": int, "lb": int | null,
     "score": float}
    score = (cover_rank - rank) / cover_rank if status == "ok" else -1.0
    order = position of the graph in the fixed evaluation order (cheap / small graphs first).

A candidate's full fitness is the mean score over all L graphs. Batches ("generations") are the units a stopping
rule acts on: the B candidates of one batch are evaluated together, graph by graph in `order`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS: tuple[str, ...] = ("run", "ts", "generation", "candidate", "graph", "order", "n", "split", "family", "budget",
                           "seconds", "status", "rank", "cover_rank", "lb", "score")
STATUSES = frozenset({"ok", "invalid", "timeout", "error"})


@dataclass
class Batch:
    """One batch of candidates with complete evaluations.

    scores:  (B, L) float, score of candidate i on the graph at evaluation position g.
    seconds: (B, L) float, measured evaluation time of that cell.
    """

    bid: int
    candidates: list[str]
    scores: np.ndarray
    seconds: np.ndarray
    graphs: list[str]

    def __post_init__(self) -> None:
        B, L = self.scores.shape
        if self.seconds.shape != (B, L) or len(self.candidates) != B or len(self.graphs) != L:
            raise ValueError(f"batch {self.bid}: inconsistent shapes {self.scores.shape}, {self.seconds.shape}, "
                             f"{len(self.candidates)} candidates, {len(self.graphs)} graphs")
        if B == 0 or L == 0:
            raise ValueError(f"batch {self.bid} is empty")
        if not np.isfinite(self.scores).all() or not np.isfinite(self.seconds).all():
            raise ValueError(f"batch {self.bid}: non-finite scores or seconds")
        if (self.seconds < 0).any():
            raise ValueError(f"batch {self.bid}: negative seconds")

    @property
    def size(self) -> int:
        return self.scores.shape[0]

    @property
    def n_graphs(self) -> int:
        return self.scores.shape[1]

    @property
    def full_fitness(self) -> np.ndarray:
        return self.scores.mean(axis=1)  # (B,)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the contract; return a copy with normalised dtypes. Raises ValueError on violations."""
    missing = [f for f in FIELDS if f not in df.columns]
    if missing:
        raise ValueError(f"trace is missing fields {missing}")
    df = df.copy()
    bad = set(df["status"].unique()) - STATUSES
    if bad:
        raise ValueError(f"unknown status values {sorted(bad)}")
    for col in ("ts", "budget", "seconds", "score"):
        df[col] = df[col].astype(float)
    for col in ("generation", "order", "n", "cover_rank"):
        df[col] = df[col].astype(int)
    df["candidate"] = df["candidate"].astype(str)
    if not np.isfinite(df["score"]).all() or not np.isfinite(df["seconds"]).all():
        raise ValueError("non-finite score or seconds in trace")
    if (df["order"] < 0).any():
        raise ValueError("negative order")
    if (df["score"] > 1.0 + 1e-9).any() or (df["score"] < -1.0 - 1e-9).any():
        raise ValueError("score outside [-1, 1]")
    return df


def load_traces(path: str | Path) -> pd.DataFrame:
    """Read a JSON Lines trace file and validate it."""
    rows = []
    with Path(path).open() as f:
        for k, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{k}: bad JSON ({e})") from None
    if not rows:
        raise ValueError(f"{path}: no records")
    return validate(pd.DataFrame(rows))


def save_traces(df: pd.DataFrame, path: str | Path) -> None:
    validate(df)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for rec in df[list(FIELDS)].to_dict(orient="records"):
            for k in ("rank", "lb"):
                if rec[k] is not None and isinstance(rec[k], float) and np.isnan(rec[k]):
                    rec[k] = None
            f.write(json.dumps(rec) + "\n")


def to_batches(df: pd.DataFrame, batch_size: int = 8) -> list[Batch]:
    """Group complete candidates into batches.

    - Duplicate (candidate, order) records: the latest by ts wins.
    - Only candidates evaluated on every evaluation position are kept (a replay needs the full row).
    - If every record has generation >= 0, a batch is one generation (ordered by its first timestamp); otherwise
      candidates are ordered by their first timestamp and cut into consecutive batches of `batch_size`.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    df = validate(df).sort_values("ts").drop_duplicates(["candidate", "order"], keep="last")
    orders = np.sort(df["order"].unique())
    L = len(orders)
    graphs = [df.loc[df["order"] == o, "graph"].mode().iloc[0] for o in orders]
    cover = df.groupby("candidate")["order"].nunique()
    complete = set(cover[cover == L].index)
    df = df[df["candidate"].isin(complete)]
    if df.empty:
        raise ValueError("no candidate has a complete evaluation over all graphs")
    first_ts = df.groupby("candidate")["ts"].min()
    gen = df.groupby("candidate")["generation"].first()
    if (df["generation"] >= 0).all():
        groups = sorted(((first_ts[gen == g].min(), list(first_ts[gen == g].sort_values().index))
                         for g in gen.unique()), key=lambda t: t[0])
        cand_groups = [c for _, c in groups]
    else:
        ordered = list(first_ts.sort_values().index)
        cand_groups = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    piv_s = df.pivot(index="candidate", columns="order", values="score")
    piv_t = df.pivot(index="candidate", columns="order", values="seconds")
    out = []
    for k, cands in enumerate(cand_groups):
        s = piv_s.loc[cands, orders].to_numpy(dtype=float)  # (B, L)
        t = piv_t.loc[cands, orders].to_numpy(dtype=float)  # (B, L)
        out.append(Batch(k, list(cands), s, t, graphs))
    return out


def synthetic_traces(n_candidates: int = 400, n_graphs: int = 24, seed: int = 0, drift: float = 0.0005,
                     quality_sd: float = 0.06, noise_sd: float = 0.08, graph_sd: float = 0.04, fail_p: float = 0.08,
                     timeout_p: float = 0.01, base_seconds: float = 0.05, cost_growth: float = 1.15,
                     generation_size: int | None = None, run: str = "synthetic") -> pd.DataFrame:
    """Trace following the contract, for tests and development before real traces exist.

    Candidate c has latent quality q_c = -0.05 + drift * c + N(0, quality_sd) (slowly improving population, so the
    incumbent keeps moving). Its score on graph g is q_c + d_g + N(0, noise_sd), clipped to [-1, 1], where d_g is a
    per-graph difficulty offset; it is then quantised through an integer rank so that
    score = (cover_rank - rank) / cover_rank holds exactly. A fraction fail_p of candidates crash everywhere
    (status "error", score -1); single cells time out with probability timeout_p. Evaluation time grows
    geometrically with the evaluation position: base_seconds * cost_growth ** order * lognormal noise.
    """
    rng = np.random.default_rng(seed)
    q = -0.05 + drift * np.arange(n_candidates) + rng.normal(0.0, quality_sd, n_candidates)  # (C,)
    d = rng.normal(0.0, graph_sd, n_graphs)  # (L,)
    fail = rng.random(n_candidates) < fail_p
    ns = 20 + 5 * np.arange(n_graphs)
    cover = np.maximum(4, ns // 3)
    rows = []
    for c in range(n_candidates):
        cand = f"c{c:05d}"
        gen_id = c // generation_size if generation_size else -1
        for g in range(n_graphs):
            ts = float(c) + g * 1e-3
            secs = float(base_seconds * cost_growth ** g * rng.lognormal(0.0, 0.2))
            if fail[c] or rng.random() < timeout_p:
                status = "error" if fail[c] else "timeout"
                rank, score = None, -1.0
                secs = 0.01 if fail[c] else secs
            else:
                s = float(np.clip(q[c] + d[g] + rng.normal(0.0, noise_sd), -1.0, 1.0))
                rank = int(np.clip(round(cover[g] * (1.0 - s)), 1, 2 * cover[g]))
                score = (cover[g] - rank) / cover[g]
                status = "ok"
            rows.append({"run": run, "ts": ts, "generation": gen_id, "candidate": cand, "graph": f"g{g:03d}",
                         "order": g, "n": int(ns[g]), "split": "train", "family": "synthetic", "budget": 1.0,
                         "seconds": secs, "status": status, "rank": rank, "cover_rank": int(cover[g]),
                         "lb": int(cover[g] // 2), "score": float(score)})
    return validate(pd.DataFrame(rows))
