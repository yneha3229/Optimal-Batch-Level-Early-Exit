"""Benchmark: Pincer policies against standard pipelines at equal wall-clock budget.

    python -m pincer.bench --split test --budget 30 --workers 3 --methods classic,satdescend,roundrobin,index
    python -m pincer.bench --split train --budget 30 --methods roundrobin --calibrate   # writes pincer/priors.json

Methods
- classic:    the standard bounds pipeline (Pincer proposal, Sec. 5 prototype): whole graph, no decomposition;
              MAIS by the CP-SAT ordering model for half the budget, TabuCol clique cover for the other half.
- satdescend: classic lower bound for half the budget, then minrank B9 of minrank/ (TabuCol + SAT descent with
              the original encoding) for the other half. The strongest single-solver minrank baseline in minrank/.
- roundrobin: Pincer with every useful tool in turn (decomposition, full toolbox, same slice schedule).
- index:      Pincer with the cost-aware index policy.
- llm:        Pincer with the LLM policy (needs --llm).
Every method's numbers come from verified certificates; baselines go through the same CompState checks.

Splits: random families use seeds 0-1 for train (calibration, FunSearch) and 5-6 for test; structured graphs are
test only.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from minrank.bench import generators as gen
from minrank.core.graph import Graph

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


# --------------------------------------------------------------------------------------------- instances
def clustered(k: int, m: int, p_in: float, p_fwd: float, p_back: float, seed: int, mutual: float = 0.5) -> Graph:
    """k clusters of m receivers. Inside a cluster: arcs with prob p_in, a fraction `mutual` of them bidirected.
    Between clusters i < j: arcs i -> j with prob p_fwd, j -> i with prob p_back (small p_back merges clusters
    into larger SCCs). Models side information that is mostly local (caching groups, cells)."""
    rng = np.random.default_rng(seed)
    n = k * m
    adj = np.zeros((n, n), dtype=bool)
    cl = np.repeat(np.arange(k), m)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cl[i] == cl[j]:
                if i < j and rng.random() < p_in:
                    if rng.random() < mutual:
                        adj[i, j] = adj[j, i] = True
                    elif rng.random() < 0.5:
                        adj[i, j] = True
                    else:
                        adj[j, i] = True
            elif cl[i] < cl[j]:
                if rng.random() < p_fwd:
                    adj[i, j] = True
                if rng.random() < p_back:
                    adj[j, i] = True
    return Graph.from_adjacency(adj, True, f"clustered_k{k}_m{m}_pb{p_back}_s{seed}", "clustered",
                                {"k": k, "m": m, "p_in": p_in, "p_fwd": p_fwd, "p_back": p_back, "seed": seed})


def structured() -> list[Graph]:
    """Vertex-transitive / symmetric graphs where beta is often fractional (the Shannon LP's home ground)."""
    gs = [gen.cycle(n) for n in (5, 7, 9, 11, 13, 15)]
    gs += [gen.cycle(n).complement() for n in (7, 9, 11, 13, 15)]
    gs += [gen.circulant(13, (1, 5)), gen.circulant(15, (1, 4)), gen.circulant(16, (1, 4)), gen.circulant(14, (1, 3)),
           gen.cycle_power(16, 2), gen.cycle_power(13, 2), gen.paley(13), gen.paley(17), gen.kneser(5, 2),
           gen.kneser(6, 2), gen.mycielski(4), gen.mycielski(5)]
    return gs


def minrank_bench(pattern: str) -> list[Graph]:
    """Graphs of the shared minrank benchmark (minrank/bench/graphs/*.json), e.g. 'planted*_e0.1_*'."""
    root = ROOT.parent / "minrank" / "bench" / "graphs"
    return [Graph.load(p) for p in sorted(root.glob(pattern + ".json"))]


def instances(split: str) -> list[Graph]:
    if split == "structured":
        return structured()
    if split == "scale":  # larger graphs, seeds 7-8 (never used for tuning)
        gs = []
        for sd in (7, 8):
            gs += [gen.gnp(200, 0.05, sd, True), gen.gnp(200, 0.15, sd, True), gen.gnp(400, 0.03, sd, True),
                   gen.planted(200, 12, 0.03, sd, True), gen.planted(200, 12, 0.03, sd, False),
                   clustered(20, 15, 0.5, 0.1, 0.0002, sd)]
        return gs
    if split == "planted_e01":  # the minrank teammate's planted extra_p = 0.1 benchmark (exact SAT results exist)
        return minrank_bench("planted*_e0.1_*")
    if split == "planted_e01_n100":
        return minrank_bench("planted*_n100_*_e0.1_*")
    seeds = {"train": (0, 1), "test": (5, 6), "test2": (7, 8)}[split]
    out: list[Graph] = []
    for s in seeds:
        for n, p in [(30, 0.3), (50, 0.1), (50, 0.3), (50, 0.6), (80, 0.3), (120, 0.1), (120, 0.3)]:
            out.append(gen.gnp(n, p, s, True))
        for n in (40, 80):
            out.append(gen.gnp(n, 0.5, s, False))
        out.append(gen.planted(50, 8, 0.05, s, True))
        out.append(gen.planted(50, 8, 0.05, s, False))
        out.append(clustered(8, 12, 0.6, 0.15, 0.0, s))
        out.append(clustered(8, 12, 0.6, 0.15, 0.0005, s))
    if split == "test":
        out += [gen.cycle(5), gen.cycle(7), gen.cycle(9), gen.cycle(7).complement(), gen.mycielski(4),
                gen.cycle_power(21, 2), gen.circulant(13, (1, 5)), gen.kneser(7, 2), gen.paley(13)]
    return out


# ------------------------------------------------------------------------------------------------ baselines
def run_baseline(g: Graph, method: str, budget: float, seed: int) -> dict:
    from minrank.baselines.clique_cover import clique_cover_rows
    from minrank.exact.lower import mais
    from minrank.exact.sat import sat_descend

    from .decompose import Component
    from .state import CompState

    t0 = time.perf_counter()
    # whole graph as one "component" (no decomposition); bundle verification needs the true SCCs, so we verify
    # the two certificates directly instead.
    st = CompState(Component(0, tuple(range(g.n)), g))
    lb = mais(g, budget / 2, seed=seed)
    st.offer_acyclic(lb.vertices, "cpsat_order", time.perf_counter() - t0)
    rng = np.random.default_rng(seed)
    info: dict = {}
    if method == "classic":
        rows = clique_cover_rows(g, "tabucol", budget / 2, rng)
    else:
        rows = sat_descend(g, budget / 2, rng, info=info)
    st.offer_rows(rows, method, time.perf_counter() - t0)
    # I + A(G): a one-line fitting matrix every method must at least match (it is optimal on planted graphs
    # with extra_p = 0 and on several Kneser graphs), so the baselines get it too.
    from minrank.baselines.identity_adj import identity_plus_adjacency_rows
    st.offer_rows(identity_plus_adjacency_rows(g), "identity_adj", time.perf_counter() - t0)
    if info.get("proven"):
        st.offer_unsat(st.mr_hi - 1, {"mode": "cols(sat_descend)"}, method, time.perf_counter() - t0)
    tot = {"mr": [st.mr_lo, st.mr_hi], "beta": [str(st.b_lo), str(st.b_hi)], "components": 1,
           "closed_mr": int(st.mr_closed), "closed_beta": int(st.b_closed)}
    last = max((e.t for e in st.events), default=0.0)
    return {"graph": g.name, "family": g.family, "n": g.n, "method": method, "budget": budget,
            "seconds": round(time.perf_counter() - t0, 2), "t_last_improve": round(last, 2), **_flat(tot)}


def _flat(t: dict) -> dict:
    gap_mr = t["mr"][1] - t["mr"][0]
    gap_b = float(Fraction(t["beta"][1]) - Fraction(t["beta"][0]))
    return {"mr_lo": t["mr"][0], "mr_hi": t["mr"][1], "b_lo": t["beta"][0], "b_hi": t["beta"][1],
            "gap_mr": gap_mr, "gap_beta": round(gap_b, 6), "closed_mr_all": int(gap_mr == 0),
            "closed_beta_all": int(gap_b == 0)}


def run_pincer(g: Graph, method: str, budget: float, seed: int, llm: str = "none") -> dict:
    from .agent import Pincer
    from .llm import make_backend
    from .policies import make_policy

    pol = make_policy(method, make_backend(llm) if method == "llm" else None)
    res = Pincer(g, budget, pol, seed).run()
    last = max((st.t for st in res.steps if st.gain > 0), default=0.0)
    closed_at = res.seconds if res.stopped == "closed" else None
    row = {"graph": g.name, "family": g.family, "n": g.n, "method": method, "budget": budget,
           "seconds": round(res.seconds, 2), "t_last_improve": round(last, 2),
           "t_closed": None if closed_at is None else round(closed_at, 2), "stopped": res.stopped,
           "components": res.totals["components"], "steps": len(res.steps), **_flat(res.totals)}
    row["tool_gain"] = _tool_gain(res)
    row["step_log"] = [[s.tool, s.status, s.seconds, s.gain, s.comp, s.slice] for s in res.steps]
    row["n_comp"] = {str(s.comp.index): s.n for s in pol.agent.states}
    row["directed"] = {str(s.comp.index): s.comp.graph.directed for s in pol.agent.states}
    if res.llm_usage:
        row["llm"] = res.llm_usage
    return row


def _tool_gain(res) -> dict:
    out: dict[str, float] = {}
    for s in res.steps:
        if s.tool != "quick":
            out[s.tool] = round(out.get(s.tool, 0.0) + s.gain, 4)
    return out


def _job(args) -> dict:
    gd, method, budget, seed, llm = args
    g = Graph.from_dict(gd)
    if method in ("classic", "satdescend"):
        return run_baseline(g, method, budget, seed)
    return run_pincer(g, method, budget, seed, llm)


def calibrate(rows: list[dict], out: Path) -> None:
    """Per (tool, size bucket): Beta pseudo-counts of success (= the call reduced the gap) and the sum / count of
    gains of successful calls, from round-robin step logs, halved so online updates can still move them."""
    from .policies import bucket

    counts: dict[str, list[float]] = {}
    for r in rows:
        for tool, status, _sec, gain, comp, _sl in r.get("step_log", []):
            if tool == "quick" or (status == "done" and gain == 0):
                continue
            b = bucket(r["n_comp"][str(comp)], r["directed"][str(comp)])
            c = counts.setdefault(f"{tool}|{b}", [1.0, 1.0, 0.0, 0.0])
            if gain > 1e-9:
                c[0] += 0.5
                c[2] += 0.5 * gain
                c[3] += 0.5
            else:
                c[1] += 0.5
    out.write_text(json.dumps({k: [round(x, 4) for x in v] for k, v in counts.items()}, indent=1, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test")
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--methods", default="classic,satdescend,roundrobin,index")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default="none")
    ap.add_argument("--only", default="", help="comma-separated substrings of graph names")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate", action="store_true", help="write pincer/priors.json from the round-robin logs")
    ap.add_argument("--calibrate-only", action="store_true", help="recompute pincer/priors.json from --out and exit")
    a = ap.parse_args(argv)
    if a.calibrate_only:
        rows = [json.loads(line) for line in Path(a.out).read_text().splitlines()]
        calibrate([r for r in rows if r["method"] == "roundrobin"], ROOT / "priors.json")
        print("wrote", ROOT / "priors.json")
        return
    gs = instances(a.split)
    if a.only:
        keys = a.only.split(",")
        gs = [g for g in gs if any(k in g.name for k in keys)]
    jobs = [(g.to_dict(), m, a.budget, a.seed, a.llm) for g in gs for m in a.methods.split(",")]
    random.Random(0).shuffle(jobs)  # mix heavy and light jobs across workers
    RESULTS.mkdir(exist_ok=True)
    out = Path(a.out) if a.out else RESULTS / f"bench_{a.split}_b{int(a.budget)}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            d = json.loads(line)
            done.add((d["graph"], d["method"]))
    jobs = [j for j in jobs if (j[0]["name"], j[1]) not in done]
    print(f"{len(jobs)} jobs -> {out}", flush=True)
    # ProcessPoolExecutor workers are not daemonic, so tools can start their own SAT child processes.
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex, \
            out.open("a") as f:
        futs = [ex.submit(_job, j) for j in jobs]
        for fut in as_completed(futs):
            row = fut.result()
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{row['method']:10s} {row['graph']:34s} mr [{row['mr_lo']},{row['mr_hi']}] "
                  f"beta [{float(Fraction(row['b_lo'])):.3f},{float(Fraction(row['b_hi'])):.3f}] "
                  f"{row['seconds']}s", flush=True)
    if a.calibrate:
        rows = [json.loads(line) for line in out.read_text().splitlines()]
        calibrate([r for r in rows if r["method"] == "roundrobin"], ROOT / "priors.json")
        print("wrote", ROOT / "priors.json")


if __name__ == "__main__":
    main()
