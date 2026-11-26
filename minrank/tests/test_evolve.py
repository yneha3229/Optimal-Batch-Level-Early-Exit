"""Tests for Option A: seed program, sandbox runner, evaluator, traces, held-out CSV.
Run: python -m pytest minrank/tests/test_evolve.py -q
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from minrank.baselines.clique_cover import conflict_graph, dsatur, num_colors
from minrank.bench import generators as gen
from minrank.bench.instances import Instance, load_index, materialize
from minrank.core.verify import verify
from minrank.evolve import evaluator as ev

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "minrank/evolve/seed_program.py"


def _load_seed():
    spec = importlib.util.spec_from_file_location("seed_candidate", SEED)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def small_bench(tmp_path, monkeypatch):
    """Three small graphs in a private graph dir, reference and traces in tmp (nothing written to the repo)."""
    gs = [gen.cycle(5), gen.gnp(20, 0.5, 4), gen.planted(30, 6, 0.0, 1, directed=True)]
    gdir = tmp_path / "graphs"
    materialize(gdir, [Instance(g.name, "train", (lambda g=g: g)) for g in gs])
    monkeypatch.setenv("MINRANK_GRAPHS", str(gdir))
    monkeypatch.setenv("MINRANK_REFERENCE", str(tmp_path / "reference.json"))
    monkeypatch.setenv("MINRANK_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("MINRANK_RUN", "test")
    ev._CACHE.clear()
    ev._GRAPHS.clear()
    rows = sorted(load_index(gdir), key=lambda r: (int(r["n"]), r["name"]))
    return rows, gs, tmp_path


def _write(tmp_path: Path, name: str, code: str) -> str:
    p = tmp_path / name
    p.write_text(code)
    return str(p)


# ---- seed program ----------------------------------------------------------------------------------------------------
def test_seed_passes_static_check():
    assert ev.static_check(SEED.read_text()) is None


@pytest.mark.parametrize("g", [gen.cycle(5), gen.gnp(20, 0.5, 4), gen.planted(30, 6, 0.0, 1, directed=True)],
                         ids=lambda g: g.name)
def test_seed_returns_fitting_rows_not_worse_than_dsatur(g):
    rows = _load_seed().solve(g.n, list(g.out), 0.8, 0)
    r = verify(g, rows)
    assert r <= num_colors(dsatur(conflict_graph(g)))


def test_seed_respects_budget():
    import time
    g = gen.gnp(50, 0.5, 0)
    t = time.perf_counter()
    _load_seed().solve(g.n, list(g.out), 0.5, 0)
    assert time.perf_counter() - t < 0.5 + 0.3


# ---- sandbox ---------------------------------------------------------------------------------------------------------
def test_sandbox_runner_end_to_end(small_bench):
    rows, gs, _ = small_bench
    r = next(x for x in rows if x["name"] == "cycle_5")
    res = ev.run_one(str(SEED), str(Path(ev.graph_dir()) / r["path"]), 0.3, 0)
    assert res["status"] == "ok"
    assert verify(gs[0], res["rows"]) == 3


def test_sandbox_is_isolated_from_repo(tmp_path, small_bench):
    rows, _, _ = small_bench
    prog = _write(tmp_path, "imp.py", "import minrank\ndef solve(n, out, budget_s, seed):\n    return []\n")
    res = ev.run_one(prog, str(Path(ev.graph_dir()) / rows[0]["path"]), 0.3, 0)
    assert res["status"] == "error" and "minrank" in res["error"]


def test_timeout_kills_infinite_loop(tmp_path, small_bench):
    import time
    rows, _, _ = small_bench
    prog = _write(tmp_path, "loop.py", "def solve(n, out, budget_s, seed):\n    while True:\n        pass\n")
    t = time.perf_counter()
    res = ev.run_one(prog, str(Path(ev.graph_dir()) / rows[0]["path"]), 0.2, 0, grace=0.5)
    assert res["status"] == "timeout"
    assert time.perf_counter() - t < 0.2 + 0.5 + 1.0


# ---- static check ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("code,why", [
    ("def solve(n, out, b, s):\n    open('x')\n", "open"),
    ("import os\ndef solve(n, out, b, s):\n    return []\n", "os"),
    ("import json\ndef solve(n, out, b, s):\n    return []\n", "import not allowed"),
    ("from subprocess import run\ndef solve(n, out, b, s):\n    return []\n", "subprocess"),
    ("import numpy as np\ndef other():\n    return 1\n", "no top-level solve"),
])
def test_static_check_rejects(code, why):
    err = ev.static_check(code)
    assert err is not None and (why in err or why.split()[0] in err)


def test_rejected_program_scores_minus_one(tmp_path, small_bench):
    rows, _, _ = small_bench
    prog = _write(tmp_path, "bad.py", "import os\ndef solve(n, out, b, s):\n    return []\n")
    rep = ev.run_evaluation(prog, rows, budget=0.3, workers=2)
    assert rep.metrics["combined_score"] == -1.0 and rep.error


# ---- evaluator scoring and traces ------------------------------------------------------------------------------------
def test_invalid_matrix_scores_minus_one(tmp_path, small_bench):
    rows, _, _ = small_bench
    # identity with one forbidden bit set: A[0][j] = 1 for a j outside S_0  -> does not fit
    code = ("def solve(n, out, budget_s, seed):\n"
            "    rows = [1 << i for i in range(n)]\n"
            "    for j in range(1, n):\n"
            "        if not (out[0] >> j) & 1:\n"
            "            rows[0] |= 1 << j\n"
            "            break\n"
            "    return rows\n")
    prog = _write(tmp_path, "invalid.py", code)
    rep = ev.run_evaluation(prog, rows, budget=0.3, workers=3)
    assert all(r["status"] == "invalid" and r["score"] == -1.0 for r in rep.records)
    assert rep.metrics["combined_score"] == -1.0 and rep.metrics["n_invalid"] == len(rows)


def test_trace_records_and_scores(small_bench):
    rows, _, tmp = small_bench
    rep = ev.run_evaluation(str(SEED), rows, budget=0.3, workers=3, stage="full")
    lines = (tmp / "traces" / "evals.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(rows)
    for ln in lines:
        rec = json.loads(ln)
        assert set(ev.TRACE_FIELDS) <= set(rec)
        assert rec["run"] == "test" and rec["status"] == "ok" and rec["budget"] == 0.3
        assert rec["score"] == pytest.approx((rec["cover_rank"] - rec["rank"]) / rec["cover_rank"])
        assert rec["lb"] <= rec["rank"]
    assert sorted(json.loads(ln)["order"] for ln in lines) == list(range(len(rows)))
    assert rep.metrics["frac_ok"] == 1.0 and rep.metrics["n_graphs"] == len(rows)
    assert {f"score_{r['family']}" for r in rows} <= set(rep.metrics)


def test_stage2_reuses_stage1_without_duplicate_traces(small_bench):
    rows, _, tmp = small_bench
    ev.run_evaluation(str(SEED), rows[:2], budget=0.3, workers=2, stage="stage1", order_base=rows)
    ev.run_evaluation(str(SEED), rows, budget=0.3, workers=2, stage="stage2", order_base=rows)
    recs = [json.loads(ln) for ln in (tmp / "traces" / "evals.jsonl").read_text().strip().splitlines()]
    assert len(recs) == len(rows)  # stage-1 graphs not re-run or re-traced
    assert {r["graph"]: r["order"] for r in recs} == {r["name"]: k for k, r in enumerate(rows)}


def test_openevolve_entry_points(small_bench, monkeypatch):
    rows, _, _ = small_bench
    monkeypatch.setattr(ev, "evolve_graphs", lambda split="train", families=None, max_n=None: rows)
    monkeypatch.setenv("MINRANK_EVO_BUDGET", "0.3")
    for fn in (ev.evaluate_stage1, ev.evaluate_stage2, ev.evaluate):
        res = fn(str(SEED))
        metrics = res if isinstance(res, dict) else res.metrics
        assert metrics["combined_score"] > -1.0


# ---- held-out CSV ----------------------------------------------------------------------------------------------------
def test_eval_heldout_writes_baseline_format(small_bench):
    rows, _, tmp = small_bench
    out = tmp / "evolved.csv"
    cmd = [sys.executable, "-m", "minrank.evolve.eval_heldout", "--program", str(SEED), "--graphs", str(ev.graph_dir()),
           "--split", "train", "--budgets", "0.3", "--seeds", "0", "--workers", "3", "--out", str(out),
           "--certs", str(tmp / "certs")]
    for _ in range(2):  # second run must add nothing
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    import csv
    from minrank.run_baselines import RESULT_FIELDS
    with out.open() as f:
        rd = csv.DictReader(f)
        assert rd.fieldnames == RESULT_FIELDS
        data = list(rd)
    assert len(data) == len(rows)
    assert all(d["method"].startswith("evolved:") and d["verified"] == "True" for d in data)
    assert len(list((tmp / "certs").glob("*.json"))) == len(rows)


# ---- re-targeted mode: score relative to SAT at the same budget (MINRANK_EVO_TARGET=sat) ---------------------------
def _write_refs(tmp: Path, rows: list[dict], sat: dict[str, tuple[int, bool]], budget: float) -> None:
    """Hand-made reference.json (cover 3 more than lb) and reference_sat.json (sat_rank, sat_proven) for `rows`."""
    from minrank.evolve.reference_sat import sat_key
    covers = {"cycle_5": (3, 2), "gnp_n20_p0.5_s4": (7, 4), "planted_d_n30_r6_e0.0_s1": (9, 6)}
    ref, sref = {}, {}
    for r in rows:
        cover, lb = covers[r["name"]]
        ref[r["name"]] = {"cover_rank": cover, "lb": lb, "digest": r["digest"], "split": r["split"],
                          "family": r["family"], "n": int(r["n"]), "cover_time": 1.0, "lb_time": 5.0}
        rank, proven = sat[r["name"]]
        sref[sat_key(r["name"], budget, 0)] = {"graph": r["name"], "sat_rank": rank, "sat_proven": proven,
                                               "budget": budget, "seed": 0, "seconds": budget,
                                               "digest": r["digest"], "split": r["split"], "family": r["family"],
                                               "n": int(r["n"])}
    (tmp / "reference.json").write_text(json.dumps(ref))
    (tmp / "reference_sat.json").write_text(json.dumps(sref))


@pytest.fixture()
def sat_target(small_bench, monkeypatch):
    rows, gs, tmp = small_bench
    monkeypatch.setenv("MINRANK_EVO_TARGET", "sat")
    monkeypatch.setenv("MINRANK_SAT_REFERENCE", str(tmp / "reference_sat.json"))
    monkeypatch.setenv("MINRANK_EVO_BUDGET", "0.3")
    monkeypatch.setenv("MINRANK_EVO_FAMILIES", "cycle,gnp,planted_d")
    return rows, gs, tmp


def test_default_target_is_cover_and_unknown_target_fails(monkeypatch):
    monkeypatch.delenv("MINRANK_EVO_TARGET", raising=False)
    assert ev.target() == "cover"
    monkeypatch.setenv("MINRANK_EVO_TARGET", "bogus")
    with pytest.raises(ValueError):
        ev.target()


def test_score_record_relative_to_ref_rank():
    g = gen.cycle(5)
    res = {"status": "ok", "rows": [1 << i for i in range(5)], "error": ""}  # identity: rank 5
    assert ev.score_record(g, {"cover_rank": 3}, res) == ("ok", 5, pytest.approx(-2 / 3), "")
    assert ev.score_record(g, {"cover_rank": 3}, res, ref_rank=4) == ("ok", 5, pytest.approx(-1 / 4), "")
    assert ev.score_record(g, {"cover_rank": 3}, {"status": "timeout", "rows": None, "error": "timeout"},
                           ref_rank=4)[2] == -1.0


def test_default_mode_traces_and_metrics_unchanged(small_bench, monkeypatch):
    monkeypatch.delenv("MINRANK_EVO_TARGET", raising=False)
    rows, _, tmp = small_bench
    rep = ev.run_evaluation(str(SEED), rows, budget=0.3, workers=3)
    rec = json.loads((tmp / "traces" / "evals.jsonl").read_text().splitlines()[0])
    assert list(rec) == ev.TRACE_FIELDS + ["stage"]
    assert set(rep.metrics) == {"combined_score", "n_graphs", "n_improved", "n_invalid", "frac_ok", "mean_seconds",
                                "mean_rank_gap_to_lb"} | {f"score_{r['family']}" for r in rows}
    assert not (tmp / "reference_sat.json").exists()


def test_hard_graphs_auto_filter(sat_target, monkeypatch):
    rows, _, tmp = sat_target
    # cycle_5: SAT proven -> easy; gnp: SAT rank == lb -> easy; planted_d: SAT 8 > lb 6, not proven -> hard
    _write_refs(tmp, rows, {"cycle_5": (3, True), "gnp_n20_p0.5_s4": (4, False),
                            "planted_d_n30_r6_e0.0_s1": (8, False)}, 0.3)
    assert [r["name"] for r in ev.hard_graphs()] == ["planted_d_n30_r6_e0.0_s1"]
    assert ev.training_graphs() == ev.hard_graphs()
    monkeypatch.setenv("MINRANK_EVO_HARD", "all")
    assert [r["name"] for r in ev.hard_graphs()] == [r["name"] for r in rows]
    monkeypatch.setenv("MINRANK_EVO_HARD", "cycle_5, gnp_n20_p0.5_s4")
    assert [r["name"] for r in ev.hard_graphs()] == ["cycle_5", "gnp_n20_p0.5_s4"]
    monkeypatch.setenv("MINRANK_EVO_HARD", "all")
    monkeypatch.setenv("MINRANK_EVO_MIN_N", "20")
    assert [r["name"] for r in ev.hard_graphs()] == ["gnp_n20_p0.5_s4", "planted_d_n30_r6_e0.0_s1"]


def test_sat_target_scores_traces_and_metrics(sat_target):
    rows, _, tmp = sat_target
    sat = {"cycle_5": (4, False), "gnp_n20_p0.5_s4": (7, False), "planted_d_n30_r6_e0.0_s1": (6, False)}
    _write_refs(tmp, rows, sat, 0.3)
    rep = ev.run_evaluation(str(SEED), rows, budget=0.3, workers=3)
    recs = [json.loads(ln) for ln in (tmp / "traces" / "evals.jsonl").read_text().splitlines()]
    assert len(recs) == len(rows)
    for rec in recs:
        assert rec["status"] == "ok" and rec["target"] == "sat" and rec["ref_rank"] == sat[rec["graph"]][0]
        assert rec["score"] == pytest.approx((rec["ref_rank"] - rec["rank"]) / rec["ref_rank"])
    m = rep.metrics
    assert m["combined_score"] == pytest.approx(sum(r["score"] for r in recs) / len(recs))
    assert m["n_improved"] == sum(r["rank"] < r["ref_rank"] for r in recs)
    assert m["n_tied_ref"] == sum(r["rank"] == r["ref_rank"] for r in recs)
    assert m["n_below_cover"] == sum(r["rank"] < r["cover_rank"] for r in recs)
    assert any("sat=" in d for d in rep.details)


def test_sat_target_cache_does_not_leak_into_cover_mode(sat_target, monkeypatch):
    rows, _, tmp = sat_target
    _write_refs(tmp, rows, {r["name"]: (30, False) for r in rows}, 0.3)
    sat_rep = ev.run_evaluation(str(SEED), rows, budget=0.3, workers=3, trace=False)
    monkeypatch.setenv("MINRANK_EVO_TARGET", "cover")
    cov_rep = ev.run_evaluation(str(SEED), rows, budget=0.3, workers=3, trace=False)
    assert all("ref_rank" not in r for r in cov_rep.records)
    assert sat_rep.metrics["combined_score"] > cov_rep.metrics["combined_score"]


def test_sat_reference_computed_and_cached(tmp_path, small_bench):
    from minrank.evolve.reference_sat import ensure_sat_reference, sat_entry
    rows, _, _ = small_bench
    path = tmp_path / "sat_ref.json"
    cyc = [r for r in rows if r["name"] == "cycle_5"]
    ref = ensure_sat_reference(cyc, 3.0, 0, ev.graph_dir(), path, workers=1)
    e = sat_entry(ref, "cycle_5", 3.0, 0)
    assert e["sat_rank"] == 3 and e["sat_proven"] is True and e["digest"] == cyc[0]["digest"]
    before = path.read_text()
    ensure_sat_reference(cyc, 3.0, 0, ev.graph_dir(), path, workers=1)  # cached: nothing recomputed
    assert path.read_text() == before


def test_sat_descend_info_reports_proof():
    import numpy as np
    from minrank.exact.sat import sat_descend
    info: dict = {}
    rows = sat_descend(gen.cycle(5), 5.0, np.random.default_rng(0), info=info)
    assert verify(gen.cycle(5), rows) == 3 and info["proven"] is True
