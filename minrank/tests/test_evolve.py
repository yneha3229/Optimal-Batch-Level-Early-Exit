"""Tests for Option A: seed program, sandbox runner, evaluator, traces, held-out CSV.
Run: python -m pytest minrank/tests/test_evolve.py -q
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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
