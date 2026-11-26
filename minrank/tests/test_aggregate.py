"""End-to-end check of the aggregation on a 2-graph benchmark (fast)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from minrank.analysis.aggregate import family_group, summarize
from minrank.bench import generators as gen
from minrank.bench.instances import Instance, materialize

ROOT = Path(__file__).resolve().parents[2]


def test_family_group():
    assert family_group("gnp") == "random" and family_group("dnp") == "random"
    assert family_group("planted_d") == "planted" and family_group("orthogonality") == "planted"
    assert family_group("kneser") == "structured"


@pytest.fixture()
def mini(tmp_path):
    g1, g2 = gen.cycle(5), gen.orthogonality(5)
    materialize(tmp_path / "graphs", [Instance(g1.name, "test", lambda: g1), Instance(g2.name, "test", lambda: g2)])
    common = ["--graphs", str(tmp_path / "graphs"), "--split", "test", "--workers", "1"]
    out = ["--out", str(tmp_path / "res" / "baselines.csv"), "--certs", str(tmp_path / "certs")]
    run = lambda *a: subprocess.run([sys.executable, "-m", "minrank.run_baselines", *common, *out, *a],  # noqa: E731
                                    cwd=ROOT, check=True, capture_output=True, text=True)
    run("--methods", "dsatur", "random_greedy", "sat", "--budgets", "1", "--seeds", "0")
    run("--lower-only", "--lb-time", "5")
    return tmp_path


def test_summarize_end_to_end(mini):
    graphs, methods = summarize(mini / "res", mini / "graphs", mini / "certs")
    g = graphs.set_index("graph")
    assert g.loc["cycle_5", "best_ub"] == 3 and g.loc["cycle_5", "lb"] == 2 and g.loc["cycle_5", "gap"] == 1
    assert g.loc["orth_5", "best_ub"] == 5 and bool(g.loc["orth_5", "proven_optimal"])  # lb 5 = ub 5
    assert g["cert_ok"].all()
    assert set(methods["method"]) == {"dsatur", "random_greedy", "sat"}
    assert (methods["mean_ratio_to_best"] >= 1.0).all()


def test_tampered_certificate_is_flagged(mini):
    p = mini / "certs" / "cycle_5.json"
    d = json.loads(p.read_text())
    d["rows"] = ["1"] * 5  # does not fit: rows 1..4 lack their diagonal
    p.write_text(json.dumps(d))
    graphs, _ = summarize(mini / "res", mini / "graphs", mini / "certs")
    assert not bool(graphs.set_index("graph").loc["cycle_5", "cert_ok"])
    # the best upper bound falls back to the sweep, never to the bad certificate
    assert graphs.set_index("graph").loc["cycle_5", "best_ub"] == 3


def test_cli_fails_loudly_on_bad_certificate(mini):
    p = mini / "certs" / "orth_5.json"
    d = json.loads(p.read_text())
    d["digest"] = "0" * 16
    p.write_text(json.dumps(d))
    r = subprocess.run([sys.executable, "-m", "minrank.analysis.aggregate", "--results", str(mini / "res"),
                        "--graphs", str(mini / "graphs"), "--certs", str(mini / "certs")],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "do not verify" in (r.stdout + r.stderr)
    assert pd.read_csv(mini / "res" / "summary_graphs.csv").shape[0] == 2


def test_evolved_rows_and_failures_are_merged(mini):
    ev = mini / "res" / "evolved.csv"
    ev.write_text("graph,split,family,n,density,method,budget,seed,rank,seconds,verified\n"
                  "cycle_5,test,cycle,5,0.5,evolved:abc,1.0,0,3,0.5,True\n"
                  "orth_5,test,orthogonality,16,0.5,evolved:abc,1.0,0,,0.5,False\n")
    graphs, methods = summarize(mini / "res", mini / "graphs", mini / "certs")
    row = methods[methods["method"] == "evolved:abc"]
    assert set(row["group"]) == {"structured"}  # the failed orth_5 run contributes no rank
    assert float(row["failure_rate"].iloc[0]) == 0.5
    assert graphs.set_index("graph").loc["orth_5", "best_ub"] == 5  # unaffected by the failed run
