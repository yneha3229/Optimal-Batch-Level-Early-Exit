"""Run the pipeline on one graph file.

    python -m kagent.run bench/graphs/<id> --mode full|no_critic|no_llm|small_model --budget 300 --cores 48-55

Writes results/runs/<mode>/<id>.json, traces/<mode>/<id>.jsonl, certificates in results/certs/<mode>/<id>/ and, for
a proven graph, the explicit code in results/codes/<id>.json. The ground truth is read only after the Solver has
finished (for the Verifier's report)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_cores(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def run_one(graph: str, mode: str, budget: float, cores: list[int], model: str | None = None, url: str | None = None,
            seed: int = 0, tag: str | None = None, verify: bool = True, listener=None) -> dict:
    from .llm import DEFAULT_URL, LLM, MODELS
    from .orchestrator import Run
    from .verify.stage import run_stage

    gpath = Path(graph)
    if not gpath.is_absolute():
        gpath = ROOT / gpath
    gid = gpath.stem
    tag = tag or mode
    llm = None
    if mode in ("full", "no_critic", "small_model"):
        m = model or (MODELS["qwen3_4b"] if mode == "small_model" else MODELS["qwen32b"])
        llm = LLM(model=m, url=url or DEFAULT_URL)
    cert_dir = ROOT / "results" / "certs" / tag / gid
    trace_path = ROOT / "traces" / tag / f"{gid}.jsonl"
    run = Run(gpath, mode=mode, llm=llm, budget=budget, cores=cores, trace_path=trace_path, seed=seed,
              cert_dir=cert_dir)
    if listener is not None:
        run.trace.listeners.append(listener)
    res = run.run()
    run.save_certificates(cert_dir)
    if verify:
        truth_all = json.loads((ROOT / "bench" / "truth.json").read_text()) if (ROOT / "bench" / "truth.json").exists() \
            else {}
        truth = truth_all.get(gid)                                   # read only now: the Solver has finished
        truth_pub = None if truth is None else {k: truth[k] for k in ("name", "family", "group") if k in truth}
        bl_path = ROOT / "results" / "baselines" / f"{gid}.json"
        baselines = json.loads(bl_path.read_text()) if bl_path.exists() else None
        t0 = time.perf_counter()
        vs = run_stage(run.A, cert_dir, res, llm=llm, trace=run.trace, baselines=baselines, truth=truth_pub,
                       code_path=ROOT / "results" / "codes" / f"{gid}.json" if mode == "full" else None)
        vs["verify_wall"] = round(time.perf_counter() - t0, 2)
        res["verification"] = vs
        res["truth"] = truth_pub
        res["llm"] = llm.usage() if llm else None
    run.trace.close()
    res.pop("describe", None)
    out = ROOT / "results" / "runs" / tag / f"{gid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, default=str, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("--mode", default="no_llm", choices=["full", "no_critic", "no_llm", "small_model"])
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--cores", default="0-7")
    ap.add_argument("--model", default=None)
    ap.add_argument("--url", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args(argv)
    cores = parse_cores(a.cores)
    os.sched_setaffinity(0, cores)
    res = run_one(a.graph, a.mode, a.budget, cores, a.model, a.url, a.seed, a.tag)
    v = res.get("verification", {})
    print(json.dumps({k: res[k] for k in ("graph", "mode", "L", "U", "proven", "field", "t_first_bound", "t_proof",
                                          "wall", "stopped")} | {"verdict": v.get("verdict"), "llm": res.get("llm")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
