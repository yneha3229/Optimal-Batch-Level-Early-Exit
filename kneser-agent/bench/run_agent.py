"""Run the agent on the benchmark, with and without the family rule.
    python bench/run_agent.py [--budget 120] [--llm none|local|"local:URL|MODEL"] [--specs K(16,4) ...]"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
args = sys.argv[1:]
get = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
budget, llm = get("--budget", "120"), get("--llm", "none")
specs = args[args.index("--specs") + 1:] if "--specs" in args else json.loads((HERE / "specs.json").read_text())["benchmark"]
for spec in specs:
    for extra in ([], ["--no-family-rule", "--tag", "nofam"]):
        p = subprocess.run([sys.executable, "-m", "kagent.pipeline", spec, "--llm", llm, "--budget", budget, *extra],
                           cwd=HERE.parent, capture_output=True, text=True, timeout=float(budget) * 3 + 600)
        print(spec, extra[:1], (p.stdout.strip().splitlines() or [p.stderr[-200:]])[-1][:200], flush=True)
