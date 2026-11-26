"""Run the agent (no LLM) on the benchmark: generic rules + family rule, and generic rules only.
    python bench/run_agent.py [--budget 120]"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
budget = sys.argv[sys.argv.index("--budget") + 1] if "--budget" in sys.argv else "120"
for spec in json.loads((HERE / "specs.json").read_text())["benchmark"]:
    for extra in ([], ["--no-family-rule", "--tag", "nofam"]):
        p = subprocess.run([sys.executable, "-m", "kagent.pipeline", spec, "--llm", "none", "--budget", budget, *extra],
                           cwd=HERE.parent, capture_output=True, text=True, timeout=float(budget) * 3 + 120)
        print(spec, extra[:1], (p.stdout.strip().splitlines() or ["?"])[-1][:160], flush=True)
