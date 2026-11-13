"""OpenEvolve evaluator: evaluate(program_path) -> metrics dict with combined_score.

Guardrails: the candidate runs in a separate isolated Python process with a timeout, and a static check rejects
programs that try to read files, import anything but numpy/math, or reach the interpreter (so they cannot read the
trace labels).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"\bopen\s*\(", r"parquet", r"\bos\b", r"\bsys\b", r"subprocess", r"__import__", r"\beval\s*\(",
             r"\bexec\s*\(", r"importlib", r"pickle", r"pandas", r"socket", r"globals\s*\(", r"builtins"]
ALLOWED_IMPORTS = {"numpy", "math"}


def static_check(code: str) -> str | None:
    for pat in FORBIDDEN:
        if re.search(pat, code):
            return f"forbidden pattern: {pat}"
    for m in re.finditer(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", code, re.M):
        mod = m.group(1) or m.group(2)
        if mod not in ALLOWED_IMPORTS:
            return f"import not allowed: {mod}"
    if "def decide" not in code:
        return "no decide() function"
    return None


def evaluate(program_path: str, split: str = "evolve", timeout: int = 120) -> dict:
    code = Path(program_path).read_text()
    err = static_check(code)
    if err:
        return {"combined_score": 0.0, "error": err}
    try:
        p = subprocess.run([sys.executable, "-I", str(ROOT / "evolve/sandbox_eval.py"), program_path, split],
                           capture_output=True, text=True, timeout=timeout, cwd=str(ROOT))
    except subprocess.TimeoutExpired:
        return {"combined_score": 0.0, "error": "timeout"}
    if p.returncode != 0:
        return {"combined_score": 0.0, "error": p.stderr.strip().splitlines()[-1][:300] if p.stderr else "crash"}
    return json.loads(p.stdout.strip().splitlines()[-1])
