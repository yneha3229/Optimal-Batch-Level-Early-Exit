"""Run one candidate program in its own process and print its metrics as JSON (called by evaluator.py).
Usage: python -I evolve/sandbox_eval.py <program.py> [split]"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from evolve.score import score_decide  # noqa: E402


def load_decide(path: str):
    spec = importlib.util.spec_from_file_location("candidate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.decide

    def decide(e, confs, n, B, left, a, b):
        mask, stop = raw(e, confs.copy(), n, B, left, a, b)
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != (n,):
            raise ValueError(f"mask has shape {mask.shape}, expected ({n},)")
        return mask, bool(stop)
    return decide


if __name__ == "__main__":
    split = sys.argv[2] if len(sys.argv) > 2 else "evolve"
    print(json.dumps(score_decide(load_decide(sys.argv[1]), split)))
