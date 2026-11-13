"""Merge per-batch-size GPU result files (main_<task>_<regime>_gpu*.csv) into main_<task>_<regime>_gpu.csv."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
task, reg = sys.argv[1], sys.argv[2]
files = sorted((ROOT / "results").glob(f"main_{task}_{reg}_gpu*.csv"))
df = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["B", "family"], keep="last").sort_values(["B", "family"])
df.to_csv(ROOT / f"results/main_{task}_{reg}_gpu.csv", index=False)
print(f"merged {[f.name for f in files]} -> {len(df)} rows, batch sizes {sorted(df.B.unique())}")
