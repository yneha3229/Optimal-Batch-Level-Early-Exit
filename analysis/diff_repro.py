"""Compare regenerated result files (results_repro/) with the committed ones (results/).

Usage: python analysis/diff_repro.py --repro results_repro/.root/results --ref results --files F1 F2 ... \
           --out results_repro/diff_summary.csv --md results_repro/DIFF.md

Tolerances (stated in docs/REPRODUCE_EARLY_EXIT.md):
  MATCH     every numeric value within atol 1e-9 + rtol 1e-6 of the committed value, every other value equal
            (CSV: same rows after sorting on the non-numeric columns; JSON: same structure)
  CLOSE     every numeric value within atol 1e-4 + rtol 1e-3 (float summation order, library versions)
  MISMATCH  anything larger; the largest difference and the column/key where it occurs are reported
  MISSING   the step did not write the file (it failed; see its log in results_repro/logs/)
Images (PNG) and the report HTML are compared by their bytes / their text and reported as IDENTICAL or DIFFERS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

TIGHT, LOOSE = (1e-9, 1e-6), (1e-4, 1e-3)


def _close(a: float, b: float, tol) -> bool:
    if (a is None or (isinstance(a, float) and math.isnan(a))) and (b is None or (isinstance(b, float) and math.isnan(b))):
        return True
    return abs(a - b) <= tol[0] + tol[1] * abs(b)


def cmp_csv(p: Path, q: Path) -> tuple[str, str]:
    a, b = pd.read_csv(p), pd.read_csv(q)
    if list(a.columns) != list(b.columns):
        return "MISMATCH", f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    if len(a) != len(b):
        return "MISMATCH", f"{len(a)} rows vs {len(b)} committed"
    num = [c for c in a.columns if pd.api.types.is_numeric_dtype(b[c]) and pd.api.types.is_numeric_dtype(a[c])]
    key = [c for c in a.columns if c not in num]
    if key:
        a = a.sort_values(key + num, kind="mergesort").reset_index(drop=True)
        b = b.sort_values(key + num, kind="mergesort").reset_index(drop=True)
        s = lambda col: col.astype(object).where(col.notna(), "<NA>").astype(str).to_numpy()   # pandas 3: NaN stays NaN
        bad = [c for c in key if (s(a[c]) != s(b[c])).any()]
        if bad:
            c = bad[0]
            i = int(np.flatnonzero(s(a[c]) != s(b[c]))[0])
            na = f" ({int(a[c].isna().sum())} empty vs {int(b[c].isna().sum())} committed)" if a[c].isna().sum() != b[c].isna().sum() else ""
            return "MISMATCH", f"column {c} row {i}: {s(a[c])[i]!r} vs {s(b[c])[i]!r} committed{na}"
    worst, where = 0.0, ""
    for c in num:
        x, y = a[c].to_numpy(float), b[c].to_numpy(float)
        both_nan = np.isnan(x) & np.isnan(y)
        if (np.isnan(x) ^ np.isnan(y)).any():
            return "MISMATCH", f"column {c}: NaN pattern differs"
        d = np.where(both_nan, 0.0, np.abs(x - y) / (1.0 + np.abs(y)))
        if d.size and d.max() > worst:
            i = int(d.argmax())
            worst, where = float(d.max()), f"column {c} row {i}: {x[i]:.6g} vs {y[i]:.6g} committed"
    for tol, lab in ((TIGHT, "MATCH"), (LOOSE, "CLOSE")):
        ok = all(np.all(np.isnan(a[c].to_numpy(float)) | (np.abs(a[c].to_numpy(float) - b[c].to_numpy(float))
                                                             <= tol[0] + tol[1] * np.abs(b[c].to_numpy(float))))
                 for c in num)
        if ok:
            return lab, ("" if lab == "MATCH" else f"largest difference {where}")
    return "MISMATCH", f"largest relative difference {worst:.3g}: {where}"


def _walk(x, y, path=""):
    """Yield (path, a, b) for every leaf; structural differences as (path, 'STRUCT', detail)."""
    if isinstance(x, dict) and isinstance(y, dict):
        if set(x) != set(y):
            yield path, "STRUCT", f"keys differ: {sorted(set(x) ^ set(y))[:5]}"
        for k in sorted(set(x) & set(y)):
            yield from _walk(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            yield path, "STRUCT", f"length {len(x)} vs {len(y)} committed"
        for i, (u, v) in enumerate(zip(x, y)):
            yield from _walk(u, v, f"{path}[{i}]")
    else:
        yield path, x, y


def cmp_json(p: Path, q: Path) -> tuple[str, str]:
    a, b = json.loads(p.read_text()), json.loads(q.read_text())
    state, msg, worst = "MATCH", "", 0.0
    for path, u, v in _walk(a, b):
        if u == "STRUCT":
            return "MISMATCH", f"{path}: {v}"
        if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool):
            if _close(u, v, TIGHT):
                continue
            rel = abs(u - v) / (1 + abs(v))
            if _close(u, v, LOOSE):
                if state == "MATCH":
                    state, msg = "CLOSE", f"{path}: {u} vs {v} committed"
                continue
            if rel > worst:
                worst, msg = rel, f"{path}: {u} vs {v} committed"
            state = "MISMATCH"
        elif u != v:
            return "MISMATCH", f"{path}: {u!r} vs {v!r} committed"
    return state, msg


def cmp_file(p: Path, q: Path) -> tuple[str, str]:
    if not p.exists():
        return "MISSING", "not regenerated (step failed or skipped)"
    if not q.exists():
        return "NO REFERENCE", "no committed file to compare with"
    if p.suffix == ".csv":
        return cmp_csv(p, q)
    if p.suffix == ".json":
        return cmp_json(p, q)
    if p.suffix == ".html":
        strip = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()
        x, y = strip(p.read_text()), strip(q.read_text())
        if x == y:
            return "IDENTICAL", "same text"
        nx, ny = re.findall(r"\d+(?:\.\d+)?", x), re.findall(r"\d+(?:\.\d+)?", y)
        diff = [(i, u, v) for i, (u, v) in enumerate(zip(nx, ny)) if u != v]
        return "DIFFERS", f"{len(nx)} vs {len(ny)} numbers in the text; first differing: {diff[:5]}"
    h = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    return ("IDENTICAL", "same bytes") if h(p) == h(q) else ("DIFFERS", "bytes differ (rendering); check visually")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", required=True, help="directory with the regenerated files")
    ap.add_argument("--ref", required=True, help="directory with the committed files")
    ap.add_argument("--files", nargs="+", required=True, help="'group:relative/path' entries")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    rows = []
    for item in a.files:
        group, f = item.split(":", 1)
        st, msg = cmp_file(Path(a.repro) / f, Path(a.ref) / f)
        rows.append(dict(group=group, file=f, status=st, detail=msg))
    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    counts = d.status.value_counts().to_dict()
    md = ["# reproduce_cpu.sh: regenerated files against the committed ones", "",
          "Tolerances: MATCH = atol 1e-9 + rtol 1e-6 on every number; CLOSE = atol 1e-4 + rtol 1e-3; "
          "MISMATCH = larger; MISSING = the step failed (see results_repro/logs/).", "",
          "Totals: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())), "",
          "| group | file | status | detail |", "|---|---|---|---|"]
    md += [f"| {r.group} | `{r.file}` | {r.status} | {str(r.detail).replace('|', '/')} |" for r in d.itertuples()]
    Path(a.md).write_text("\n".join(md) + "\n")
    print(d.status.value_counts().to_string())


if __name__ == "__main__":
    main()
