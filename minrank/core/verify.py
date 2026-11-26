"""Fit check, rank verification and certificate files.

Every reported upper bound must pass `verify`. CLI:
  python -m minrank.core.verify GRAPH.json CERT.json
prints "fits: yes, rank: r" or exits with an error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .f2 import rank_f2
from .graph import Graph


def fit_error(g: Graph, rows: Sequence[int]) -> str | None:
    """None if rows fit g, else a description of the first violation."""
    if len(rows) != g.n:
        return f"matrix has {len(rows)} rows, graph has {g.n} vertices"
    for i, row in enumerate(rows):
        if row < 0 or row >> g.n:
            return f"row {i} has bits outside 0..n-1"
        if not (row >> i) & 1:
            return f"A[{i},{i}] = 0"
        bad = row & g.forbidden(i)
        if bad:
            j = (bad & -bad).bit_length() - 1
            return f"A[{i},{j}] = 1 but {j} is not in S_{i}"
    return None


def verify(g: Graph, rows: Sequence[int], claimed_rank: int | None = None) -> int:
    """Return rank_F2(rows) after checking the fit; raise ValueError on any violation or a wrong claimed rank."""
    err = fit_error(g, rows)
    if err:
        raise ValueError(f"{g.name}: matrix does not fit: {err}")
    r = rank_f2(rows)
    if claimed_rank is not None and r != claimed_rank:
        raise ValueError(f"{g.name}: claimed rank {claimed_rank}, actual {r}")
    return r


def save_certificate(path: str | Path, g: Graph, rows: Sequence[int], method: str, extra: dict | None = None) -> int:
    """Verify and write {graph digest, rank, rows in hex}. Returns the rank."""
    r = verify(g, rows)
    payload = {"graph": g.name, "digest": g.digest(), "n": g.n, "rank": r, "method": method,
               "rows": [f"{x:x}" for x in rows], **(extra or {})}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload))
    return r


def load_certificate(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    d["rows"] = [int(x, 16) for x in d["rows"]]
    return d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("graph")
    ap.add_argument("cert")
    a = ap.parse_args(argv)
    g = Graph.load(a.graph)
    c = load_certificate(a.cert)
    if c["digest"] != g.digest():
        print(f"digest mismatch: certificate {c['digest']}, graph {g.digest()}", file=sys.stderr)
        return 1
    r = verify(g, c["rows"], c.get("rank"))
    print(f"fits: yes, rank: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
