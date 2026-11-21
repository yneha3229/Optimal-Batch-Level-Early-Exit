"""The fixed benchmark: instance list, splits, and materialisation to JSON files.

Splits (frozen after day 2; changing them makes results incomparable):
- random G(n, p): for each (n, p), seeds 0-3 train, 4 val, 5-7 test.
- structured families: all test (family-held-out generalisation check).

Usage:
  python -m minrank.bench.instances --out minrank/bench/graphs        # write every graph + index.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..core.graph import Graph
from . import generators as gen

ROOT = Path(__file__).resolve().parents[1]
SIZES = (20, 30, 50, 100)
DENSITIES = (0.3, 0.5, 0.7)
SPLIT_OF_SEED = {0: "train", 1: "train", 2: "train", 3: "train", 4: "val", 5: "test", 6: "test", 7: "test"}


@dataclass(frozen=True)
class Instance:
    name: str
    split: str
    build: Callable[[], Graph]


def instances(sizes: tuple[int, ...] = SIZES, densities: tuple[float, ...] = DENSITIES,
              directed: bool = False) -> list[Instance]:
    out: list[Instance] = []
    kind = "dnp" if directed else "gnp"
    for n in sizes:
        for p in densities:
            for seed, split in SPLIT_OF_SEED.items():
                out.append(Instance(f"{kind}_n{n}_p{p}_s{seed}", split,
                                    lambda n=n, p=p, s=seed: gen.gnp(n, p, s, directed)))
    if directed:
        return out
    structured: list[tuple[str, Callable[[], Graph]]] = [
        ("cycle_5", lambda: gen.cycle(5)), ("cycle_7", lambda: gen.cycle(7)),
        ("kneser_6_2", lambda: gen.kneser(6, 2)), ("kneser_7_2", lambda: gen.kneser(7, 2)),
        ("kneser_8_2", lambda: gen.kneser(8, 2)), ("kneser_8_3", lambda: gen.kneser(8, 3)),
        ("kneser_9_3", lambda: gen.kneser(9, 3)),
        ("paley_13", lambda: gen.paley(13)), ("paley_17", lambda: gen.paley(17)), ("paley_29", lambda: gen.paley(29)),
        ("paley_37", lambda: gen.paley(37)), ("paley_41", lambda: gen.paley(41)), ("paley_53", lambda: gen.paley(53)),
        ("paley_61", lambda: gen.paley(61)), ("paley_73", lambda: gen.paley(73)), ("paley_89", lambda: gen.paley(89)),
        ("paley_97", lambda: gen.paley(97)),
        ("mycielski_4", lambda: gen.mycielski(4)), ("mycielski_5", lambda: gen.mycielski(5)),
        ("mycielski_6", lambda: gen.mycielski(6)),
        ("cyclepow_21_2", lambda: gen.cycle_power(21, 2)), ("cyclepow_31_3", lambda: gen.cycle_power(31, 3)),
        ("cyclepow_50_4", lambda: gen.cycle_power(50, 4)),
        ("circ_41_1-5-8", lambda: gen.circulant(41, (1, 5, 8))), ("circ_61_1-3-11", lambda: gen.circulant(61, (1, 3, 11))),
    ]
    out += [Instance(name, "test", b) for name, b in structured]
    return out


def materialize(out_dir: Path, insts: list[Instance]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.csv"
    with index.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "split", "family", "n", "density", "digest", "path"])
        for inst in insts:
            g = inst.build()
            if g.name != inst.name:
                raise RuntimeError(f"instance name {inst.name} != graph name {g.name}")
            path = out_dir / f"{g.name}.json"
            g.save(path)
            w.writerow([g.name, inst.split, g.family, g.n, f"{g.density():.4f}", g.digest(), path.name])
    return index


def load_index(graph_dir: Path) -> list[dict]:
    with (graph_dir / "index.csv").open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "bench/graphs")
    ap.add_argument("--directed", action="store_true")
    a = ap.parse_args()
    idx = materialize(a.out, instances(directed=a.directed))
    print(f"wrote {idx}")


if __name__ == "__main__":
    main()
