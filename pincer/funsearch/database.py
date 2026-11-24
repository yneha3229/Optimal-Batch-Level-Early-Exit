"""FunSearch program database (Romera-Paredes et al., Nature 2024, Methods "Programs database").

- `num_islands` islands evolve independently. A new program goes to the island its prompt came from.
- Inside an island, programs are grouped into clusters by their score signature (per-graph scores, rounded).
- Prompt sampling: pick an island uniformly; pick `k` clusters with softmax(score / T), T decaying linearly from
  T0 to 0 over each period of `temp_period` registered programs; inside a cluster prefer shorter programs
  (softmax of negative normalised length). The k programs go into the prompt sorted by ascending score.
- Every `reset_every` registrations the worse half of the islands (by best score) is wiped and each is re-seeded
  with the best program of a random surviving island.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Program:
    code: str
    score: float
    signature: tuple
    island: int
    iteration: int
    parent_ids: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.code)


def _softmax(xs: list[float], temp: float) -> list[float]:
    temp = max(temp, 1e-6)
    m = max(xs)
    ws = [math.exp((x - m) / temp) for x in xs]
    s = sum(ws)
    return [w / s for w in ws]


class Island:
    def __init__(self) -> None:
        self.clusters: dict[tuple, list[Program]] = {}

    def register(self, p: Program) -> None:
        self.clusters.setdefault(p.signature, []).append(p)

    def best(self) -> Program | None:
        progs = [p for c in self.clusters.values() for p in c]
        return max(progs, key=lambda p: (p.score, -p.length)) if progs else None

    def size(self) -> int:
        return sum(len(c) for c in self.clusters.values())


class ProgramDatabase:
    def __init__(self, num_islands: int = 4, k: int = 2, t0: float = 0.1, temp_period: int = 300,
                 reset_every: int = 120, signature_digits: int = 3, seed: int = 0) -> None:
        self.islands = [Island() for _ in range(num_islands)]
        self.k, self.t0, self.temp_period, self.reset_every = k, t0, temp_period, reset_every
        self.digits = signature_digits
        self.rng = random.Random(seed)
        self.registered = 0
        self.all: list[Program] = []

    def signature(self, per_graph: list[float]) -> tuple:
        return tuple(round(x, self.digits) for x in per_graph)

    def register(self, code: str, score: float, per_graph: list[float], island: int | None, iteration: int,
                 parents: tuple = (), meta: dict | None = None) -> Program:
        if island is None:  # seed programs go to every island
            p = None
            for i, isl in enumerate(self.islands):
                p = Program(code, score, self.signature(per_graph), i, iteration, parents, meta or {})
                isl.register(p)
            self.all.append(p)
            return p
        p = Program(code, score, self.signature(per_graph), island, iteration, parents, meta or {})
        self.islands[island].register(p)
        self.all.append(p)
        self.registered += 1
        if self.reset_every and self.registered % self.reset_every == 0:
            self.reset_islands()
        return p

    def temperature(self) -> float:
        return self.t0 * (1 - (self.registered % self.temp_period) / self.temp_period)

    def sample(self) -> tuple[int, list[Program]]:
        i = self.rng.randrange(len(self.islands))
        isl = self.islands[i]
        sigs = list(isl.clusters)
        scores = [max(p.score for p in isl.clusters[s]) for s in sigs]
        probs = _softmax(scores, self.temperature())
        k = min(self.k, len(sigs))
        chosen: list[tuple] = []
        pool = list(range(len(sigs)))
        w = list(probs)
        for _ in range(k):
            j = self.rng.choices(pool, weights=w)[0]
            chosen.append(sigs[j])
            idx = pool.index(j)
            pool.pop(idx)
            w.pop(idx)
            if not pool:
                break
        progs = []
        for s in chosen:
            c = isl.clusters[s]
            lens = [p.length for p in c]
            lo, hi = min(lens), max(lens)
            norm = [-(x - lo) / (hi - lo + 1e-6) for x in lens]
            progs.append(self.rng.choices(c, weights=_softmax(norm, 1.0))[0])
        progs.sort(key=lambda p: p.score)
        return i, progs

    def reset_islands(self) -> None:
        order = sorted(range(len(self.islands)), key=lambda i: (self.islands[i].best().score
                                                                 if self.islands[i].best() else -1)
                       + self.rng.random() * 1e-9)
        half = len(order) // 2
        losers, keep = order[:half], order[half:]
        for i in losers:
            founder = self.islands[self.rng.choice(keep)].best()
            self.islands[i] = Island()
            if founder is not None:
                self.islands[i].register(Program(founder.code, founder.score, founder.signature, i,
                                                 founder.iteration, founder.parent_ids, founder.meta))

    def best(self) -> Program:
        return max(self.all, key=lambda p: (p.score, -p.length))
