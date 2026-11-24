"""Side-information digraphs, strongly connected components, and the block-length-t blow-up.

Receivers 0..n-1; receiver i wants x_i. out[i] is a bitmask: bit j set <=> receiver i already has x_j (arc i -> j).

Blow-up. A vector-linear code with block length t on G (every message split into t symbols over the field) is the
same thing as a scalar-linear code on G^(t): receiver i becomes t sub-receivers (i, 0..t-1), and sub-receiver
(i, a) has symbol (j, b) exactly when receiver i has message j in G. A fitting matrix of G^(t) with rank r is a
vector-linear code of rate r / t, so lambda_F(G) <= minrk_F(G^(t)) / t, and lambda_F is the infimum over t.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def bits(m: int):
    while m:
        low = m & -m
        yield low.bit_length() - 1
        m ^= low


@dataclass(frozen=True)
class Digraph:
    n: int
    out: tuple[int, ...]
    name: str = ""

    @staticmethod
    def from_arcs(n: int, arcs, name: str = "") -> "Digraph":
        out = [0] * n
        for i, j in arcs:
            if i != j:
                out[i] |= 1 << j
        return Digraph(n, tuple(out), name)

    def arcs(self) -> list[tuple[int, int]]:
        return [(i, j) for i in range(self.n) for j in bits(self.out[i])]

    def knows(self, i: int, j: int) -> bool:
        return bool((self.out[i] >> j) & 1)

    def mutual(self, i: int, j: int) -> bool:
        return self.knows(i, j) and self.knows(j, i)

    def to_json(self) -> dict:
        return {"name": self.name, "n": self.n, "arcs": self.arcs()}

    @staticmethod
    def from_json(d: dict) -> "Digraph":
        return Digraph.from_arcs(d["n"], [tuple(a) for a in d["arcs"]], d.get("name", ""))


def induced(g: Digraph, verts: list[int], name: str = "") -> Digraph:
    pos = {v: k for k, v in enumerate(verts)}
    out = []
    for v in verts:
        m = 0
        for u in bits(g.out[v]):
            if u in pos:
                m |= 1 << pos[u]
        out.append(m)
    return Digraph(len(verts), tuple(out), name or g.name)


def scc(g: Digraph) -> list[list[int]]:
    """Strongly connected components (iterative Tarjan), largest first."""
    n = g.n
    index, low, on, stack, comps, counter = [-1] * n, [0] * n, [False] * n, [], [], 0
    succ = [list(bits(g.out[v])) for v in range(n)]
    for root in range(n):
        if index[root] != -1:
            continue
        work = [(root, 0)]
        while work:
            v, pi = work[-1]
            if pi == 0:
                index[v] = low[v] = counter
                counter += 1
                stack.append(v)
                on[v] = True
            if pi < len(succ[v]):
                work[-1] = (v, pi + 1)
                w = succ[v][pi]
                if index[w] == -1:
                    work.append((w, 0))
                elif on[w]:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on[w] = False
                    comp.append(w)
                    if w == v:
                        break
                comps.append(sorted(comp))
    return sorted(comps, key=lambda c: (-len(c), c[0]))


def blowup(g: Digraph, t: int) -> Digraph:
    """G^(t): sub-receiver i*t + a has symbol j*t + b iff receiver i has message j in G."""
    full = (1 << t) - 1
    out = []
    for i in range(g.n):
        m = 0
        for j in bits(g.out[i]):
            m |= full << (j * t)
        out.extend([m] * t)
    return Digraph(g.n * t, tuple(out), f"{g.name}^({t})")


def load_suite(path: str | Path) -> list[Digraph]:
    return [Digraph.from_json(d) for d in json.loads(Path(path).read_text())]
