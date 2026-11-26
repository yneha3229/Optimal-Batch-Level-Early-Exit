"""Non-LLM baseline for the FunSearch loop: genetic programming over priority expressions.

Recent studies of LLM-evolved heuristics (EoH, ReEvo follow-ups; and our own first OpenEvolve run, where a tuned
random search tied the LLM) find that simple non-LLM search can match LLM proposals, so every LLM run must be
compared with this at the same number of evaluated programs.

A program is an expression tree over per-vertex features; it is stored in the code itself as a `# expr:` comment
so the GP operators can read it back from any program in the database. Programs without that comment (seed or
LLM-written ones) are only used as crossover donors via the default tree.
Features (all cheap bit counts on the Ctx of sandbox_runner.py):
  i, o, m  : in-, out-, mutual degree among undecided vertices
  ki, ko   : in-/out-neighbours already kept in the acyclic set
  a, s, n  : number of undecided vertices, deletion step, graph size
  oi, io   : sum over undecided out-neighbours of their in-degree / over in-neighbours of their out-degree
"""
from __future__ import annotations

import ast
import random

TERMINALS = ["i", "o", "m", "ki", "ko", "a", "s", "n", "oi", "io"]
CONSTS = [0.5, 1.0, 2.0, 3.0]
BINARY = ["+", "-", "*", "/", "max", "min"]
UNARY = ["log", "sqrt", "sq"]
MAX_NODES = 25

PRELUDE = '''import math


def priority(v, ctx):
    al = ctx.alive
    i = ctx.in_deg(v)
    o = ctx.out_deg(v)
    m = ctx.mutual_deg(v)
    ki = (ctx.inn[v] & ctx.kept).bit_count()
    ko = (ctx.out[v] & ctx.kept).bit_count()
    a = al.bit_count()
    s = ctx.step
    n = ctx.n
    oi = 0
    x = ctx.out[v] & al
    while x:
        low = x & -x
        oi += ctx.in_deg(low.bit_length() - 1)
        x ^= low
    io = 0
    x = ctx.inn[v] & al
    while x:
        low = x & -x
        io += ctx.out_deg(low.bit_length() - 1)
        x ^= low
    return {expr}
'''


def to_code(t) -> str:
    if isinstance(t, str):
        return t
    if isinstance(t, float):
        return repr(t)
    op = t[0]
    if op in ("+", "-", "*"):
        return f"({to_code(t[1])} {op} {to_code(t[2])})"
    if op == "/":
        return f"({to_code(t[1])} / (abs({to_code(t[2])}) + 1.0))"
    if op in ("max", "min"):
        return f"{op}({to_code(t[1])}, {to_code(t[2])})"
    if op == "log":
        return f"math.log1p(abs({to_code(t[1])}))"
    if op == "sqrt":
        return f"math.sqrt(abs({to_code(t[1])}))"
    if op == "sq":
        return f"({to_code(t[1])}) ** 2"
    raise ValueError(op)


SA_TERMINALS = ["delta", "ci", "co", "ik", "ok", "ideg", "odeg", "kept", "n", "temp", "step"]
SA_PRELUDE = '''import math


def score(f):
    delta, ci, co, ik, ok, ideg, odeg, kept, n, temp, step = f
    return {expr}
'''
TARGET_TERMINALS = {"greedy": TERMINALS, "sa": SA_TERMINALS}
DEFAULT_TREES = {"greedy": ("*", "i", "o"), "sa": ("-", 0.5, "delta")}


def program(t, target: str = "greedy") -> str:
    pre = SA_PRELUDE if target == "sa" else PRELUDE
    return f"# expr: {t!r}\n" + pre.format(expr=to_code(t))


def parse(code: str):
    for line in code.splitlines():
        if line.startswith("# expr:"):
            return ast.literal_eval(line[len("# expr:"):].strip())
    return None


def size(t) -> int:
    return 1 if not isinstance(t, tuple) else 1 + sum(size(c) for c in t[1:])


def random_tree(rng: random.Random, depth: int, terminals: list[str] | None = None):
    terminals = terminals or TERMINALS
    if depth <= 0 or rng.random() < 0.3:
        return rng.choice(terminals) if rng.random() < 0.85 else rng.choice(CONSTS)
    if rng.random() < 0.25:
        return (rng.choice(UNARY), random_tree(rng, depth - 1, terminals))
    return (rng.choice(BINARY), random_tree(rng, depth - 1, terminals), random_tree(rng, depth - 1, terminals))


def _paths(t, path=()):
    yield path
    if isinstance(t, tuple):
        for k, c in enumerate(t[1:], start=1):
            yield from _paths(c, path + (k,))


def _get(t, path):
    for k in path:
        t = t[k]
    return t


def _set(t, path, new):
    if not path:
        return new
    k = path[0]
    lst = list(t)
    lst[k] = _set(t[k], path[1:], new)
    return tuple(lst)


def mutate(t, rng: random.Random, terminals: list[str] | None = None):
    p = rng.choice(list(_paths(t)))
    r = rng.random()
    sub = _get(t, p)
    if r < 0.5:
        new = random_tree(rng, 2, terminals)
    elif r < 0.8 and isinstance(sub, tuple):  # point mutation of the operator
        ops = UNARY if sub[0] in UNARY else BINARY
        new = (rng.choice(ops),) + sub[1:]
    else:  # wrap
        new = (rng.choice(BINARY), sub, random_tree(rng, 1, terminals))
    return _set(t, p, new)


def crossover(a, b, rng: random.Random):
    pa = rng.choice(list(_paths(a)))
    pb = rng.choice(list(_paths(b)))
    return _set(a, pa, _get(b, pb))


DEFAULT_TREE = ("*", "i", "o")


class GPProposer:
    """Drop-in for an LLM sampler: propose(parents) -> program code."""

    name = "gp"

    def __init__(self, seed: int = 0, target: str = "greedy") -> None:
        self.rng = random.Random(seed)
        self.target = target
        self.terms = TARGET_TERMINALS[target]

    def propose(self, parents: list) -> str:
        trees = [parse(p.code) or DEFAULT_TREES[self.target] for p in parents]
        for _ in range(50):
            if len(trees) >= 2 and self.rng.random() < 0.4:
                t = crossover(trees[-1], trees[0], self.rng)
                if self.rng.random() < 0.5:
                    t = mutate(t, self.rng, self.terms)
            else:
                t = mutate(trees[-1], self.rng, self.terms)
            if size(t) <= MAX_NODES and t not in trees:
                return program(t, self.target)
        return program(random_tree(self.rng, 3, self.terms), self.target)
