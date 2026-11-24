"""Linear-programming bounds on the broadcast rate beta, each with an exactly checkable rational certificate.

1. Shannon (polymatroid) lower bound, for small components (n <= ~12).
   Take any valid index code with t-bit messages and broadcast Y, and let g(S) = H(Y, X_S) / t for S subset of V.
   Every such g satisfies (Blasiak, Kleinberg, Lubetzky, arXiv:1004.1379, Sec. 3; the "polymatroid bound"):
     g(V) = n                                       (Y is a function of all messages)
     g(N(i) + i) = g(N(i))                          (receiver i decodes x_i from Y and its side information)
     g(K + i) + g(K + j) >= g(K + i + j) + g(K)     (submodularity, elemental form, i != j not in K)
     g(K + x) <= g(K) + 1                           (H(X_x | Y, X_K) <= H(X_x) = 1)
     |K| <= g(K) <= n                               (H(Y, X_K) >= H(X_K) = |K|, monotone)
   and beta >= g(empty) = H(Y) / t. So min g(empty) over this polytope is a lower bound on beta. These are all
   Shannon inequalities on (Y, X_1..X_n) once the message-only entropies (= |K|, independent uniform messages) are
   substituted.
   Certificate: Lagrange multipliers (y >= 0 on inequalities, mu free on equalities), rationalised. For ANY such
   multipliers, weak duality gives the bound
     g(empty) >= -y.b_ub - mu.b_eq + sum_k min(d_k l_k, d_k u_k),  d = c + A_ub^T y + A_eq^T mu,
   computed here in exact rational arithmetic, so float error in the LP solver cannot make the bound wrong.

2. Fractional clique cover upper bound (time sharing over cliques of the mutual graph), any n.
   Weights w_C >= 0 on cliques with sum_{C containing v} w_C >= 1 for all v give a vector-linear code of rate
   sum w_C (Blasiak et al.). Column generation with a max-weight-clique pricing step (CP-SAT). Certificate: the
   cliques and rational weights; the checker tests every clique and every coverage sum exactly.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from minrank.core.graph import Graph, iter_bits

SHANNON_MAX_N = 12


# ---------------------------------------------------------------------------------------------- Shannon bound
@dataclass
class ShannonLP:
    n: int
    c: np.ndarray
    A_ub: list[dict[int, int]]  # sparse rows, integer coefficients; constraint: row . g <= b
    b_ub: list[int]
    A_eq: list[dict[int, int]]
    b_eq: list[int]
    lb: list[int]
    ub: list[int]


def build_shannon(g: Graph) -> ShannonLP:
    n = g.n
    if n > SHANNON_MAX_N:
        raise ValueError(f"Shannon LP has 2^{n} variables; limit is n <= {SHANNON_MAX_N}")
    N = 1 << n
    full = N - 1
    A_ub: list[dict[int, int]] = []
    b_ub: list[int] = []
    for K in range(N):
        free = full & ~K
        for i in iter_bits(free):
            # g(K + i) - g(K) <= 1
            A_ub.append({K | 1 << i: 1, K: -1})
            b_ub.append(1)
            for j in iter_bits(free >> (i + 1) << (i + 1)):
                # g(K+i+j) + g(K) - g(K+i) - g(K+j) <= 0
                A_ub.append({K | 1 << i | 1 << j: 1, K: 1, K | 1 << i: -1, K | 1 << j: -1})
                b_ub.append(0)
    for i in range(n):
        A_ub.append({full & ~(1 << i): 1, full: -1})  # g(V - i) <= g(V)
        b_ub.append(0)
    A_eq = [{full: 1}]
    b_eq = [n]
    for i in range(n):
        Ni = g.out[i]
        A_eq.append({Ni | 1 << i: 1, Ni: -1})
        b_eq.append(0)
    c = np.zeros(N)
    c[0] = 1.0
    lb = [K.bit_count() for K in range(N)]
    ub = [n] * N
    return ShannonLP(n, c, A_ub, b_ub, A_eq, b_eq, lb, ub)


def _sparse(rows: list[dict[int, int]], ncols: int):
    r, cidx, v = [], [], []
    for k, row in enumerate(rows):
        for j, a in row.items():
            r.append(k)
            cidx.append(j)
            v.append(a)
    return coo_matrix((v, (r, cidx)), shape=(len(rows), ncols)).tocsr()


def certified_bound(lp: ShannonLP, y: list[Fraction], mu: list[Fraction]) -> Fraction:
    """Exact weak-duality bound for multipliers y >= 0 (inequalities) and mu (equalities). See module docstring."""
    if any(v < 0 for v in y):
        raise ValueError("inequality multipliers must be >= 0")
    N = 1 << lp.n
    d = [Fraction(int(x)) for x in lp.c]
    const = Fraction(0)
    for row, b, yk in zip(lp.A_ub, lp.b_ub, y):
        if yk:
            const -= yk * b
            for j, a in row.items():
                d[j] += yk * a
    for row, b, mk in zip(lp.A_eq, lp.b_eq, mu):
        if mk:
            const -= mk * b
            for j, a in row.items():
                d[j] += mk * a
    for k in range(N):
        dk = d[k]
        if dk:
            const += min(dk * lp.lb[k], dk * lp.ub[k])
    return const


@dataclass
class ShannonResult:
    bound: Fraction           # certified lower bound on beta
    lp_value: float           # float optimum reported by the solver
    seconds: float
    y: dict[int, Fraction] = field(default_factory=dict)   # sparse multipliers (certificate)
    mu: dict[int, Fraction] = field(default_factory=dict)


def _on_grid(x: float, den: int) -> Fraction:
    """Round to the grid k / den. A common denominator keeps exact sums small (no lcm blow-up)."""
    return Fraction(int(round(float(x) * den)), den)


# Denominators tried for the multipliers; highly composite ones make exact optimal duals (2, 3, 4, 6 ...) land on
# the grid, so the certified bound usually equals the LP optimum exactly.
GRID_DENOMINATORS = (720720, 2**20 * 45045, 10**9)


def shannon_bound(g: Graph) -> ShannonResult:
    """Certified Shannon lower bound on beta(g). Raises ValueError for n > SHANNON_MAX_N."""
    t0 = time.perf_counter()
    lp = build_shannon(g)
    N = 1 << lp.n
    # Interior point + crossover: about 40x faster than dual simplex on these LPs (n = 11: 2 s vs 88 s).
    res = linprog(lp.c, A_ub=_sparse(lp.A_ub, N), b_ub=lp.b_ub, A_eq=_sparse(lp.A_eq, N), b_eq=lp.b_eq,
                  bounds=list(zip(lp.lb, lp.ub)), method="highs-ipm")
    if res.status != 0:
        raise RuntimeError(f"Shannon LP failed on {g.name}: {res.message}")
    # scipy marginals are d(objective)/d(rhs): <= 0 for <= rows. Our y multiplies (b - A x) >= 0, so y = -marginal.
    best: Fraction | None = None
    best_y: dict[int, Fraction] = {}
    best_mu: dict[int, Fraction] = {}
    target = Fraction(res.fun).limit_denominator(1000)
    # Two ways to round the float multipliers: entry by entry to small denominators (exact when the optimal duals
    # are "nice", e.g. halves for odd cycles), or onto one common grid (never blows up the denominators). Any
    # rounding gives a valid bound; keep the best.
    candidates = [("entry", d) for d in (60, 1000, 10**5)] + [("grid", d) for d in GRID_DENOMINATORS]
    for how, den in candidates:
        rnd = (lambda x, d=den: Fraction(float(x)).limit_denominator(d)) if how == "entry" else \
              (lambda x, d=den: _on_grid(x, d))
        y = [max(Fraction(0), rnd(-m)) for m in res.ineqlin.marginals]
        mu = [rnd(-m) for m in res.eqlin.marginals]
        b = certified_bound(lp, y, mu)
        if best is None or b > best:
            best = b
            best_y = {k: v for k, v in enumerate(y) if v}
            best_mu = {k: v for k, v in enumerate(mu) if v}
        if b >= target:
            break
    if best.denominator > 10**12:  # report a short number below the certified one (still a valid bound)
        best = Fraction(math.floor(best * 10**9), 10**9)
    return ShannonResult(best, float(res.fun), time.perf_counter() - t0, best_y, best_mu)


def check_shannon_certificate(g: Graph, y: dict[int, Fraction], mu: dict[int, Fraction]) -> Fraction:
    """Rebuild the LP from g and recompute the certified bound from the stored multipliers."""
    lp = build_shannon(g)
    yy = [Fraction(0)] * len(lp.A_ub)
    mm = [Fraction(0)] * len(lp.A_eq)
    for k, v in y.items():
        yy[int(k)] = Fraction(v)
    for k, v in mu.items():
        mm[int(k)] = Fraction(v)
    return certified_bound(lp, yy, mm)


# ---------------------------------------------------------------------------------- fractional clique cover
@dataclass
class FracCover:
    value: Fraction
    cliques: list[list[int]]
    weights: list[Fraction]
    optimal: bool           # pricing proved no improving clique exists
    lp_value: float
    seconds: float
    iterations: int


def mutual_masks(g: Graph) -> list[int]:
    return list(g.mutual().out)


def check_frac_cover(g: Graph, cliques: list[list[int]], weights: list[Fraction]) -> Fraction:
    """Exact check: every clique is a clique of the mutual graph, every vertex covered with weight >= 1.
    Returns the value (sum of weights); raises ValueError otherwise."""
    mut = mutual_masks(g)
    cover = [Fraction(0)] * g.n
    total = Fraction(0)
    for C, w in zip(cliques, weights):
        w = Fraction(w)
        if w < 0:
            raise ValueError("negative weight")
        for a in C:
            for b in C:
                if a != b and not (mut[a] >> b) & 1:
                    raise ValueError(f"{a} and {b} do not know each other's messages")
            cover[a] += w
        total += w
    for v in range(g.n):
        if cover[v] < 1:
            raise ValueError(f"vertex {v} covered with weight {cover[v]} < 1")
    return total


def _max_weight_clique(mut: list[int], weights: np.ndarray, time_limit: float) -> tuple[list[int], bool]:
    """Max-weight clique of the mutual graph by CP-SAT (weights scaled to integers). Returns (clique, proven)."""
    from ortools.sat.python import cp_model

    n = len(mut)
    cand = [v for v in range(n) if weights[v] > 1e-12]
    if not cand:
        return [], True
    m = cp_model.CpModel()
    x = {v: m.NewBoolVar(f"x{v}") for v in cand}
    for a_i, a in enumerate(cand):
        for b in cand[a_i + 1:]:
            if not (mut[a] >> b) & 1:
                m.AddBoolOr([x[a].Not(), x[b].Not()])
    scale = 10**6
    m.Maximize(sum(int(round(weights[v] * scale)) * x[v] for v in cand))
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = max(0.05, float(time_limit))
    s.parameters.num_workers = 1
    st = s.Solve(m)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], False
    return [v for v in cand if s.Value(x[v])], st == cp_model.OPTIMAL


def _greedy_extend(mut: list[int], clique: list[int], weights: np.ndarray) -> list[int]:
    """Extend a clique to a maximal one, heaviest vertices first (more coverage per column)."""
    common = (1 << len(mut)) - 1
    for v in clique:
        common &= mut[v]
    cl = list(clique)
    for v in sorted(iter_bits(common), key=lambda v: -weights[v]):
        if (common >> v) & 1:
            cl.append(v)
            common &= mut[v]
    return sorted(cl)


def frac_clique_cover(g: Graph, time_limit: float = 10.0, init_cliques: list[list[int]] | None = None,
                      max_den: int = 1000) -> FracCover:
    """Fractional clique cover by column generation, with an exact rational certificate."""
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    n = g.n
    mut = mutual_masks(g)
    cols: list[list[int]] = [[v] for v in range(n)]
    seen = {tuple(c) for c in cols}
    for C in init_cliques or []:
        C = _greedy_extend(mut, sorted(C), np.ones(n))
        if tuple(C) not in seen:
            seen.add(tuple(C))
            cols.append(C)
    optimal = False
    it = 0
    w = None
    lp_val = float(n)
    while True:
        it += 1
        A = np.zeros((n, len(cols)))
        for k, C in enumerate(cols):
            A[C, k] = 1.0
        res = linprog(np.ones(len(cols)), A_ub=-A, b_ub=-np.ones(n), bounds=(0, None), method="highs")
        if res.status != 0:
            raise RuntimeError(f"fractional cover LP failed: {res.message}")
        w, lp_val = res.x, float(res.fun)
        pi = -np.asarray(res.ineqlin.marginals)  # duals >= 0
        left = deadline - time.perf_counter()
        if left <= 0.05:
            break
        C, proven = _max_weight_clique(mut, pi, left)
        if not C or pi[C].sum() <= 1 + 1e-7:
            optimal = proven
            break
        C = _greedy_extend(mut, C, pi)
        if tuple(C) in seen:
            optimal = False
            break
        seen.add(tuple(C))
        cols.append(C)
    # rationalise, then scale so every vertex is covered >= 1 exactly
    used = [(cols[k], Fraction(float(w[k])).limit_denominator(max_den)) for k in range(len(cols)) if w[k] > 1e-9]
    cover = [Fraction(0)] * n
    for C, wk in used:
        for v in C:
            cover[v] += wk
    worst = min(cover)
    if worst <= 0:
        raise RuntimeError("rounded weights leave a vertex uncovered")
    if worst < 1:
        used = [(C, wk / worst) for C, wk in used]
    cliques = [C for C, _ in used]
    weights = [wk for _, wk in used]
    value = check_frac_cover(g, cliques, weights)
    return FracCover(value, cliques, weights, optimal, lp_val, time.perf_counter() - t0, it)
