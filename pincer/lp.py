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
# The exact basis fallback (python-flint) is automatic up to these sizes (it costs up to ~1 min above them):
EXACT_FULL_MAX_N = 11
EXACT_SYM_MAX_COLS = 2500


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


def shannon_bound(g: Graph, exact: bool | None = None) -> ShannonResult:
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
    candidates = None
    # Two ways to round the float multipliers: entry by entry to small denominators (exact when the optimal duals
    # are "nice", e.g. halves for odd cycles), or onto one common grid (never blows up the denominators). Any
    # rounding gives a valid bound; keep the best.
    if candidates is None:
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
    if exact is None:
        exact = lp.n <= EXACT_FULL_MAX_N
    if exact and best < target:  # rounding missed: exact dual vertex from the optimal basis (slower, no rounding)
        ex = exact_basis_duals(lp, N)
        if ex is not None:
            b = certified_bound(lp, *ex)
            if b > best:
                best = b
                best_y = {k: v for k, v in enumerate(ex[0]) if v}
                best_mu = {k: v for k, v in enumerate(ex[1]) if v}
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
    # Rationalise. Two candidates, keep the smaller value: (a) each weight to a small denominator (exact when the
    # optimum is nice, e.g. 5/2 for C5), (b) every weight rounded UP onto one common grid 1/D (denominators stay
    # bounded; an lcm of many different denominators can reach 100 digits). Either is then scaled up, if needed,
    # by a factor on the same grid, so every vertex is covered >= 1 exactly.
    D = 720720
    idx = [k for k in range(len(cols)) if w[k] > 1e-9]

    def scaled(ws: list[Fraction]) -> list[Fraction]:
        cover = [Fraction(0)] * n
        for k, wk in zip(idx, ws):
            for v in cols[k]:
                cover[v] += wk
        worst = min(cover)
        if worst <= 0:
            raise RuntimeError("rounded weights leave a vertex uncovered")
        if worst < 1:
            f = Fraction(math.ceil(D / worst), D)  # >= 1 / worst
            ws = [wk * f for wk in ws]
        return ws

    cands = [scaled([Fraction(float(w[k])).limit_denominator(max_den) for k in idx]),
             scaled([Fraction(math.ceil(float(w[k]) * D), D) for k in idx])]
    ws = min(cands, key=lambda c: (sum(c), max(x.denominator for x in c)))
    used = list(zip([cols[k] for k in idx], ws))
    cliques = [C for C, _ in used]
    weights = [wk for _, wk in used]
    value = check_frac_cover(g, cliques, weights)
    return FracCover(value, cliques, weights, optimal, lp_val, time.perf_counter() - t0, it)


# ------------------------------------------------------------------------- symmetry-reduced Shannon bound
SHANNON_SYM_MAX_N = 16


def subset_orbits(n: int, gens: list[list[int]]) -> list[int]:
    """orbit[S] = smallest subset (as int) in the orbit of S under the group generated by `gens` (a vertex
    permutation p maps S to {p[v] : v in S})."""
    N = 1 << n
    parent = list(range(N))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in gens:
        bit = [1 << p[v] for v in range(n)]
        for S in range(N):
            T = 0
            x = S
            while x:
                low = x & -x
                T |= bit[low.bit_length() - 1]
                x ^= low
            a, b = find(S), find(T)
            if a != b:
                if a < b:
                    parent[b] = a
                else:
                    parent[a] = b
    return [find(S) for S in range(N)]


def build_shannon_sym(g: Graph, gens: list[list[int]]) -> tuple[ShannonLP, list[int], dict[int, int]]:
    """The Shannon LP restricted to Aut-invariant g: one variable per subset orbit, constraints projected and
    de-duplicated. Valid because the full system is permuted by every automorphism (submodularity and the bounds
    are label-free, and the decoding constraint of i maps to that of p(i) since N(p(i)) = p(N(i))), so averaging a
    feasible point over the group gives an invariant feasible point with the same objective g(empty)."""
    from minrank.exact.symmetry import is_automorphism

    n = g.n
    if n > SHANNON_SYM_MAX_N:
        raise ValueError(f"symmetric Shannon LP limited to n <= {SHANNON_SYM_MAX_N}")
    for p in gens:
        if not is_automorphism(g, p):
            raise ValueError("not an automorphism")
    orb = subset_orbits(n, gens)
    reps = sorted(set(orb))
    col = {r: k for k, r in enumerate(reps)}
    N = 1 << n
    full = N - 1
    seen: dict[tuple, None] = {}

    def add(row: dict[int, int], b: int, store: list, bs: list) -> None:
        red: dict[int, int] = {}
        for S, a in row.items():
            c = col[orb[S]]
            red[c] = red.get(c, 0) + a
        red = {c: a for c, a in red.items() if a}
        if not red:
            if b < 0:
                raise RuntimeError("infeasible projected constraint")
            return
        key = (tuple(sorted(red.items())), b, id(store))
        if key in seen:
            return
        seen[key] = None
        store.append(red)
        bs.append(b)

    A_ub: list[dict[int, int]] = []
    b_ub: list[int] = []
    for K in range(N):
        if orb[K] != K:
            continue  # constraints based at K are images of those at its orbit representative... only for the
            # base set; the pair (i, j) is free, so this covers every orbit of (K, i, j) at least once
        free = full & ~K
        for i in iter_bits(free):
            add({K | 1 << i: 1, K: -1}, 1, A_ub, b_ub)
            for j in iter_bits(free >> (i + 1) << (i + 1)):
                add({K | 1 << i | 1 << j: 1, K: 1, K | 1 << i: -1, K | 1 << j: -1}, 0, A_ub, b_ub)
    for i in range(n):
        add({full & ~(1 << i): 1, full: -1}, 0, A_ub, b_ub)
    A_eq: list[dict[int, int]] = []
    b_eq: list[int] = []
    add({full: 1}, n, A_eq, b_eq)
    for i in range(n):
        Ni = g.out[i]
        add({Ni | 1 << i: 1, Ni: -1}, 0, A_eq, b_eq)
    c = np.zeros(len(reps))
    c[col[0]] = 1.0
    lb = [r.bit_count() for r in reps]
    ub = [n] * len(reps)
    lp = ShannonLP(n, c, A_ub, b_ub, A_eq, b_eq, lb, ub)
    return lp, reps, col


def _certified_bound_cols(lp: ShannonLP, y: list[Fraction], mu: list[Fraction]) -> Fraction:
    """certified_bound for an LP whose variable count is len(lp.c) (not 2^n)."""
    if any(v < 0 for v in y):
        raise ValueError("inequality multipliers must be >= 0")
    d = [Fraction(int(x)) for x in lp.c]
    const = Fraction(0)
    for rows, bs, mult in ((lp.A_ub, lp.b_ub, y), (lp.A_eq, lp.b_eq, mu)):
        for row, b, m in zip(rows, bs, mult):
            if m:
                const -= m * b
                for j, a in row.items():
                    d[j] += m * a
    for k, dk in enumerate(d):
        if dk:
            const += min(dk * lp.lb[k], dk * lp.ub[k])
    return const


def shannon_bound_sym(g: Graph, gens: list[list[int]] | None = None, ds_time_limit: float = 20.0,
                      exact: bool | None = None) -> ShannonResult:
    """Certified Shannon bound via the orbit-reduced LP. Certificate: y / mu on the reduced constraints plus the
    generators (stored in result.y["gens"]-free form: see check_shannon_sym)."""
    from minrank.exact.symmetry import automorphisms

    t0 = time.perf_counter()
    gens = [list(p) for p in (gens if gens is not None else automorphisms(g))]
    lp, reps, _ = build_shannon_sym(g, gens)
    M = len(reps)
    best, by, bmu, lp_val = None, {}, {}, None
    # Round the float duals of interior point (then, time-limited, dual simplex); if that misses the optimum, solve
    # the optimal basis exactly. Both: their dual vertices differ, and which one rounds to an exact certificate varies by graph
    # (C13: interior point exact at denominator 1000, dual simplex never).
    # Dual simplex can stall on the larger LPs (circulant n = 16: > 300 s vs 9 s for interior point), so it only
    # runs, time-limited, when the interior-point duals did not round to the optimum.
    for method, limit in (("highs-ipm", None), ("highs-ds", ds_time_limit)):
        opts = {"time_limit": limit} if limit else {}
        res = linprog(lp.c, A_ub=_sparse(lp.A_ub, M), b_ub=lp.b_ub, A_eq=_sparse(lp.A_eq, M), b_eq=lp.b_eq,
                      bounds=list(zip(lp.lb, lp.ub)), method=method, options=opts)
        if res.status != 0:
            continue
        lp_val = res.fun if lp_val is None else lp_val
        target = Fraction(res.fun).limit_denominator(1000)
        if best is not None and best >= target:
            break
        for how, den in [("entry", d) for d in (60, 1000, 10**5, 10**7)] + [("grid", d) for d in GRID_DENOMINATORS]:
            rnd = (lambda x, d=den: Fraction(float(x)).limit_denominator(d)) if how == "entry" else \
                  (lambda x, d=den: _on_grid(x, d))
            y = [max(Fraction(0), rnd(-m)) for m in res.ineqlin.marginals]
            mu = [rnd(-m) for m in res.eqlin.marginals]
            bnd = _certified_bound_cols(lp, y, mu)
            if best is None or bnd > best:
                best, by, bmu = bnd, {k: v for k, v in enumerate(y) if v}, {k: v for k, v in enumerate(mu) if v}
            if bnd >= target:
                break
        if best is not None and best >= target:
            break
    if exact is None:
        exact = M <= EXACT_SYM_MAX_COLS
    if (best is None or best < Fraction(lp_val).limit_denominator(1000)) and (exact or best is None):
        ex = exact_basis_duals(lp, M)
        if ex is not None:
            b = _certified_bound_cols(lp, *ex)
            if best is None or b > best:
                best = b
                by, bmu = {k: v for k, v in enumerate(ex[0]) if v}, {k: v for k, v in enumerate(ex[1]) if v}
    if best is None:
        raise RuntimeError(f"symmetric Shannon LP failed on {g.name}")
    if best.denominator > 10**12:
        best = Fraction(math.floor(best * 10**9), 10**9)
    r = ShannonResult(best, float(lp_val), time.perf_counter() - t0, by, bmu)
    r.gens = gens  # type: ignore[attr-defined]
    r.reduced_size = (M, len(lp.A_ub) + len(lp.A_eq))  # type: ignore[attr-defined]
    return r


def check_shannon_sym(g: Graph, gens: list[list[int]], y: dict[int, Fraction], mu: dict[int, Fraction]) -> Fraction:
    """Rebuild the reduced LP from g and the generators (which must be automorphisms) and recompute the bound."""
    lp, _, _ = build_shannon_sym(g, gens)
    yy = [Fraction(0)] * len(lp.A_ub)
    mm = [Fraction(0)] * len(lp.A_eq)
    for k, v in y.items():
        yy[int(k)] = Fraction(v)
    for k, v in mu.items():
        mm[int(k)] = Fraction(v)
    return _certified_bound_cols(lp, yy, mm)


def exact_basis_duals(lp: ShannonLP, ncols: int, time_limit: float = 120.0):
    """Exact dual vertex: solve the LP with HiGHS (interior point + crossover), take the optimal basis and solve its
    square system B^T y = c_B in exact rational arithmetic (python-flint). Returns (y, mu) as Fraction lists, or
    None if flint/highspy is missing or the basis is unusable. Whatever comes back is only *used* through
    `_certified_bound_cols`, so a wrong basis can lower the bound but never make it invalid."""
    try:
        import flint
        import highspy
    except ImportError:
        return None
    m_ub, m_eq = len(lp.A_ub), len(lp.A_eq)
    rows = lp.A_ub + lp.A_eq
    inf = highspy.kHighsInf
    rl = [-inf] * m_ub + [float(b) for b in lp.b_eq]
    ru = [float(b) for b in lp.b_ub] + [float(b) for b in lp.b_eq]
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", "ipm")
    h.setOptionValue("run_crossover", "on")
    h.setOptionValue("time_limit", float(time_limit))
    h.addVars(ncols, np.array(lp.lb, float), np.array(lp.ub, float))
    h.changeColsCost(ncols, np.arange(ncols, dtype=np.int32), np.asarray(lp.c, float))
    M = _sparse(rows, ncols)
    h.addRows(len(rows), np.array(rl), np.array(ru), M.nnz, M.indptr[:-1].astype(np.int32),
              M.indices.astype(np.int32), M.data.astype(float))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return None
    basis = h.getBasis()
    if not basis.valid:
        return None
    B = highspy.HighsBasisStatus.kBasic
    bcols = [j for j, st in enumerate(basis.col_status) if st == B]
    nrows = [i for i, st in enumerate(basis.row_status) if st != B]
    if len(bcols) != len(nrows):
        return None
    if not bcols:
        return [Fraction(0)] * m_ub, [Fraction(0)] * m_eq
    k = len(bcols)
    cpos = {j: t for t, j in enumerate(bcols)}
    # rows of the system: one per basic column j:  sum_{i in nrows} A[i, j] * yhat_i = c_j   (HiGHS convention
    # c - A^T yhat - z = 0 with z_j = 0 on basic columns)
    mat = flint.fmpz_mat(k, k)
    for t, i in enumerate(nrows):
        for j, a in rows[i].items():
            p = cpos.get(j)
            if p is not None:
                mat[p, t] = a
    rhs = flint.fmpz_mat(k, 1, [int(lp.c[j]) for j in bcols])
    try:
        sol = mat.solve(rhs)  # fmpq_mat
    except Exception:  # singular (numerically chosen basis not exactly a basis)
        return None
    yhat = [Fraction(0)] * len(rows)
    for t, i in enumerate(nrows):
        q = sol[t, 0]
        yhat[i] = Fraction(int(q.p), int(q.q))
    # our multipliers: y (on b - A x >= 0) = -yhat for <= rows; mu = -yhat for equalities
    y = [max(Fraction(0), -v) for v in yhat[:m_ub]]
    mu = [-v for v in yhat[m_ub:]]
    return y, mu
