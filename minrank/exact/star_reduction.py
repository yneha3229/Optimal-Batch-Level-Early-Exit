"""Star-basis reduction for deciding minrk_{F_q}(K(n,k)) = alpha.

Soundness.  Let S be the star {A : 1 in A}, an independent set of size alpha.
For any matrix M that fits K(n,k), the block M[S,S] is diagonal with a nonzero
diagonal, so the rows indexed by S are linearly independent and rank M >= alpha.
If rank M = alpha, these rows are a basis.  Scale rows so M[S,S] = I.  Then every
other row v is v = sum_s c_{v,s} M[s,:], and reading columns S gives c_{v,s} = M[v,s].
So rank M = alpha iff there are
    X[s,w] = M[s,w]   (s in S, w not in S; forced 0 when s,w intersect)
    C[v,s] = M[v,s]   (v not in S, s in S; forced 0 when v,s intersect)
with (C X)[v,v] != 0 and (C X)[v,w] = 0 for distinct intersecting v,w not in S.
Adjacent (disjoint) pairs are unconstrained.  This system is far smaller than the
generic rank-r SAT encoding, so it decides cases the generic encoding cannot.

UNSAT here means minrk_{F_q}(K(n,k)) >= alpha + 1.

Usage:
    python -m minrank.exact.star_reduction --case 9,2,2 --case 7,2,3 [--cnf-dir DIR]
F_2 uses an XOR-chain CNF, other primes q a one-hot CNF; both are solved with
CaDiCaL and can be written out (--cnf-dir) for DRAT checking. Every SAT answer is
re-assembled into M and checked (fits K(n,k), exact rank alpha mod q).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import os
import time

import numpy as np


def kneser_sets(n, k):
    return [frozenset(c) for c in itertools.combinations(range(1, n + 1), k)]


def structure(n, k):
    V = kneser_sets(n, k)
    S = [A for A in V if 1 in A]
    R = [A for A in V if 1 not in A]
    xs = {(s, w) for s in S for w in R if not (s & w)}
    cs = {(v, s) for v in R for s in S if not (v & s)}
    cons = []  # (v, w, terms, rhs) ; rhs 'nz' for diagonal, 0 otherwise
    for v in R:
        for w in R:
            if v != w and not (v & w):
                continue
            terms = [s for s in S if (v, s) in cs and (s, w) in xs]
            cons.append((v, w, terms, "nz" if v == w else 0))
    return V, S, R, xs, cs, cons


def rank_mod_p(A, p):
    A = A.copy() % p
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if A[i, c]), None)
        if piv is None:
            continue
        A[[r, piv]] = A[[piv, r]]
        A[r] = (A[r] * pow(int(A[r, c]), -1, p)) % p
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % p
        r += 1
        if r == rows:
            break
    return r


def build_and_check(n, k, q, Xv, Cv):
    """Assemble M = [I | X ; C | C X] and verify it fits and has rank alpha."""
    V, S, R, xs, cs, cons = structure(n, k)
    idx = {A: i for i, A in enumerate(S + R)}
    a = len(S)
    Xm = np.zeros((a, len(R)), dtype=np.int64)
    Cm = np.zeros((len(R), a), dtype=np.int64)
    for (s, w), val in Xv.items():
        Xm[idx[s], idx[w] - a] = val
    for (v, s), val in Cv.items():
        Cm[idx[v] - a, idx[s]] = val
    M = np.block([[np.eye(a, dtype=np.int64), Xm], [Cm, (Cm @ Xm) % q]]) % q
    order = S + R
    for i, A in enumerate(order):
        if M[i, i] == 0:
            return False, "zero diagonal"
        for j, B in enumerate(order):
            if i != j and (A & B) and M[i, j]:
                return False, "nonzero on non-edge"
    rk = rank_mod_p(M, q)
    return rk == a, f"rank {rk}"


# ---------------- F_2: CNF ----------------
def cnf_f2(n, k):
    V, S, R, xs, cs, cons = structure(n, k)
    var = {}

    def new(key=None):
        var_id = len(var) + 1
        var[key if key is not None else ("aux", var_id)] = var_id
        return var_id

    for e in xs:
        new(("X",) + e)
    for e in cs:
        new(("C",) + e)
    clauses = []
    for v, w, terms, rhs in cons:
        prods = []
        for s in terms:
            c, x = var[("C", v, s)], var[("X", s, w)]
            p = new()
            clauses += [[-p, c], [-p, x], [p, -c, -x]]
            prods.append(p)
        target = 1 if rhs == "nz" else 0
        if not prods:
            if target:
                clauses.append([])  # trivially UNSAT
            continue
        y = prods[0]
        for p in prods[1:]:
            z = new()
            clauses += [[-z, y, p], [-z, -y, -p], [z, -y, p], [z, y, -p]]
            y = z
        clauses.append([y] if target else [-y])
    return var, clauses


def solve_f2(n, k, cnf_dir=None, solver_name="cadical195"):
    from pysat.solvers import Solver

    var, clauses = cnf_f2(n, k)
    nv = len(var)
    if cnf_dir:
        os.makedirs(cnf_dir, exist_ok=True)
        path = os.path.join(cnf_dir, f"star_kneser_{n}_{k}_q2.cnf.gz")
        with gzip.open(path, "wt") as f:
            f.write(f"p cnf {nv} {len(clauses)}\n")
            for cl in clauses:
                f.write(" ".join(map(str, cl)) + " 0\n")
    with Solver(name=solver_name, bootstrap_with=clauses) as sol:
        sat = sol.solve()
        model = set(l for l in sol.get_model() if l > 0) if sat else None
    if not sat:
        return "UNSAT", None
    Xv = {key[1:]: 1 if vid in model else 0 for key, vid in var.items() if key[0] == "X"}
    Cv = {key[1:]: 1 if vid in model else 0 for key, vid in var.items() if key[0] == "C"}
    ok, msg = build_and_check(n, k, 2, Xv, Cv)
    return ("SAT" if ok else f"SAT-BUT-CHECK-FAILED({msg})"), (Xv, Cv)


# ---------------- F_q (q prime): one-hot CNF ----------------
def cnf_fq(n, k, q):
    """One-hot CNF over F_q. Each unknown is q booleans with exactly one true."""
    V, S, R, xs, cs, cons = structure(n, k)
    var = {}
    clauses = []

    def new(key=None):
        vid = len(var) + 1
        var[key if key is not None else ("aux", vid)] = vid
        return vid

    def onehot(key):
        lits = [new(key + (a,)) for a in range(q)]
        clauses.append(lits[:])
        for i in range(q):
            for j in range(i + 1, q):
                clauses.append([-lits[i], -lits[j]])
        return lits

    X = {e: onehot(("X",) + e) for e in xs}
    C = {e: onehot(("C",) + e) for e in cs}
    for v, w, terms, rhs in cons:
        acc = None  # one-hot of partial sum
        for s in terms:
            c, x = C[(v, s)], X[(s, w)]
            prod = onehot(("P", v, w, s))
            for a in range(q):
                for b in range(q):
                    clauses.append([-c[a], -x[b], prod[(a * b) % q]])
            if acc is None:
                acc = prod
                continue
            nxt = onehot(("S", v, w, s))
            for a in range(q):
                for b in range(q):
                    clauses.append([-acc[a], -prod[b], nxt[(a + b) % q]])
            acc = nxt
        if acc is None:
            if rhs == "nz":
                clauses.append([])
            continue
        clauses.append([-acc[0]] if rhs == "nz" else [acc[0]])
    return var, clauses


def solve_fq(n, k, q, cnf_dir=None, solver_name="cadical195"):
    from pysat.solvers import Solver

    var, clauses = cnf_fq(n, k, q)
    if cnf_dir:
        os.makedirs(cnf_dir, exist_ok=True)
        path = os.path.join(cnf_dir, f"star_kneser_{n}_{k}_q{q}.cnf.gz")
        with gzip.open(path, "wt") as f:
            f.write(f"p cnf {len(var)} {len(clauses)}\n")
            for cl in clauses:
                f.write(" ".join(map(str, cl)) + " 0\n")
    with Solver(name=solver_name, bootstrap_with=clauses) as sol:
        sat = sol.solve()
        model = set(l for l in sol.get_model() if l > 0) if sat else None
    if not sat:
        return "UNSAT", None
    Xv, Cv = {}, {}
    for key, vid in var.items():
        if key[0] in ("X", "C") and vid in model:
            (Xv if key[0] == "X" else Cv)[key[1:3]] = key[3]
    ok, msg = build_and_check(n, k, q, Xv, Cv)
    return ("SAT" if ok else f"SAT-BUT-CHECK-FAILED({msg})"), (Xv, Cv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", required=True, help="n,k,q")
    ap.add_argument("--cnf-dir", default=None)
    ap.add_argument("--out", default="minrank/results/field_sep/star_reduction.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["graph", "q", "alpha", "result", "meaning", "seconds"])
        for case in a.case:
            n, k, q = map(int, case.split(","))
            alpha = len([1 for A in kneser_sets(n, k) if 1 in A])
            t0 = time.time()
            if q == 2:
                res, _ = solve_f2(n, k, a.cnf_dir)
            else:
                res, _ = solve_fq(n, k, q, a.cnf_dir)
            dt = time.time() - t0
            meaning = {"SAT": f"minrk = {alpha}", "UNSAT": f"minrk >= {alpha + 1}"}.get(res, "undecided")
            row = [f"K({n},{k})", q, alpha, res, meaning, f"{dt:.1f}"]
            print(*row, flush=True)
            wr.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
