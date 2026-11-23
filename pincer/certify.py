"""Certificate bundle: everything needed to re-check Pincer's final intervals from the graph alone.

`make_bundle` serialises the component states; `verify_bundle` rebuilds the SCCs from the graph and re-checks every
certificate with code that does not depend on how it was found:
- global fitting matrix (block diagonal over SCCs): fit + exact F_2 rank  -> minrk_2 <= U_mr
- global acyclic set (union over SCCs): Kahn's algorithm                -> MAIS >= |set|
- per-SCC Shannon multipliers: exact rational weak duality               -> beta(SCC) >= bound
- per-SCC fractional clique covers: exact coverage                       -> beta(SCC) <= value
- per-SCC UNSAT records: listed as solver-trusted (re-run or DRAT-check them with `--recheck-unsat`)
CLI:  python -m pincer.certify GRAPH.json BUNDLE.json [--recheck-unsat SECONDS]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

from minrank.core.graph import Graph
from minrank.core.verify import verify
from minrank.exact.lower import is_acyclic_induced

from .decompose import decompose, lift_rows
from .lp import check_frac_cover, check_shannon_certificate
from .state import CompState


def _frac(x) -> str:
    return str(Fraction(x))


def make_bundle(g: Graph, states: list[CompState], meta: dict | None = None) -> dict:
    comps = [s.comp for s in states]
    rows = lift_rows(g, comps, [s.rows for s in states])
    acyc = sorted(v for s in states for v in s.comp.to_global(s.acyclic))
    out = {"graph": g.name, "digest": g.digest(), "n": g.n, "meta": meta or {},
           "rows": [f"{x:x}" for x in rows], "acyclic": acyc, "components": []}
    for s in states:
        c = {"verts": list(s.comp.verts), "mr": [s.mr_lo, s.mr_hi], "beta": [_frac(s.b_lo), _frac(s.b_hi)],
             "sources": s.sources, "unsat": s.unsat,
             "events": [[e.t, e.tool, e.what] for e in s.events]}
        if s.shannon:
            c["shannon"] = {"bound": _frac(s.shannon["bound"]),
                            "y": {str(k): _frac(v) for k, v in s.shannon["y"].items()},
                            "mu": {str(k): _frac(v) for k, v in s.shannon["mu"].items()}}
        if s.frac:
            c["frac"] = {"value": _frac(s.frac["value"]), "cliques": s.frac["cliques"],
                         "weights": [_frac(w) for w in s.frac["weights"]]}
        out["components"].append(c)
    out["totals"] = totals(states)
    return out


def totals(states: list[CompState]) -> dict:
    return {"mr": [sum(s.mr_lo for s in states), sum(s.mr_hi for s in states)],
            "beta": [_frac(sum((s.b_lo for s in states), Fraction(0))),
                     _frac(sum((s.b_hi for s in states), Fraction(0)))],
            "components": len(states), "closed_mr": sum(s.mr_closed for s in states),
            "closed_beta": sum(s.b_closed for s in states)}


def verify_bundle(g: Graph, b: dict, recheck_unsat: float | None = None) -> dict:
    """Re-check every certificate. Returns the verified totals; raises ValueError on any failure."""
    if b["digest"] != g.digest():
        raise ValueError("bundle is for a different graph")
    comps = decompose(g)
    by_verts = {tuple(c.verts): c for c in comps}
    if sorted(tuple(c["verts"]) for c in b["components"]) != sorted(by_verts):
        raise ValueError("components in the bundle are not the SCCs of the graph")
    rows = [int(x, 16) for x in b["rows"]]
    U_mr = verify(g, rows)
    acyc = [int(v) for v in b["acyclic"]]
    if not is_acyclic_induced(g, acyc):
        raise ValueError("global acyclic set has a cycle")
    L_mais = len(acyc)
    mr_lo_total = 0
    b_lo_total = Fraction(0)
    b_hi_total = Fraction(0)
    unsat_trusted = 0
    for c in b["components"]:
        comp = by_verts[tuple(c["verts"])]
        cg = comp.graph
        vs = set(comp.verts)
        loc = {v: k for k, v in enumerate(comp.verts)}
        # per-component pieces of the global certificates
        a_c = [loc[v] for v in acyc if v in vs]
        r_c = verify(cg, [_restrict(rows[v], loc) for v in comp.verts])
        lo_mr, lo_b, hi_b = len(a_c), Fraction(len(a_c)), Fraction(r_c)
        if "shannon" in c:
            y = {int(k): Fraction(v) for k, v in c["shannon"]["y"].items()}
            mu = {int(k): Fraction(v) for k, v in c["shannon"]["mu"].items()}
            sb = check_shannon_certificate(cg, y, mu)
            claimed = Fraction(c["shannon"]["bound"])
            if sb < claimed:
                raise ValueError("Shannon certificate does not reproduce its bound")
            lo_b = max(lo_b, claimed)
            lo_mr = max(lo_mr, math.ceil(claimed))
        if "frac" in c:
            fv = check_frac_cover(cg, c["frac"]["cliques"], [Fraction(w) for w in c["frac"]["weights"]])
            hi_b = min(hi_b, fv)
        for u in c.get("unsat", []):
            if recheck_unsat is not None:
                _recheck_unsat(cg, u, recheck_unsat)
            else:
                unsat_trusted += 1
            lo_mr = max(lo_mr, int(u["r"]) + 1)
        if lo_mr > r_c or lo_b > hi_b:
            raise ValueError(f"component {comp.index}: lower bound above upper bound")
        if [lo_mr, r_c] != c["mr"] or [lo_b, hi_b] != [Fraction(x) for x in c["beta"]]:
            raise ValueError(f"component {comp.index}: claimed {c['mr']} {c['beta']}, "
                             f"verified {[lo_mr, r_c]} {[str(lo_b), str(hi_b)]}")
        mr_lo_total += lo_mr
        b_lo_total += lo_b
        b_hi_total += hi_b
    res = {"mr": [mr_lo_total, U_mr], "beta": [str(b_lo_total), str(b_hi_total)], "mais": L_mais,
           "unsat_records_trusted": unsat_trusted}
    if res["mr"] != b["totals"]["mr"] or res["beta"] != b["totals"]["beta"]:
        raise ValueError(f"totals do not match: claimed {b['totals']}, verified {res}")
    return res


def _restrict(row: int, loc: dict[int, int]) -> int:
    out = 0
    for v, k in loc.items():
        if (row >> v) & 1:
            out |= 1 << k
    return out


def _recheck_unsat(cg: Graph, u: dict, limit: float) -> None:
    from minrank.exact.symmetry import rank_at_most_sym

    res = rank_at_most_sym(cg, int(u["r"]), time_limit=limit, mode=u.get("mode", "cols"), indep=u.get("indep"))
    if res is not False:
        raise ValueError(f"UNSAT record r={u['r']} not reproduced (got {'timeout' if res is None else 'SAT'})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("graph")
    ap.add_argument("bundle")
    ap.add_argument("--recheck-unsat", type=float, default=None, help="re-run every UNSAT call with this limit")
    a = ap.parse_args(argv)
    g = Graph.load(a.graph)
    b = json.loads(Path(a.bundle).read_text())
    res = verify_bundle(g, b, a.recheck_unsat)
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
