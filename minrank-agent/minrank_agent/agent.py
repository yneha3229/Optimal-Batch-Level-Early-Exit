"""The adaptive multi-field minrank agent.

Per graph it keeps a certified bracket  alpha <= beta <= upper  and tries to close it:

  1. alpha (max independent set) -- the lower bound and its certificate.
  2. clique cover -- a field-independent upper bound (the classic one).
  3. the multi-field spectral construction  I + cA.  Computing it over every field is wasteful on a large graph
     (each field costs a mod-p rank, O(n^3)).  Instead the agent computes the integer spectrum ONCE and uses the
     eigenvalue multiplicities to PREDICT, for each field p, how small  n - (nullity of A - lambda I mod p)  can be;
     it then tries fields in predicted order and stops as soon as the spectral bound meets alpha (gap closed, with an
     explicit optimal code over that field) or the budget runs out.
  4. optional F_2 minrank SAT when a small gap remains over F_2.

The field-ordering policy is the learned/adaptive part: a model maps graph features to the order of fields to try.
`spectrum_order` is the spectrum-based heuristic; `LearnedOrder` is trained offline (learn.py) and compared to it.
"""
from __future__ import annotations

import time

import numpy as np

from . import bounds as B

FIELDS = (2, 3, 5, 7, 11, 13)


def real_spectrum(A):
    """Integer/clustered eigenvalues of A with multiplicities (structured graphs are usually integral)."""
    w = np.linalg.eigvalsh(A.astype(float))
    out = []
    for v in sorted(w):
        r = round(v)
        if out and abs(out[-1][0] - r) < 1e-6 and abs(v - r) < 1e-4:
            out[-1][1] += 1
        elif out and abs(out[-1][2] - v) < 1e-4:
            out[-1][1] += 1
            out[-1][2] = v
        else:
            out.append([r, 1, v])
    return [(r, m) for r, m, _ in out]


def predicted_bound(n, spectrum, p):
    """n - max multiplicity of a nonzero-mod-p integer eigenvalue: the spectral bound we expect over F_p."""
    best = 0
    for lam, m in spectrum:
        if lam % p != 0 and abs(lam - round(lam)) < 1e-6:
            best = max(best, m)
    return n - best if best else n


def spectrum_order(n, spectrum, fields=FIELDS):
    """Heuristic field order: smallest predicted spectral bound first, ties broken by smaller field."""
    return sorted(fields, key=lambda p: (predicted_bound(n, spectrum, p), p))


def features(g, spectrum):
    n = g.n
    mults = sorted((m for _, m in spectrum), reverse=True)
    top = (mults + [0, 0, 0])[:3]
    return {
        "n": n, "density": g.density, "n_distinct_eig": len(spectrum),
        "max_mult": top[0] / n, "mult2": top[1] / n, "mult3": top[2] / n,
        "max_mult_abs": top[0], "integral": all(abs(lam - round(lam)) < 1e-6 for lam, _ in spectrum),
        **{f"pred_F{p}": predicted_bound(n, spectrum, p) / n for p in FIELDS},
    }


class SpectrumAgent:
    """Field order from the integer spectrum (no training)."""
    name = "spectrum-ordered"

    def order(self, g, spectrum, feats):
        return spectrum_order(g.n, spectrum, FIELDS)


class NaiveAgent:
    """Fixed ascending field order (the no-insight baseline)."""
    name = "ascending-fields"

    def order(self, g, spectrum, feats):
        return list(FIELDS)


class LearnedOrder:
    """Field order from a model trained offline to predict the gap-closing field (see learn.py)."""
    name = "learned"

    def __init__(self, model):
        self.model = model

    def order(self, g, spectrum, feats):
        import numpy as np

        X = np.array([[feats[k] for k in self.model["cols"]]])
        scores = self.model["clf"].predict_proba(X)[0]
        classes = self.model["clf"].classes_
        rank = {p: s for p, s in zip(classes, scores)}
        return sorted(FIELDS, key=lambda p: (-rank.get(p, 0), predicted_bound(g.n, spectrum, p), p))


def gate_says_closable(gate, feats):
    """The learned gate (results/gate.pkl): predict from cheap features whether multi-field spectral can close."""
    if gate is None:
        return True
    X = np.array([[float(feats[c]) for c in gate["cols"]]])
    return bool(gate["clf"].predict(X)[0])


def solve(g, policy, fields=FIELDS, alpha_limit=10.0, clique_limit=10.0, try_f2_sat=True, sat_budget=30000,
          gate=None):
    """Run the agent on graph g. Returns a certified result dict. If a gate is given and predicts the gap cannot
    close, the expensive mod-p search is skipped and only the clique-cover bracket is reported."""
    t0 = time.perf_counter()
    if getattr(g, "alpha_known", None) is not None:        # a proven formula (e.g. EKR for Kneser)
        a, aopt = g.alpha_known, True
    else:
        a, S, aopt = B.independence(g.A, alpha_limit)
    cc, colour, ccopt = B.clique_cover(g.A, clique_limit)
    lower, upper = a, cc
    best = {"by": "clique_cover", "p": None, "cost": cc}
    spectrum = real_spectrum(g.A)
    feats = features(g, spectrum)
    gated_out = not gate_says_closable(gate, feats)
    order = [] if gated_out else policy.order(g, spectrum, feats)
    tried = []
    for p in order:
        if upper <= lower:
            break
        cost, c, lam = B.spectral_field(g.A, p)
        assert B.verify_spectral(g.A, p, c, cost)
        tried.append((p, cost))
        if cost < upper:
            upper, best = cost, {"by": "spectral", "p": p, "cost": cost, "c": int(c)}
    f2 = None
    if try_f2_sat and upper > lower and upper - lower <= 3:
        r = lower
        while r < upper:
            res = B.f2_minrank_at(g.A, r, budget=sat_budget)
            if res is None or res is False:        # unknown or no code of this rank
                break
            upper, best, f2 = r, {"by": "f2_sat", "p": 2, "cost": r}, True
            r = lower                              # found rank r, try to go lower (but lower is alpha)
            break
    proven = (upper <= lower) and aopt
    return {
        "graph": g.name, "family": g.family, "n": g.n, "alpha": a, "alpha_opt": aopt, "clique_cover": cc,
        "lower": lower, "upper": upper, "gap": upper - lower, "proven_beta": upper if proven else None,
        "best_field": best["p"], "best_by": best["by"], "fields_tried": tried, "gated_out": gated_out,
        "seconds": round(time.perf_counter() - t0, 3), "policy": policy.name,
        "spectrum_distinct": len(spectrum), "feats": feats,
    }
