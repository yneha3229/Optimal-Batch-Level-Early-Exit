"""Verification stage: independent re-checks of the saved certificates (verify/checks.py only, no Solver code),
the explicit code file, the Verifier LLM's analysis, and the verdict computed by code.

verdict = "invalid" if any check failed; else "proven" if L == U; else "gap".
"""
from __future__ import annotations

import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import checks as V


def run_checks(A: np.ndarray, cert_dir: Path, result: dict) -> dict:
    t0 = time.perf_counter()
    certs = json.loads((cert_dir / "certificates.json").read_text())
    out: dict = {"checks": [], "ok": True}

    def add(name, passed, **info):
        info = {("sub_ok" if k == "ok" else k): v for k, v in info.items()}
        out["checks"].append({"check": name, "ok": bool(passed), **info})
        out["ok"] &= bool(passed)

    lo = certs.get("lower")
    if lo is not None:
        ok, msg, v = V.check_independent_set(A, lo["vertices"])
        add("independent_set", ok and v == result["L"], detail=msg, value=v)
    up = certs.get("upper")
    spectral = None
    if up is not None:
        k = up["kind"]
        if k == "identity":
            add("trivial_upper", up["value"] == A.shape[0], value=up["value"])
        elif k == "clique_cover":
            ok, msg, v = V.check_clique_cover(A, up["cliques"])
            add("clique_cover", ok and v == Fraction(result["U"]), detail=msg, value=v)
            sim = V.simulate_clique_decoding(A, up["cliques"])
            add("decode_simulation_clique_code", sim["ok"], **sim)
        elif k == "fractional_cover":
            ok, msg, v = V.check_fractional_cover(A, up["cliques"], up["weights"])
            add("fractional_cover", ok and v == Fraction(result["U"]), detail=msg, value=str(v))
        elif k == "spectral":
            z = np.load(cert_dir / up["file"])
            S, C, p, c = z["S"], z["C"], int(z["p"]), int(z["c"])
            spectral = (p, c, S, C)
            r = V.check_spectral_code(A, p, c, S, C)
            add("spectral_code", r["ok"] and len(S) == Fraction(result["U"]), **r)
            sim = V.simulate_decoding(A, p, c, S, C)
            add("decode_simulation", sim["ok"], **sim)
        elif k == "f2_rows":
            M = np.array([[(m >> j) & 1 for j in range(A.shape[0])] for m in up["rows"]])
            r = V.check_dense_code(A, M, 2, up["rank"])
            add("f2_code", r["ok"], **r)
        elif k == "blowup_f2":
            t = up["t"]
            At = np.kron(A.astype(np.int8), np.ones((t, t), dtype=np.int8)).astype(bool)
            N = A.shape[0] * t
            M = np.array([[(m >> j) & 1 for j in range(N)] for m in up["rows"]])
            r = V.check_dense_code(At, M, 2, up["rank"])
            add("blowup_code", r["ok"], **r)
    for u in certs.get("minrk2_unsat", []):
        if u.get("cnf"):
            r = V.check_unsat(u["cnf"])
            add("unsat_resolve", r["ok"], r=u["r"], **r)
    out["seconds"] = round(time.perf_counter() - t0, 2)
    out["_spectral"] = spectral
    return out


def extra_check(A: np.ndarray, spec, req: dict, lower_vertices) -> dict:
    name = req.get("check")
    if name == "indep_recheck" and lower_vertices is not None:
        ok, msg, v = V.check_independent_set(A, lower_vertices)
        return {"check": name, "ok": ok, "detail": msg}
    if spec is None:
        return {"check": name, "ok": True, "detail": "not applicable (no spectral code)"}
    p, c, S, C = spec
    if name == "freivalds_more":
        r = V.check_spectral_code(A, p, c, S, C, rounds=int(req.get("rounds", 30)), seed=777, exact_max=0)
        return {"check": name, "ok": r["ok"], "passed": r["freivalds_passed"]}
    if name == "fit_sample":
        rng = np.random.default_rng(5)
        n = A.shape[0]
        k = int(min(200000, req.get("samples", 20000)))
        u, v = rng.integers(0, n, k), rng.integers(0, n, k)
        M_uv = np.where(u == v, 1, (c % p) * A[u, v]) % p
        bad = int(((u != v) & ~A[u, v] & (M_uv != 0)).sum() + ((u == v) & (M_uv == 0)).sum())
        return {"check": name, "ok": bad == 0, "samples": k, "violations": bad}
    if name == "decode_more":
        r = V.simulate_decoding(A, p, c, S, C, trials=int(req.get("trials", 5)), seed=4242)
        return {"check": name, **r}
    if name == "exact_rank":
        if A.shape[0] > 5000:
            return {"check": name, "ok": True, "detail": "skipped (n > 5000)"}
        r = V.own_rank_mod_p(V.spectral_matrix(A, p, c), p)
        return {"check": name, "ok": r <= len(S), "rank": r}
    return {"check": name, "ok": True, "detail": "unknown check ignored"}


def write_code_file(A: np.ndarray, spec, path: Path, graph: str, result: dict, sim: dict | None) -> None:
    """Explicit optimal code: field, transmissions, decoding coefficients, simulation result. Rows of C go into a
    companion .npz (uint16) when n is large; transmissions are listed explicitly when small, else by their rule."""
    if spec is None:
        return
    p, c, S, C = spec
    n = A.shape[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"graph": graph, "field": f"F_{p}", "rule": f"M = I + {c} * A over F_{p}", "length": int(len(S)),
           "receivers": n, "beta_proven": result.get("proven"), "basis_rows_S": [int(s) for s in S],
           "transmission_rule": f"y_s = x_s + {c} * sum_{{j ~ s}} x_j (mod {p}) for every s in S",
           "decoding_rule": f"receiver v: x_v = sum_k C[v,k] y_k - {c} * sum_{{u ~ v}} x_u (mod {p})",
           "simulation": sim}
    nnz = int(A[S].sum()) + len(S)
    if nnz <= 2_000_000:
        doc["transmissions"] = [{"s": int(s), "terms": [[int(s), 1]] + [[int(j), int(c % p)]
                                                                       for j in np.nonzero(A[s])[0]]}
                                for s in S]
    else:
        doc["transmissions"] = f"{nnz} nonzero terms: given by transmission_rule (graph file has the neighbours)"
    if n * len(S) <= 300_000:
        doc["decoding_coefficients"] = {str(v): [int(x) for x in C[v]] for v in range(n)}
    else:
        f = path.with_name(path.stem + "_decode.npz")
        np.savez_compressed(f, C=np.asarray(C, dtype=np.uint16 if p < 65536 else np.int64), S=np.asarray(S))
        doc["decoding_coefficients"] = f"rows of C in {f.name} (C[v] for receiver v, {n} x {len(S)}, mod {p})"
    path.write_text(json.dumps(doc))


def verdict(checks: dict, result: dict) -> str:
    if not checks["ok"]:
        return "invalid"
    return "proven" if result["proven"] else "gap"


def verifier_payload(result: dict, checks: dict, baselines: dict | None, truth: dict | None) -> str:
    def fmt_checks():
        return "\n".join(f"- {c['check']}: {'PASS' if c['ok'] else 'FAIL'} "
                         + json.dumps({k: v for k, v in c.items() if k not in ('check', 'ok')}, default=str)[:300]
                         for c in checks["checks"])

    lines = [f"GRAPH {result['graph']}: n = {result['n']} receivers",
             f"AGENT RESULT: beta in [{result['L']}, {result['U']}], proven = {result['proven']}, "
             f"upper bound from {result['upper_kind']} (field {result['field']}); "
             f"time to first certified bound {result['t_first_bound']} s, time to proof {result['t_proof']} s, "
             f"wall {result['wall']} s; LLM {json.dumps(result['llm'])}",
             f"Solver family hypothesis: {json.dumps(result['hypothesis'])[:400]}",
             "INDEPENDENT CHECKS\n" + fmt_checks()]
    if truth is not None:
        lines.append(f"GROUND TRUTH (revealed after the Solver finished): {json.dumps(truth)}")
    if baselines:
        bl = []
        for name, b in baselines.items():
            bl.append(f"- {name}: {json.dumps({k: b.get(k) for k in ('status', 'lower', 'upper', 'time', 'exact')})}")
        lines.append("BASELINES (each 120 s, same cores)\n" + "\n".join(bl))
    else:
        lines.append("BASELINES: not available for this run")
    return "\n\n".join(lines)


def run_stage(A: np.ndarray, cert_dir: Path, result: dict, llm=None, trace=None, baselines=None, truth=None,
              code_path: Path | None = None) -> dict:
    from ..board import Trace

    trace = trace or Trace(None)
    ch = run_checks(A, cert_dir, result)
    spec = ch.pop("_spectral")
    for c in ch["checks"]:
        trace.log("verify_check", **{k: v for k, v in c.items()})
    certs = json.loads((cert_dir / "certificates.json").read_text())
    lower_vertices = (certs.get("lower") or {}).get("vertices")
    analysis, extra = None, []
    if llm is not None:
        from .. import agents as AG

        payload = verifier_payload(result, ch, baselines, truth)
        rep = llm.ask("verifier", AG.system("verifier"), payload, AG.validate_verifier, trace)
        if rep is not None and rep.get("checks_requested"):
            for req in rep["checks_requested"][:3]:
                r = extra_check(A, spec, req, lower_vertices)
                extra.append(r)
                trace.log("verify_check", source="verifier_request", **r)
                if not r.get("ok", True):
                    ch["ok"] = False
            payload2 = payload + "\n\nEXTRA CHECKS YOU REQUESTED\n" + json.dumps(extra, default=str)
            rep2 = llm.ask("verifier", AG.system("verifier"), payload2 + "\nNow write the final analysis "
                           "(checks_requested must be empty).", AG.validate_verifier, trace)
            rep = rep2 or rep
        analysis = rep.get("analysis") if rep else None
    v = verdict(ch, result)
    sim = next((c for c in ch["checks"] if c["check"] == "decode_simulation"), None)
    if code_path is not None and v == "proven":
        write_code_file(A, spec, code_path, result["graph"], result, sim)
    trace.log("verdict", verdict=v, analysis=analysis, checks_ok=ch["ok"])
    return {"verdict": v, "checks": ch["checks"], "extra_checks": extra, "checks_seconds": ch["seconds"],
            "analysis": analysis, "all_certificates_verified": ch["ok"]}
