"""Compare the adaptive agent against the baselines on accuracy (gap closed, correct field) and work (fields tried).

Baselines, all starting from the same alpha (lower) and clique cover (upper):
  clique_cover_only   no minrank construction (the classic upper bound)
  fixed_F2 / F3 / F5  the spectral construction over one fixed field only
  brute_all_fields    the spectral construction over every field (the accuracy ceiling for this construction)
  agent (spectrum)    multi-field, fields tried in spectrum-predicted order, stop when the gap closes

    python minrank-agent/study.py run      -> results/study.jsonl
    python minrank-agent/study.py report
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from benchmark import benchmark  # noqa: E402
from minrank_agent import bounds as B  # noqa: E402
from minrank_agent.agent import FIELDS, SpectrumAgent, real_spectrum, solve, spectrum_order  # noqa: E402

OUT = HERE / "results" / "study.jsonl"


def alpha_of(g):
    if getattr(g, "alpha_known", None) is not None:
        return g.alpha_known, True
    a, _, opt = B.independence(g.A, 15.0)
    return a, opt


def analyse(g):
    a, aopt = alpha_of(g)
    cc, _, _ = B.clique_cover(g.A, 15.0)
    spec = {}
    for p in FIELDS:
        t = time.perf_counter()
        cost, c, lam = B.spectral_field(g.A, p)
        assert B.verify_spectral(g.A, p, c, cost)
        spec[p] = {"cost": cost, "s": time.perf_counter() - t}
    spectrum = real_spectrum(g.A)
    # agent: spectrum-ordered, stop as soon as spectral meets alpha
    order = spectrum_order(g.n, spectrum, FIELDS)
    best, tried, agent_field = cc, 0, None
    for p in order:
        tried += 1
        if spec[p]["cost"] < best:
            best, agent_field = spec[p]["cost"], p
        if best <= a:
            break
    best_brute = min(cc, min(spec[p]["cost"] for p in FIELDS))
    brute_field = next((p for p in FIELDS if spec[p]["cost"] == best_brute), None)
    return {
        "graph": g.name, "family": g.family, "n": g.n, "alpha": a, "alpha_opt": aopt, "clique_cover": cc,
        "spectral": {str(p): spec[p]["cost"] for p in FIELDS},
        "spectral_time": {str(p): round(spec[p]["s"], 4) for p in FIELDS},
        "agent_upper": best, "agent_fields_tried": tried, "agent_field": agent_field,
        "brute_upper": best_brute, "brute_field": brute_field,
        "closed_agent": best <= a and aopt, "closed_brute": best_brute <= a and aopt,
        "order": order,
    }


def cmd_run():
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        for g in benchmark():
            r = analyse(g)
            f.write(json.dumps(r) + "\n")
            f.flush()
            tag = f"beta={r['alpha']}/F{r['agent_field']}" if r["closed_agent"] else f"[{r['alpha']},{r['agent_upper']}]"
            print(f"{r['graph']:12s} n={r['n']:3d} a={r['alpha']:3d} clique={r['clique_cover']:3d} "
                  f"spectral={r['spectral']} agent={tag} tried={r['agent_fields_tried']}", flush=True)


def cmd_report():
    rows = [json.loads(x) for x in OUT.read_text().splitlines() if x.strip()]
    def closed_by_field(p):
        return sum(1 for r in rows if r["alpha_opt"] and min(r["clique_cover"], r["spectral"][str(p)]) <= r["alpha"])
    n = len(rows)
    lines = [f"{n} structured graphs. 'closed' = beta proven exactly (alpha = upper, explicit verified code).", ""]
    lines.append("Accuracy -- how many gaps each strategy closes:")
    lines.append(f"  clique cover only          {sum(r['clique_cover'] <= r['alpha'] and r['alpha_opt'] for r in rows):2d}/{n}")
    for p in FIELDS:
        lines.append(f"  fixed field F{p:<2d}             {closed_by_field(p):2d}/{n}")
    lines.append(f"  multi-field (agent/brute)  {sum(r['closed_brute'] for r in rows):2d}/{n}")
    lines.append("")
    tot_gap = lambda key: sum(r[key] - r["alpha"] for r in rows)
    lines.append(f"Total remaining gap (sum of upper-alpha): clique-only {tot_gap('clique_cover')}, "
                 f"best fixed field {min(sum(min(r['clique_cover'], r['spectral'][str(p)]) - r['alpha'] for r in rows) for p in FIELDS)}, "
                 f"multi-field {tot_gap('agent_upper')}")
    closed = [r for r in rows if r["closed_agent"]]
    if closed:
        avg_tried = sum(r["agent_fields_tried"] for r in closed) / len(closed)
        lines.append(f"\nAmong {len(closed)} closed graphs: agent tries {avg_tried:.2f} fields on average "
                     f"(ascending-order would try {sum(min([i+1 for i,p in enumerate(FIELDS) if r['spectral'][str(p)]<=r['alpha']] or [len(FIELDS)]) for r in closed)/len(closed):.2f}); "
                     f"brute tries all {len(FIELDS)}.")
        from collections import Counter
        lines.append("  gap-closing field used: " + str(dict(Counter(r["agent_field"] for r in closed))))
    lines.append("\nBy family (closed / count, and which fields close them):")
    from collections import Counter, defaultdict
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    for f in sorted(fam):
        rs = fam[f]
        c = sum(r["closed_brute"] for r in rs)
        fields = Counter(r["agent_field"] for r in rs if r["closed_agent"])
        lines.append(f"  {f:12s} {c:2d}/{len(rs):<2d}  fields: {dict(fields)}")
    text = "\n".join(lines)
    (HERE / "results" / "study_report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    {"run": cmd_run, "report": cmd_report}[sys.argv[1]]()
