"""Tables for RESULTS.md from results/runs/<tag>/*.json and results/baselines/*.json.

    python -m kagent.report > results/tables.md
"""
from __future__ import annotations

import json
import statistics as st
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODES = ["full", "no_critic", "no_llm", "small_model"]


def load_runs(tag: str) -> dict:
    d = ROOT / "results" / "runs" / tag
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))} if d.exists() else {}


def truth() -> dict:
    return json.loads((ROOT / "bench" / "truth.json").read_text())


def baselines(gid: str) -> dict:
    f = ROOT / "results" / "baselines" / f"{gid}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def fmt_br(L, U) -> str:
    if L is None and U is None:
        return "n/a"
    if L is not None and U is not None and Fraction(str(L)) >= Fraction(str(U)):
        return f"**{U}**"
    return f"[{L if L is not None else '?'}, {U if U is not None else '?'}]"


def agent_time(r: dict):
    return r["t_proof"] if r["proven"] else r.get("t_final_change")


def hyp_correct(r: dict, t: dict) -> bool | None:
    h = (r.get("hypothesis") or {})
    name = (h.get("name") or "").lower()
    if not name or r.get("plan_source", "").startswith("default"):
        return None
    if t["family"] == "kneser":
        if "kneser" not in name:
            return False
        p = h.get("params") or {}
        vals = {str(v) for v in p.values()}
        return str(t["n"]) in vals and str(t["k"]) in vals if p else True
    perturbed = any(w in name for w in ("perturb", "near", "almost", "delet", "minus", "modified", "noisy"))
    return ("kneser" not in name) or perturbed


def main_table(tag: str = "full") -> tuple[str, dict]:
    T = truth()
    runs = load_runs(tag)
    rows = ["| graph | receivers | group | agent | proven | agent time (s) | field | standard best | std time (s) | "
            "Pincer | minrank-agent |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    stats = {"agent": 0, "standard": 0, "pincer": 0, "minrank_agent": 0, "speedups": [], "only_agent": [],
             "agent_lost": [], "n": 0}
    for gid, t in sorted(T.items(), key=lambda x: (x[1]["group"] != "main", x[1]["receivers"])):
        if t["seed"] != 0 or gid not in runs:
            continue
        r = runs[gid]
        b = baselines(gid)
        sb = b.get("standard_best", {})
        pc = b.get("pincer", {})
        ma = b.get("minrank_agent", {})
        stats["n"] += 1
        ag_exact = r["proven"] and r.get("verification", {}).get("verdict") == "proven"
        sb_exact = bool(sb.get("exact"))
        pc_exact = pc.get("status") == "ok" and Fraction(str(pc["lower"])) >= Fraction(str(pc["upper"]))
        ma_exact = ma.get("status") == "ok" and bool(ma.get("exact"))
        stats["agent"] += ag_exact
        stats["standard"] += sb_exact
        stats["pincer"] += pc_exact
        stats["minrank_agent"] += ma_exact
        at = agent_time(r)
        if ag_exact and sb_exact and at and sb.get("time"):
            stats["speedups"].append((t["name"], sb["time"] / max(at, 1e-3)))
        if ag_exact and not (sb_exact or pc_exact or ma_exact):
            stats["only_agent"].append(t["name"])
        if (sb_exact and not ag_exact) or (ag_exact and sb_exact and at and sb.get("time") and sb["time"] < at):
            stats["agent_lost"].append(t["name"])
        pcs = fmt_br(pc.get("lower"), pc.get("upper")) if pc.get("status") == "ok" else pc.get("status", "-")
        mas = fmt_br(ma.get("lower"), ma.get("upper")) if ma.get("status") == "ok" else ma.get("status", "-")
        rows.append(f"| {t['name']} | {t['receivers']} | {t['group']} | {fmt_br(r['L'], r['U'])} | "
                    f"{'yes' if ag_exact else 'no'} | {at if at is None else round(at, 1)} | {r.get('field')} | "
                    f"{fmt_br(sb.get('lower'), sb.get('upper')) if sb else '-'} | {sb.get('time', '-')} | "
                    f"{pcs} ({pc.get('time', '-')} s) | {mas} ({ma.get('time', '-')} s) |")
    return "\n".join(rows), stats


def ablation_table() -> str:
    T = truth()
    rows = ["| run | graphs | proven | median time to proof (s) | total gap | LLM calls / graph | tokens in+out / graph | "
            "LLM s / graph | hypothesis correct | critic rounds / graph | all certificates verified |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in MODES:
        runs = {g: r for g, r in load_runs(tag).items() if g in T and T[g]["seed"] == 0}
        if not runs:
            continue
        pr = [r for r in runs.values() if r["proven"]]
        tp = [r["t_proof"] for r in pr if r.get("t_proof") is not None]
        gap = sum(float(Fraction(r["U"]) - r["L"]) for r in runs.values())
        llm = [r.get("llm") or {} for r in runs.values()]
        calls = st.mean([u.get("calls", 0) for u in llm]) if llm else 0
        toks = st.mean([u.get("tokens_in", 0) + u.get("tokens_out", 0) for u in llm]) if llm else 0
        secs = st.mean([u.get("seconds", 0) for u in llm]) if llm else 0
        hc = [hyp_correct(r, T[g]) for g, r in runs.items()]
        hc = [x for x in hc if x is not None]
        cr = st.mean([r.get("critic_rounds", 0) for r in runs.values()])
        ver = all(r.get("verification", {}).get("all_certificates_verified") for r in runs.values())
        rows.append(f"| {tag} | {len(runs)} | {len(pr)} | {st.median(tp) if tp else '-'} | {gap:g} | {calls:.1f} | "
                    f"{toks:.0f} | {secs:.1f} | {f'{sum(hc)}/{len(hc)}' if hc else '-'} | {cr:.1f} | "
                    f"{'yes' if ver else 'NO'} |")
    lo = ROOT / "results" / "llm_only.json"
    if lo.exists():
        d = json.loads(lo.read_text())
        ok = sum(1 for x in d if x.get("beta_correct"))
        fok = sum(1 for x in d if x.get("field_gives_beta"))
        rows.append(f"| llm_only | {len(d)} | beta right {ok}/{len(d)}, field right {fok}/{len(d)} | - | - | 1 | - | "
                    f"- | - | - | answers checked, not trusted |")
    return "\n".join(rows)


def seeds_table(names: list[str] | None = None) -> str:
    T = truth()
    rows = ["| graph | mode | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|---|"]
    by_name = {}
    for gid, t in T.items():
        by_name.setdefault(t["name"], {})[t["seed"]] = gid
    for name, sd in sorted(by_name.items()):
        if len(sd) < 3:
            continue
        for mode in ("full", "no_llm"):
            cells = []
            for s in (0, 1, 2):
                tag = mode if s == 0 else f"{mode}_s{s}"
                r = load_runs(tag).get(sd[s])
                cells.append("-" if r is None else (f"{r['t_proof']} s" if r["proven"] else f"[{r['L']}, {r['U']}]"))
            rows.append(f"| {name} | {mode} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main() -> int:
    tab, s = main_table("full" if load_runs("full") else "no_llm")
    print("## Main table\n\n" + tab + "\n")
    sp = [x for _, x in s["speedups"]]
    print(f"proven: agent {s['agent']}/{s['n']}, standard best {s['standard']}/{s['n']}, Pincer {s['pincer']}/{s['n']}, "
          f"minrank-agent {s['minrank_agent']}/{s['n']}")
    if sp:
        print(f"speed-up where both exact ({len(sp)} graphs): median {st.median(sp):.1f}x, min {min(sp):.2f}x, "
              f"max {max(sp):.1f}x")
    print(f"only the agent proves: {', '.join(s['only_agent']) or 'none'}")
    print(f"agent lost (accuracy or time): {', '.join(s['agent_lost']) or 'none'}\n")
    print("## Ablations\n\n" + ablation_table() + "\n")
    print("## Seeds\n\n" + seeds_table() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
