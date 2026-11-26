"""Build docs/IE624_progress_report_v2.{html,pdf} from result files (WeasyPrint; no LaTeX on the local machine).

Usage: python3 analysis/report_figures.py && python3 analysis/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAMS = ["threshold", "threshold+plan", "unified", "majority", "random_collective", "theory", "theory+plan"]
NAME = {"threshold": "Per-sample threshold", "threshold+plan": "Threshold + exit plan", "unified": "Unified exit",
        "majority": "Majority vote", "random_collective": "Random search (150 rules)",
        "theory": "Cost rule, all exits (ours)", "theory+plan": "Cost rule + exit plan (ours)"}


def local() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("results/local_time_*_cuda_T*.csv"))], ignore_index=True)
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    return d


def cell_table(d: pd.DataFrame, task: str) -> str:
    """Measured speedup per (sequence length, batch size) for one task; best in bold; * = held-out drop > 1 pt."""
    g = d[d.task == task]
    cols = sorted({(t, b) for t, b in zip(g["T"], g.B)})
    h = "<tr><th rowspan=2>Policy</th>" + "".join(
        f"<th colspan={sum(1 for c in cols if c[0] == T)}>sequence length {T}</th>" for T in sorted({c[0] for c in cols})) + "</tr>"
    h += "<tr>" + "".join(f"<th>B={b}</th>" for _, b in cols) + "</tr>"
    best = {c: g[(g["T"] == c[0]) & (g.B == c[1]) & g.family.isin(FAMS)].real_speedup.max() for c in cols}
    body = ""
    for f in ["heads_only"] + FAMS:
        row = f"<td>{NAME.get(f, 'All heads, no exits')}</td>"
        for c in cols:
            r = g[(g["T"] == c[0]) & (g.B == c[1]) & (g.family == f)]
            if r.empty:
                row += "<td>–</td>"
                continue
            r = r.iloc[0]
            s = f"{r.real_speedup:.2f}"
            if f in FAMS and r.real_speedup >= best[c] - 1e-9:
                s = f"<b>{s}</b>"
            if r.ho_drop_pts > 1.0:
                s += "*"
            row += f"<td>{s}</td>"
        body += f"<tr class='{'ours' if 'ours' in NAME.get(f, '') else ''}'>{row}</tr>"
    return f"<table class='num'>{h}{body}</table>"


def regret_table(d: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Per family over all measured cells: mean and worst speedup relative to the best policy of the cell."""
    g = d[d.family.isin(FAMS)].copy()
    g["best"] = g.groupby(["task", "T", "B"]).real_speedup.transform("max")
    g["rel"] = g.real_speedup / g.best
    g["err"] = (g.model_speedup - g.real_speedup).abs() / g.real_speedup
    g["aerr"] = (g.affine_speedup - g.real_speedup).abs() / g.real_speedup
    s = g.groupby("family").agg(mean=("rel", "mean"), worst=("rel", "min"), wins=("rel", lambda x: int((x > 0.97).sum())),
                                n=("rel", "size"), viol=("ho_drop_pts", lambda x: int((x > 1.0).sum())),
                                maxdrop=("ho_drop_pts", "max")).reindex(FAMS)
    rows = "".join(
        f"<tr class='{'ours' if 'ours' in NAME[f] else ''}'><td>{NAME[f]}</td><td>{r['mean']:.3f}</td><td>{r['worst']:.3f}</td>"
        f"<td>{int(r['wins'])} / {int(r['n'])}</td><td>{int(r['viol'])} / {int(r['n'])}</td><td>{r['maxdrop']:.2f}</td></tr>"
        for f, r in s.iterrows())
    t = ("<table class='num'><tr><th>Policy</th><th>Mean speed relative to the cell's best</th><th>Worst cell</th>"
         "<th>Cells within 3% of the best</th><th>Cells over the 1-pt budget (held-out)</th><th>Largest held-out drop (pts)</th></tr>"
         + rows + "</table>")
    return t, g


def resplit_table() -> str:
    out = ""
    for f in sorted(ROOT.glob("results/local_resplit_*_cuda_T16.csv")):
        d = pd.read_csv(f)
        task = f.name.split("_")[2].upper()
        for (B, fam), g in d.groupby(["B", "family"]):
            out += (f"<tr><td>{task}</td><td>{B}</td><td>{NAME[fam].replace(' (ours)', '')}</td>"
                    f"<td>{100 * (g.naive_drop > 0.01).mean():.0f}%</td><td>{100 * (g.cert_drop > 0.01).mean():.0f}%</td>"
                    f"<td>{100 * g.naive_drop.mean():.2f}</td><td>{100 * g.cert_drop.mean():.2f}</td>"
                    f"<td>{g.naive_speedup.mean():.2f}×</td><td>{g.cert_speedup.mean():.2f}×</td><td>{len(g)}</td></tr>")
    if not out:
        return ""
    return ("<table class='num'><tr><th>Task</th><th>B</th><th>Policy</th><th>Over budget, old protocol</th>"
            "<th>Over budget, certified</th><th>Mean drop, old (pts)</th><th>Mean drop, certified (pts)</th>"
            "<th>Speedup, old (cost model)</th><th>Speedup, certified (cost model)</th><th>Re-splits</th></tr>" + out + "</table>")


def front() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("results/local_front_*_cuda_T*.csv"))], ignore_index=True)
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    return d


def matched(d: pd.DataFrame, cap: float = 1.0) -> pd.DataFrame:
    """Fastest front point of each family whose held-out drop is at most `cap` points; index family, columns (task, T, B)."""
    return d[d.ho_drop_pts <= cap].groupby(["family", "task", "T", "B"]).ho_speedup.max().unstack(["task", "T", "B"])


def front_table(d: pd.DataFrame, task: str) -> str:
    m = matched(d)[task]
    cols = sorted(m.columns)
    h = "<tr><th rowspan=2>Policy</th>" + "".join(
        f"<th colspan={sum(1 for c in cols if c[0] == T)}>sequence length {T}</th>" for T in sorted({c[0] for c in cols})) + "</tr>"
    h += "<tr>" + "".join(f"<th>B={b}</th>" for _, b in cols) + "</tr>"
    body = ""
    for f in FAMS:
        row = f"<td>{NAME[f]}</td>"
        for c in cols:
            x = m.loc[f, c] if f in m.index else np.nan
            row += "<td>–</td>" if np.isnan(x) else (f"<td><b>{x:.2f}</b></td>" if x >= m[c].max() - 0.005 else f"<td>{x:.2f}</td>")
        body += f"<tr class='{'ours' if 'ours' in NAME[f] else ''}'>{row}</tr>"
    return f"<table class='num'>{h}{body}</table>"


def hard_table(d: pd.DataFrame) -> str:
    a = d[d.alpha_pts == 1.0].groupby(["family", "task", "T"])[["ho_drop_pts", "hard_drop_pts", "easy_drop_pts"]].mean()
    groups = [(t, T) for t in ("qqp", "mnli") for T in (16, 128) if (t, T) in {(i[1], i[2]) for i in a.index}]
    h = "<tr><th rowspan=2>Policy</th>" + "".join(f"<th colspan=3>{t.upper()}, length {T}</th>" for t, T in groups) + "</tr>"
    h += "<tr>" + "<th>all</th><th>hard 25%</th><th>other 75%</th>" * len(groups) + "</tr>"
    body = ""
    for f in FAMS:
        row = f"<td>{NAME[f]}</td>"
        for t, T in groups:
            r = a.loc[(f, t, T)] if (f, t, T) in a.index else None
            row += "<td>–</td>" * 3 if r is None else f"<td>{r.ho_drop_pts:.2f}</td><td>{r.hard_drop_pts:.2f}</td><td>{r.easy_drop_pts:.2f}</td>"
        body += f"<tr class='{'ours' if 'ours' in NAME[f] else ''}'>{row}</tr>"
    return f"<table class='num'>{h}{body}</table>"


def cost_rows() -> str:
    out = ""
    for T in (16, 32, 128):
        d = json.loads((ROOT / f"results/local_cost_cuda_T{T}.json").read_text())
        y = {p["B"]: p["full_ms"] / 12 for p in d["points"]}
        out += (f"<tr><td>{T}</td><td>{y[1]:.2f}</td><td>{y[8]:.2f}</td><td>{y[64]:.2f}</td><td>{d['head']:.2f}</td>"
                f"<td>{max(d['rho8'], 0):.2f}</td><td>{max(d['rho64'], 0):.3f}</td><td>{100 * d['affine_max_rel_err']:.0f}%</td></tr>")
    return out


# ---- Section 5 results (A6000 runs and CPU replays, 16-23 Nov) -----------------------------------------------------
OURS = ["theory", "theory+plan"]
BASE = [f for f in FAMS if f not in OURS]
MODEL = {"roberta": "RoBERTa-base", "llama": "Llama-3.1-8B", "qwen": "Qwen2.5-1.5B"}
short = lambda f: NAME[f].replace(" (ours)", "").replace(" (150 rules)", "")


def a6000_time() -> pd.DataFrame:
    """Timed A6000 runs (eager, certified selection, held-out): results/local_time_<task>[_<llm>]_a6000_T<T>.csv."""
    import re
    out = []
    for f in sorted(ROOT.glob("results/local_time_*_a6000_T*.csv")):
        m = re.fullmatch(r"local_time_([a-z0-9]+?)(?:_(llama|qwen))?_a6000_T(\d+)\.csv", f.name)
        d = pd.read_csv(f)
        d["model"], d["T"] = m.group(2) or "roberta", int(m.group(3))
        d["task"] = m.group(1)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def timed_grid(groups: list[tuple[str, pd.DataFrame]]) -> str:
    """Speedup per family (rows) and (group, B) (columns). Bold: fastest policy within the 1-pt budget in the column;
    * held-out drop above one point; –: no certified policy."""
    h = "<tr><th rowspan=2>Policy</th>" + "".join(
        f"<th colspan={g.B.nunique()}>{lab}</th>" for lab, g in groups) + "</tr>"
    h += "<tr>" + "".join(f"<th>B={b}</th>" for _, g in groups for b in sorted(g.B.unique())) + "</tr>"
    body = ""
    for f in FAMS:
        row = f"<td>{NAME[f]}</td>"
        for _, g in groups:
            for b in sorted(g.B.unique()):
                c = g[(g.B == b) & g.family.isin(FAMS)]
                ok = c[c.ho_drop_pts <= 1.0]
                r = c[c.family == f]
                if r.empty:
                    row += "<td>–</td>"
                    continue
                r = r.iloc[0]
                s = f"{r.real_speedup:.2f}"
                if r.ho_drop_pts <= 1.0 and r.real_speedup >= ok.real_speedup.max() - 1e-9:
                    s = f"<b>{s}</b>"
                if r.ho_drop_pts > 1.0:
                    s += "*"
                row += f"<td>{s}</td>"
        body += f"<tr class='{'ours' if f in OURS else ''}'>{row}</tr>"
    return f"<table class='num'>{h}{body}</table>"


def timed_extra() -> str:
    """5.1 runs that need a working A6000: CUDA graphs (RoBERTa) and CIFAR-100 (ViT-B/16, eager and graphs).
    Rendered from results/local_time_{qqp,mnli}_a6000_graph_T*.csv and results/local_time_cifar100_a6000[_graph]_T197.csv
    when they exist; otherwise a pending note."""
    def load(pattern):
        fs = sorted(ROOT.glob(pattern))
        return pd.concat([pd.read_csv(f).assign(src=f.name) for f in fs], ignore_index=True) if fs else None
    html, have = "", []
    g = load("results/local_time_*_a6000_graph_T*.csv")
    rob_g = None if g is None else g[g.task.isin(["qqp", "mnli"])]
    if rob_g is not None and len(rob_g):
        rob_g = rob_g.assign(T=rob_g.device.str.extract(r"_T(\d+)$")[0].astype(int))
        groups = [(f"{t.upper()}, length {T}", rob_g[(rob_g.task == t) & (rob_g["T"] == T)])
                  for t in ("qqp", "mnli") for T in sorted(rob_g["T"].unique()) if len(rob_g[(rob_g.task == t) & (rob_g["T"] == T)])]
        html += (timed_grid(groups) + "<p class='cap'><b>Table 11b:</b> the same as Tables 11–12 under CUDA graphs "
                 "(<code>--exec graph</code>: every segment replayed as a captured graph, rows padded to fixed buckets). "
                 "<code>results/local_time_{qqp,mnli}_a6000_graph_T*.csv</code>.</p>")
        have.append("CUDA graphs")
    c = load("results/local_time_cifar100_a6000*_T197.csv")
    if c is not None and len(c):
        groups = [(("CUDA graphs" if "graph" in dv else "eager"), c[c.device == dv]) for dv in sorted(c.device.unique())]
        html += (timed_grid(groups) + "<p class='cap'><b>Table 12b:</b> CIFAR-100, ViT-B/16 (197 tokens), RTX A6000, "
                 "same protocol. <code>results/local_time_cifar100_a6000[_graph]_T197.csv</code>.</p>")
        have.append("CIFAR-100")
    if not have:
        return ("<p class='fb'><i class='l'>Pending:</i> the CUDA-graph and CIFAR-100 (ViT) timings. The code and scripts "
                "are ready (<code>scripts/run_server_a6000_graph.sh</code>, <code>scripts/run_server_a6000_vit.sh</code>), "
                "and <code>scripts/wait_gpu_then_graph.sh</code> starts them as soon as an A6000 can initialise CUDA.</p>")
    rows = []
    for name, d in (("graph", rob_g), ("cifar", c)):
        if d is None or not len(d):
            continue
        for k, gg in d[d.family.isin(FAMS)].groupby(["task", "device", "B"]):
            ok = gg[gg.ho_drop_pts <= 1.0]
            o, b = ok[ok.family.isin(OURS)], ok[ok.family.isin(BASE)]
            if len(o) and len(b):
                rows.append(100 * (o.real_speedup.max() / b.real_speedup.max() - 1))
    summ = (f" Within the one-point budget, the cost rule is ahead of the best baseline in {sum(r > 0 for r in rows)} "
            f"of {len(rows)} columns where both have a certified policy (median {np.median(rows):+.0f}%)." if rows else "")
    return html + f"<p class='fb'><i class='l'>Result ({' and '.join(have)}):</i> Tables above.{summ}</p>"


def in_budget_best(d: pd.DataFrame) -> pd.DataFrame:
    """Per (model, task, T, B): fastest of ours and fastest baseline within the 1-pt budget (held-out, timed)."""
    rows = []
    for k, g in d[d.family.isin(FAMS)].groupby(["model", "task", "T", "B"]):
        ok = g[g.ho_drop_pts <= 1.0]
        o, b = ok[ok.family.isin(OURS)], ok[ok.family.isin(BASE)]
        r = dict(zip(["model", "task", "T", "B"], k))
        r["ours"] = o.real_speedup.max() if len(o) else np.nan
        r["ours_fam"] = o.loc[o.real_speedup.idxmax(), "family"] if len(o) else ""
        r["base"] = b.real_speedup.max() if len(b) else np.nan
        r["base_fam"] = b.loc[b.real_speedup.idxmax(), "family"] if len(b) else ""
        r["theory"] = g[g.family == "theory"].real_speedup.max()
        r["plan"] = g[g.family == "theory+plan"].real_speedup.max()
        r["ours_drop_min"] = g[g.family.isin(OURS)].ho_drop_pts.min()
        r["ours_drop_max"] = g[g.family.isin(OURS)].ho_drop_pts.max()
        rows.append(r)
    s = pd.DataFrame(rows)
    s["gain"] = 100 * (s.ours / s.base - 1)
    return s


def cert_table() -> tuple[str, dict]:
    """Item 5.2 from results/cert_check_<task>_summary.csv (30 re-splits x 4 batch sizes; cost-table speedups)."""
    rows, x = "", {}
    for task, lab in [("qqp", "QQP"), ("mnli", "MNLI"), ("cifar100", "CIFAR-100"), ("mnli_d05", "MNLI, δ = 5%")]:
        p = ROOT / f"results/cert_check_{task}_summary.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)

        def cell(m, key="drop", fam="theory+plan", frac=1.0):
            return d[(d.frac == frac) & (d.key == key) & (d.method == m) & (d.family == fam)]
        t = d[(d.frac == 1.0) & (d.key == "drop") & (d.method == "t")].groupby("family").agg(
            viol=("viol", "mean"), sp=("speedup", "mean"))
        wsr, hb, lf, small = cell("wsr"), cell("hb"), cell("t", "disagree"), cell("t", fam="theory", frac=0.1)
        small_rc = cell("t", fam="random_collective", frac=0.1)
        e = x[task] = dict(viol=t.viol.mean(), viol_max=t.viol.max(), viol_max_fam=t.viol.idxmax(), t=t,
                           wsr=wsr.speedup.mean() if len(wsr) else np.nan,
                           wsr_chose=wsr.chose.mean() if len(wsr) else np.nan,
                           hb_chose=hb.chose.max() if len(hb) else np.nan, lf=lf.speedup.mean(), lf_viol=lf.viol.mean(),
                           small=small.speedup.mean(), small_viol=small.viol.mean(), small_rc=small_rc.speedup.mean())
        wsr_s = "–" if np.isnan(e["wsr"]) else f"{e['wsr']:.2f}× ({100 * e['wsr_chose']:.0f}%)"
        hb_s = "–" if np.isnan(e["hb_chose"]) else ("never" if e["hb_chose"] == 0 else f"{100 * e['hb_chose']:.0f}%")
        rows += (f"<tr><td>{lab}</td><td>{100 * e['viol']:.1f}% ({100 * e['viol_max']:.1f}%, {short(e['viol_max_fam'])})</td>"
                 f"<td>{t.loc['theory+plan', 'sp']:.2f}× ({100 * t.loc['theory+plan', 'viol']:.1f}%)</td><td>{wsr_s}</td>"
                 f"<td>{hb_s}</td><td>{e['lf']:.2f}× ({100 * e['lf_viol']:.1f}%)</td>"
                 f"<td>{t.loc['theory', 'sp']:.2f}× → {e['small']:.2f}× ({100 * e['small_viol']:.1f}%)</td></tr>")
    tab = ("<table class='num'><tr><th>Task</th><th>t-test: share over budget, mean of 5 methods (worst)</th>"
           "<th>Cost rule + plan, t-test (share over budget)</th><th>Same, WSR bound (share of cases with a certified policy)</th>"
           "<th>Hoeffding–Bentkus</th><th>Cost rule + plan, label-free t-test (over budget)</th>"
           "<th>Cost rule, all → 10% of calibration data (over budget)</th></tr>" + rows + "</table>")
    return tab, x


def load_table() -> tuple[str, dict]:
    """Item 5.3 from results/load_trace_<task>_levels.csv: tuned-once rule vs re-tuned and tuned-once baselines."""
    rows, x = "", {}
    for task, lab in [("qqp", "QQP"), ("mnli", "MNLI"), ("mnli_d05", "MNLI, δ = 5%"), ("cifar100", "CIFAR-100")]:
        p = ROOT / f"results/load_trace_{task}_levels.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        ok = d[d.drop_pts <= 1.0]
        get = lambda f, tu: d[(d.family == f) & (d.tuning == tu)].iloc[0]
        rb = ok[ok.family.isin(BASE) & (ok.tuning == "per_B")].sort_values("speedup").iloc[-1]
        ob = ok[ok.family.isin(BASE) & ok.tuning.str.startswith("once")].sort_values("speedup").iloc[-1]
        cells, verdict = "", []
        for f in OURS:
            r = get(f, "once@64")
            within = r.drop_pts <= 1.0
            passed = within and r.speedup >= 0.95 * rb.speedup and r.speedup > ob.speedup
            verdict.append(passed)
            cells += f"<td>{r.speedup:.2f}× ({r.drop_pts:.2f}{'' if within else '*'})</td>"
            x[(task, f)] = dict(sp=r.speedup, drop=r.drop_pts, passed=passed)
        x[task] = dict(per_b=rb, once=ob)
        res = "pass" if any(verdict) else "fail"
        if any(verdict) and not all(verdict):
            res += " (plan only)" if verdict[1] else " (all exits only)"
        rows += (f"<tr><td>{lab}</td>{cells}<td>{rb.speedup:.2f}× ({short(rb.family)}, {rb.drop_pts:.2f})</td>"
                 f"<td>{ob.speedup:.2f}× ({short(ob.family)} {ob.tuning}, {ob.drop_pts:.2f})</td><td>{res}</td></tr>")
    tab = ("<table class='num'><tr><th>Task</th><th>Cost rule, tuned once at B = 64 (drop)</th>"
           "<th>Cost rule + plan, tuned once (drop)</th><th>Best baseline re-tuned per batch size, within budget</th>"
           "<th>Best baseline tuned once, within budget</th><th>Criterion</th></tr>" + rows + "</table>")
    return tab, x


def bandit_summary() -> dict:
    """Item 5.4 from results/bandit_slopes_<setting>_B64_10seeds.csv."""
    out = {}
    for f in sorted(ROOT.glob("results/bandit_slopes_*_B64_10seeds.csv")):
        d = pd.read_csv(f).set_index("variant")
        out[f.name[len("bandit_slopes_"):-len("_B64_10seeds.csv")]] = d
    return out


def robust_summary() -> tuple[pd.DataFrame, int, int]:
    """Item 5.5 from results/robust_<setting>_B64_cert.csv and the pathwise-bound checks next to them."""
    rows, viol, checks = [], 0, 0
    for f in sorted(ROOT.glob("results/robust_*_B64_cert.csv")):
        if f.name.startswith("robust_violations"):
            continue
        d = pd.read_csv(f)
        d["setting"] = f.name[len("robust_"):-len("_B64_cert.csv")]
        rows.append(d)
        v = pd.read_csv(f.with_name(f.name.replace("robust_", "robust_violations_")))
        viol, checks = viol + int(v.violations.sum()), checks + int(v.batches.sum())
    return pd.concat(rows, ignore_index=True), viol, checks


def rng(x, fmt="{:.2f}", unit="") -> str:
    """'lo–hi' (or one value) of a series, formatted."""
    lo, hi = fmt.format(min(x)), fmt.format(max(x))
    return f"{lo}{unit}" if lo == hi else f"{lo}–{hi}{unit}"


def section5() -> dict:
    """HTML pieces with the results of items 5.1-5.5, the LLM classifiers and the MNLI delta check, all computed from
    the result files named in each caption."""
    out = {}
    d = a6000_time()
    s = in_budget_best(d)
    rob = d[d.model == "roberta"]
    out["t51"] = (timed_grid([(f"QQP, length {T}", rob[(rob.task == "qqp") & (rob["T"] == T)]) for T in (128, 32)])
                  + "<p class='cap'><b>Table 11:</b> QQP, RTX A6000, eager PyTorch, trained RoBERTa-base checkpoint, "
                  "real inputs: measured speedup of each method's certified policy on held-out inputs (2,048 per pass, "
                  "5 rounds, median paired speedup). Bold: fastest policy within the one-point budget in the column. "
                  "* held-out drop above one point. <code>results/local_time_qqp_a6000_T{128,32}.csv</code>.</p>"
                  + timed_grid([(f"MNLI, length {T}", rob[(rob.task == "mnli") & (rob["T"] == T)]) for T in (128, 32)])
                  + "<p class='cap'><b>Table 12:</b> MNLI, same layout. <code>results/local_time_mnli_a6000_T{128,32}.csv</code>.</p>")
    out["t51x"] = timed_extra()
    q = s[(s.model == "roberta") & (s.task == "qqp") & (s["T"] == 128)]
    m = s[(s.model == "roberta") & (s.task == "mnli") & (s["T"] == 128)]
    q32 = s[(s.model == "roberta") & (s.task == "qqp") & (s["T"] == 32) & (s.B <= 16)]
    m32 = s[(s.model == "roberta") & (s.task == "mnli") & (s["T"] == 32) & (s.B <= 16)]
    assert (q.ours_fam == "theory+plan").all() and m.ours.isna().all() and (q32.ours_fam == "theory").all()
    rq = rob[(rob.task == "qqp") & (rob["T"] == 128) & (rob.family == "theory+plan")]
    p32 = rob[(rob.task == "qqp") & (rob["T"] == 32) & (rob.B <= 16) & rob.family.isin(["theory+plan", "unified"])]
    assert (p32.ho_drop_pts > 1.0).all() and (m32.base_fam == "unified").all()
    rm = rob[(rob.task == "mnli") & (rob["T"] == 128)]
    rc_over = rm[(rm.family == "random_collective") & (rm.B >= 16)]
    assert (rc_over.ho_drop_pts > 1.0).all()
    out["r51"] = f"""<p class='fb'><i class='l'>Result (16–22 Nov, eager only):</i> run for QQP and MNLI at sequence
lengths 128 and 32 (Tables 11–12). CUDA graphs and CIFAR-100 (ViT) are reported separately after the tables: both needed a working A6000, and
on 23–24 Nov none could start CUDA (one needed a reset, the idle ones failed to initialise). On QQP at length 128 the cost rule with the exit plan is the fastest policy within budget at every
batch size, {rng(q.ours, unit='×')} at a held-out drop of {rng(rq.ho_drop_pts)} points, {rng(q.gain, '{:.0f}')}% ahead of
the best baseline within budget. At length 32 and B ≤ 16 the plan's policy is {rng(p32.ho_drop_pts - 1)} points over
budget (so is the unified exit's), and the rule with all exits is the fastest policy within budget ({rng(q32.ours, unit='×')}). On MNLI
at length 128 the cost rule is over budget at every batch size (held-out drop {rng(m.ours_drop_min.tolist() + m.ours_drop_max.tolist())}
points), and so are the per-sample threshold and, at B ≥ 16, random search ({rng(rc_over.ho_drop_pts)} points). The
fastest policies within budget there are the unified exit, the threshold with an exit plan and majority vote, at
{rng(m.base, unit='×')}. At length 32 and B ≤ 16 the plan's policy is within budget but {rng(-m32.gain, '{:.0f}')}% behind
the unified exit. Section 5.2 checks whether the MNLI overshoot is a property of this held-out half.</p>"""

    tab, c = cert_table()
    qx, mx, cx = c["qqp"], c["mnli"], c["cifar100"]
    out["t52"] = (tab + "<p class='cap'><b>Table 13:</b> Certified selection on 30 random re-splits of the validation "
                  "traces (20% tables and paths, 30% certification, 50% test) at B = 8, 16, 64, 128; share of the "
                  "120 cases per method whose test drop exceeds one point, and mean speedup from the cost table (not "
                  "timed). δ = 10% unless stated. <code>results/cert_check_{qqp,mnli,cifar100,mnli_d05}_summary.csv</code>.</p>")
    dc = pd.read_csv(ROOT / "results/delta_check_mnli.csv")
    dd = lambda f, de: dc[(dc.family == f) & (dc.delta == de)].set_index("B")
    assert (dc[(dc.delta == 0.10) & (dc.family == "random_collective") & (dc.B >= 16)].ho_drop_pts > 1.0).all()
    over10 = dc[(dc.delta == 0.10) & dc.family.isin(["threshold", "theory", "theory+plan"])].ho_drop_pts
    under10 = dc[(dc.delta == 0.10) & dc.family.isin(["unified", "threshold+plan"])].ho_drop_pts
    th2, th10 = dd("theory", 0.02), dd("theory", 0.10)
    small2 = th2[th2.index <= 64]
    loss2 = 100 * (1 - th2.ho_speedup / th10.ho_speedup)[th2.index <= 64]
    d05 = c.get("mnli_d05")
    t10 = mx["t"]
    d05_txt = ""
    if d05 is not None:
        t05 = d05["t"]
        d05_txt = (f" Lowering δ to 5% cuts the re-split violation rate of the cost rule from {100 * t10.loc['theory', 'viol']:.1f}% "
                   f"to {100 * t05.loc['theory', 'viol']:.1f}% (with the plan {100 * t10.loc['theory+plan', 'viol']:.1f}% → "
                   f"{100 * t05.loc['theory+plan', 'viol']:.1f}%, random search {100 * t10.loc['random_collective', 'viol']:.1f}% → "
                   f"{100 * t05.loc['random_collective', 'viol']:.1f}%) for about {100 * (1 - t05.loc['theory', 'sp'] / t10.loc['theory', 'sp']):.0f}% "
                   f"less speed ({t10.loc['theory', 'sp']:.2f}× → {t05.loc['theory', 'sp']:.2f}×).")
    out["r52"] = f"""<p class='fb'><i class='l'>Result (Table 13):</i> the t-test keeps the share of re-splits over budget
at or below δ = 10% on average on all three tasks ({100 * qx['viol']:.1f}% QQP, {100 * mx['viol']:.1f}% MNLI,
{100 * cx['viol']:.1f}% CIFAR-100, mean over five methods). On CIFAR-100 two methods are slightly above δ (per-sample
threshold {100 * cx['t'].loc['threshold', 'viol']:.1f}%, cost rule + plan {100 * cx['t'].loc['theory+plan', 'viol']:.1f}%),
which 120 cases per method cannot separate from 10% (the 95% interval is about ±5 points). The finite-sample bounds are
too conservative at these sample sizes: for the accuracy drop, Hoeffding–Bentkus certifies no policy on any task, and
the WSR bound certifies
one in {100 * qx['wsr_chose']:.0f}% of the QQP cases, at {qx['wsr']:.2f}× against {qx['t'].loc['theory+plan', 'sp']:.2f}×
with the t-test, and never on MNLI or CIFAR-100. The label-free version (disagreement with the full model) had no
violations and costs speed: {qx['lf']:.2f}× against {qx['t'].loc['theory+plan', 'sp']:.2f}× on QQP. With 10% of the
calibration data the cost rule falls from {qx['t'].loc['theory', 'sp']:.2f}× to {qx['small']:.2f}× on QQP, and random
search loses about as much ({qx['t'].loc['random_collective', 'sp']:.2f}× → {qx['small_rc']:.2f}×), so the data-efficiency hypothesis (one parameter needs less data than three) is not
supported. We therefore state the certificate as asymptotic, checked on re-splits.</p>
<p><b>The MNLI held-out half.</b> On the fixed split (<code>results/delta_check_mnli.csv</code>) the cost rule, the
per-sample threshold and random search at B ≥ 16 all exceed the budget at δ = 10% (cost rule, with or without the
plan, and threshold: {rng(over10)} points), while the unified exit and the threshold with an exit plan stay at or just
above it ({rng(under10)} points). At
δ = 2% the cost rule is within budget at B ≤ 64 ({rng(small2.ho_drop_pts)} points) for {rng(loss2, '{:.0f}')}% less
speed, but not at B = 128 ({th2.loc[128, 'ho_drop_pts']:.2f}). The split is random (<code>traces/train_exit_heads.py</code>),
and on 30 random re-splits at δ = 10% the cost rule is over budget in {100 * t10.loc['theory', 'viol']:.1f}% of cases.{d05_txt}
We read the fixed-split overshoot as an unlucky split, not a distribution shift; these numbers are consistent with that
reading and do not prove it.</p>"""

    tab, l = load_table()
    out["t53"] = (tab + "<p class='cap'><b>Table 14:</b> Fluctuating load: 3 arrival traces of 3,000 batches with B from 1 "
                  "to 128, throughput speedup over all batches and accuracy drop over all served inputs (held-out, cost-table "
                  "replay). Criterion of <code>docs/DIRECTION.md</code>: tuned once, within budget, within 5% of the best "
                  "per-batch-size re-tuned baseline and ahead of every tuned-once baseline. * over the one-point budget. "
                  "<code>results/load_trace_{qqp,mnli,mnli_d05,cifar100}_levels.csv</code>.</p>")
    lq, lc, lm = l["qqp"], l["cifar100"], l[("mnli", "theory")]
    out["r53"] = f"""<p class='fb'><i class='l'>Result (Table 14):</i> passes on QQP: the cost rule tuned once reaches
{l[('qqp', 'theory')]['sp']:.2f}× against {lq['per_b'].speedup:.2f}× for the best baseline re-tuned per batch size
({short(lq['per_b'].family).lower()}). CIFAR-100 passes only with the exit plan ({l[('cifar100', 'theory+plan')]['sp']:.2f}×
against {lc['per_b'].speedup:.2f}×); with all exits the rule reaches {l[('cifar100', 'theory')]['sp']:.2f}× and is behind
the best tuned-once baseline. MNLI fails on the budget: tuned once, the cost rule loses {lm['drop']:.2f} points (with the
plan {l[('mnli', 'theory+plan')]['drop']:.2f}), and δ = 5% does not change the policy tuned at B = 64.</p>"""

    b = bandit_summary()
    live = {k: v for k, v in b.items() if v.loc["bandit", "regret"] > 1.0}
    dead = [k for k in b if k not in live]
    sched = ["bandit", "bandit_exp1/2", "bandit_const5pct"]
    best_sched = {k: v.loc[sched, "regret"].idxmin() for k, v in b.items()}
    n13 = sum(1 for k in live if best_sched[k] == "bandit")
    n_band = sum(1 for x in live.values() if 0.5 <= x.loc["bandit", "slope"] <= 0.7)
    warm = [100 * (1 - v.loc["bandit_warm", "regret"] / v.loc["bandit", "regret"]) for v in live.values()]
    out["r54"] = f"""<p class='fb'><i class='l'>Result (ten seeds, B = 64, 3,000 batches; <code>results/bandit_slopes_*_10seeds.csv</code>):</i>
the log–log slope of cumulative regret is {rng([x.loc['bandit', 'slope'] for x in live.values()])} in the {len(live)}
settings where the bandit has regret to measure (95% intervals ±{rng([x.loc['bandit', 'slope_ci95'] for x in live.values()])});
in the remaining one (<code>{', '.join(dead)}</code>) cumulative regret stays below
{max(b[k].loc['bandit', 'regret'] for k in dead):.2f} and the slope means nothing. {n_band} of the {len(live)} slopes are
in the 0.5–0.7 band we expected; a slope is not a proof of a rate. The <i class='v'>t</i><sup>−1/3</sup> probe schedule has
the lowest regret of the three schedules in {n13} of these {len(live)} settings; <i class='v'>t</i><sup>−1/2</sup> is
lower in the others. The warm start reduces regret by {rng(warm, '{:.0f}')}%.</p>"""

    r, nv, nb = robust_summary()
    text = r[r.setting.str.startswith(("qqp", "mnli"))]
    cif = r[r.setting.str.startswith("cifar")]
    k_major = text[text.policy == "majority"][["k16", "k32"]].max(axis=1)
    cons8 = text[text.policy == "consensus"].k8
    th0 = text[text.policy == "theory m=0"]
    thr = text[text.policy == "threshold"]
    m8, m0 = cif[cif.policy == "theory m=8"].set_index("setting"), cif[cif.policy == "theory m=0"].set_index("setting")
    fast = m0.index[m0.clean_speedup > 1.1]
    m8, m0 = m8.loc[fast], m0.loc[fast]
    k32 = text.pivot_table(index="setting", columns="policy", values="k32")
    assert (k32[["theory m=4", "theory m=8"]].min(axis=1) >= k32["theory m=0"] - 1e-9).all()
    mnli_drop = 100 * r[r.setting.str.startswith("mnli") & r.policy.str.startswith("theory")].clean_acc_drop
    out["r55"] = f"""<p class='fb'><i class='l'>Result (<code>experiments/robust_bound.py --tune cert</code>, 3 tasks × 4 cost
profiles, B = 64; <code>results/robust_*_B64_cert.csv</code>):</i> the pathwise bound held in all {nb:,} checked batches
({nv} violations). Collective rules that count votes collapse: on the text tasks majority vote reaches
{rng(k_major)}× full-model latency at 16–32 attackers and consensus {rng(cons8)}× at 8. The cost rule degrades
smoothly (victims' latency at 32 attackers {rng(th0.k32)}× without trimming). Two findings narrow the claim: the
per-sample threshold is also robust ({rng(thr[~thr.setting.str.contains('dec_eager')].k32)}× at 32 attackers, outside the
decoder-cost profile where it gives no speedup at all), and trimming buys little. At 32 attackers the trimmed rules
are not faster than the untrimmed one on the text tasks, and on CIFAR-100 trimming eight gains costs most of the clean
speedup ({rng(m8.clean_speedup)}× against {rng(m0.clean_speedup)}×). On MNLI the certified cost rule is again over
budget on the held-out half ({rng(mnli_drop, '{:.1f}')} points).</p>"""

    groups = lambda task: [(MODEL[mo], d[(d.model == mo) & (d.task == task)]) for mo in ("llama", "qwen")]
    out["t57"] = (timed_grid(groups("qqp")) + "<p class='cap'><b>Table 15:</b> LLMs as batched classifiers, QQP: "
                  "LoRA + one linear exit head per exit on the last prompt token (prefill only, prompts padded to 128 "
                  "tokens, bf16, eager), RTX A6000, trained weights and real prompts, held-out, 2,048 inputs per pass, "
                  "5 rounds. Layout as Table 11. <code>results/local_time_qqp_{llama,qwen}_a6000_T128.csv</code>.</p>"
                  + timed_grid(groups("mnli")) + "<p class='cap'><b>Table 16:</b> MNLI, same layout. "
                  "<code>results/local_time_mnli_{llama,qwen}_a6000_T128.csv</code>.</p>")
    ll = lambda mo, t: s[(s.model == mo) & (s.task == t)]
    lq, lm, qq, qm = ll("llama", "qqp"), ll("llama", "mnli"), ll("qwen", "qqp"), ll("qwen", "mnli")
    for x in (lq, lm, qq):
        assert (x.gain > 0).all()
    qm_thr = d[(d.model == "qwen") & (d.task == "mnli") & (d.family == "threshold")]
    ahead, behind = qm[qm.gain > 0], qm[qm.gain <= 0]
    pct = lambda x: ", ".join(f"{g:+.0f}%".replace("-", "−") for g in x.gain)
    rho = json.loads((ROOT / "results/local_cost_a6000_T128_llama.json").read_text())["rho8"]
    agree = min(min(json.loads(f.read_text())["pred_agreement_per_exit"])
                for f in ROOT.glob("results/local_check_*_a6000_T128.json"))
    acc = lambda run: pd.read_csv(ROOT / f"results/train_wikitext_{run}_exit_accuracy.csv").query("split == 'all'")
    fz, lo = acc("qfrozen"), acc("qlora")
    gap = lambda a: 100 * (a[a.layer == a.layer.max()].accuracy.iloc[0] - a[a.layer == 24].accuracy.iloc[0])
    gap_fz = 100 * (fz.original_model_accuracy.iloc[0] - fz[fz.layer == 24].accuracy.iloc[0])
    ad = pd.read_csv(ROOT / "results/adaptivity_vs_batch.csv").query("task == 'wikitext_qlora' and budget_pts == 1.0")
    out["r57"] = f"""<p class='fb'><i class='l'>Result, decoding (15–16 Nov):</i> steps 0–2 of the runbook ran. Training
the exits with LoRA narrows the gap of frozen exits (Qwen2.5-1.5B, layer 24: {gap(lo):.1f} points below the fine-tuned
model against {gap_fz:.1f} frozen), but on free-text next-token prediction a batch-level rule has almost nothing to
adapt to: the gain of an adaptive depth over a fixed exit at a one-point budget falls from
{ad[ad.B == ad.B.min()].gain_over_static_pct.iloc[0]:.0f}% at B = {ad.B.min()} to
{ad[ad.B == ad.B.max()].gain_over_static_pct.iloc[0]:.1f}% at B = {ad.B.max()} (<code>results/adaptivity_vs_batch.csv</code>). Decoding stays a negative result (<code>docs/PLAN_LLM.md</code> §7).</p>
<p class='fb'><i class='l'>Result, the LLM as a batched classifier (16–22 Nov, Tables 15–16):</i> on Llama-3.1-8B
(compute-bound: ρ = {rho:.2f} at B = 8) the cost rule is the fastest policy within budget at every batch size on both
tasks: QQP {rng(lq.ours, unit='×')} against {rng(lq.base, unit='×')} ({rng(lq.gain, '{:.0f}')}% ahead), MNLI
{rng(lm.ours, unit='×')} against {rng(lm.base, unit='×')} ({rng(lm.gain, '{:.0f}')}% ahead), at held-out drops of
{rng(lq.ours_drop_min.tolist() + lq.ours_drop_max.tolist())} and {rng(lm.ours_drop_min.tolist() + lm.ours_drop_max.tolist())}
points. On Qwen2.5-1.5B QQP it is {rng(qq.gain, '{:.0f}')}% ahead ({rng(qq.ours, unit='×')}). Qwen2.5-1.5B MNLI is the
exception: with all exits the rule is the slowest certified policy ({rng(qm.theory, unit='×')}); with the exit plan it
reaches {rng(qm.plan, unit='×')} at a 0.2-point drop, ahead of the best baseline within budget at
B = {', '.join(str(x) for x in ahead.B)} ({pct(ahead)}) and behind at B = {', '.join(str(x) for x in behind.B)}
({pct(behind)}). The per-sample threshold is faster there ({rng(qm_thr.real_speedup, unit='×')}) but
over budget ({rng(qm_thr.ho_drop_pts)} points). Decisions are replayed from the recorded traces on the trained weights;
the device's predictions agree with the traces on {100 * agree:.0f}–100% of inputs at every exit
(<code>results/local_check_*_a6000_T128.json</code>).</p>"""
    out["claims_nearbest"] = (f"<span class='tag'>HOLDS</span> on QQP (RoBERTa, Qwen2.5-1.5B, Llama-3.1-8B) and Llama "
                              f"MNLI, strongest on the LLM classifiers ({rng(pd.concat([lq.gain, lm.gain]), '{:.0f}')}% "
                              f"ahead on Llama). Exceptions: RoBERTa MNLI (over budget on the fixed split) and Qwen MNLI "
                              f"(needs the plan; {pct(qm)} at B = 8, 16, 64, 128)")
    out["r58"] = theory58()
    return out


def theory58() -> str:
    """Item 5.8 from theory/THEORY_58.md: results/theory58_checks.json (exact DP on small instances) and
    results/theory58_front_*.csv (the corrected rule (2) on real traces, cost-table replay)."""
    v = lambda x: f"<i class='v'>{x}</i>"
    ck = json.loads((ROOT / "results/theory58_checks.json").read_text())
    d = pd.concat([pd.read_csv(f) for f in sorted(ROOT.glob("results/theory58_front_*.csv"))], ignore_index=True)
    k = ["task", "device", "B"]
    at1 = d[d.alpha_pts == 1.0].pivot_table(index=k, columns="family", values="ho_speedup")
    dr1 = d[d.alpha_pts == 1.0].pivot_table(index=k, columns="family", values="ho_drop_pts")
    best = d[d.ho_drop_pts <= 1.0].groupby(k + ["family"]).ho_speedup.max().unstack()
    r_cert = at1.theory_la / at1.theory - 1
    r_match = (best.theory_la / best.theory - 1).dropna()
    t1, t2, p3, ja, jf = ck["T1"], ck["T2"], ck["P3"], ck["J_affine"], ck["J_floor"]
    assert t1["mono_gap"] == 0 and t2["la_gap"] == 0 and t2["r2_late"] == 0 and t2["bound_viol"] == 0
    return f"""<p class='fb'><i class='l'>Result (24 Nov):</i> proofs and exact checks in <code>theory/THEORY_58.md</code>.
All statements assume exact calibration, inputs that are independent given their confidences, and expected gains
that are not negative (which the clipping in the rule imposes). <b>(i) Collective stop.</b> With the active set fixed,
the latency of each segment is deterministic, and rule (3) is the lookahead to every later exit. In the monotone case
this lookahead equals the one-step lookahead and is optimal, <i>for any cost</i> {v('c')}({v('n')}): the cost shape
matters only through individual exits. <b>(ii) Affine cost, the two limits.</b> At ρ = 0 the problem splits into one
stopping problem per input, and at ρ = ∞ it is pure collective stopping. At ρ = 0 rule (2) is <i>not</i> the optimal
per-input rule: it compares only with the last exit, so in the monotone case it never leaves later than the optimum
but can leave earlier. Its expected loss is below λ{v('b')}(L − {v('d')}<sub>τ+1</sub>)/{v('F')}, the cost of the
layers after the next exit. The optimal per-input rule tests every later exit, as (3) does. <b>(iii) Floor
cost.</b> Below the knee an individual exit saves nothing and can only lose accuracy, so there the rule is optimal in
the monotone case; this proves the regime switch inside a draining batch. Above the knee, rule (2) prices every
leaving input at today's marginal cost and over-counts the saving by {v('b')}·max(0, {v('k')} − ({v('n')} −
{v('n')}<sub>k</sub>)) per layer when {v('k')} inputs leave together. An instance with {p3['n_k']} easy inputs and one hard
one attains the bound ({p3['simultaneous']:.2f} in units of λ{v('b')}(L − {v('d')}<sub>e</sub>)/{v('F')}). Leaving one at a time, each priced at its own marginal cost,
fixes it ({p3['sequential']:+.3f}). <b>Exact checks</b> (dynamic programming on small instances,
<code>results/theory58_checks.json</code>): rule (3) is optimal in all {t1['mono']} monotone instances and
suboptimal in {t1['nonmono_gap']} of {t1['nonmono']} non-monotone ones. The all-exits per-input rule is optimal in all
{t2['mono']} monotone single-input instances, while rule (2) is suboptimal in {t2['r2_gap']}, never late, and within the
bound in all {t2['bound_n']} instances where it applies. For 0 &lt; ρ &lt; ∞ there is no theorem: in the joint problem
with {ja['n']} instances of B = 3, rules (2)+(3) are optimal in {ja['rules_optimal']} (affine cost) and
{jf['rules_optimal']} (floor cost), with mean gaps in the objective of {100 * ja['gap_mean']:.1f} and {100 * jf['gap_mean']:.1f}
points. <b>On real traces</b> (QQP and MNLI; GTX 1650 and A6000 cost tables; B = 8–128; {len(r_cert)} settings;
<code>results/theory58_front_*.csv</code>) the all-exits rule (2) is {100 * r_cert.mean():+.1f}% faster at the
one-point certified target, but it spends more of the budget (held-out drop {dr1.theory_la.mean():.2f} vs
{dr1.theory.mean():.2f} points). At matched held-out drop (≤ 1 point) it is {100 * r_match.mean():+.1f}% on average
({int((r_match > 0.005).sum())} wins, {int((r_match < -0.005).sum())} losses): re-tuning λ absorbs the difference, so
the implemented rule is kept. Not proved: optimality for 0 &lt; ρ &lt; ∞ and above the knee, and whether the real
calibration tables are monotone. "Optimal" stays out of the title.</p>"""


CSS = """
@page { size: A4; margin: 21mm 20mm 20mm 20mm; @bottom-center { content: counter(page); font: 9pt 'C059', serif; } }
body { font: 9.6pt/1.36 'C059', 'Century Schoolbook L', 'DejaVu Serif', serif; color: #111; text-align: justify; hyphens: auto; }
h1 { font-size: 19pt; text-align: center; margin: 0 0 2pt; }
.sub { text-align: center; font-size: 12.5pt; margin: 0; }
.sub2 { text-align: center; font-size: 10pt; margin: 3pt 0 12pt; }
.auth { text-align: center; font-size: 10.5pt; margin: 0; }
.inst { text-align: center; font-size: 9pt; margin: 2pt 0 10pt; }
.abs { margin: 0 10mm 8pt; font-size: 9pt; }
.abs b.t { display: block; text-align: center; margin-bottom: 3pt; }
h2 { font-size: 12.5pt; margin: 13pt 0 4pt; page-break-after: avoid; }
h3 { font-size: 10.3pt; margin: 9pt 0 3pt; page-break-after: avoid; }
p { margin: 0 0 5pt; }
ul { margin: 2pt 0 5pt; padding-left: 14pt; } li { margin-bottom: 2pt; }
table { border-collapse: collapse; margin: 5pt auto 3pt; font-size: 8.3pt; text-align: left; page-break-inside: avoid; }
th { border-top: 0.9pt solid #111; border-bottom: 0.5pt solid #111; padding: 2.5pt 5pt; font-weight: 600; vertical-align: bottom; }
td { padding: 2pt 5pt; vertical-align: top; }
table tr:last-child td { border-bottom: 0.9pt solid #111; }
table.num td:not(:first-child), table.num th:not(:first-child) { text-align: center; }
tr.ours td { background: #eef4fc; }
.cap { font-size: 8.3pt; margin: 2pt 6mm 9pt; text-align: justify; }
.fig { text-align: center; margin: 7pt 0 0; page-break-inside: avoid; } .fig img { max-width: 100%; }
.eq { text-align: center; margin: 5pt 0; font-size: 10pt; }
.fb { margin-left: 12pt; } .fb i.l { font-style: italic; }
.tag { font-size: 7.6pt; font-weight: 600; padding: 0 3pt; border: 0.5pt solid #555; border-radius: 2pt; white-space: nowrap; }
.refs p { font-size: 8.3pt; margin: 0 0 2pt; padding-left: 16pt; text-indent: -16pt; text-align: left; }
code { font: 8.2pt 'DejaVu Sans Mono', monospace; }
i.v { font-style: italic; }
"""


def renumber(html: str) -> str:
    """Number references in order of first citation and drop the ones that are never cited."""
    import re
    body, refs = html.split("<h2>References</h2>")
    pat = re.compile(r"\[(\d+(?:, \d+)*)\]")
    new: dict[str, int] = {}
    for m in pat.finditer(body):
        for k in m.group(1).split(", "):
            new.setdefault(k, len(new) + 1)
    body = pat.sub(lambda m: "[" + ", ".join(map(str, sorted(new[k] for k in m.group(1).split(", ")))) + "]", body)
    items = dict(re.findall(r"<p>\[(\d+)\] (.*?)</p>", refs))
    missing = [k for k in new if k not in items]
    assert not missing, missing
    print("references dropped (never cited):", sorted(set(items) - set(new), key=int))
    lst = "".join(f"<p>[{n}] {items[k]}</p>\n" for k, n in sorted(new.items(), key=lambda kv: kv[1]))
    return body + "<h2>References</h2><div class='refs'>\n" + lst + "</div></body></html>"


def build() -> None:
    d = local()
    reg_html, g = regret_table(d)
    ours = g[g.family == "theory+plan"]
    plain = g[g.family == "theory"]
    n_cells = len(ours)
    tab_err = g.err.median() * 100
    aff_err = g.aerr.median() * 100
    tab_err_max, aff_err_max = g.err.max() * 100, g.aerr.max() * 100
    best_other = g[~g.family.isin(["theory", "theory+plan"])].groupby("family").rel.min().max()
    rs = resplit_table()
    fd = front()
    mt = matched(fd)
    gain = 100 * (mt.loc[["theory", "theory+plan"]].max() / mt.drop(["theory", "theory+plan"]).max() - 1)
    q_lo, q_hi = gain["qqp"].min(), gain["qqp"].max()
    m_behind = -gain["mnli"][gain["mnli"] < -2]
    m_lo, m_hi = m_behind.min(), m_behind.max()
    assert (gain["mnli"] >= -2).sum() == 4 and len(m_behind) == 2 and q_lo > 0
    print("matched-accuracy gain of the rule over the best baseline (%):\n", gain.round(1).to_string())
    fig = lambda name, cap: (f"<div class='fig'><img src='../figures/{name}'></div><p class='cap'>{cap}</p>"
                             if (ROOT / 'figures' / name).exists() else "")
    v = lambda x: f"<i class='v'>{x}</i>"
    r5 = section5()
    html = f"""<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style><title>IE624 progress report</title></head><body>
<h1>When Should a Batch Stop?</h1>
<p class='sub'>Cost-Derived Batch-Level Early Exit, with Bounded Damage from Slowdown Inputs</p>
<p class='sub2'>IE624 Course Project: Progress Report and Plan</p>
<p class='auth'>Divyanshu Gupta (23B0306) &nbsp;&nbsp; Kushagra Gupta (23B3956) &nbsp;&nbsp; Neha Yadav (23B1807)</p>
<p class='inst'>Course instructor: Prof. Manjesh K. Hanawal, IIT Bombay<br>13 November 2026; results in Section 5 updated 23 November 2026</p>

<div class='abs'><b class='t'>Abstract</b>
Early-exit networks answer easy inputs at intermediate layers, but under batched GPU inference one hard input keeps
the whole batch running, so most of the promised speedup disappears. We measure the per-layer cost {v('c')}({v('n')})
of running {v('n')} active inputs and derive from it a rule that decides when single inputs leave, when the whole
batch stops, and which exits are evaluated at all. Its parameters are chosen with a statistical test on the
accuracy drop. Evidence so far comes in three parts. On an RTX A6000 with trained RoBERTa and ViT checkpoints, no
existing rule is fastest across batch sizes and execution modes, and early exit that waits for the whole batch is
slower than no early exit. On a GTX 1650 the measured cost is a floor followed by a linear part, not
{v('a')} + {v('b n')}. In wall-clock runs on that GPU (two text tasks, three batch sizes, sequence length varied as
a timing knob, decisions replayed from the trained models) the rule is at most {100 * (1 - ours.rel.min()):.1f}%
behind the fastest method, but the methods are not at equal accuracy there. Compared at a matched held-out drop of
at most one point, using the measured cost table, the rule is {q_lo:.0f}–{q_hi:.0f}% faster than the best baseline on
QQP; on MNLI it is within 2% of the best baseline in four of six settings and {m_lo:.1f}–{m_hi:.1f}% slower in the
other two. The rule is not shown to be optimal. Section 5 adds the results obtained since (23 November): timed,
certified runs of trained checkpoints on an RTX A6000 (RoBERTa, and Llama-3.1-8B and Qwen2.5-1.5B used as batched
classifiers), where the rule is the fastest policy within budget on QQP and on Llama MNLI and not on RoBERTa or Qwen
MNLI; a certificate that keeps the budget on re-splits but only asymptotically; and the load, bandit and robustness
checks. A label-free bandit and a bound on the damage from
slowdown inputs are secondary. Section 4.1 lists the claims of our first report that we corrected.</div>

<h2>1. Problem statement</h2>
<p>Early-exit networks attach small classifiers (exits) after some layers, so an easy input can stop after a few
layers instead of all of them [1, 2]. On a GPU, inputs are served in batches and a batch runs layer by layer for all
its members together. A single hard input, a <i>straggler</i>, keeps the batch running to the last layer, so early
exits save compute (FLOPs) but little wall-clock time. DREX, GEEP and Miao et al. report the same effect for batched
BERT and LLM inference [3, 5, 4].</p>
<p><b>Cost model.</b> One layer run on the {v('n')} inputs still active costs {v('c')}({v('n')}). Exits sit after layers
{v('l')}<sub>1</sub> &lt; … &lt; {v('l')}<sub>E</sub> = {v('L')}. Only a subset is evaluated, the <i>exit plan</i>
{v('S')} = {{{v('s')}<sub>1</sub> &lt; … &lt; {v('s')}<sub>m</sub> = {v('E')}}}, and each evaluated exit costs {v('h')}. Write
{v('d')}<sub>i</sub> for the layer of the {v('i')}-th evaluated exit ({v('d')}<sub>0</sub> = 0) and {v('n')}<sub>i</sub> for
the number of inputs active on the layers between evaluated exits {v('i')}−1 and {v('i')}. If the batch ends at its
{v('k')}-th evaluated exit, its latency is</p>
<p class='eq'>{v('T')}<sub>batch</sub> = Σ<sub>i = 1…k</sub> [ ({v('d')}<sub>i</sub> − {v('d')}<sub>i−1</sub>) · {v('c')}({v('n')}<sub>i</sub>) + {v('h')} ]. &nbsp;&nbsp;&nbsp; (1)</p>
<p>Every layer up to the stopping depth is paid for, including layers whose exits are not in the plan; {v('h')} is
paid only for exits that are evaluated.</p>
<p>The first report assumed {v('c')}({v('n')}) = {v('a')} + {v('b n')} and described the regime by ρ = {v('a')}/({v('b n')}).
When ρ is small (compute-bound), removing an exited input saves real time, so per-input exits work. When ρ is large
(fixed-cost-bound), removing one input saves almost nothing and only stopping the whole batch saves time. That
picture stands, but Section 4.3 shows the affine form is a poor fit on a second GPU, so the method now takes the
measured {v('c')}({v('n')}) directly.</p>
<p><b>Gap.</b> Existing batch-level rules are heuristics: Consensus waits until every input is confident, and
Majority and Greedy stop when half or one of them is [3, 4]. None is derived from the cost. Apparate chooses which
exits to keep by their saving minus their overhead, but all inputs still run to the end of the model and accuracy
is monitored without a guarantee [6]. Risk-controlled early exit gives a guarantee, but for per-input exits with no
batch coupling [29].</p>
<p><b>Setting, and what we do not compare against.</b> A batch is formed once, runs, and all its answers are returned;
the quantity of interest is the latency of that batch. The other way to deal with stragglers is to rebatch: DREX
regroups the inputs that continue past an exit into new batches with other requests waiting at that depth, and E3
splits the model at its exits so that each part runs at a constant batch size [3, 7]. Rebatching needs other
requests to regroup with, so it applies under sustained load;
it does not apply to a fixed offline batch, at low load, or when a batch has a latency deadline. The two are
complementary, and this report has no rebatching baseline, which is a limitation. Stopping a batch also has a
cost that rebatching does not: the inputs forced out are the hard ones, so the accuracy loss is not spread evenly.
Section 4.5 reports the loss on hard inputs separately.</p>
<p><b>Security coupling.</b> A slowdown attacker crafts inputs that never become confident [8, 9, 10]. In a
synchronous batch, such an input delays every co-batched user, not only itself.</p>
<p><b>Research questions.</b> Given the measured {v('c')}({v('n')}): (1) when should a batch stop, and which exits are
worth evaluating; (2) can the rule's parameters be set with a guarantee on accuracy, and learned online without
labels; (3) can the damage from slowdown inputs be bounded without detecting them?</p>

<h2>2. Approach</h2>
<p><b>(a) Batch-stopping rule for a measured cost.</b> The objective for one batch is its mean accuracy minus
λ · {v('T')}<sub>batch</sub> / {v('F')}, with {v('F')} the latency of the full model at the same batch size. λ has no
unit: it is the accuracy we are willing to give up to save one full-model latency. Let {v('g')}<sub>j</sub>({v('e')}→{v('e′')}) be the expected accuracy gain
of carrying input {v('j')} from exit {v('e')} to a later exit {v('e′')}, estimated from a calibration table of
P(correct at {v('e′')} | confidence at {v('e')}). One input contributes 1/{v('B')} of the batch's mean accuracy,
which is why its gain is divided by {v('B')}. Input {v('j')} leaves alone if</p>
<p class='eq'>{v('g')}<sub>j</sub>({v('e')}→{v('E')}) / {v('B')} &lt; λ · [{v('c')}({v('n')}) − {v('c')}({v('n')}−1)] · ({v('L')} − {v('l')}<sub>e</sub>) / {v('F')}, &nbsp;&nbsp;&nbsp; (2)</p>
<p>and the remaining inputs {v('R')} stop together if no later evaluated exit is worth reaching:</p>
<p class='eq'>∀ {v('e′')} ∈ {v('S')}, {v('e′')} &gt; {v('e')}: &nbsp; (1/{v('B')}) Σ<sub>j ∈ R</sub> {v('g')}<sub>j</sub>({v('e')}→{v('e′')}) &lt; λ · [({v('l')}<sub>e′</sub> − {v('l')}<sub>e</sub>) · {v('c')}(|{v('R')}|) + {v('m')}({v('e')}, {v('e′')}) · {v('h')}] / {v('F')}, &nbsp;&nbsp;&nbsp; (3)</p>
<p>where {v('m')}({v('e')}, {v('e′')}) counts the evaluated exits after {v('e')} up to and including {v('e′')}.
An individual exit saves only the marginal cost {v('c')}({v('n')}) − {v('c')}({v('n')}−1); a collective stop saves all of
{v('c')}(|{v('R')}|). For the affine cost these are {v('b')} and {v('a')} + {v('b')}|{v('R')}|, which recovers the rule of
the first report and its threshold ratio of about 1/(1 + ρ). For a cost with a floor the marginal cost is zero below
the knee, so the rule switches from per-input exits to whole-batch stopping <i>inside a batch</i> as it drains.</p>
<p>Two approximations are built in. Rule (2) prices the saving as if the input would otherwise run to layer {v('L')}
at today's marginal cost; when the batch later stops early the true saving is smaller, so (2) overestimates it.
Rule (3) looks ahead to each later exit separately and uses only the current confidences. Lookahead rules of this
kind are optimal in the monotone case of optimal stopping [11], which needs, among other conditions, gains that
diminish with depth and a cost that is linear in {v('n')}. We have not shown these conditions for a measured
{v('c')}({v('n')}), and the floor-shaped cost of Section 4.3 violates the linear one. We therefore do not claim that
the rule is optimal; Section 5.8 states what we plan to prove.</p>
<p><b>(b) Exit plan.</b> The first rule evaluated every exit and paid {v('h')} for each. The plan {v('S')} is now a
choice: any subset of the intermediate exits, with the rule (2)–(3) run on that subset. With one exit kept the plan
behaves like a unified exit; with all exits kept it is the earlier rule. The plan and λ are chosen together by
replaying calibration traces under the measured {v('c')}({v('n')}) and {v('h')}; this needs no GPU time.</p>
<p><b>(c) Certified selection.</b> The first report chose the fastest setting whose accuracy on the tuning
split was within one point of the full model, and it then lost 0.8–2.5 points on held-out data. We now split the
tuning data in two. On the first part each method's candidates are reduced to a path ordered from safest to
fastest. On the second part the path is tested in order: for candidate θ, the drop in accuracy against the full
model is computed per batch on {v('K')} disjoint batches, {v('Z')}<sub>1</sub>, …, {v('Z')}<sub>K</sub>, and θ is certified
if a one-sided {v('t')}-test rejects "mean drop &gt; α" at level δ. Testing stops at the first candidate that fails
(fixed-sequence testing, as in Learn-then-Test [30]); the fastest certified candidate is used. We use α = 1 point
and δ = 10%. Batches, not inputs, are the unit, because a batch-level rule couples the inputs of a batch. The test
treats the {v('K')} batch drops as independent draws from one distribution. That holds here because the batches are
disjoint random groups from one split, and it means the guarantee covers inputs from the same distribution only.
The {v('t')}-test is also valid only for large {v('K')}; a finite-sample bound is planned (Section 5.2). The certificate holds for the
batch size and cost it was computed for, and recomputing it for a new deployment is a replay on the CPU.</p>
<p><b>(d) Label-free, partial-feedback bandit (secondary).</b> The bandit replaces "correct" with "agrees with the full
model", which upper-bounds the accuracy loss. Feedback is censored: once a batch stops, its deeper exits are never
seen. Probe batches run to full depth with probability {v('c t')}<sup>−1/3</sup>, the partial-monitoring rate for a
revealing action [12]. Prior bandits for early exit tune per-input thresholds [13, 14].</p>
<p><b>(e) Bounded damage from slowdown inputs.</b> The robust rule leaves the {v('m')} largest gains out of the collective
sum in (3). A trimmed sum is monotone, so for {v('k')} ≤ {v('m')} slowdown inputs {v('K')} among clean inputs {v('C')},
TS<sub>m</sub>({v('C')}) ≤ TS<sub>m</sub>({v('C')} ∪ {v('K')}) ≤ TS<sub>m−k</sub>({v('C')}), hence
depth({v('C')} ∪ {v('K')}; {v('m')}) ≤ depth({v('C')}; {v('m')} − {v('k')}) for every batch. The property is simple, and
it bounds the damage without removing it: the batch can still run deeper as {v('k')} grows.</p>

<h2>3. Experimental setup</h2>
<table><tr><th>Model</th><th>Task</th><th>Exits after layer</th><th>Full-model acc.</th><th>Used in</th></tr>
<tr><td>RoBERTa-base [21]</td><td>MNLI [22]</td><td>2, 4, …, 12</td><td>87.8%</td><td>server and local</td></tr>
<tr><td>RoBERTa-base</td><td>QQP [23]</td><td>2, 4, …, 12</td><td>91.1%</td><td>server and local</td></tr>
<tr><td>ViT-B/16 [24]</td><td>CIFAR-100 [25]</td><td>2, 4, …, 12</td><td>93.3%</td><td>server</td></tr>
<tr><td>Qwen2.5-1.5B, Llama-3.1-8B (frozen) [26, 27]</td><td>WikiText-103 [28]</td><td>6 exits</td><td>49.9%, 53.6%</td><td>server (negative result)</td></tr></table>
<p class='cap'><b>Table 1:</b> Models and tasks. Encoder backbones were fine-tuned jointly with their six exits.</p>
<ul>
<li><b>Server.</b> One RTX A6000, fp16. Two execution modes: eager PyTorch and CUDA graphs (a compiled stack). All
server numbers come from trained checkpoints running on real inputs.</li>
<li><b>Local.</b> One GTX 1650 (4 GB), fp32 (fp16 is about four times slower on this card). The trained checkpoints
are not on this machine. The timed network has the trained model's architecture and tensor shapes with random
weights; every layer, exit head, host synchronisation, batch compaction and policy call is executed and timed, and
the exit decisions and predictions are those recorded for the trained model on the same inputs. Accuracy is
therefore exact for the trained model and time is exact for this hardware. Sequence lengths 16 and 32 use decisions
recorded at length 128; they show how the same policies behave when per-input compute is smaller, and they are not a
claim about truncated inputs.</li>
<li><b>Protocol.</b> Each validation set is split in half. Parameters are chosen on the tuning half by certified
selection (Section 2c; 40% for tables and candidate paths, 60% for the test) and reported on the held-out half.
Local timing runs 1,920 held-out inputs per pass at lengths 16 and 32 (5 rounds) and 768 at length 128 (3 rounds),
with policies interleaved within a round; we report the median over rounds of the paired speedup and its range.</li>
<li><b>Baselines.</b> Per-sample threshold (DeeBERT-style), unified exit (SkipDecode-style [20], one head evaluated),
Majority (as in DREX), a 150-trial random search over a 3-parameter collective rule, and the per-sample threshold
given the same exit-plan search as our rule (an Apparate-style selection of exits [6]).</li>
</ul>

<h2>4. Current status</h2>
<h3>4.1 Corrections to the first report</h3>
<table><tr><th>First report said</th><th>What the evidence shows</th><th>Measured on</th></tr>
<tr><td>The rule matches or beats the best heuristic in every cell</td><td>With CUDA graphs it is never the single fastest rule: unified exit wins at B ≤ 16 on text, a random search ties or wins at B ≥ 64. At B = 8 on MNLI it was the slowest rule (0.86×)</td><td>Server, CUDA graphs</td></tr>
<tr><td>1.41×, 2.10×, 1.46× at B = 64 within one accuracy point</td><td>Those were simulated and selected on the held-out split. The real-GPU runs lost 0.8–2.5 points on held-out data</td><td>Server, eager</td></tr>
<tr><td>The simulator tracks the GPU within about 5%</td><td>Only at B ≥ 64 in eager mode; 15–32% off at B ≤ 16, and it overestimates in graph mode</td><td>Server, both modes</td></tr>
<tr><td>Every collective rule beats per-sample exits at B ≤ 16</td><td>True on MNLI and QQP, not on CIFAR-100</td><td>Server, ViT</td></tr>
<tr><td>Robustness costs no clean speedup</td><td>True at high ρ. At low ρ on CIFAR-100 the robust rule reaches 1.12× against 1.46×</td><td>Server, ViT</td></tr>
<tr><td>Tuned parameters carry over between deployments (assumed)</td><td>Tested and false for every method: in simulation the rule broke the one-point budget in 29–31% of transfers</td><td>Simulation, 3 tasks</td></tr></table>
<p class='cap'><b>Table 2:</b> Claims of the first report that were narrowed or withdrawn.</p>

<h3>4.2 No rule wins everywhere, and the winner flips with the deployment (server)</h3>
{fig('report2_server_speedups.png', '<b>Figure 1:</b> Real wall-clock speedup against the full model on an RTX A6000, held-out split. Left: eager PyTorch. Right: CUDA graphs. These runs used the old selection protocol and are not accuracy-matched (on MNLI the cost rule lost 1.8–2.5 points, unified exit 0.8–0.9); Section 5.1 reruns them with certified selection.')}
<p>Figure 1 shows the server measurements. Three statements hold in both execution modes. Wait-for-all early exit,
which keeps exited inputs in the batch, is slower than no early exit (0.78–0.97×). Per-sample exits fall below 1× at
small batches on text (0.77–0.90× eager, 0.89× on MNLI with CUDA graphs). And the ordering of rules changes between
B = 8 and B = 128. Compiling cuts the fixed per-layer cost about six-fold, and the cause of the small-batch loss
changes with it: under CUDA graphs it is the exit heads, which alone cost 19% of the runtime at B = 8.</p>

<h3>4.3 The cost is a floor followed by a linear part (GTX 1650)</h3>
{fig('report2_cost_gtx1650.png', '<b>Figure 2:</b> Measured time of one RoBERTa-base layer on a GTX 1650 (fp32) against the number of active inputs, at three sequence lengths, with the affine fit of the first report. Both axes are logarithmic.')}
<table class='num'><tr><th>Sequence length</th><th>c(1) ms</th><th>c(8) ms</th><th>c(64) ms</th><th>h ms</th><th>ρ at n=8</th><th>ρ at n=64</th><th>Affine fit, largest error</th></tr>{cost_rows()}</table>
<p class='cap'><b>Table 3:</b> Per-layer cost and exit-head cost on the GTX 1650.</p>
<p>A layer costs about 1.8 ms however few inputs are active, and then grows in proportion to the number of tokens in
the batch. The knee is near 128 tokens: one input at length 128, four at length 32, eight at length 16. The affine
model is off by up to 41% at short sequences. One GPU therefore covers both regimes: at length 128 it is
compute-bound at every batch size, and at length 16 with B = 8 a per-input exit saves nothing. Policies below are
chosen with the measured table. The gain in predicting <i>speedups</i> is small, because a speedup is a ratio and
errors partly cancel: over the {len(g)} policy runs timed in Section 4.4 the table's median error is {tab_err:.1f}%
(largest {tab_err_max:.0f}%), against {aff_err:.1f}% (largest {aff_err_max:.0f}%) for the affine fit applied to the
same decisions. The floor matters for the rule's decisions more than for the prediction.</p>

<h3>4.4 Exit plan and certified selection, real wall-clock (GTX 1650)</h3>
{fig('report2_local_speedups.png', '<b>Figure 3:</b> Real wall-clock speedup on the GTX 1650 of the certified policy of each method, held-out inputs. Markers are the median over rounds; vertical bars are the range. Dashed blue: the cost rule with all exits; solid blue: with an exit plan.')}
{cell_table(d, 'qqp')}
<p class='cap'><b>Table 4:</b> QQP, GTX 1650, measured speedup of each method's certified policy. Bold: fastest in the column. * held-out accuracy drop above one point.</p>
{cell_table(d, 'mnli')}
<p class='cap'><b>Table 5:</b> MNLI, same layout.</p>
{reg_html}
<p class='cap'><b>Table 6:</b> Summary over 2 tasks × 3 batch sizes × 3 sequence lengths ({n_cells} timed cells; lengths 16 and 32 change the timing only, Section 3). Speed is relative to the fastest method of each cell. The methods are not at the same accuracy here: the fifth column differs by method, so this table is not a ranking. Section 4.5 compares speed at matched accuracy.</p>
RESULTS_TEXT
<h3>4.5 Speed at matched accuracy, and where the accuracy loss lands (cost table, GTX 1650)</h3>
{fig('report2_front.png', '<b>Figure 4:</b> Held-out accuracy drop against speedup. Each point is the policy that certified selection returns for one accuracy target between 0.25 and 3 points, scored once on the held-out half. Up and to the left is better. The dotted line is the one-point budget. Speedups are computed from the measured cost table, not timed.')}
{front_table(fd, 'qqp')}
<p class='cap'><b>Table 7:</b> QQP. Speedup of each method's fastest policy whose held-out drop is at most one point (cost table). Bold: best in the column. –: no such policy.</p>
{front_table(fd, 'mnli')}
<p class='cap'><b>Table 8:</b> MNLI, same layout.</p>
{hard_table(fd)}
<p class='cap'><b>Table 9:</b> Held-out accuracy drop in points at the one-point target, mean over B = 8, 16, 64: all inputs, the hard quarter (lowest mean confidence over the five intermediate exits), and the rest. The rows of the threshold, the unified exit and majority vote are the same at both lengths because their certified parameters do not change with the cost table.</p>
FRONT_TEXT
<h3>4.6 Does the certificate hold on re-splits?</h3>
RESPLIT_TEXT
<h3>4.7 Secondary: the label-free bandit (server, single seeds)</h3>
<p>Against an oracle that knows the true tables, cumulative regret over 3,000 batches of 64 grows with log–log slope
0.49–0.65 on MNLI and QQP, while a CeeBERT-style per-input bandit's regret grows linearly. On the GPU the bandit uses
no labels, costs 0.2–0.3 accuracy points, and reaches 1.26–1.77× at B = 64–128. Two cautions that the first report
did not state: these are single runs fitted on a few points, and a slope below 2/3 does not "confirm" the
{v('T')}<sup>2/3</sup> rate, which is a worst-case bound. Section 5.4 replaces them with ten seeds.</p>
<h3>4.8 Bounded damage from slowdown inputs (server)</h3>
<p>The bound depth({v('C')} ∪ {v('K')}; {v('m')}) ≤ depth({v('C')}; {v('m')} − {v('k')}) held in all 10,500 batches tested
across five tasks, including real DeepSloth inputs on the ViT; the property test passes on the local machine as
well (19 of 19 unit tests). Consensus loses its whole speedup to a single attacker. Victims' latency under our rule
still rises with {v('k')}, and the robust rule has a clean-speed cost in the low-ρ regime (Table 2).</p>
<h3>4.9 What did not work</h3>
<ul>
<li><b>A faster rule at a fixed setting.</b> Random search, program search (FunSearch-style, 300 programs),
trajectory-aware calibration and a hindsight-best batch depth all tie the rule. A hidden-state probe adds at most
0.026 AUROC over confidence. The limit is the exit signal, not the stopping rule.</li>
<li><b>Decoders.</b> The cost model holds on Qwen2.5-1.5B and Llama-3.1-8B, but exits are too weak: frozen heads and
public LayerSkip checkpoints [18] are both about 5 points short at 75% depth, and no rule beats the full model within
one point.</li>
<li><b>Transfer of tuned parameters.</b> See Table 2. This is why selection is now redone per deployment with a
certificate, which costs a CPU replay.</li>
<li><b>Straggler rescue</b> brought no measurable gain.</li>
</ul>

<h2>5. Planned work until 26 November, and results so far</h2>
<p>Each item lists what we planned, what we expected and the fallback, as written on 13 November, followed by the
result obtained by 23 November. All numbers are on held-out data with a one-point accuracy budget.</p>
<h3>5.1 Accuracy-matched server tables with the exit plan (16–18 Nov)</h3>
<p>Rerun the 24 server settings (3 tasks × 4 batch sizes × eager and CUDA graphs) with the exit plan, the measured
cost table and certified selection, on trained checkpoints, and report each as a timed accuracy–speed front like
Figure 4. This is the main evidence still missing: trained weights, matched accuracy and a serving GPU in one run.</p>
<p class='fb'><i class='l'>Expected:</i> at a matched drop of one point, the rule ahead on QQP and within 10% of the best
baseline elsewhere, as in Section 4.5.<br>
<i class='l'>Fallback:</i> if it is more than 10% behind at B ≤ 16 under CUDA graphs, report the plan as a partial fix and
state the settings where a unified exit should be used.</p>
{r5['r51']}
{r5['t51']}
{r5['t51x']}
<h3>5.2 Tighten the certificate (13–15 Nov)</h3>
<p>Replace the {v('t')}-test with a finite-sample bound for bounded batch losses (Hoeffding–Bentkus), add the
label-free version that controls disagreement with the full model, and repeat the re-split check on CIFAR-100. In
the same runs, repeat the selection with 10%, 25% and 50% of the calibration data for the cost rule and for the
random search, to test whether a rule with one parameter needs less data than a searched rule with three.</p>
<p class='fb'><i class='l'>Expected:</i> the same violation rates with a speed cost of a few percent.<br>
<i class='l'>Fallback:</i> keep the {v('t')}-test and state it as an asymptotic guarantee with the measured violation rate.</p>
{r5['r52']}
{r5['t52']}
<h3>5.3 Fluctuating load (19–21 Nov)</h3>
<p>Replay an arrival trace in which the batch size moves between 1 and 128. Compare the rule, re-certified for each
batch size from one set of calibration traces, against heuristics tuned at a single batch size and against
heuristics re-tuned per batch size.</p>
<p class='fb'><i class='l'>Expected:</i> within 5% of per-batch-size re-tuned heuristics and ahead of heuristics tuned once.<br>
<i class='l'>Fallback:</i> if tuned-once heuristics are as good, drop the load claim and keep the per-setting results.</p>
{r5['r53']}
{r5['t53']}
<h3>5.4 Bandit with error bars (22–23 Nov)</h3>
<p>Ten seeds with confidence bands, a probe-rate ablation ({v('t')}<sup>−1/2</sup>, {v('t')}<sup>−1/3</sup>, constant), and a
warm start from a small labelled set.</p>
<p class='fb'><i class='l'>Expected:</i> slopes between 0.5 and 0.7 where the oracle stops batches.<br>
<i class='l'>Fallback:</i> report the measured slopes without a rate claim.</p>
{r5['r54']}
<h3>5.5 Robustness in the compute-bound regime (22–23 Nov)</h3>
<p>Run the attack experiments where majority vote loses its clean speedup, and test an attacker that uses
{v('k')} &gt; {v('m')} inputs.</p>
<p class='fb'><i class='l'>Expected:</i> the bound holds; the clean-speed cost at low ρ is quantified per task.<br>
<i class='l'>Fallback:</i> claim only the certificate.</p>
{r5['r55']}
<h3>5.6 Trained checkpoints on the local GPU (when checkpoints are copied)</h3>
<p>Repeat Section 4.4 with the trained weights and real tokenised inputs, with batches padded to their own longest
input, so that the short-sequence settings are measured on real short inputs.</p>
<p class='fb'><i class='l'>Expected:</i> the same ordering; timing does not depend on weight values.<br>
<i class='l'>Fallback:</i> keep the replayed-decision results, labelled as such.</p>
<h3>5.7 LLMs: decoders (second attempt) and batched classifiers</h3>
<p>The first attempt failed for reasons we can now name: the backbone was frozen while its exits were trained (the
same fault the encoders had before joint training), LayerSkip checkpoints are trained for drafting and not for
returning the early answer, the test was run only at batch sizes 16 and 64, and the budget was one point of
next-token accuracy on free text. The second attempt trains LoRA adapters and exit heads together with
self-distillation from the final layer on Qwen2.5-1.5B, keeps a frozen baseline at the same budget, and sweeps
batch sizes from 2 to 64. The training script and a runbook with pass and kill conditions for each step are
written. Steps 0–2 have since been run; the result is below.</p>
<p><b>Key–value cache.</b> A token that exits at layer {v('l')} leaves the layers above it without keys and values,
which later tokens attend to. Our decoding loop already handles this the way CALM does [19]: the exit hidden state
is projected into the keys and values of every skipped layer. In the first attempt this cost about 2% of a layer
per skipped layer. It is an approximation, and its effect on accuracy is part of what the exit accuracy measures.</p>
<p><b>Alternative considered.</b> Instead of returning the early answer, the exits can be used as drafts that the
remaining layers verify, as in LayerSkip's self-speculative decoding [18]. The output then equals the full model's
greedy output, so weak exits cost speed and not accuracy. Batched speculation has the same straggler problem,
because sequences in a batch accept different numbers of draft tokens, and the draft depth and length could be
chosen per batch from the measured {v('c')}({v('n')}). We have not started this. If the first step of the second
attempt fails its exit-quality check, this is the direction we would take, with agreement with the full model's
greedy output as the metric.</p>
<p class='fb'><i class='l'>Expected:</i> the 75%-depth exit within about 2 points of the full model (5.6 with a frozen
backbone), and a speedup at small batch sizes. This is a forecast.<br>
<i class='l'>Fallback:</i> if the exit stays more than 4 points short, the LLM track ends as a negative result with its
cause.</p>
{r5['r57']}
{r5['t57']}
<h3>5.8 Theory and write-up (24–26 Nov)</h3>
<p>State and prove when the lookahead rule is optimal: the monotone case [11] for an affine cost with gains that
diminish with depth. For a general non-decreasing {v('c')}({v('n')}) we aim only at the regime switch at the knee of a
floor-shaped cost, and at a bound on what rule (2) loses by overestimating its saving.</p>
<p class='fb'><i class='l'>Expected:</i> a proof for the affine case and a counterexample or a bound for the floor case.<br>
<i class='l'>Fallback:</i> clearly labelled proof sketches. The word "optimal" stays out of the title unless the proof is complete.</p>
{r5['r58']}
<h3>5.9 Timeline</h3>
<table><tr><th>Dates</th><th>Work</th></tr>
<tr><td>13–15 Nov</td><td>Local timing, matched-accuracy fronts and re-split check (done, Sections 4.4–4.6); finite-sample certificate; label-free certificate</td></tr>
<tr><td>16–18 Nov</td><td>Server rerun of all 24 settings with the exit plan and certified selection</td></tr>
<tr><td>19–21 Nov</td><td>Fluctuating-load experiment</td></tr>
<tr><td>22–23 Nov</td><td>Bandit with ten seeds; robustness at low ρ</td></tr>
<tr><td>24–26 Nov</td><td>Theory write-up, final figures, presentation</td></tr>
<tr><td>When a server GPU is free</td><td>LLM second attempt, first three steps of the runbook (about 4 GPU-hours, estimate). It does not take time from the server rerun</td></tr></table>
<h3>5.10 Claims and their status (24 November)</h3>
<table><tr><th>Claim</th><th>Status</th><th>Evidence, and what is still missing</th></tr>
<tr><td>Which rule is fastest flips with batch size, execution mode and sequence length</td><td><span class='tag'>HOLDS</span> on two GPUs</td><td>Sections 4.2, 4.4, 5.1. CUDA-graph rerun on the A6000 still missing</td></tr>
<tr><td>The per-layer cost is a floor plus a linear part, not affine</td><td><span class='tag'>HOLDS</span> on the GTX 1650; Qwen2.5-1.5B on the A6000 has a floor up to B = 4</td><td>Section 4.3; <code>results/local_cost_a6000_T128_qwen.json</code></td></tr>
<tr><td>Re-certified per setting (a CPU replay), the rule is near the best certified method everywhere</td><td>{r5['claims_nearbest']}</td><td>Timed A6000 runs, Tables 11–12 and 15–16. CUDA graphs and CIFAR-100 on the A6000 still missing</td></tr>
<tr><td>Stopping a batch spreads the accuracy loss evenly</td><td><span class='tag'>FALSE</span> loss lands on hard inputs</td><td>Table 9. A per-input floor on the loss is not planned before 26 Nov</td></tr>
<tr><td>The exit plan fixes the small-batch weakness</td><td><span class='tag'>PARTLY</span> needed on Qwen MNLI and CIFAR-100 under load; at length 32 and B ≤ 16 on QQP its policy is just over budget</td><td>Tables 11, 14, 16. Under CUDA graphs, where heads cost 19%, not yet measured</td></tr>
<tr><td>Certified selection keeps the accuracy budget</td><td><span class='tag'>WEAKENED</span> to an asymptotic certificate (t-test), within δ on average over 30 re-splits on all three tasks</td><td>Table 13: finite-sample bounds (Hoeffding–Bentkus, WSR) are too conservative at these sample sizes; the label-free version costs speed</td></tr>
<tr><td>The MNLI held-out overshoot is an unlucky split</td><td><span class='tag'>PARTLY</span> consistent: the re-split violation rate is within δ, and δ = 2% fixes B ≤ 64 at about 10% less speed</td><td>Section 5.2; not proved, one fixed split</td></tr>
<tr><td>Under fluctuating load, one tuning of the rule is enough</td><td><span class='tag'>HOLDS</span> on QQP; CIFAR-100 only with the plan; <span class='tag'>FAILS</span> on MNLI (budget)</td><td>Table 14 (cost-table replay, not timed)</td></tr>
<tr><td>The label-free bandit learns the rule (secondary)</td><td><span class='tag'>HOLDS</span> ten seeds, sublinear regret</td><td>Section 5.4; slopes are not a rate proof</td></tr>
<tr><td>The rule is optimal</td><td><span class='tag'>PARTLY</span> proved in the monotone case for the collective stop (any cost), at the two limits of the affine cost and below the knee of the floor cost; rule (2) is not the optimal per-input rule (never late, loss bounded)</td><td>Section 5.8, <code>theory/THEORY_58.md</code>. Open for 0 &lt; ρ &lt; ∞ and above the knee; monotonicity of the real tables not checked; the fix does not help at matched accuracy</td></tr>
<tr><td>Bounded damage from slowdown inputs</td><td><span class='tag'>HOLDS</span> 0 violations of the bound; majority and consensus collapse. The per-sample threshold is also robust, and trimming buys little</td><td>Section 5.5</td></tr>
<tr><td>Benefit on LLMs</td><td><span class='tag'>FAILED</span> for decoding; <span class='tag'>HOLDS</span> for classification on Llama-3.1-8B (both tasks) and Qwen2.5-1.5B QQP, mixed on Qwen MNLI</td><td>Section 5.7, Tables 15–16</td></tr></table>

<h2>6. Expected outcome</h2>
<p>By 26 November we expect a timed, accuracy-matched comparison on two GPUs showing where the cost-derived rule
is ahead of existing rules and where it is not; a selection procedure with a stated
guarantee on the accuracy drop; and, as secondary results, a label-free bandit with error bars and the bounded-damage property. We do not
expect a rule that is fastest everywhere. A positive result on LLM decoders depends on the second attempt (5.7) and is not assumed. For a conference submission the
evidence would still need more models, trained checkpoints timed on both GPUs, and the load experiment; with those,
the work fits a systems-leaning venue.</p>

<h2>References</h2><div class='refs'>
<p>[1] J. Xin, R. Tang, J. Lee, Y. Yu, J. Lin. DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. ACL 2020.</p>
<p>[2] W. Zhou, C. Xu, T. Ge, J. McAuley, K. Xu, F. Wei. BERT Loses Patience: Fast and Robust Inference with Early Exit. NeurIPS 2020.</p>
<p>[3] X. Liu et al. Dynamic Rebatching for Efficient Early-Exit Inference with DREX. arXiv:2512.15705, 2025.</p>
<p>[4] R. Miao, Y. Yan, X. Yao, T. Yang. An Efficient Inference Framework for Early-exit Large Language Models. arXiv:2407.20272, 2024.</p>
<p>[5] L. Li et al. Accelerating BERT Inference with GPU-Efficient Exit Prediction. Frontiers of Computer Science 18(3), 2024.</p>
<p>[6] Y. Dai, R. Pan, A. Iyer, K. Li, R. Netravali. Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving. SOSP 2024.</p>
<p>[7] A. Iyer et al. Improving DNN Inference Throughput Using Practical, Per-Input Compute Adaptation (E3). SOSP 2024.</p>
<p>[8] S. Hong, Y. Kaya, I.-V. Modoranu, T. Dumitraş. A Panda? No, It's a Sloth: Slowdown Attacks on Adaptive Multi-Exit Neural Network Inference. ICLR 2021.</p>
<p>[9] Z. Coalson, G. Ritter, R. Bobba, S. Hong. BERT Lost Patience Won't Be Robust to Adversarial Slowdown. NeurIPS 2023.</p>
<p>[10] Y. Chen et al. Dynamic Transformers Provide a False Sense of Efficiency. ACL 2023.</p>
<p>[11] Y. S. Chow, H. Robbins, D. Siegmund. Great Expectations: The Theory of Optimal Stopping. Houghton Mifflin, 1971.</p>
<p>[12] N. Cesa-Bianchi, G. Lugosi, G. Stoltz. Minimizing Regret with Label Efficient Prediction. IEEE Trans. Information Theory 51(6), 2005.</p>
<p>[13] D. J. Bajpai, M. K. Hanawal. CeeBERT: Cross-Domain Inference in Early Exit BERT. Findings of ACL 2024.</p>
<p>[14] R. G. Pacheco et al. UCBEE: A Multi-Armed Bandit Approach for Early-Exit in Neural Networks. IEEE Trans. Network and Service Management, 2024.</p>
<p>[19] T. Schuster et al. Confident Adaptive Language Modeling. NeurIPS 2022.</p>
<p>[18] M. Elhoushi et al. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. ACL 2024.</p>
<p>[20] L. Del Corro et al. SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference. arXiv:2307.02628, 2023.</p>
<p>[21] Y. Liu et al. RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692, 2019.</p>
<p>[22] A. Williams, N. Nangia, S. Bowman. A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference. NAACL 2018.</p>
<p>[23] A. Wang et al. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. ICLR 2019.</p>
<p>[24] A. Dosovitskiy et al. An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.</p>
<p>[25] A. Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Tech. report, University of Toronto, 2009.</p>
<p>[26] Qwen Team. Qwen2.5 Technical Report. arXiv:2412.15115, 2024.</p>
<p>[27] Llama Team, Meta AI. The Llama 3 Herd of Models. arXiv:2407.21783, 2024.</p>
<p>[28] S. Merity, C. Xiong, J. Bradbury, R. Socher. Pointer Sentinel Mixture Models. ICLR 2017.</p>
<p>[29] M. Jazbec, A. Timans, et al. Fast yet Safe: Early-Exiting with Risk Control. NeurIPS 2024.</p>
<p>[30] A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control. arXiv:2110.01052, 2021.</p>
</div></body></html>"""
    extra = json.loads((ROOT / "docs/report_v2_text.json").read_text())
    html = html.replace("RESULTS_TEXT", extra["results"]).replace("FRONT_TEXT", extra["front"]).replace("RESPLIT_TEXT", (rs + extra["resplit"]) if rs else extra["resplit_pending"])
    html = html.replace("CERT_STATUS", extra["cert_status"] if rs else "<span class='tag'>PENDING</span>")
    html = renumber(html)
    out = ROOT / "docs/IE624_progress_report_v2.html"
    out.write_text(html)
    from weasyprint import HTML
    HTML(str(out)).write_pdf(str(out.with_suffix(".pdf")))
    print("wrote", out.with_suffix(".pdf"))
    print(g.groupby("family").agg(mean=("rel", "mean"), worst=("rel", "min")).round(3))


if __name__ == "__main__":
    build()
