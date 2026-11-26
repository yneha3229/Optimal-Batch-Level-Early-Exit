"""Recompute every number of the early-exit progress report from the result files and compare with the report.

Reported values are read from the committed report, docs/IE624_progress_report_v2.html (tables parsed cell by cell,
text numbers quoted in the CHECKS list below). Recomputed values are computed here from results/*, independently of
analysis/make_report.py (same definitions, separate code). Each number gets one status:

  MATCH     recomputed value, rounded to the precision printed in the report, equals the reported value
            (ranges "a–b": both ends; "about x": within 15%)
  MISMATCH  it does not (both values are listed)
  NO FILE   no result file in the repository holds the number (the reason and the raw log, if any, are listed)

Part C re-renders the report with analysis/make_report.py from the committed results in a scratch copy and compares
every number in its text with the committed HTML (the generated sections of the report).

Usage:  python analysis/check_report_numbers.py [--out results_repro] [--no-render]
Output: <out>/report_numbers.csv (one row per number), <out>/REPORT_NUMBERS.md (summary per item)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
FAMS = ["threshold", "threshold+plan", "unified", "majority", "random_collective", "theory", "theory+plan"]
OURS, BASE = ["theory", "theory+plan"], ["threshold", "threshold+plan", "unified", "majority", "random_collective"]
ROWNAME = {"All heads, no exits": "heads_only", "Per-sample threshold": "threshold", "Threshold + exit plan": "threshold+plan",
           "Unified exit": "unified", "Majority vote": "majority", "Random search (150 rules)": "random_collective",
           "Cost rule, all exits (ours)": "theory", "Cost rule + exit plan (ours)": "theory+plan"}
ROWS: list[dict] = []


# ---------------------------------------------------------------------------------------------------- comparison
def decimals(s: str) -> int:
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def as_num(s: str) -> float:
    return float(s.replace("−", "-").replace(",", ""))


def add(item: str, where: str, reported: str, value, files: str, note: str = "", approx: bool = False) -> None:
    """Compare one reported number (string as printed) with a recomputed value (float, or None = NO FILE)."""
    if value is None:
        ROWS.append(dict(item=item, where=where, reported=reported, recomputed="", status="NO FILE", files=files, note=note))
        return
    rep = as_num(re.sub(r"[^\d.\-−,]", "", reported) or "nan")
    k = decimals(reported)
    if approx:
        ok = abs(value - rep) <= 0.15 * abs(rep)
    else:
        ok = abs(round(value + 1e-12, k) - rep) < 0.5 * 10 ** -k + 1e-9 or abs(value - rep) <= 0.5 * 10 ** -k + 1e-9
    ROWS.append(dict(item=item, where=where, reported=reported, recomputed=f"{value:.{max(k, 2) + 2}f}",
                     status="MATCH" if ok else "MISMATCH", files=files, note=note))


def add_range(item, where, reported: str, values, files, note="") -> None:
    """reported 'a–b' (or a single value) against min and max of values."""
    v = [x for x in np.asarray(values, float).ravel() if np.isfinite(x)]
    parts = re.split(r"[–-](?=\d)", reported.replace("−", "-"))
    if not v:
        add(item, where, reported, None, files, note or "no values")
        return
    if len(parts) == 1:
        add(item, where + " (all values)", reported, min(v) if abs(min(v) - as_num(parts[0])) < abs(max(v) - as_num(parts[0])) else max(v), files, note)
        if round(min(v), decimals(reported)) != round(max(v), decimals(reported)):
            ROWS[-1]["status"] = "MISMATCH"
            ROWS[-1]["recomputed"] = f"{min(v):.3f}–{max(v):.3f}"
        return
    add(item, where + " (low end)", parts[0], min(v), files, note)
    add(item, where + " (high end)", parts[1], max(v), files, note)


def add_bool(item, where, claim: str, ok: bool | None, files, note="") -> None:
    ROWS.append(dict(item=item, where=where, reported=claim, recomputed="" if ok is None else str(bool(ok)),
                     status="NO FILE" if ok is None else ("MATCH" if ok else "MISMATCH"), files=files, note=note))


def cell(s) -> tuple[float | None, bool]:
    s = str(s).strip()
    if s in ("–", "-", "nan", ""):
        return None, False
    return as_num(re.sub(r"[^\d.\-−]", "", s)), "*" in s


def cmp_grid(item, files, html_tab: pd.DataFrame, want: dict) -> None:
    """html_tab: read_html table with a 2-level header (group, B=b) and a Policy column; want[(row, col_index)] =
    (value or None, star). Cells compared on value (printed precision) and on the '*' flag."""
    cols = list(html_tab.columns)
    for _, r in html_tab.iterrows():
        fam = ROWNAME.get(str(r.iloc[0]).strip(), str(r.iloc[0]).strip())
        for j, c in enumerate(cols[1:], start=1):
            rep_s = str(r.iloc[j]).strip()
            v, star = cell(rep_s)
            key = (fam, j)
            if key not in want:
                continue
            wv, wstar = want[key]
            where = f"{fam} / {' '.join(map(str, c))}"
            if v is None or wv is None:
                add_bool(item, where, rep_s, (v is None) == (wv is None), files,
                         "" if (v is None) == (wv is None) else f"recomputed {wv}")
                continue
            add(item, where, re.sub(r"[*×]", "", rep_s), wv, files)
            if star != wstar:
                ROWS[-1]["status"] = "MISMATCH"
                ROWS[-1]["note"] = f"over-budget flag: report {'*' if star else 'none'}, file {'*' if wstar else 'none'}"


# ---------------------------------------------------------------------------------------------------- loaders
def rd(f: str) -> pd.DataFrame:
    return pd.read_csv(R / f)


def local_time() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(R.glob("local_time_*_cuda_T*.csv"))], ignore_index=True)
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    return d


def local_front() -> pd.DataFrame:
    d = pd.concat([pd.read_csv(f) for f in sorted(R.glob("local_front_*_cuda_T*.csv"))], ignore_index=True)
    d["T"] = d.device.str.extract(r"T(\d+)").astype(int)
    return d


def a6000_time() -> pd.DataFrame:
    out = []
    for f in sorted(R.glob("local_time_*_a6000_T*.csv")):
        m = re.fullmatch(r"local_time_([a-z0-9]+?)(?:_(llama|qwen))?_a6000_T(\d+)\.csv", f.name)
        if not m:
            continue
        d = pd.read_csv(f)
        d["model"], d["T"], d["task"] = m.group(2) or "roberta", int(m.group(3)), m.group(1)
        out.append(d)
    return pd.concat(out, ignore_index=True)


def grid_want(groups: list[pd.DataFrame], rows=FAMS) -> dict:
    """Expected cells of a timed grid (Tables 4-5, 11-12, 15-16): column order = groups x sorted B."""
    want, j = {}, 1
    for g in groups:
        for b in sorted(g.B.unique()):
            for f in rows:
                r = g[(g.B == b) & (g.family == f)]
                want[(f, j)] = (None, False) if r.empty else (float(r.real_speedup.iloc[0]), bool(r.ho_drop_pts.iloc[0] > 1.0))
            j += 1
    return want


# ---------------------------------------------------------------------------------------------------- part A: tables
def tables(tabs: list[pd.DataFrame]) -> None:
    T = {i + 1: t for i, t in enumerate(tabs)}       # report tables in order: 1..16, then timeline, claims
    # Table 1: full-model accuracy (last exit, whole validation set)
    for (task, f), rep in zip([("MNLI", "train_mnli_exit_accuracy.csv"), ("QQP", "train_qqp_exit_accuracy.csv"),
                               ("CIFAR-100", "train_cifar100_exit_accuracy.csv")], ["87.8", "91.1", "93.3"]):
        d = rd(f)
        add("Table 1", f"{task} full-model accuracy", rep, 100 * d[(d.split == "all") & (d.layer == d.layer.max())].accuracy.iloc[0], f"results/{f}")
    q = rd("train_wikitext_qfrozen_exit_accuracy.csv")
    add("Table 1", "Qwen2.5-1.5B WikiText next-token accuracy (full model)", "49.9", 100 * q.original_model_accuracy.iloc[0],
        "results/train_wikitext_qfrozen_exit_accuracy.csv")
    ll = rd("train_wikitext_llama_exit_accuracy.csv")
    ll_full = ll[(ll.split == "all") & (ll.layer == ll.layer.max())].accuracy.iloc[0]
    add("Table 1", "Llama-3.1-8B WikiText next-token accuracy (full model = last exit)", "53.6", 100 * ll_full,
        "results/train_wikitext_llama_exit_accuracy.csv")

    # Table 3: GTX 1650 cost table
    for i, Tq in enumerate((16, 32, 128)):
        d = json.loads((R / f"local_cost_cuda_T{Tq}.json").read_text())
        y = {p["B"]: p["full_ms"] / 12 for p in d["points"]}
        rep = T[3].iloc[i]
        f = f"results/local_cost_cuda_T{Tq}.json"
        for col, val in [("c(1) ms", y[1]), ("c(8) ms", y[8]), ("c(64) ms", y[64]), ("h ms", d["head"]),
                         ("ρ at n=8", max(d["rho8"], 0)), ("ρ at n=64", max(d["rho64"], 0)),
                         ("Affine fit, largest error", 100 * d["affine_max_rel_err"])]:
            add("Table 3", f"T={Tq} {col}", str(rep[col]).rstrip("%"), val, f + " (GTX 1650 profile)")

    # Tables 4-5: GTX 1650 timed cells; Table 6: summary
    lt = local_time()
    for n, task in ((4, "qqp"), (5, "mnli")):
        g = lt[lt.task == task]
        groups = [g[g["T"] == t] for t in sorted(g["T"].unique())]
        cmp_grid(f"Table {n}", f"results/local_time_{task}_cuda_T{{16,32,128}}.csv (GTX 1650, laptop; raw logs logs/local_time_{task}_T*.log)",
                 T[n], grid_want(groups, ["heads_only"] + FAMS))
    g = lt[lt.family.isin(FAMS)].copy()
    g["rel"] = g.real_speedup / g.groupby(["task", "T", "B"]).real_speedup.transform("max")
    s = g.groupby("family").agg(mean=("rel", "mean"), worst=("rel", "min"), wins=("rel", lambda x: int((x > 0.97).sum())),
                                n=("rel", "size"), viol=("ho_drop_pts", lambda x: int((x > 1.0).sum())), maxdrop=("ho_drop_pts", "max"))
    for _, r in T[6].iterrows():
        f = ROWNAME[r.iloc[0]]
        add("Table 6", f"{f} mean relative speed", str(r.iloc[1]), s.loc[f, "mean"], "results/local_time_*_cuda_T*.csv")
        add("Table 6", f"{f} worst cell", str(r.iloc[2]), s.loc[f, "worst"], "results/local_time_*_cuda_T*.csv")
        add("Table 6", f"{f} cells within 3%", str(r.iloc[3]).split("/")[0], s.loc[f, "wins"], "results/local_time_*_cuda_T*.csv")
        add("Table 6", f"{f} cells over budget", str(r.iloc[4]).split("/")[0], s.loc[f, "viol"], "results/local_time_*_cuda_T*.csv")
        add("Table 6", f"{f} largest held-out drop", str(r.iloc[5]), s.loc[f, "maxdrop"], "results/local_time_*_cuda_T*.csv")

    # Tables 7-8: matched-accuracy fronts; Table 9: hard inputs
    fd = local_front()
    m = fd[fd.ho_drop_pts <= 1.0].groupby(["family", "task", "T", "B"]).ho_speedup.max()
    for n, task in ((7, "qqp"), (8, "mnli")):
        cols = sorted({(t, b) for (_, tk, t, b) in m.index if tk == task})
        want = {}
        for j, (t, b) in enumerate(cols, start=1):
            for f in FAMS:
                want[(f, j)] = ((float(m[(f, task, t, b)]) if (f, task, t, b) in m.index else None), False)
        cmp_grid(f"Table {n}", f"results/local_front_{task}_cuda_T{{16,128}}.csv (CPU replay, GTX 1650 cost table)", T[n], want)
    a = fd[fd.alpha_pts == 1.0].groupby(["family", "task", "T"])[["ho_drop_pts", "hard_drop_pts", "easy_drop_pts"]].mean()
    groups = [(t, Tq) for t in ("qqp", "mnli") for Tq in (16, 128)]
    for _, r in T[9].iterrows():
        f = ROWNAME[r.iloc[0]]
        for gi, (t, Tq) in enumerate(groups):
            for k, col in enumerate(["ho_drop_pts", "hard_drop_pts", "easy_drop_pts"]):
                rep = str(r.iloc[1 + 3 * gi + k])
                if (f, t, Tq) in a.index:
                    add("Table 9", f"{f} {t} T{Tq} {col}", rep, a.loc[(f, t, Tq), col], "results/local_front_*_cuda_T*.csv")

    # Table 10: re-splits
    rs = {}
    for f in sorted(R.glob("local_resplit_*_cuda_T16.csv")):
        d = pd.read_csv(f)
        for (B, fam), gg in d.groupby(["B", "family"]):
            rs[(f.name.split("_")[2].upper(), B, fam)] = gg
    names = {v: k for k, v in ROWNAME.items()}
    for _, r in T[10].iterrows():
        fam = ROWNAME[r["Policy"] + (" (ours)" if r["Policy"].startswith("Cost rule") else "")]
        gg = rs[(r["Task"], int(r["B"]), fam)]
        f = f"results/local_resplit_{r['Task'].lower()}_cuda_T16.csv"
        w = f"{r['Task']} B={r['B']} {fam}"
        add("Table 10", w + " over budget, old", str(r.iloc[3]).rstrip("%"), 100 * (gg.naive_drop > 0.01).mean(), f)
        add("Table 10", w + " over budget, certified", str(r.iloc[4]).rstrip("%"), 100 * (gg.cert_drop > 0.01).mean(), f)
        add("Table 10", w + " mean drop, old", str(r.iloc[5]), 100 * gg.naive_drop.mean(), f)
        add("Table 10", w + " mean drop, certified", str(r.iloc[6]), 100 * gg.cert_drop.mean(), f)
        add("Table 10", w + " speedup, old", str(r.iloc[7]).rstrip("×"), gg.naive_speedup.mean(), f)
        add("Table 10", w + " speedup, certified", str(r.iloc[8]).rstrip("×"), gg.cert_speedup.mean(), f)
        add("Table 10", w + " re-splits", str(r.iloc[9]), len(gg), f)
    del names

    # Tables 11-12 (RoBERTa, A6000 eager), 15-16 (LLM classifiers)
    d = a6000_time()
    rob = d[d.model == "roberta"]
    for n, task in ((11, "qqp"), (12, "mnli")):
        cmp_grid(f"Table {n}", f"results/local_time_{task}_a6000_T{{128,32}}.csv (A6000 eager; logs/server_time_{task}_a6000_T*.log)",
                 T[n], grid_want([rob[(rob.task == task) & (rob["T"] == t)] for t in (128, 32)]))
    for n, task in ((15, "qqp"), (16, "mnli")):
        cmp_grid(f"Table {n}", f"results/local_time_{task}_{{llama,qwen}}_a6000_T128.csv (A6000; logs/cls_time_{task}_*_a6000.log)",
                 T[n], grid_want([d[(d.model == mo) & (d.task == task)] for mo in ("llama", "qwen")]))

    # Table 13: certificates on re-splits
    lab = {"QQP": "qqp", "MNLI": "mnli", "CIFAR-100": "cifar100", "MNLI, δ = 5%": "mnli_d05"}
    for _, r in T[13].iterrows():
        task = lab[r.iloc[0]]
        f = f"results/cert_check_{task}_summary.csv"
        d = pd.read_csv(ROOT / f)
        c = lambda m, key="drop", fam="theory+plan", frac=1.0: d[(d.frac == frac) & (d.key == key) & (d.method == m) & (d.family == fam)]
        t = d[(d.frac == 1.0) & (d.key == "drop") & (d.method == "t")].groupby("family").agg(viol=("viol", "mean"), sp=("speedup", "mean"))
        nums = lambda s: re.findall(r"\d+(?:\.\d+)?", str(s))
        x = nums(r.iloc[1])
        add("Table 13", f"{task} t-test mean violation (5 methods)", x[0], 100 * t.viol.mean(), f)
        add("Table 13", f"{task} t-test worst violation", x[1], 100 * t.viol.max(), f)
        x = nums(r.iloc[2])
        add("Table 13", f"{task} cost rule + plan speedup (t)", x[0], t.loc["theory+plan", "sp"], f)
        add("Table 13", f"{task} cost rule + plan violation (t)", x[1], 100 * t.loc["theory+plan", "viol"], f)
        wsr = c("wsr")
        x = nums(r.iloc[3])
        if x:
            add("Table 13", f"{task} WSR speedup", x[0], wsr.speedup.mean(), f)
            add("Table 13", f"{task} WSR share with a certified policy", x[1], 100 * wsr.chose.mean(), f)
        else:
            add_bool("Table 13", f"{task} WSR not run", str(r.iloc[3]), len(wsr) == 0, f)
        hb = c("hb")
        add_bool("Table 13", f"{task} Hoeffding-Bentkus certifies nothing", str(r.iloc[4]),
                 (len(hb) == 0) if str(r.iloc[4]) == "–" else bool(hb.chose.max() == 0), f)
        lf = c("t", "disagree")
        x = nums(r.iloc[5])
        add("Table 13", f"{task} label-free speedup", x[0], lf.speedup.mean(), f)
        add("Table 13", f"{task} label-free violation", x[1], 100 * lf.viol.mean(), f)
        sm = c("t", fam="theory", frac=0.1)
        x = nums(r.iloc[6])
        add("Table 13", f"{task} cost rule, all data speedup", x[0], t.loc["theory", "sp"], f)
        add("Table 13", f"{task} cost rule, 10% data speedup", x[1], sm.speedup.mean(), f)
        add("Table 13", f"{task} cost rule, 10% data violation", x[2], 100 * sm.viol.mean(), f)

    # Table 14: fluctuating load
    lab = {"QQP": "qqp", "MNLI": "mnli", "MNLI, δ = 5%": "mnli_d05", "CIFAR-100": "cifar100"}
    for _, r in T[14].iterrows():
        task = lab[r.iloc[0]]
        f = f"results/load_trace_{task}_levels.csv"
        d = pd.read_csv(ROOT / f)
        ok = d[d.drop_pts <= 1.0]
        nums = lambda s: re.findall(r"\d+(?:\.\d+)?", str(s))
        for k, fam in enumerate(OURS):
            x = nums(r.iloc[1 + k])
            rr = d[(d.family == fam) & (d.tuning == "once@64")].iloc[0]
            add("Table 14", f"{task} {fam} once@64 speedup", x[0], rr.speedup, f)
            add("Table 14", f"{task} {fam} once@64 drop", x[1], rr.drop_pts, f)
        rb = ok[ok.family.isin(BASE) & (ok.tuning == "per_B")].sort_values("speedup").iloc[-1]
        ob = ok[ok.family.isin(BASE) & ok.tuning.str.startswith("once")].sort_values("speedup").iloc[-1]
        x = nums(r.iloc[3])
        add("Table 14", f"{task} best per-B baseline speedup", x[0], rb.speedup, f)
        add("Table 14", f"{task} best per-B baseline drop", x[-1], rb.drop_pts, f)
        x = nums(r.iloc[4])
        add("Table 14", f"{task} best tuned-once baseline speedup", x[0], ob.speedup, f)
        add("Table 14", f"{task} best tuned-once baseline drop", x[-1], ob.drop_pts, f)


# ---------------------------------------------------------------------------------------------------- part B: text
def text_checks() -> None:
    lt = local_time()
    fl = "results/local_time_*_cuda_T*.csv (GTX 1650, laptop)"
    sp = lambda task, Tq, B, f: float(lt[(lt.task == task) & (lt["T"] == Tq) & (lt.B == B) & (lt.family == f)].real_speedup.iloc[0])
    it = "§4.3 text"
    c1 = [json.loads((R / f"local_cost_cuda_T{t}.json").read_text())["points"] for t in (16, 32, 128)]
    add(it, "a layer costs about 1.8 ms however few inputs are active (c(1), median over lengths)", "1.8",
        float(np.median([[p["full_ms"] / 12 for p in pts if p["B"] == 1][0] for pts in c1])),
        "results/local_cost_cuda_T{16,32,128}.json", "c(1) = 2.32, 1.97, 1.76 ms", approx=True)
    it = "§4.4 text"
    add(it, "QQP T16 B8 threshold", "1.27", sp("qqp", 16, 8, "threshold"), fl)
    add(it, "QQP T16 B8 unified", "1.91", sp("qqp", 16, 8, "unified"), fl)
    add(it, "QQP T16 B64 threshold", "2.13", sp("qqp", 16, 64, "threshold"), fl)
    add(it, "QQP T16 B64 unified", "1.97", sp("qqp", 16, 64, "unified"), fl)
    g = lt[lt.family.isin(FAMS)].copy()
    g["rel"] = g.real_speedup / g.groupby(["task", "T", "B"]).real_speedup.transform("max")
    tp = g[g.family == "theory+plan"]
    add(it, "cost rule + plan, largest lag behind the fastest (%)", "7.5", 100 * (1 - tp.rel.min()), fl)
    add(it, "cost rule + plan, cells within 3% of the fastest", "11", int((tp.rel > 0.97).sum()), fl)
    for f, rep in [("random_collective", "15"), ("threshold+plan", "15"), ("unified", "22"), ("threshold", "34"), ("majority", "43")]:
        add(it, f"{f} worst lag (%)", rep, 100 * (1 - g[g.family == f].rel.min()), fl)
    add(it, "cost rule + plan cells over 1 pt", "7", int((tp.ho_drop_pts > 1).sum()), fl)
    add(it, "majority cells over 1 pt", "3", int((g[g.family == "majority"].ho_drop_pts > 1).sum()), fl)
    add(it, "threshold + plan cells over 1 pt", "3", int((g[g.family == "threshold+plan"].ho_drop_pts > 1).sum()), fl)
    q = g[g.task == "qqp"]
    best_is_ours = q.loc[q.groupby(["T", "B"]).real_speedup.idxmax()].family.isin(OURS)
    add(it, "QQP settings where a rule variant is fastest", "8", int(best_is_ours.sum()), fl)
    add(it, "QQP T16 B64 rule (best variant)", "2.52", max(sp("qqp", 16, 64, "theory"), sp("qqp", 16, 64, "theory+plan")), fl)
    nxt = q[(q["T"] == 16) & (q.B == 64) & ~q.family.isin(OURS)].real_speedup.max()
    add(it, "QQP T16 B64 next method", "2.15", nxt, fl)
    add_range(it, "QQP held-out drop of the rule with a plan", "0.52–0.82", q[q.family == "theory+plan"].ho_drop_pts, fl)
    mn = g[g.task == "mnli"]
    add(it, "MNLI T16 B8 unified ahead of the rule (%)", "5",
        100 * (sp("mnli", 16, 8, "unified") / max(sp("mnli", 16, 8, "theory"), sp("mnli", 16, 8, "theory+plan")) - 1), fl)
    lead = [100 * (sp("mnli", t, 64, "random_collective") / max(sp("mnli", t, 64, "theory"), sp("mnli", t, 64, "theory+plan")) - 1) for t in (16, 32, 128)]
    add_range(it, "MNLI B64 random search ahead of the rule (%)", "7–8", [round(x) for x in lead], fl)
    add(it, "QQP T16 B8 rule all exits", "1.79", sp("qqp", 16, 8, "theory"), fl)
    add(it, "QQP T16 B8 rule + plan", "1.85", sp("qqp", 16, 8, "theory+plan"), fl)
    add(it, "MNLI T16 B8 rule all exits", "1.37", sp("mnli", 16, 8, "theory"), fl)
    add(it, "MNLI T16 B8 rule + plan", "1.41", sp("mnli", 16, 8, "theory+plan"), fl)
    add(it, "exit heads cost at most (% of runtime, GTX 1650)", "7", 100 * (1 - lt[lt.family == "heads_only"].real_speedup.min()), fl)
    add_range(it, "MNLI held-out drop of the rule with a plan", "0.88–1.28", mn[mn.family == "theory+plan"].ho_drop_pts, fl)
    add(it, "MNLI settings where the rule + plan is over 1 pt", "7", int((mn[mn.family == "theory+plan"].ho_drop_pts > 1).sum()), fl)
    add_range(it, "MNLI per-sample threshold held-out drop", "1.21–1.23", mn[mn.family == "threshold"].ho_drop_pts, fl)
    add(it, "MNLI random search largest drop", "1.44", mn[mn.family == "random_collective"].ho_drop_pts.max(), fl)
    add_range(it, "QQP unified exit held-out drop", "1.05", q[q.family == "unified"].ho_drop_pts, fl)

    fd = local_front()
    ff = "results/local_front_*_cuda_T*.csv (CPU replay)"
    it = "§4.5 text"
    lt2 = lt[lt.family.isin(FAMS)].copy()
    err = (lt2.model_speedup - lt2.real_speedup).abs() / lt2.real_speedup
    add(it, "cost table median error over timed runs (%)", "1.3", 100 * err.median(), fl)
    add(it, "cost table largest error (%)", "17", 100 * err.max(), fl)
    m = fd[fd.ho_drop_pts <= 1.0].groupby(["family", "task", "T", "B"]).ho_speedup.max().unstack(["task", "T", "B"])
    gain = 100 * (m.loc[OURS].max() / m.drop(OURS).max() - 1)
    add_range(it, "QQP lead of the rule over the best baseline (%)", "2–17", gain["qqp"], ff)
    add(it, "QQP T16 B64 rule", "2.63", m.loc[OURS, ("qqp", 16, 64)].max(), ff)
    add(it, "QQP T16 B64 best baseline", "2.26", m.drop(OURS)[("qqp", 16, 64)].max(), ff)
    add(it, "QQP T128 B8 rule", "2.55", m.loc[OURS, ("qqp", 128, 8)].max(), ff)
    add(it, "QQP T128 B8 best baseline", "2.32", m.drop(OURS)[("qqp", 128, 8)].max(), ff)
    add(it, "MNLI settings within 2% of the best baseline", "4", int((gain["mnli"] >= -2).sum()), ff)
    add(it, "MNLI T16 B64 behind the threshold (%)", "8.5", -gain[("mnli", 16, 64)], ff)
    add(it, "MNLI T128 B16 behind random search (%)", "6.5", -gain[("mnli", 128, 16)], ff)
    add(it, "MNLI T16 B8 rule without plan", "1.23", m.loc["theory", ("mnli", 16, 8)], ff)
    add(it, "QQP T16 B8 rule without plan", "1.78", m.loc["theory", ("qqp", 16, 8)], ff)
    add(it, "MNLI T16 B8 rule with plan", "1.49", m.loc["theory+plan", ("mnli", 16, 8)], ff)
    add(it, "QQP T16 B8 rule with plan", "1.91", m.loc["theory+plan", ("qqp", 16, 8)], ff)
    add(it, "MNLI T16 B8 threshold without plan", "1.05", m.loc["threshold", ("mnli", 16, 8)], ff)
    add(it, "MNLI T16 B8 threshold with plan", "1.48", m.loc["threshold+plan", ("mnli", 16, 8)], ff)
    add(it, "MNLI T16 B8 unified", "1.49", m.loc["unified", ("mnli", 16, 8)], ff)
    tp = fd[(fd.task == "mnli") & (fd["T"] == 128) & (fd.B == 16) & (fd.family == "theory+plan") & (fd.alpha_pts <= 2.0)]
    add_range(it, "MNLI T128 B16 plan policy held-out drop at every target up to 2 pts", "1.16", tp.ho_drop_pts, ff)
    nc = fd.groupby("family").n_cands.max()
    add(it, "candidates, rule without plan", "40", nc["theory"], ff)
    add(it, "candidates, random search", "150", nc["random_collective"], ff)
    add(it, "candidates, rule with plan", "620", nc["theory+plan"], ff)
    add(it, "candidates, threshold with plan", "558", nc["threshold+plan"], ff)
    add(it, "QQP settings where the rule without a plan is ahead", "5",
        int((m.loc["theory", "qqp"] > m.drop(OURS)["qqp"].max()).sum()), ff)
    add(it, "QQP T16 B8 threshold with plan", "1.88", m.loc["threshold+plan", ("qqp", 16, 8)], ff)
    a = fd[fd.alpha_pts == 1.0].groupby(["family", "task", "T"])[["hard_drop_pts", "easy_drop_pts"]].mean()
    th = lambda t: a.loc[("theory", t)].hard_drop_pts.tolist() + a.loc[("theory+plan", t)].hard_drop_pts.tolist()
    add_range(it, "MNLI hard quarter, rule", "2.0–2.7", th("mnli"), ff)
    add_range(it, "QQP hard quarter, rule", "1.3–2.0", th("qqp"), ff)
    add_range(it, "MNLI hard quarter, threshold", "0.6", a.loc[("threshold", "mnli")].hard_drop_pts, ff)
    add_range(it, "QQP hard quarter, threshold", "1.0", a.loc[("threshold", "qqp")].hard_drop_pts, ff)
    add_range(it, "MNLI other three quarters, threshold", "1.4", a.loc[("threshold", "mnli")].easy_drop_pts, ff)
    add_range(it, "QQP other three quarters, threshold", "0.5", a.loc[("threshold", "qqp")].easy_drop_pts, ff)
    add_range(it, "unified exit hard quarter, MNLI", "2.8", a.loc[("unified", "mnli")].hard_drop_pts, ff)
    add_range(it, "unified exit hard quarter, QQP", "4.1", a.loc[("unified", "qqp")].hard_drop_pts, ff)

    sz = {t: pd.read_parquet(ROOT / f"traces/{t}_val.parquet", columns=["split"]).split.value_counts() for t in ("mnli", "qqp")}
    add("§4.4-4.6 text", "MNLI held-out half (examples)", "4,900", sz["mnli"]["heldout"], "traces/mnli_val.parquet", approx=True)
    add("§4.4-4.6 text", "MNLI tuning half (examples)", "4,900", sz["mnli"]["evolve"], "traces/mnli_val.parquet", approx=True)
    add("§4.4-4.6 text", "QQP tuning half (examples)", "20,200", sz["qqp"]["evolve"], "traces/qqp_val.parquet", approx=True)
    it = "§4.6 text"
    rs = pd.concat([pd.read_csv(f).assign(task=f.name.split("_")[2]) for f in sorted(R.glob("local_resplit_*_cuda_T16.csv"))])
    fr = "results/local_resplit_{qqp,mnli}_cuda_T16.csv"
    s = rs.groupby(["task", "B", "family"]).agg(nv=("naive_drop", lambda x: 100 * (x > 0.01).mean()),
                                                 cv=("cert_drop", lambda x: 100 * (x > 0.01).mean()),
                                                 nd=("naive_drop", "mean"), cd=("cert_drop", "mean"),
                                                 ns=("naive_speedup", "mean"), cs=("cert_speedup", "mean"))
    th = s.xs("theory", level="family")
    add_range(it, "cost rule, certified, share over budget (%)", "0–7", th.cv.round(), fr)
    add_range(it, "cost rule, old protocol, share over budget (%)", "13–40", th.nv.round(), fr)
    add_range(it, "cost rule, certified, mean drop (pts)", "0.33–0.64", 100 * th.cd, fr)
    add(it, "rows at or below 10%", "11", int((s.cv <= 10 + 1e-9).sum()), fr)
    add(it, "majority MNLI B64 certified share over budget (%)", "13", s.loc[("mnli", 64, "majority")].cv, fr)
    loss = 100 * (1 - th.cs / th.ns)
    add_range(it, "speed cost of certification, QQP (%)", "2–4", loss.loc["qqp"].round(), fr)
    add_range(it, "speed cost of certification, MNLI (%)", "8–12", loss.loc["mnli"].round(), fr)
    add_range(it, "old protocol MNLI share over budget, every method (%)", "33–43", s.loc["mnli"].nv.round(), fr)
    add_range(it, "old protocol mean drop on re-splits (pts)", "0.7–1.0", 100 * s.nd, fr)

    # ----- §4.2, Figure 1, Table 2: server runs of 6-12 Nov (old selection protocol)
    it = "§4.2 / Figure 1"
    fs = "results/main_{mnli,qqp}_real_*_gpu_B8.csv, _B16-128.csv (eager), results/graph_{mnli,qqp}_gpu.csv (CUDA graphs); A6000"
    eag = pd.concat([pd.concat([rd(f"main_{t}_real_{t}_gpu_B8.csv"), rd(f"main_{t}_real_{t}_gpu_B16-128.csv")]).assign(task=t)
                     for t in ("mnli", "qqp")])
    gra = pd.concat([rd(f"graph_{t}_gpu.csv").assign(task=t) for t in ("mnli", "qqp")])
    both = pd.concat([eag.assign(mode="eager"), gra.assign(mode="graph")])
    add_range(it, "wait-for-all (threshold_nocompact) speedup, eager", "0.78–0.97",
              eag[eag.family == "threshold_nocompact"].real_speedup, fs,
              "under CUDA graphs the range is 0.77–0.96; the report gives one range for both modes")
    add_bool(it, "wait-for-all slower than no early exit in both modes", "< 1x everywhere",
             bool((both[both.family == "threshold_nocompact"].real_speedup < 1).all()), fs)
    add_range(it, "per-sample threshold at B <= 16, eager", "0.77–0.90",
              eag[(eag.family == "threshold") & (eag.B <= 16)].real_speedup, fs)
    add(it, "per-sample threshold MNLI B8, CUDA graphs", "0.89",
        float(gra[(gra.task == "mnli") & (gra.B == 8) & (gra.family == "threshold")].real_speedup.iloc[0]), fs)
    ratio = [json.loads((R / f"runner_cost_{t}.json").read_text())["a"] / json.loads((R / f"runner_cost_graph_{t}.json").read_text())["a"]
             for t in ("mnli", "qqp", "cifar100")]
    add(it, "fixed per-layer cost, eager / graphs (about six-fold)", "6", float(np.mean(ratio)),
        "results/runner_cost_{task}.json, runner_cost_graph_{task}.json", f"ratios {np.round(ratio, 2).tolist()}", approx=True)
    for t in ("mnli", "qqp"):
        add(it, f"exit heads' share of runtime at B = 8 under CUDA graphs, {t.upper()} (%)", "19",
            100 * (1 - float(gra[(gra.task == t) & (gra.B == 8) & (gra.family == "heads_only")].real_speedup.iloc[0])), fs,
            "eager B = 8: 18.8% on both tasks")
    fm = both[(both.task == "mnli")]
    add_range(it, "Figure 1 caption: MNLI cost-rule held-out drop (both panels)", "1.8–2.5",
              fm[fm.family == "theory"].acc_drop_pts, fs, "eager only: 1.77–2.41")
    add_range(it, "Figure 1 caption: MNLI unified exit held-out drop (both panels)", "0.8–0.9",
              fm[fm.family == "unified"].acc_drop_pts, fs)

    it = "Table 2"
    add(it, "CUDA graphs, MNLI B8: cost rule speedup (slowest rule)", "0.86",
        float(gra[(gra.task == "mnli") & (gra.B == 8) & (gra.family == "theory")].real_speedup.iloc[0]), fs)
    rules = ["threshold", "majority", "consensus", "unified", "random_collective", "theory"]
    g8 = gra[(gra.task == "mnli") & (gra.B == 8) & gra.family.isin(rules)]
    add_bool(it, "CUDA graphs, MNLI B8: the cost rule is the slowest rule", "slowest",
             g8.loc[g8.real_speedup.idxmin(), "family"] == "theory", fs)
    w16 = gra[(gra.B <= 16) & gra.family.isin(rules)]
    add_bool(it, "CUDA graphs: unified exit fastest at B <= 16 on text", "unified wins",
             bool((w16.loc[w16.groupby(["task", "B"]).real_speedup.idxmax(), "family"] == "unified").all()), fs)
    f1 = ["theory", "unified", "majority", "threshold"]
    add_range(it, "real-GPU runs lost (pts): MNLI, rules of Figure 1, eager and graphs", "0.8–2.5",
              fm[fm.family.isin(f1)].acc_drop_pts, fs,
              "the row says 'Server, eager'; eager only gives 0.82–2.41; with QQP 0.67–2.45")
    sm = eag[(eag.B <= 16) & eag.family.isin(f1)]
    add_range(it, "simulator error at B <= 16, eager (%): rules of Figure 1", "15–32",
              (100 * (sm.sim_speedup - sm.real_speedup).abs() / sm.real_speedup).round(), fs,
              "definition not stated in the report; per policy and task the error is 1.5–47% (cost rule 23–47%)")
    gg = gra[gra.family.isin(f1)]
    add_bool(it, "simulator overestimates in graph mode", "overestimates",
             bool(((gg.sim_speedup - gg.real_speedup) > 0).mean() > 0.9), fs, "sim > real in all but 2 of 40 cells")
    lo = rd("robust_cifar100_real_cifar100_B64_lowrho.csv").set_index("policy")
    add(it, "low rho, CIFAR-100: robust rule (m = 8)", "1.12", lo.loc["theory m=8", "clean_speedup"],
        "results/robust_cifar100_real_cifar100_B64_lowrho.csv",
        "this file is the 22 Nov rerun (--tag _lowrho); the 13 Nov run that printed 1.12x is not in results/")
    add(it, "low rho, CIFAR-100: plain rule", "1.46", lo.loc["theory m=0", "clean_speedup"], "results/robust_cifar100_real_cifar100_B64_lowrho.csv")
    for t, rep in (("mnli", "1.41"), ("qqp", "2.10"), ("cifar100", "1.46")):
        md = (R / f"matched_table_{t}.md").read_text()
        sec = re.search(r"## Real GPU runner cost.*?(?=\n## |\Z)", md, re.S)
        row = re.search(r"\| Theory rule \(ours\) \| [^|]+ \| ([\d.]+)×", sec.group(0)) if sec else None
        add(it, f"first report: cost rule at B = 64 within one point, {t} (simulated, selected on held-out)", rep,
            float(row.group(1)) if row else None, f"results/matched_table_{t}.md (section 'Real GPU runner cost')",
            "" if row else "section or row not found")
    ts = rd("transfer_summary.csv")
    tv = ts[(ts.family == "theory") & (ts.kind != "in_domain") & (ts.margin == 0.005)].violation_rate
    add_range(it, "transfer: rule over budget (%)", "29–31", 100 * tv, "results/transfer_summary.csv (simulator)")

    it = "§4.7 bandit (single runs)"
    fb = "results/bandit_{mnli,qqp}_{dec_eager,enc_eager_meas}_B64.csv"
    slopes = []
    for f in ("bandit_mnli_dec_eager_B64.csv", "bandit_mnli_enc_eager_meas_B64.csv", "bandit_qqp_dec_eager_B64.csv"):
        d = rd(f)
        for _, gg in d.groupby("seed"):
            cr = (gg.oracle_fid_fid_score - gg.bandit_fid_score).cumsum().to_numpy()
            t = np.arange(1, len(cr) + 1)
            ok = (t >= 100) & (cr > 0)
            slopes.append(np.polyfit(np.log(t[ok]), np.log(cr[ok]), 1)[0])
            break                     # the first seed: "single runs"
    add_range(it, "log-log regret slope, MNLI and QQP (seed 0, current fit)", "0.49–0.65", slopes, fb,
              "refit with run_bandit.py's method (seed 0, t >= 100; t >= 1 gives 0.62-0.69); the committed files hold a later "
              "3-seed run with ablation variants, and the 6 Nov single-run fit that printed 0.49/0.52/0.65 is not in results/ or logs/")
    bt = eag[(eag.family == "bandit") & (eag.B >= 64)]
    add_range(it, "bandit accuracy cost (pts, A6000)", "0.2–0.3", bt.acc_drop_pts, fs)
    add_range(it, "bandit speedup at B = 64-128 (A6000)", "1.26–1.77", bt.real_speedup, fs)

    it = "§4.8 robustness"
    vs = [rd(f.name) for f in sorted(R.glob("robust_violations_*_B64.csv"))]
    add(it, "batches checked", "10,500", sum(int(v.batches.sum()) for v in vs), "results/robust_violations_*_B64.csv (5 tasks)")
    add(it, "violations", "0", sum(int(v.violations.sum()) for v in vs), "results/robust_violations_*_B64.csv")
    try:
        out = subprocess.run([sys.executable, "-m", "pytest", "tests", "simulator/test_simulator.py", "-q", "-p", "no:cacheprovider"],
                             cwd=ROOT, capture_output=True, text=True, timeout=900).stdout
        mm = re.search(r"(\d+) passed", out)
        add(it, "unit tests passing", "19", int(mm.group(1)) if mm else 0, "tests/test_core.py, simulator/test_simulator.py (run now)",
            out.strip().splitlines()[-1] if out.strip() else "")
    except Exception as e:  # noqa: BLE001
        add(it, "unit tests passing", "19", None, "tests/", f"pytest failed to run: {e}")
    for t in ("cifar100", "mnli", "qqp"):
        c = rd(f"robust_{t}_dec_eager_B64.csv").set_index("policy").loc["consensus"]
        add_bool(it, f"consensus loses its whole speedup to one attacker, {t} (victims' latency >= full model at k = 1)",
                 "k=1", bool(c.k1 >= 0.99), f"results/robust_{t}_dec_eager_B64.csv",
                 f"consensus latency k=0 {c.k0:.2f}, k=1 {c.k1:.2f}")

    it = "§5.7 plan text"
    rc = json.loads((R / "runner_cost_wikitext.json").read_text())
    add(it, "KV propagation for a skipped layer, share of a layer (%)", "2", 100 * rc["skip"] / rc["a"],
        "results/runner_cost_wikitext.json (skip / a)", approx=True)
    qf = rd("train_wikitext_qfrozen_exit_accuracy.csv").query("split == 'all'").set_index("layer").accuracy
    add(it, "frozen Qwen2.5-1.5B, gap of the layer-24 exit (pts), 'Expected' paragraph", "5.6", 100 * (qf[28] - qf[24]),
        "results/train_wikitext_qfrozen_exit_accuracy.csv",
        "the 5.7 result paragraph of the same report prints 4.0 from this file; 5.6 was written on 13 Nov")
    it = "§4.9 what did not work"
    add_bool(it, "program search: 300 programs", "300", "checkpoint_300" in (ROOT / "logs/evolve_mnli.log").read_text(), "logs/evolve_mnli.log")
    pr = pd.concat([rd(f"probe_{t}.csv") for t in ("mnli", "qqp", "cifar100")])
    add(it, "hidden-state probe, largest AUROC gain over confidence", "0.026", pr.best_gain.max(), "results/probe_{mnli,qqp,cifar100}.csv")
    ll = rd("train_wikitext_llama_exit_accuracy.csv").query("split == 'all'").set_index("layer").accuracy
    add(it, "frozen Llama-3.1-8B heads, gap at 75% depth (layer 24 of 32, pts)", "5", 100 * (ll[32] - ll[24]),
        "results/train_wikitext_llama_exit_accuracy.csv", "layer 28 (88%): 3.7; frozen Qwen layer 20 (71%): 9.5, layer 24 (86%): 4.0",
        approx=True)
    for m in ("ls8b", "ls1b"):
        ls = rd(f"layerskip_{m}_layer_accuracy.csv")
        add(it, f"LayerSkip {m[2:]}, gap at 75% depth (pts)", "5", float(ls[ls.depth_frac == 0.75].drop_pts.iloc[0]),
            f"results/layerskip_{m}_layer_accuracy.csv", approx=True)


# ---------------------------------------------------------------------------------------------------- part C: render
def render_check(committed: str) -> None:
    """Re-render the report from the committed results with analysis/make_report.py in a scratch copy, and compare
    the numbers in its text with the committed HTML."""
    tmp = Path(tempfile.mkdtemp(prefix="report_render_"))
    try:
        for d in ("analysis", "docs", "simulator", "experiments"):
            shutil.copytree(ROOT / d, tmp / d)
        (tmp / "results").symlink_to(R)
        (tmp / "figures").symlink_to(ROOT / "figures")
        p = subprocess.run([sys.executable, "analysis/make_report.py"], cwd=tmp, capture_output=True, text=True, timeout=900)
        new = tmp / "docs/IE624_progress_report_v2.html"
        if p.returncode != 0 or not new.exists():
            add_bool("Part C", "re-render of the report from committed results", "renders", None, "analysis/make_report.py",
                     (p.stderr or p.stdout)[-300:])
            return
        strip = lambda s: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))
        a, b = re.findall(r"\d+(?:\.\d+)?", strip(new.read_text())), re.findall(r"\d+(?:\.\d+)?", strip(committed))
        diff = [(i, u, v) for i, (u, v) in enumerate(zip(a, b)) if u != v]
        ROWS.append(dict(item="Part C", where="all numbers in the re-rendered report text", reported=f"{len(b)} numbers",
                         recomputed=f"{len(a)} numbers, {len(diff)} differ",
                         status="MATCH" if not diff and len(a) == len(b) else "MISMATCH",
                         files="analysis/make_report.py on results/", note=str(diff[:8])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results_repro"))
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args()
    html = (ROOT / "docs/IE624_progress_report_v2.html").read_text()
    tabs = pd.read_html(StringIO(html))
    tables(tabs)
    text_checks()
    if not a.no_render:
        render_check(html)
    d = pd.DataFrame(ROWS)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "report_numbers.csv", index=False)
    s = d.groupby(["item", "status"]).size().unstack(fill_value=0)
    for c in ("MATCH", "MISMATCH", "NO FILE"):
        if c not in s:
            s[c] = 0
    s = s[["MATCH", "MISMATCH", "NO FILE"]]
    md = ["# Report numbers recomputed from the result files", "",
          f"`analysis/check_report_numbers.py` on `docs/IE624_progress_report_v2.html`. Totals: "
          f"**{int(s.MATCH.sum())} MATCH, {int(s.MISMATCH.sum())} MISMATCH, {int(s['NO FILE'].sum())} NO FILE** "
          f"({len(d)} numbers).", "", "| item | MATCH | MISMATCH | NO FILE |", "|---|---|---|---|"]
    md += [f"| {i} | {r.MATCH} | {r.MISMATCH} | {r['NO FILE']} |" for i, r in s.iterrows()]
    bad = d[d.status != "MATCH"]
    md += ["", "## Every number that is not MATCH", "", "| item | where | reported | recomputed | status | file | note |",
           "|---|---|---|---|---|---|---|"]
    md += [f"| {r.item} | {r.where} | {r.reported} | {r.recomputed} | {r.status} | `{r.files}` | {str(r.note).replace('|', '/')} |"
           for r in bad.itertuples()]
    (out / "REPORT_NUMBERS.md").write_text("\n".join(md) + "\n")
    print(s.to_string())
    print(f"total: {int(s.MATCH.sum())} MATCH, {int(s.MISMATCH.sum())} MISMATCH, {int(s['NO FILE'].sum())} NO FILE")
    print(bad.to_string(max_colwidth=70))


if __name__ == "__main__":
    main()
