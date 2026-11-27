"""Write the large speedup tables of the report as LaTeX (report/tables/*.tex).

Values are copied from the progress report of 13 Nov (Tables 4, 5, 7, 8) and from the regenerated report of
26 Nov (Tables 11, 12, 12b, 15, 16); each table names the result file it comes from. A trailing '*' marks a
held-out accuracy drop above one point; '-' means no policy. Bold is computed here:
  mode "max"     fastest value in the column (Tables 4, 5, 7, 8, as in the 13 Nov report)
  mode "budget"  fastest value in the column among cells without '*' (Tables 11, 12, 12b, 15, 16)
Usage: python report/make_tables.py
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "tables"

ROWS = ["Per-sample threshold", "Threshold + exit plan", "Unified exit", "Majority vote",
        "Random search (150 rules)", "Cost rule, all exits (ours)", "Cost rule + exit plan (ours)"]

T4 = {  # QQP, GTX 1650, timed; columns: len16 B8 B16 B64 | len32 ... | len128 ...
    "All heads, no exits": "0.93 0.96 0.99 0.96 0.97 0.99 0.99 0.99 1.00",
    "Per-sample threshold": "1.27 1.73 2.13 1.87 1.83 2.20 2.26 2.20 2.28",
    "Threshold + exit plan": "1.86* 1.92* 2.15 1.93* 1.90 2.19 2.27 2.20 2.30",
    "Unified exit": "1.91* 1.95* 1.97* 1.94* 1.94* 1.97* 1.98* 2.00* 2.00*",
    "Majority vote": "1.30 1.38 1.80 1.35 1.42 1.80 1.39 1.42 1.88",
    "Random search (150 rules)": "1.63 1.92 2.14 1.97 1.94 2.14 2.28 2.25 2.33",
    "Cost rule, all exits (ours)": "1.79 2.00 2.52 2.12 2.24 2.51 2.46 2.50 2.56",
    "Cost rule + exit plan (ours)": "1.85 2.04 2.44 2.11 2.24 2.53 2.39 2.46 2.55",
}
T5 = {
    "All heads, no exits": "0.93 0.96 0.98 0.95 0.98 0.99 0.98 0.98 1.00",
    "Per-sample threshold": "1.24* 1.44* 1.72* 1.30* 1.57* 1.71* 1.75* 1.80* 1.81*",
    "Threshold + exit plan": "1.46 1.45 1.63 1.47 1.48 1.68 1.67 1.72 1.77",
    "Unified exit": "1.48 1.46 1.49 1.44 1.50 1.50 1.48 1.49 1.50",
    "Majority vote": "1.26* 1.36 1.47 1.33* 1.43 1.48 1.39* 1.47 1.49",
    "Random search (150 rules)": "1.37 1.60* 1.86* 1.48 1.68* 1.84* 1.70 1.74 1.91*",
    "Cost rule, all exits (ours)": "1.37* 1.60* 1.70* 1.60* 1.56* 1.70* 1.73* 1.73* 1.78",
    "Cost rule + exit plan (ours)": "1.41 1.63* 1.72* 1.60* 1.62* 1.72* 1.73* 1.74* 1.77",
}
T7 = {  # QQP, cost table, matched held-out drop <= 1 point; len16 B8 B16 B64 | len128 B8 B16 B64
    "Per-sample threshold": "1.08 1.48 2.26 2.30 2.32 2.44",
    "Threshold + exit plan": "1.88 1.89 2.16 2.26 2.29 2.35",
    "Unified exit": "1.49 1.49 1.50 1.50 1.50 1.50",
    "Majority vote": "1.34 1.45 1.72 1.43 1.51 1.74",
    "Random search (150 rules)": "1.48 1.89 2.14 2.32 2.36 2.29",
    "Cost rule, all exits (ours)": "1.78 2.13 2.63 2.55 2.48 2.67",
    "Cost rule + exit plan (ours)": "1.91 2.08 2.63 2.55 2.48 2.67",
}
T8 = {
    "Per-sample threshold": "1.05 1.30 1.66 1.68 1.68 1.74",
    "Threshold + exit plan": "1.48 1.49 1.63 1.67 1.68 1.72",
    "Unified exit": "1.49 1.49 1.50 1.50 1.50 1.50",
    "Majority vote": "1.20 1.41 1.48 1.28 1.47 1.50",
    "Random search (150 rules)": "1.27 1.38 - 1.68 1.69 -",
    "Cost rule, all exits (ours)": "1.23 1.48 1.51 1.55 1.58 1.76",
    "Cost rule + exit plan (ours)": "1.49 1.52 1.52 1.65 - 1.76",
}
T11 = {  # QQP, A6000 eager; len128 B8 B16 B64 B128 | len32 ...
    "Per-sample threshold": "1.48 1.79 2.11 2.13 0.84 0.85 1.75 2.03",
    "Threshold + exit plan": "1.90* 1.83 2.15 2.13 1.73* 1.80* 1.86 2.04",
    "Unified exit": "1.91* 1.96* 1.99* 1.99* 1.79* 1.83* 1.97* 1.98*",
    "Majority vote": "1.33 1.40 1.80 1.87 1.06 1.16 1.73 1.83",
    "Random search (150 rules)": "1.77 1.99 2.11 2.26 1.09 1.14 1.75 2.34",
    "Cost rule, all exits (ours)": "1.84 2.20 2.45 2.39 1.33 1.30 2.11 2.37",
    "Cost rule + exit plan (ours)": "1.95 2.20 2.46 2.44 1.68* 1.76* 2.11 2.30",
}
T12 = {
    "Per-sample threshold": "1.33* 1.54* 1.69* 1.73* 0.85* 0.93* 1.48* 1.65*",
    "Threshold + exit plan": "1.46 1.51 1.66 1.67* 1.38 1.41 1.47 1.63*",
    "Unified exit": "1.46 1.49 1.50 1.50 1.42 1.45 1.48 1.49",
    "Majority vote": "1.32* 1.40 1.48 1.51 1.05* 1.20 1.44 1.49",
    "Random search (150 rules)": "1.44 1.66* 1.80* 1.83* 1.01 1.22* 1.71* 1.79*",
    "Cost rule, all exits (ours)": "1.47* 1.63* 1.72* 1.71* 1.07 1.20* 1.61* 1.67*",
    "Cost rule + exit plan (ours)": "1.54* 1.65* 1.71* 1.72* 1.35 1.40 1.65* 1.69*",
}
T15 = {  # QQP; Llama B8 B16 B64 B128 | Qwen ...
    "Per-sample threshold": "3.08 3.07 3.32 3.32 1.52 1.97 2.54 2.58",
    "Threshold + exit plan": "3.04 3.03 3.27 3.26 2.00 2.00 2.55 2.58",
    "Unified exit": "2.65 2.66 2.66 2.66 1.73 1.74 1.75 1.75",
    "Majority vote": "1.94 1.64 1.81 1.66 1.45 1.60 1.90 1.86",
    "Random search (150 rules)": "3.13 3.11 3.50 3.43 2.02* 2.41 2.56 2.60",
    "Cost rule, all exits (ours)": "3.48 3.52 3.66 3.62 2.15 2.52 2.96 3.02",
    "Cost rule + exit plan (ours)": "3.48 3.49 3.64 3.62 2.18 2.52 2.96 3.01",
}
T16 = {
    "Per-sample threshold": "2.38 2.44* 2.62 2.61 1.34* 1.65* 1.89* 1.94*",
    "Threshold + exit plan": "2.56 2.47 2.53 2.53 1.70* 1.72* 1.73* 1.72*",
    "Unified exit": "2.64* 2.66* 2.66* 2.66* 1.39 1.40 1.40 1.40",
    "Majority vote": "1.62 1.72 2.75* 2.72* 1.29 - 1.54 -",
    "Random search (150 rules)": "2.40 2.66 2.66 2.66 1.48 1.37 - 1.33",
    "Cost rule, all exits (ours)": "2.97 3.05 3.12 3.14 1.10 1.16 1.18 1.19",
    "Cost rule + exit plan (ours)": "2.97 3.04 3.12 3.14 1.38 1.41 1.44 1.48",
}
T12B = {
    "Per-sample threshold": "1.46", "Threshold + exit plan": "1.48", "Unified exit": "-",
    "Majority vote": "1.00", "Random search (150 rules)": "1.46", "Cost rule, all exits (ours)": "1.40",
    "Cost rule + exit plan (ours)": "1.44",
}


def val(s: str) -> float | None:
    return None if s == "-" else float(s.rstrip("*"))


def cell(s: str, bold: bool) -> str:
    if s == "-":
        return "--"
    v = s.rstrip("*")
    txt = f"\\textbf{{{v}}}" if bold else v
    return txt + ("$^{*}$" if s.endswith("*") else "")


def body(data: dict, mode: str, groups: int) -> str:
    rows = list(data)
    cells = {r: data[r].split() for r in rows}
    ncol = len(cells[rows[0]])
    best = []
    for j in range(ncol):
        vals = [(val(cells[r][j]), r) for r in rows
                if val(cells[r][j]) is not None and (mode == "max" or not cells[r][j].endswith("*"))]
        best.append(max(v for v, _ in vals) if vals else None)
    lines = []
    for r in rows:
        cs = [cell(cells[r][j], best[j] is not None and val(cells[r][j]) == best[j]
                   and (mode == "max" or not cells[r][j].endswith("*"))) for j in range(ncol)]
        name = r.replace(" (ours)", " \\textit{(ours)}")
        lines.append(name + " & " + " & ".join(cs) + r" \\")
        if r == "Random search (150 rules)" or r == "All heads, no exits":
            lines.append(r"\midrule")
    return "\n".join(lines)


def table(name: str, data: dict, mode: str, head1: list[tuple[str, int]], bs: list[str]) -> None:
    ncol = sum(n for _, n in head1)
    spec = "@{}l" + "".join("c" * n for _, n in head1) + "@{}"
    h1 = " & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{t}}}" for t, n in head1) + r" \\"
    rules, c0 = [], 2
    for _, n in head1:
        rules.append(f"\\cmidrule(lr){{{c0}-{c0 + n - 1}}}")
        c0 += n
    h2 = "Policy & " + " & ".join(f"$B={b}$" for b in bs) + r" \\"
    tex = "\n".join([f"\\begin{{tabular}}{{{spec}}}", r"\toprule", h1, "".join(rules), h2, r"\midrule",
                     body(data, mode, len(head1)), r"\bottomrule", r"\end{tabular}"])
    assert len(bs) == ncol
    (OUT / f"{name}.tex").write_text(tex + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    b3, b4 = ["8", "16", "64"], ["8", "16", "64", "128"]
    L = lambda t: f"sequence length {t}"
    table("t04", T4, "max", [(L(16), 3), (L(32), 3), (L(128), 3)], b3 * 3)
    table("t05", T5, "max", [(L(16), 3), (L(32), 3), (L(128), 3)], b3 * 3)
    table("t07", T7, "max", [(L(16), 3), (L(128), 3)], b3 * 2)
    table("t08", T8, "max", [(L(16), 3), (L(128), 3)], b3 * 2)
    table("t11", T11, "budget", [("QQP, length 128", 4), ("QQP, length 32", 4)], b4 * 2)
    table("t12", T12, "budget", [("MNLI, length 128", 4), ("MNLI, length 32", 4)], b4 * 2)
    table("t15", T15, "budget", [("Llama-3.1-8B", 4), ("Qwen2.5-1.5B", 4)], b4 * 2)
    table("t16", T16, "budget", [("Llama-3.1-8B", 4), ("Qwen2.5-1.5B", 4)], b4 * 2)
    table("t12b", T12B, "budget", [("eager", 1)], ["8"])
    print("written:", sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
