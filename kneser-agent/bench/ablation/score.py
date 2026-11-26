"""Score the model ablation from the traces: one row per (graph, configuration) -> scores.jsonl and a table.

Automatic metrics come from the trace. False Critic objections cannot be judged by code; they are labelled by hand
in false_objections.json ({"<graph>|<config>": n}), after reading every objection (python score.py --objections).
"""
import json, math, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TR = HERE.parents[1] / "traces"
TRUTH = {"K_16_4_s3": (16, 4, "rule applies"), "K_16_4-del0.01_s0": (16, 4, "perturbed"),
         "K_14_4_s0": (14, 4, "rule exception")}
CONFIGS = {"Qwen2.5-32B": "{g}_Qwen2.5-Coder-32B-Instruct-AWQ.jsonl", "Qwen3-4B": "{g}_Qwen3-4B-Instruct-2507.jsonl"}
for c in ("haiku-med", "sonnet-med", "opus-low", "opus-med", "opus-high"):
    CONFIGS[c] = "{g}_%s_chat.jsonl" % c
BIN = re.compile(r"C\((\d+)\s*,\s*(\d+)\)\s*=\s*(\d+)")


def named_nk(reply):
    """(n, k) pairs the Solver named in a reply: hypothesis params, or family_rule arguments in its plan."""
    p = (reply.get("family_hypothesis") or {}).get("params") or {}
    out = {(p.get("n", p.get("n_guess")), p.get("k", p.get("k_guess")))}
    out |= {((s.get("args") or {}).get("n"), (s.get("args") or {}).get("k"))
            for s in reply.get("plan", []) if s.get("tool") == "family_rule"}
    return out


def score(g, cfg, path, labels):
    ev = [json.loads(l) for l in open(path)]
    n, k, kind = TRUTH[g]
    rep = [e for e in ev if e["ev"] == "llm_reply"]
    sol = [e["reply"] for e in rep if e.get("agent") == "solver"]
    cri = [e["reply"] for e in rep if e.get("agent") == "critic"]
    ver = [e for e in ev if e["ev"] == "verdict"][-1]
    text = json.dumps(sol + cri)
    return {
        "graph": g, "kind": kind, "config": cfg, "verdict": ver["verdict"], "bracket": ver["bracket"],
        "llm_calls": sum(1 for e in ev if e["ev"] == "llm_request"),
        "invalid": sum(1 for e in ev if e["ev"] == "llm_invalid"),
        "fallback": sum(1 for e in ev if e["ev"] == "llm_fallback"),
        "first_reply_names_nk": bool(sol) and (n, k) in named_nk(sol[0]),
        "critic_rounds": len(cri), "critic_objections": sum(len(c.get("objections", [])) for c in cri),
        "false_objections": labels.get(f"{g}|{cfg}"),
        "wrong_binomials": sum(1 for a, b, x in BIN.findall(text) if int(b) <= int(a) and math.comb(int(a), int(b)) != int(x)),
        "safety_net": any(e["ev"] == "event" and e.get("name") == "safety_net" for e in ev),
        "codes_tried": sum(1 for e in ev if e["ev"] == "start" and e.get("tool") == "code"),
    }


def main():
    lab_f = HERE / "false_objections.json"
    labels = json.loads(lab_f.read_text()) if lab_f.exists() else {}
    rows = []
    for g in TRUTH:
        for cfg, pat in CONFIGS.items():
            f = TR / pat.format(g=g)
            if not f.exists():
                continue
            if "--objections" in sys.argv:
                for e in map(json.loads, open(f)):
                    if e["ev"] == "llm_reply" and e.get("agent") == "critic":
                        for o in e["reply"].get("objections", []):
                            print(f"{g}|{cfg}: {o}")
                continue
            rows.append(score(g, cfg, f, labels))
    if "--objections" in sys.argv:
        return
    (HERE / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    cols = ["graph", "config", "verdict", "bracket", "llm_calls", "invalid", "fallback", "first_reply_names_nk",
            "critic_objections", "false_objections", "wrong_binomials", "safety_net", "codes_tried"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
