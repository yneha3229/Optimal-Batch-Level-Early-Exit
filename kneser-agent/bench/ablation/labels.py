"""Per-objection labels for the model ablation -> label_detail.json (one record per Critic objection).

Rule (agreed 26 Nov): an objection is FALSE if it asserts something wrong about the data or the tools, including a
wrong inference (e.g. "family_rule returned null, so K(14,4) is not confirmed": a rule naming no field says nothing
about the family). A true statement, even an irrelevant one, is not false; a pure process request ("query X first")
is not false. Labelled by hand: the pilot rows by the teammate who built the ablation, every other row by Claude Code
(26 Nov evening), with the reviewer's two corrections applied. score.py counts the false labels from this file.

    python labels.py            # writes label_detail.json and prints any objection without a label
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from score import CONFIGS, TR, TRUTH  # noqa: E402

# (cell|config, substring of the objection) -> (false?, reason). Unlisted objections of the Claude configurations
# default to "not false" (checked one by one: true statements or process requests); unlisted Qwen objections are an
# error, so nothing is labelled silently.
L = {
    # ---- Qwen 32B, GPU (reference traces *_ref_*, code before d22d46c)
    ("K_16_4_s3|Qwen2.5-32B", "does not specify (n, k)"): (False, "true: params were empty"),
    ("K_16_4_s3|Qwen2.5-32B", "should be C(12,4) = 495"): (True, "says 495 does not match, then that it should be 495"),
    ("K_16_4_s3|Qwen2.5-32B", "C(12,3) = 220"): (True, "the K(16,4) degree is C(12,4) = 495, not C(12,3)"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B", "distinct eigenvalues count"): (True, "a Kneser graph has few distinct eigenvalues, not N"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B", "degree = C(n-k, k) = 482"): (True, "C(16,4) = 1820 does match; 482 is the perturbed minimum degree"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B", "inconsistent with the number of receivers"): (True, "binomial_match (16,4) fits 1820 receivers"),
    ("K_14_4_s0|Qwen2.5-32B", "does not specify (n, k)"): (False, "true: params were empty"),
    # ---- Qwen 32B, CPU (fixed code)
    ("K_16_4_s3|Qwen2.5-32B-cpu", "does not specify (n, k)"): (False, "true: params were empty"),
    ("K_16_4_s3|Qwen2.5-32B-cpu", "do not match the expected pattern"): (True, "70, 126, 210, 330 = C(8,4), C(9,4), C(10,4), C(11,4): they match"),
    ("K_16_4_s3|Qwen2.5-32B-cpu", "do not match the expected values"): (True, "same: the counts match K(16,4)"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B-cpu", "regular by definition"): (False, "true: Kneser graphs are regular (reviewer's correction)"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B-cpu", "within the range 482-495"): (True, "binomial_match (16,4) does fit 1820 receivers"),
    ("K_16_4-del0.01_s0|Qwen2.5-32B-cpu", "does not match the number of receivers (1820) and the degree range"): (True, "binomial_match (16,4) does fit 1820 receivers"),
    ("K_14_4_s0|Qwen2.5-32B-cpu", "relies on family_rule without querying it first"): (False, "true at that point; process request"),
    ("K_14_4_s0|Qwen2.5-32B-cpu", "do not match the expected spectrum"): (True, "the spectrum it lists as expected is the one in the summary"),
    ("K_14_4_s0|Qwen2.5-32B-cpu", "even though the result is already in check_history"): (True, "the rule had been queried; its result was in the history"),
    ("K_14_4_s0|Qwen2.5-32B-cpu", "necessary to confirm the hypothesis"): (True, "wrong inference: a rule naming no field does not refute K(14,4) (reviewer's rule)"),
    # ---- Claude: one borderline case, recorded
    ("K_16_4-del0.01_s1|haiku-med", "do not fit any exact overlap"): (False, "borderline: true that 66-70 fit no exact overlap (they come from deleted edges); claim is about the Solver's evidence, which indeed did not explain them"),
    ("K_16_4-del0.01_s1|haiku-med", "not every number is explained"): (False, "borderline, as above"),
    ("K_16_4-del0.01_s1|haiku-med", "Not every number is explained"): (False, "borderline, as above"),
}


def main():
    out, missing = {}, []
    for g in TRUTH:
        for cfg, pat in CONFIGS.items():
            f = TR / pat.format(g=g)
            if not f.exists():
                continue
            recs = []
            for e in map(json.loads, open(f)):
                if e["ev"] == "llm_reply" and e.get("agent") == "critic":
                    for o in e["reply"].get("objections", []):
                        hit = [v for (cell, sub), v in L.items() if cell == f"{g}|{cfg}" and sub in o]
                        if hit:
                            fl, why = hit[0]
                        elif not cfg.startswith("Qwen"):
                            fl, why = False, "true statement or process request (checked)"
                        else:
                            missing.append(f"{g}|{cfg}: {o[:160]}")
                            continue
                        recs.append({"objection": o, "false": fl, "why": why})
            out[f"{g}|{cfg}"] = recs
    (HERE / "label_detail.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"labelled {sum(len(v) for v in out.values())} objections in {len(out)} runs; "
          f"false: {sum(r['false'] for v in out.values() for r in v)}")
    for m in missing:
        print("NO LABEL:", m)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
