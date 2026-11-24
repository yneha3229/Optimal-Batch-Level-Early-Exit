"""Probe: Shannon LP on 10-receiver subsets of open components (beta(G) >= beta(G[S]) for any subset S).
   Run from the repository root: PYTHONPATH=lambda-agent python lambda-agent/subset_lp_probe.py"""
import json, glob, random, time
from fractions import Fraction
from lambda_agent.graph import Digraph, induced, scc
from lambda_agent.lower import shannon
from lambda_agent.replay import Sim
from lambda_agent.controllers import OracleController
suites = {}
for which in ("train", "hard"):
    for d in json.load(open(f"lambda-agent/portfolio_data/{which}.json")): suites[d["name"]] = d
rows = []
for f in sorted(glob.glob("lambda-agent/portfolio_data/logs_train/*.json")) + sorted(glob.glob("lambda-agent/portfolio_data/logs_hard/*.json")):
    L = json.load(open(f))
    g = Digraph.from_json(suites[L["graph"]])
    for c in L["comps"]:
        if c.get("single"): continue
        evs = [e for m in c["moves"].values() for e in m["events"]]
        lo = max([Fraction(e[2]) for e in evs if e[1] == "lower"], default=Fraction(0))
        hi = min([Fraction(e[2]) for e in evs if e[1] == "upper"], default=Fraction(99))
        if lo >= hi or len(c["verts"]) <= 10: continue
        h = induced(g, c["verts"])
        acyc = next((e[3] for e in reversed(c["moves"]["acyclic"]["events"]) if e[1] == "lower" and e[3]), None) or []
        rng = random.Random(0); best = Fraction(0); t0 = time.time()
        for trial in range(25):
            base = list(acyc)[: 10]
            rest = [v for v in range(h.n) if v not in base]; rng.shuffle(rest)
            # grow by neighbours of the base first (they create cycles the LP can exploit)
            rest.sort(key=lambda v: -sum(h.knows(v, u) and h.knows(u, v) for u in base) + rng.random())
            S = sorted(base + rest[: 10 - len(base)]) if len(base) < 10 else sorted(rng.sample(range(h.n), 10))
            best = max(best, shannon(induced(h, S)))
            if time.time() - t0 > 40: break
        rows.append((L["graph"], h.n, str(lo), str(hi), str(best)))
        print(L["graph"], "n", h.n, "MAIS/lower", lo, "upper", hi, "best LP on a 10-subset", best, "improves" if best > lo else "", flush=True)
print(f"{sum(Fraction(r[4]) > Fraction(r[2]) for r in rows)}/{len(rows)} open components get a better lower bound; "
      f"{sum(Fraction(r[4]) >= Fraction(r[3]) for r in rows)} get pinned")
