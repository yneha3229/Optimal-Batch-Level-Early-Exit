"""Learn a field-ordering policy: from graph features, predict which field closes the minrank gap, so the agent
tries the right field first instead of in spectrum-heuristic or ascending order.

    python minrank-agent/learn.py data     build results/dataset.jsonl (features + per-field spectral + closing field)
    python minrank-agent/learn.py train     train the classifier, 5-fold, and compare ordering policies

The dataset is many structured graphs (mostly Kneser, where the closing field is graph-dependent). The comparison
metric is the number of fields the agent must try before the gap closes, on held-out graphs:
  ascending (no insight)  vs  spectrum heuristic  vs  learned.
"""
from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from minrank_agent import bounds as B  # noqa: E402
from minrank_agent import structgraph as SG  # noqa: E402
from minrank_agent.agent import FIELDS, features, predicted_bound, real_spectrum, spectrum_order  # noqa: E402

DS = HERE / "results" / "dataset.jsonl"


def graphs():
    g = []
    for k in range(2, 6):
        for n in range(2 * k + 1, 40):
            if comb(n, k) <= 330:
                g.append(SG.kneser(n, k))
    for n, k in [(5, 2), (6, 2), (7, 2), (6, 3), (7, 3), (8, 3), (9, 4)]:
        if comb(n, k) <= 330:
            g.append(SG.johnson(n, k))
    for q in (5, 13, 17, 29, 37, 41):
        g.append(SG.paley(q))
    for n in range(7, 20):
        g.append(SG.comp_cycle(n))
    for n in range(9, 24):
        for r in (2, 3):
            if r < n // 2:
                g.append(SG.cycle_power(n, r))
    return g


def cmd_data():
    DS.parent.mkdir(exist_ok=True)
    with open(DS, "w") as f:
        for g in graphs():
            a = g.alpha_known if getattr(g, "alpha_known", None) is not None else B.independence(g.A, 10)[0]
            spec = {p: B.spectral_field(g.A, p)[0] for p in FIELDS}
            closing = [p for p in FIELDS if spec[p] <= a]
            feats = features(g, real_spectrum(g.A))
            rec = {"graph": g.name, "family": g.family, "n": g.n, "alpha": a, "spectral": spec,
                   "closing_fields": closing, "best_field": min(closing, key=lambda p: (spec[p], p)) if closing else None,
                   "feats": feats}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(g.name, "closes@", closing or "-", flush=True)


def _order_from_scores(scores, n, spectrum_feats):
    return sorted(FIELDS, key=lambda p: (-scores.get(p, 0.0), spectrum_feats[p], p))


def cmd_train():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    rows = [json.loads(x) for x in DS.read_text().splitlines() if x.strip()]
    closed = [r for r in rows if r["best_field"] is not None]
    cols = sorted(closed[0]["feats"].keys())
    X = np.array([[float(r["feats"][c]) for c in cols] for r in closed])
    y = np.array([r["best_field"] for r in closed])
    print(f"{len(rows)} graphs, {len(closed)} with a closing field; closing-field distribution: "
          f"{dict(zip(*np.unique(y, return_counts=True)))}")

    def closes(r, p):
        return r["spectral"][str(p)] <= r["alpha"]

    def fields_tried(order, r):
        for i, p in enumerate(order, 1):
            if closes(r, p):
                return i
        return len(FIELDS)

    def sp_feats(r):
        return {p: r["feats"][f"pred_F{p}"] for p in FIELDS}

    asc, spec_h, learned = [], [], []
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=0)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        classes = list(clf.classes_)
        for idx, i in enumerate(te):
            r = closed[i]
            spf = sp_feats(r)
            asc.append(fields_tried(list(FIELDS), r))
            spec_h.append(fields_tried(sorted(FIELDS, key=lambda p: (spf[p], p)), r))
            sc = {p: proba[idx][classes.index(p)] if p in classes else 0.0 for p in FIELDS}
            learned.append(fields_tried(_order_from_scores(sc, r["n"], spf), r))
    # The learned GATE: from cheap features (one eigendecomposition, no mod-p ranks), predict whether multi-field
    # spectral will close the gap -- so the agent skips the expensive search on graphs where it cannot pay.
    import pickle

    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import accuracy_score, precision_score, recall_score

    Xg = np.array([[float(r["feats"][c]) for c in cols] for r in rows])
    yg = np.array([1 if r["best_field"] is not None else 0 for r in rows])
    gate = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0)
    predg = cross_val_predict(gate, Xg, yg, cv=5)
    gate.fit(Xg, yg)
    pickle.dump({"clf": gate, "cols": cols}, open(HERE / "results" / "gate.pkl", "wb"))

    out = (f"closed graphs: {len(closed)} / {len(rows)}\n\n"
           f"LEARNED GATE -- predict closability from cheap spectral features (5-fold):\n"
           f"  accuracy {accuracy_score(yg, predg):.2f}  precision {precision_score(yg, predg):.2f}  "
           f"recall {recall_score(yg, predg):.2f}\n"
           f"  (lets the agent skip the mod-p search on the {int((yg==0).sum())} graphs whose gap cannot close here)\n\n"
           f"FIELD ORDER -- avg fields tried before the gap closes (held-out, lower better):\n"
           f"  ascending order    {np.mean(asc):.2f}\n"
           f"  spectrum heuristic {np.mean(spec_h):.2f}\n"
           f"  learned ranker     {np.mean(learned):.2f}   (does not beat the heuristic: too few closed "
           f"instances across 6 field-classes, and F2 is modal)\n"
           f"  oracle             1.00")
    (HERE / "results" / "learn_report.txt").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    {"data": cmd_data, "train": cmd_train}[sys.argv[1]]()
