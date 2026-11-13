"""Step 1 of docs/PLAN_PAPER.md: can a probe on the hidden state tell which exits are correct better than confidence?

For each exit, fit probes for "this exit's prediction is correct" and report held-out AUROC against plain confidence.
  conf            max softmax probability (what every policy uses now)
  lr_evolve       logistic regression on [CLS, conf, entropy], fitted on the validation evolve split
  lr_train        the same, fitted on 50k training inputs (more data, but the model has seen them)
  mlp_train       2-layer MLP on the same features, fitted on the training inputs
  mlp_train+prev  MLP on the CLS of this exit and the previous one
Usage: ~/venv/bin/python analysis/probe_correctness.py mnli qqp cifar100     (CPU only)
Output: results/probe_<task>.csv, and held-out probe scores in traces/hidden/<task>_probe.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def feats(cls, probs, e, prev=False):
    p = probs[:, e]
    extra = np.stack([p.max(1), -(p * np.log(p + 1e-9)).sum(1)], 1)
    x = [cls[:, e].astype(np.float32), extra]
    if prev and e > 0:
        x.append(cls[:, e - 1].astype(np.float32))
    return np.concatenate(x, 1)


def main() -> None:
    for task in sys.argv[1:]:
        va, trn = np.load(ROOT / f"traces/hidden/{task}_val.npz"), np.load(ROOT / f"traces/hidden/{task}_train.npz")
        split = pd.read_parquet(ROOT / f"traces/{task}_val.parquet", columns=["idx", "split"]).sort_values("idx").split.to_numpy()
        ev, ho = split == "evolve", split == "heldout"
        vc, vp, vy = va["cls"], va["probs"], va["label"]
        ty = trn["label"]
        tc, tp = trn["cls"][:len(ty)], trn["probs"][:len(ty)]
        rows, scores = [], {}
        for e in range(vp.shape[1] - 1):
            yv = vp[:, e].argmax(1) == vy
            yt = tp[:, e].argmax(1) == ty
            r = {"task": task, "exit": e, "heldout_acc": float(yv[ho].mean()), "train_acc": float(yt.mean()),
                 "conf": roc_auc_score(yv[ho], vp[ho, e].max(1))}
            Xv, Xt = feats(vc, vp, e), feats(tc, tp, e)
            lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=300))
            mlp = lambda: make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=1e-2, early_stopping=True,
                                                                        max_iter=60, random_state=0))
            m = lr().fit(Xv[ev], yv[ev]); r["lr_evolve"] = roc_auc_score(yv[ho], m.predict_proba(Xv[ho])[:, 1])
            m = lr().fit(Xt, yt); r["lr_train"] = roc_auc_score(yv[ho], m.predict_proba(Xv[ho])[:, 1])
            m = mlp().fit(Xt, yt); s = m.predict_proba(Xv)[:, 1]; r["mlp_train"] = roc_auc_score(yv[ho], s[ho])
            scores[f"mlp_train_{e}"] = s.astype(np.float32)
            m = mlp().fit(feats(tc, tp, e, True), yt)
            r["mlp_train+prev"] = roc_auc_score(yv[ho], m.predict_proba(feats(vc, vp, e, True)[ho])[:, 1])
            r["best_gain"] = max(r[k] for k in ["lr_evolve", "lr_train", "mlp_train", "mlp_train+prev"]) - r["conf"]
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        pd.DataFrame(rows).to_csv(ROOT / f"results/probe_{task}.csv", index=False)
        np.savez(ROOT / f"traces/hidden/{task}_probe.npz", **scores)


if __name__ == "__main__":
    main()
