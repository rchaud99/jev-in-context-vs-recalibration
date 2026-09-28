"""Single-feature leakage screen (no API calls).

For every task and every structured feature: 5-fold cross-validated AUROC of a one-hot logistic regression on that
feature alone, over all 1,500 rows, plus the share of rows in the most label-pure level. A feature that alone
predicts the label very well is not proof of a leak, but every such feature is inspected by hand
(see AUTONOMOUS_LOG.md). Output: results/v2/leak_audit.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks import ROOT, TASKS  # noqa: E402

OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
rows = []
for t in TASKS:
    df = pd.read_csv(os.path.join(ROOT, "data", f"{t}.csv"))
    feats = [c for c in df.columns if c not in ("text", "label")]
    y = df.label.values
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    for f in feats:
        X = df[[f]].astype(str)
        m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=1.0, max_iter=2000))
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        g = df.groupby(X[f]).label.agg(["mean", "size"])
        pure = g[(g["size"] >= 20) & ((g["mean"] <= 0.03) | (g["mean"] >= 0.97))]
        rows.append(dict(task=t, feature=f, levels=X[f].nunique(), cv_auroc=roc_auc_score(y, p),
                         pure_levels=", ".join(f"{i}={r['mean']:.2f}(n={int(r['size'])})" for i, r in pure.iterrows())[:300]))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "leak_audit.csv"), index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 140)
print(R.to_string())
