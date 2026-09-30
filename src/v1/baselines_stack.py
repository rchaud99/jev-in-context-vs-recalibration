""""Stack, don't stuff": use Jev's cached zero-shot probability as ONE extra feature of a small model
trained on the k labels, instead of putting the k examples in Jev's context.

  stack_lr      one-hot features + logit(Jev zero-shot) -> logistic regression
  stack_tabpfn  ordinal features + logit(Jev zero-shot) -> TabPFN

Costs one short zero-shot Jev call per case (~70 input tokens/prediction when batched) vs ~1,000
tokens/prediction for in-context Jev at k=200. Local only; run with .venv_tabpfn2/bin/python, JEV_OFFLINE=1.
Output: results/v1/stack_<task>.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from tabpfn import TabPFNClassifier

import jev
from study import DRAWS, KS, OUT, draw_order, logit, smoothed_rate
from tasks import TASKS, load

warnings.filterwarnings("ignore")

for name in sys.argv[1:] or list(TASKS):
    t = TASKS[name]
    pool, test, feats = load(name)
    z_test = logit(jev.noul_batch(t["context"], list(test.text), t["question"]))
    z_pool = logit(jev.noul_batch(t["context"], list(pool.text), t["question"]))
    both = pd.concat([pool, test])[feats]
    oh = OneHotEncoder(handle_unknown="ignore").fit(both)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(both)
    rows = []
    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            if k > len(pool):
                continue
            idx = order[:k]
            ex, y = pool.iloc[idx], pool.label.values[idx]
            if len(np.unique(y)) < 2:
                p_lr = p_tab = np.full(len(test), smoothed_rate(y))
            else:
                Xtr = hstack([oh.transform(ex[feats]), csr_matrix(z_pool[idx][:, None])]).tocsr()
                Xte = hstack([oh.transform(test[feats]), csr_matrix(z_test[:, None])]).tocsr()
                p_lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr, y).predict_proba(Xte)[:, 1]
                clf = TabPFNClassifier(device="cpu", categorical_features_indices=list(range(len(feats))), random_state=d)
                p_tab = clf.fit(np.c_[enc.transform(ex[feats]), z_pool[idx]], y).predict_proba(
                    np.c_[enc.transform(test[feats]), z_test])[:, 1]
            rows.extend((name, k, d, "stack_lr", i, float(v)) for i, v in enumerate(p_lr))
            rows.extend((name, k, d, "stack_tabpfn", i, float(v)) for i, v in enumerate(p_tab))
        print(name, "draw", d, "done", flush=True)
    pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(
        os.path.join(OUT, f"stack_{name}.csv"), index=False)
