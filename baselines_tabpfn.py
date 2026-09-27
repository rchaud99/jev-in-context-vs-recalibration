"""No-Jev baselines on the same k labels as study.py. Run with .venv_tabpfn2/bin/python
(TabPFN 2.0.9: newer releases gate the weights behind a license login).

  tabpfn    TabPFN on the ordinal-encoded categorical features
  embed_lr  logistic regression on bge-small sentence embeddings of the case text
            (the standard few-label baseline for text; uses exactly what Jev sees)

Output: results/tabpfn_<task>.csv in the same format as study.py predictions.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

from study import DRAWS, KS, OUT, draw_order, smoothed_rate
from tasks import TASKS, load

warnings.filterwarnings("ignore")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")

for name in sys.argv[1:] or list(TASKS):
    pool, test, feats = load(name)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(pd.concat([pool, test])[feats])
    E_pool = embedder.encode(list(pool.text), normalize_embeddings=True, batch_size=64)
    E_test = embedder.encode(list(test.text), normalize_embeddings=True, batch_size=64)
    rows = []
    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            if k > len(pool):
                continue
            idx = order[:k]
            ex = pool.iloc[idx]
            y = ex.label.values
            if len(np.unique(y)) < 2:
                p_tab = p_emb = np.full(len(test), smoothed_rate(y))
            else:
                clf = TabPFNClassifier(device="cpu", categorical_features_indices=list(range(len(feats))),
                                       random_state=d)
                p_tab = clf.fit(enc.transform(ex[feats]), y).predict_proba(enc.transform(test[feats]))[:, 1]
                p_emb = LogisticRegression(C=1.0, max_iter=2000).fit(E_pool[idx], y).predict_proba(E_test)[:, 1]
            rows.extend((name, k, d, "tabpfn", i, float(v)) for i, v in enumerate(p_tab))
            rows.extend((name, k, d, "embed_lr", i, float(v)) for i, v in enumerate(p_emb))
        print(name, "draw", d, "done", flush=True)
    pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(
        os.path.join(OUT, f"tabpfn_{name}.csv"), index=False)
