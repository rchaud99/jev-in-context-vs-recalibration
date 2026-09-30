"""TabPFN on the synthetic tasks, trained on the same k labels (same nested draws) as synthetic_study.py.
Run with .venv_tabpfn2/bin/python (TabPFN 2.0.9, CPU). Output: results/v1/synth_part_tabpfn_<task>.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

import synthetic_tasks as S

warnings.filterwarnings("ignore")
KS, DRAWS = [5, 20, 100, 400], [0, 1]

for name in sys.argv[1:] or S.ORDER:
    pool, test, feats = S.load(name)
    enc = OrdinalEncoder().fit(pd.concat([pool, test])[feats])
    Xp, Xt = enc.transform(pool[feats]), enc.transform(test[feats])
    rows = []
    for d in DRAWS:
        order = S.draw_order(d, len(pool))
        for k in KS:
            idx = order[:k]
            y = pool.label.values[idx]
            if len(np.unique(y)) < 2:
                p = np.full(len(test), y.mean())
            else:
                clf = TabPFNClassifier(device="cpu", categorical_features_indices=list(range(len(feats))), random_state=d)
                p = clf.fit(Xp[idx], y).predict_proba(Xt)[:, 1]
            rows.extend((name, k, d, "tabpfn", "trained", i, float(v)) for i, v in enumerate(p))
        print(name, "draw", d, "done", flush=True)
    pd.DataFrame(rows, columns=["task", "k", "draw", "model", "arm", "i", "p"]).to_csv(
        os.path.join(S.ROOT, "results", "v1", f"synth_part_tabpfn_{name}.csv"), index=False)
