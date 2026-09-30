"""Label-noise comparison: the non-in-context fixes and baselines trained on the SAME noisy labels
as study.py's `noisy` arm (20% of the k labels flipped, same RNG). Local only (Jev zero-shot scores
come from the cache). Run with .venv_tabpfn2/bin/python with JEV_OFFLINE=1.

Output: results/v1/noisy_<task>.csv  (arms: platt_noisy, base_rate_noisy, logreg_noisy, tabpfn_noisy)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

import jev
from study import DRAWS, KS, NOISE, OUT, draw_order, fit_logreg, fit_platt, smoothed_rate
from tasks import TASKS, load

warnings.filterwarnings("ignore")

for name in sys.argv[1:] or list(TASKS):
    t = TASKS[name]
    pool, test, feats = load(name)
    zs_test = np.array(jev.noul_batch(t["context"], list(test.text), t["question"]))
    zs_pool = np.array(jev.noul_batch(t["context"], list(pool.text), t["question"]))
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(pd.concat([pool, test])[feats])
    rows = []
    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            if k > len(pool):
                continue
            idx = order[:k]
            ex = pool.iloc[idx]
            noisy = ex.label.values.copy()
            flip = np.random.default_rng([k, d, 11]).choice(k, int(round(NOISE * k)), replace=False)
            noisy[flip] = 1 - noisy[flip]
            arms = {"platt_noisy": fit_platt(zs_pool[idx], noisy, zs_test),
                    "base_rate_noisy": np.full(len(test), smoothed_rate(noisy)),
                    "logreg_noisy": fit_logreg(ex[feats], noisy, test[feats])}
            if len(np.unique(noisy)) < 2:
                arms["tabpfn_noisy"] = np.full(len(test), smoothed_rate(noisy))
            else:
                clf = TabPFNClassifier(device="cpu", categorical_features_indices=list(range(len(feats))), random_state=d)
                arms["tabpfn_noisy"] = clf.fit(enc.transform(ex[feats]), noisy).predict_proba(enc.transform(test[feats]))[:, 1]
            for arm, p in arms.items():
                rows.extend((name, k, d, arm, i, float(v)) for i, v in enumerate(p))
        print(name, "draw", d, "done", flush=True)
    pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(
        os.path.join(OUT, f"noisy_{name}.csv"), index=False)
