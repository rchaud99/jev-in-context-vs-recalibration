"""Memorization check: zero-shot Jev on records whose outcomes happened after Jev's release
(jev-1.13-20260917; released 2026-09-15) vs the study's test set. If Jev had memorized outcomes,
it should do noticeably worse on records it cannot have seen. (No flights: BTS data lags ~2 months.)
"""
import numpy as np

import jev
import prepare_data as pdx
from tasks import TASKS, load

pdx.SHELTER_FROM = pdx.INSPECT_FROM = "2026-09-16"
pdx.NYC_DAYS = ["2026-09-16", "2026-09-17", "2026-09-18"]  # 7-day closure outcome known by the 26th


def auroc_ci(y, p, B=2000):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            bs.append(roc_auc_score(y[i], p[i]))
    return roc_auc_score(y, p), *np.percentile(bs, [2.5, 97.5])


for name, fn in [("inspections", pdx.inspections), ("shelter", pdx.shelter), ("nyc311", pdx.nyc311)]:
    t = TASKS[name]
    post = fn().dropna(subset=["label"])
    post = post.sample(n=min(300, len(post)), random_state=0)
    _, pre, _ = load(name)
    rows = []
    for label, df in [("study test set (pre-release)", pre), ("post-release", post)]:
        y = df.label.values.astype(int)
        p = np.array(jev.noul_batch(t["context"], list(df.text), t["question"]))
        a, lo, hi = auroc_ci(y, p)
        print(f"{name:12s} {label:30s} n={len(y):3d} base {y.mean():.2f}  mean p {p.mean():.2f}  "
              f"AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]")
print("spent:", jev.spent())
