"""Analyses on cached predictions (no API calls).

1. Context ensembling: does averaging two k=m contexts (different examples) match one k=2m context?
2. Mechanism: how well does "prior + per-group label rate" explain Jev's in-context predictions?
   For each task we pick the single feature whose group label rates best predict the outcome on the
   pool (data-driven, not hand-picked). For each (k, draw, arm in {in_context, shuffled, noisy}) we
   compute each test case's smoothed group rate *from the labels shown in that context* and regress
   logit(Jev in-context p) on logit(Jev zero-shot p) and logit(group rate).
Outputs: results/ensemble.csv, results/mechanism.csv, results/fig_mechanism.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score

from study import NOISE, draw_order, logit
from tasks import ROOT, TASKS, load

OUT = os.path.join(ROOT, "results")
preds = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "pred_*.csv"))])
labels = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "labels_*.csv"))])


def mat(task, arm, k):
    g = preds[(preds.task == task) & (preds.arm == arm) & (preds.k == k)]
    return g.pivot(index="draw", columns="i", values="p") if len(g) else None


def brier(p, y):
    return float(np.mean((p - y) ** 2))


# ---------- 1. context ensembling
rows = []
for task in TASKS:
    y = labels[labels.task == task].sort_values("i").label.values
    for m in (50, 100, 200):
        small, big = mat(task, "in_context", m), mat(task, "in_context", 2 * m)
        if small is None or big is None:
            continue
        ds = sorted(set(small.index) & set(big.index))
        for a, b in zip(ds[0::2], ds[1::2]):
            ens = (small.loc[a].values + small.loc[b].values) / 2
            for label, p in [(f"one context, k={2 * m}", big.loc[a].values), (f"two contexts of k={m}, averaged", ens),
                             (f"single context, k={m}", small.loc[a].values)]:
                rows.append(dict(task=task, m=m, variant=label.split(",")[0], auroc=roc_auc_score(y, p), brier=brier(p, y)))
E = pd.DataFrame(rows)
E.to_csv(os.path.join(OUT, "ensemble.csv"), index=False)
print("=== context ensembling (mean over draw pairs)")
print(E.groupby(["task", "m", "variant"])[["auroc", "brier"]].mean().round(3).to_string())


# ---------- 2. mechanism
def group_rate(keys_ex, y_ex, keys_test, prior, m=2.0):
    df = pd.DataFrame({"g": keys_ex, "y": y_ex}).groupby("g").y.agg(["sum", "count"])
    s = df["sum"].reindex(keys_test).fillna(0).values
    n = df["count"].reindex(keys_test).fillna(0).values
    return (s + m * prior) / (n + m)


mrows = []
for task in TASKS:
    pool, test, feats = load(task)
    # key feature: best single-feature group-rate AUROC on the pool (5-fold, to avoid overfitting)
    best, key = -1, None
    for f in feats:
        folds = np.arange(len(pool)) % 5
        pr = np.zeros(len(pool))
        for fo in range(5):
            tr, te = folds != fo, folds == fo
            pr[te] = group_rate(pool[f][tr].values, pool.label[tr].values, pool[f][te].values, pool.label[tr].mean())
        a = roc_auc_score(pool.label, pr)
        if a > best:
            best, key = a, f
    zs = logit(mat(task, "zero_shot", 0).values[0])
    for k in (20, 50, 100, 200, 400):
        for arm in ("in_context", "shuffled", "noisy"):
            M = mat(task, arm, k)
            if M is None:
                continue
            X, Y = [], []
            for d in M.index:
                ex = pool.iloc[draw_order(task, d, len(pool))[:k]]
                yl = ex.label.values.copy()
                if arm == "shuffled":
                    yl = np.random.default_rng([k, d, 7]).permutation(yl)
                elif arm == "noisy":
                    flip = np.random.default_rng([k, d, 11]).choice(k, int(round(NOISE * k)), replace=False)
                    yl[flip] = 1 - yl[flip]
                gr = group_rate(ex[key].values, yl, test[key].values, (yl.sum() + 0.5) / (k + 1))
                X.append(np.c_[zs, logit(gr)])
                Y.append(logit(M.loc[d].values))
            X, Y = np.vstack(X), np.concatenate(Y)
            full = LinearRegression().fit(X, Y)
            r2 = lambda cols: LinearRegression().fit(X[:, cols], Y).score(X[:, cols], Y)
            mrows.append(dict(task=task, key=key, key_auroc=best, k=k, arm=arm,
                              r2_both=full.score(X, Y), r2_prior_only=r2([0]), r2_group_only=r2([1]),
                              w_prior=full.coef_[0], w_group=full.coef_[1]))
Mdf = pd.DataFrame(mrows)
Mdf.to_csv(os.path.join(OUT, "mechanism.csv"), index=False)
pd.set_option("display.width", 200)
print("\n=== mechanism: logit(Jev in-context) ~ w_prior*logit(zero-shot) + w_group*logit(group rate in context)")
print(Mdf.round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for task, g in Mdf[Mdf.arm == "in_context"].groupby("task"):
    axes[0].plot(g.k, g.w_prior, "o-", label=task)
    axes[1].plot(g.k, g.w_group, "o-", label=task)
for ax, ttl in zip(axes, ["Weight on Jev's own prior (zero-shot logit)", "Weight on the in-context group label rate"]):
    ax.set_xscale("log")
    ax.set_xlabel("k examples in context")
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.25)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_mechanism.png"), dpi=140)
