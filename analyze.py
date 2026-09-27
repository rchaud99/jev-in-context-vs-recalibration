"""Score every arm: Brier, ECE (10 equal-width bins), AUROC, with a two-level bootstrap
(resample the 5 draws and the 300 test cases; the same resamples are used for every arm,
so differences between arms are paired).

Outputs: results/summary.csv, results/diffs.csv, results/curves.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tasks import ROOT

OUT = os.path.join(ROOT, "results")
B = 1000
ARMS = ["zero_shot", "in_context", "shuffled", "self_feedback", "platt", "isotonic", "both",
        "base_rate", "logreg", "tabpfn"]
COMPARE = [("in_context", "platt"), ("in_context", "shuffled"), ("self_feedback", "platt"),
           ("self_feedback", "in_context"), ("both", "platt"), ("isotonic", "platt"),
           ("platt", "base_rate"), ("in_context", "zero_shot"), ("platt", "logreg"), ("platt", "tabpfn")]


def brier(P, y):
    return ((P - y) ** 2).mean(-1)


def ece(P, y, bins=10):
    b = np.clip((P * bins).astype(int), 0, bins - 1)
    onehot = b[..., None] == np.arange(bins)
    cnt = onehot.sum(-2)
    gap = np.abs((onehot * P[..., None]).sum(-2) - (onehot * y[..., None]).sum(-2))
    return gap.sum(-1) / cnt.sum(-1)


def auroc(P, y):
    """Tie-aware AUROC along the last axis; y can be broadcast (same shape as P)."""
    y = np.broadcast_to(y, P.shape)
    r = rankdata(P, axis=-1)
    n1 = y.sum(-1)
    n0 = y.shape[-1] - n1
    return ((r * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


METRICS = {"brier": brier, "ece": ece, "auroc": auroc}


def load():
    preds = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "pred_*.csv"))
                       + glob.glob(os.path.join(OUT, "tabpfn_*.csv"))])
    labels = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "labels_*.csv"))])
    return preds, labels


def main():
    preds, labels = load()
    ks = sorted(k for k in preds.k.unique() if k > 0)
    summary, diffs = [], []
    for task, tp in preds.groupby("task"):
        y = labels[labels.task == task].sort_values("i").label.values
        n = len(y)
        rng = np.random.default_rng(0)
        I = rng.integers(0, n, (B, n))           # test-case resamples
        draws = sorted(tp[tp.k > 0].draw.unique())
        J = rng.integers(0, len(draws), (B, len(draws)))  # draw resamples
        yb = y[I]

        def arr(arm, k):
            g = tp[(tp.arm == arm) & (tp.k == (0 if arm == "zero_shot" else k))]
            if g.empty:
                return None
            m = g.pivot(index="draw", columns="i", values="p").values  # draws x n
            if arm == "zero_shot":
                m = np.repeat(m, len(draws), axis=0)
            return m

        for k in ks:
            boots = {}
            for arm in ARMS:
                M = arr(arm, k)
                if M is None:
                    continue
                for mname, f in METRICS.items():
                    point = f(M, y[None, :]).mean()
                    per = f(M[:, I], yb[None])                     # draws x B
                    boot = per[J.T, np.arange(B)].mean(0)          # B
                    boots[(arm, mname)] = boot
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    summary.append(dict(task=task, k=k, arm=arm, metric=mname, value=point, lo=lo, hi=hi))
            for a, b_ in COMPARE:
                for mname in METRICS:
                    if (a, mname) in boots and (b_, mname) in boots:
                        d = boots[(a, mname)] - boots[(b_, mname)]
                        pa = [s["value"] for s in summary if s["task"] == task and s["k"] == k
                              and s["metric"] == mname and s["arm"] in (a, b_)]
                        diffs.append(dict(task=task, k=k, a=a, b=b_, metric=mname,
                                          diff=np.mean(d), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    S, D = pd.DataFrame(summary), pd.DataFrame(diffs)
    S.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    D.to_csv(os.path.join(OUT, "diffs.csv"), index=False)
    plot(S, labels)
    return S, D


STYLE = {"zero_shot": ("#888888", "--"), "in_context": ("#1f77b4", "-"), "shuffled": ("#9ecae1", ":"),
         "self_feedback": ("#9467bd", "-"), "platt": ("#d62728", "-"), "isotonic": ("#ff9896", "-"),
         "both": ("#2ca02c", "-"), "base_rate": ("#000000", ":"), "logreg": ("#8c564b", "--"),
         "tabpfn": ("#e377c2", "--")}


def plot(S, labels):
    tasks = sorted(S.task.unique())
    fig, axes = plt.subplots(3, len(tasks), figsize=(4.2 * len(tasks), 10), squeeze=False)
    for c, task in enumerate(tasks):
        for r, m in enumerate(["brier", "ece", "auroc"]):
            ax = axes[r, c]
            for arm in ARMS:
                g = S[(S.task == task) & (S.arm == arm) & (S.metric == m)].sort_values("k")
                if g.empty:
                    continue
                col, ls = STYLE[arm]
                ax.plot(g.k, g.value, ls, color=col, marker="o", ms=3, label=arm)
                ax.fill_between(g.k, g.lo, g.hi, color=col, alpha=0.08)
            ax.set_xscale("log")
            ax.set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
            if r == 0:
                ax.set_title(f"{task} (base rate {labels[labels.task == task].label.mean():.2f})")
            if c == 0:
                ax.set_ylabel(m.upper() + (" (lower better)" if m != "auroc" else " (higher better)"))
            if r == 2:
                ax.set_xlabel("k labeled examples")
    axes[0, -1].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "curves.png"), dpi=130)


if __name__ == "__main__":
    S, D = main()
    pd.set_option("display.width", 200)
    for m in ["brier", "ece", "auroc"]:
        print(f"\n=== {m} (mean over draws)")
        print(S[S.metric == m].pivot_table(index=["task", "k"], columns="arm", values="value")
              .reindex(columns=[a for a in ARMS if a in S.arm.unique()]).round(3).to_string())
