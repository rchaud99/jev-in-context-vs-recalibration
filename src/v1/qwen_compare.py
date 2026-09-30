"""Label sensitivity: Jev vs a local general-purpose LLM (Qwen2.5-7B-Instruct) given the SAME examples
with correct, 20%-flipped, and fully shuffled labels (draws 0-2, k in {20, 100}).
Label sensitivity = AUROC(correct labels) - AUROC(shuffled labels): how much of the in-context gain
comes from the labels themselves rather than from seeing example cases.
Output: results/v1/qwen_vs_jev.csv, results/v1/fig_qwen_vs_jev.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from tasks import ROOT

OUT = os.path.join(ROOT, "results", "v1")
q = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "qwen_*.csv"))])
j = pd.concat([pd.read_csv(os.path.join(OUT, f"pred_{t}.csv")) for t in q.task.unique()])
labels = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "labels_*.csv"))])
B = 1000

rows = []
sens_boot = {}  # (task, k, model) -> bootstrap label sensitivity; same resamples for both models, so paired
for task in q.task.unique():
    y = labels[labels.task == task].sort_values("i").label.values
    rng = np.random.default_rng(0)
    I = rng.integers(0, len(y), (B, len(y)))

    def auc_of(df, arm, k, draws):
        g = df[(df.arm == arm) & (df.k == k)]
        M = g.pivot(index="draw", columns="i", values="p")
        M = M.loc[[d for d in draws if d in M.index]].values
        point = np.mean([roc_auc_score(y, m) for m in M])
        boot = np.array([np.mean([roc_auc_score(y[ii], m[ii]) for m in M]) for ii in I[:200]])
        return point, boot

    for k in (20, 100):
        draws = sorted(set(q[(q.task == task) & (q.k == k)].draw) & set(j[(j.task == task) & (j.k == k) & (j.arm == "shuffled")].draw))
        if not draws:
            continue
        for model, df, zs, pre in [("Jev", j, "zero_shot", ""), ("Qwen2.5-7B", q, "qwen_zero_shot", "qwen_")]:
            df = df[df.task == task]
            zg = df[(df.arm == zs)].sort_values("i").p.values
            res = {"zero_shot": roc_auc_score(y, zg)}
            boots = {}
            for arm in ("in_context", "noisy", "shuffled"):
                res[arm], boots[arm] = auc_of(df, pre + arm, k, draws)
            sens = boots["in_context"] - boots["shuffled"]
            sens_boot[(task, k, model)] = sens
            rows.append(dict(task=task, k=k, model=model, n_draws=len(draws), **{f"auroc_{a}": v for a, v in res.items()},
                             label_sensitivity=res["in_context"] - res["shuffled"],
                             sens_lo=np.percentile(sens, 2.5), sens_hi=np.percentile(sens, 97.5),
                             gain_from_examples=res["in_context"] - res["zero_shot"]))
R = pd.DataFrame(rows)
# paired bootstrap CI for Jev - Qwen label sensitivity (same test-case resamples for both models)
for (task, k, model), b in sens_boot.items():
    other = sens_boot.get((task, k, "Qwen2.5-7B" if model == "Jev" else "Jev"))
    if model != "Jev" or other is None:
        continue
    m = (R.task == task) & (R.k == k)
    d = b - other
    R.loc[m, "sens_jev_minus_qwen"] = R.loc[m & (R.model == "Jev"), "label_sensitivity"].iloc[0] - R.loc[m & (R.model != "Jev"), "label_sensitivity"].iloc[0]
    R.loc[m, "sens_diff_lo"], R.loc[m, "sens_diff_hi"] = np.percentile(d, 2.5), np.percentile(d, 97.5)
R.to_csv(os.path.join(OUT, "qwen_vs_jev.csv"), index=False)
pd.set_option("display.width", 220)
print(R.round(3).to_string(index=False))

tasks = sorted(R.task.unique())
fig, axes = plt.subplots(1, len(tasks), figsize=(max(7.0, 3.5 * len(tasks)), 4.4), squeeze=False)
for ax, task in zip(axes[0], tasks):
    for model, col in [("Jev", "#1f77b4"), ("Qwen2.5-7B", "#ff7f0e")]:
        g = R[(R.task == task) & (R.model == model) & (R.k == 100)]
        if g.empty:
            continue
        r = g.iloc[0]
        xs = ["zero-shot", "correct labels", "20% flipped", "shuffled"]
        ax.plot(xs, [r.auroc_zero_shot, r.auroc_in_context, r.auroc_noisy, r.auroc_shuffled], "o-", color=col, label=model)
    ax.set_title(f"{task} (k=100)", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(alpha=0.25)
axes[0][0].set_ylabel("AUROC")
axes[0][0].legend(fontsize=8)
fig.suptitle("Same examples, different labels:\nhow much do Jev and Qwen2.5-7B rely on the labels?", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(os.path.join(OUT, "fig_qwen_vs_jev.png"), dpi=140)
