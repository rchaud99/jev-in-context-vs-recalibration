"""Analysis for the expanded study (8 tasks, k up to 400, 10 draws, format/noise arms, LLM comparison).

Scores every (task, k, arm): Brier, ECE (10 equal-width bins), AUROC, with a two-level bootstrap
(resample draws and the 300 test cases; resamples are shared across arms so differences are paired).

Outputs (results/v1/):
  summary2.csv, diffs2.csv
  fig_auroc.png     does Jev learn from examples? (in-context vs shuffled vs zero-shot vs baselines vs LLM)
  fig_brier.png     cheapest fix: in-context vs Platt vs both vs base rate vs baselines
  fig_formats.png   prompt-format robustness
  fig_noise.png     label noise / shuffled labels
  fig_when.png      when does in-context beat recalibration? (vs Jev's zero-shot AUROC)
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze import METRICS
from tasks import ROOT

OUT = os.path.join(ROOT, "results", "v1")
B = 1000
ZERO = {"zero_shot", "llm_zero_shot"}
ARMS = ["zero_shot", "in_context", "fmt_compact", "fmt_json", "shuffled", "noisy", "self_feedback", "platt",
        "isotonic", "both", "base_rate", "logreg", "tabpfn", "embed_lr",
        "llm_zero_shot", "llm_in_context", "llm_platt", "llm_shuffled",
        "stack_lr", "stack_tabpfn", "platt_noisy", "logreg_noisy", "tabpfn_noisy", "base_rate_noisy"]
COMPARE = [("in_context", "platt"), ("in_context", "shuffled"), ("in_context", "zero_shot"),
           ("self_feedback", "in_context"), ("self_feedback", "platt"), ("both", "platt"), ("both", "in_context"),
           ("noisy", "in_context"), ("fmt_compact", "in_context"), ("fmt_json", "in_context"),
           ("isotonic", "platt"), ("platt", "base_rate"), ("in_context", "base_rate"), ("both", "base_rate"),
           ("in_context", "tabpfn"), ("in_context", "embed_lr"), ("in_context", "logreg"),
           ("both", "tabpfn"), ("both", "embed_lr"),
           ("in_context", "llm_in_context"), ("llm_in_context", "llm_shuffled"), ("llm_in_context", "llm_zero_shot"),
           ("llm_in_context", "llm_platt"),
           ("stack_lr", "in_context"), ("stack_tabpfn", "in_context"), ("stack_tabpfn", "both"), ("stack_lr", "both"),
           ("stack_tabpfn", "tabpfn"), ("stack_lr", "logreg"),
           ("platt_noisy", "platt"), ("logreg_noisy", "logreg"), ("tabpfn_noisy", "tabpfn"),
           ("base_rate_noisy", "base_rate")]
TASK_ORDER = ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]


def load():
    files = [f for pat in ("pred_*.csv", "tabpfn_*.csv", "llm_*.csv", "stack_*.csv", "noisy_*.csv") for f in glob.glob(os.path.join(OUT, pat))]
    preds = pd.concat([pd.read_csv(f) for f in files])
    labels = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(OUT, "labels_*.csv"))])
    return preds, labels


def main():
    """Every arm at a given (task, k) is scored on the same draws: those where Jev's in-context arm
    completed (the API budget ran out mid-run, so Jev arms have 4-7 of the 10 planned draws;
    baselines have all 10 but are restricted to the same draws so comparisons stay paired)."""
    preds, labels = load()
    summary, diffs = [], []
    for task, tp in preds.groupby("task"):
        y = labels[labels.task == task].sort_values("i").label.values
        n = len(y)
        rng = np.random.default_rng(0)
        I = rng.integers(0, n, (B, n))
        yb = y[I]
        ks = sorted(k for k in tp.k.unique() if k > 0)
        for k in ks:
            ref = tp[(tp.arm == "in_context") & (tp.k == k)].draw.unique()
            if len(ref) == 0:
                ref = tp[(tp.arm == "llm_in_context") & (tp.k == k)].draw.unique()
            draws = sorted(ref) if len(ref) else sorted(tp[tp.k == k].draw.unique())
            J = rng.integers(0, len(draws), (B, len(draws)))
            per_arm = {}
            for arm in ARMS:
                g = tp[(tp.arm == arm) & (tp.k == (0 if arm in ZERO else k))]
                if g.empty:
                    continue
                M = g.pivot(index="draw", columns="i", values="p")
                if arm in ZERO:
                    M = pd.DataFrame(np.repeat(M.values, len(draws), 0), index=draws)
                elif not set(draws) <= set(M.index):
                    continue  # this arm lacks some of the reference draws at this k
                else:
                    M = M.loc[draws]
                M = M.values
                for mname, f in METRICS.items():
                    per = f(M[:, I], yb[None])                 # draws x B
                    boot = per[J.T, np.arange(B)].mean(0)
                    per_arm[(arm, mname)] = boot
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    summary.append(dict(task=task, k=k, arm=arm, metric=mname, value=f(M, y[None, :]).mean(),
                                        lo=lo, hi=hi, n_draws=len(draws)))
            for a, b in COMPARE:
                for mname in METRICS:
                    if (a, mname) in per_arm and (b, mname) in per_arm:
                        d = per_arm[(a, mname)] - per_arm[(b, mname)]
                        diffs.append(dict(task=task, k=k, a=a, b=b, metric=mname, diff=d.mean(),
                                          lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), n_draws=len(draws)))
    S, D = pd.DataFrame(summary), pd.DataFrame(diffs)
    S.to_csv(os.path.join(OUT, "summary2.csv"), index=False)
    D.to_csv(os.path.join(OUT, "diffs2.csv"), index=False)
    return S, D, labels


STYLE = {
    "zero_shot": ("#7f7f7f", "--", "Jev zero-shot"), "in_context": ("#1f77b4", "-", "Jev + k examples in context"),
    "fmt_compact": ("#17becf", "-", "Jev + examples (compact format)"), "fmt_json": ("#aec7e8", "-", "Jev + examples (JSON format)"),
    "shuffled": ("#1f77b4", ":", "Jev + examples, labels shuffled"), "noisy": ("#6baed6", "-.", "Jev + examples, 20% labels flipped"),
    "self_feedback": ("#9467bd", "-", "Jev + examples + its own past estimates"),
    "platt": ("#d62728", "-", "Jev zero-shot + Platt fit on k labels"), "isotonic": ("#ff9896", "-", "Jev zero-shot + isotonic"),
    "both": ("#2ca02c", "-", "Jev + examples + Platt"), "base_rate": ("#000000", ":", "Base rate of the k labels"),
    "logreg": ("#8c564b", "--", "Logistic regression (k labels)"), "tabpfn": ("#e377c2", "--", "TabPFN (k labels)"),
    "embed_lr": ("#bcbd22", "--", "Embeddings + LR (k labels)"),
    "llm_zero_shot": ("#ff7f0e", "--", "DeepSeek V4 Flash zero-shot"), "llm_in_context": ("#ff7f0e", "-", "DeepSeek + k examples"),
    "llm_platt": ("#ffbb78", "-", "DeepSeek zero-shot + Platt"), "llm_shuffled": ("#ff7f0e", ":", "DeepSeek + shuffled labels"),
    "stack_lr": ("#d62728", "-.", "LR on features + Jev zero-shot score"),
    "stack_tabpfn": ("#9467bd", "-.", "TabPFN on features + Jev zero-shot score"),
}


def grid(S, labels, arms, metric, fname, title):
    tasks = [t for t in TASK_ORDER if t in S.task.unique()]
    cols = 4
    rows = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 3.6 * rows), squeeze=False)
    for ax, task in zip(axes.flat, tasks):
        for arm in arms:
            g = S[(S.task == task) & (S.arm == arm) & (S.metric == metric)].sort_values("k")
            if g.empty:
                continue
            col, ls, lab = STYLE[arm]
            ax.plot(g.k, g.value, ls, color=col, marker="o", ms=3, lw=1.8, label=lab)
            ax.fill_between(g.k, g.lo, g.hi, color=col, alpha=0.10)
        ax.set_xscale("log")
        ks = sorted(S[S.task == task].k.unique())
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_title(f"{task}  (base rate {labels[labels.task == task].label.mean():.2f})", fontsize=10)
        ax.set_xlabel("k labeled examples", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(tasks):]:
        ax.axis("off")
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="lower center", ncol=min(4, len(labs)), fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0.07 + 0.02 * (len(labs) > 4), 1, 0.96))
    fig.savefig(os.path.join(OUT, fname), dpi=140)
    plt.close(fig)


def noise_bars(S, k=100):
    """Damage from flipping 20% of the k labels, per method (same flipped labels for every method)."""
    pairs = [("in_context", "noisy", "Jev + examples in context"), ("platt", "platt_noisy", "Jev zero-shot + Platt"),
             ("logreg", "logreg_noisy", "Logistic regression"), ("tabpfn", "tabpfn_noisy", "TabPFN")]
    tasks = [t for t in TASK_ORDER if t in S.task.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, m in zip(axes, ["auroc", "brier"]):
        w = 0.2
        for j, (clean, noisy, lab) in enumerate(pairs):
            vals = []
            for t in tasks:
                a = S[(S.task == t) & (S.k == k) & (S.metric == m) & (S.arm == clean)].value
                b = S[(S.task == t) & (S.k == k) & (S.metric == m) & (S.arm == noisy)].value
                vals.append(b.iloc[0] - a.iloc[0] if len(a) and len(b) else np.nan)
            ax.bar(np.arange(len(tasks)) + (j - 1.5) * w, vals, w, label=lab)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(tasks)), tasks, rotation=30, fontsize=8)
        ax.set_title(f"Change in {m.upper()} when 20% of the k={k} labels are flipped "
                     + ("(below 0 = worse)" if m == "auroc" else "(above 0 = worse)"), fontsize=10)
        ax.grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_noise_damage.png"), dpi=140)
    plt.close(fig)


def when_plot(S, D, k=100):
    """x: what the labels know that Jev doesn't = best label-trained model's AUROC at k=400 (TabPFN / LR /
    embeddings) minus Jev's zero-shot AUROC. y: Brier(in-context) - Brier(Platt) at k (below 0: examples win)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for task in TASK_ORDER:
        get = lambda arm, kk, m: S[(S.task == task) & (S.arm == arm) & (S.k == kk) & (S.metric == m)].value
        zs = get("zero_shot", k, "auroc")
        best = max([v.iloc[0] for v in (get(a, 400, "auroc") for a in ("tabpfn", "logreg", "embed_lr")) if len(v)])
        d = D[(D.task == task) & (D.a == "in_context") & (D.b == "platt") & (D.metric == "brier") & (D.k == k)]
        if zs.empty or d.empty:
            continue
        x, r = best - zs.iloc[0], d.iloc[0]
        sig = r.hi < 0 or r.lo > 0
        col = "#1f77b4" if r["diff"] < 0 else "#d62728"
        ax.errorbar(x, r["diff"], yerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", color=col,
                    mfc=col if sig else "white", capsize=3)
        ax.annotate(task, (x, r["diff"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("What the labels know that Jev doesn't\n(best label-trained model AUROC at k=400 − Jev zero-shot AUROC)")
    ax.set_ylabel(f"Brier: examples in context − Platt, k={k}\n(below 0: examples win; filled = significant)")
    ax.set_title("Examples beat recalibration when the labels carry local signal", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_when.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    S, D, labels = main()
    grid(S, labels, ["zero_shot", "in_context", "shuffled", "platt", "tabpfn", "embed_lr", "llm_in_context", "llm_shuffled"],
         "auroc", "fig_auroc.png", "Ranking (AUROC, higher is better): does the model learn from k labeled examples?")
    grid(S, labels, ["zero_shot", "in_context", "platt", "both", "base_rate", "tabpfn", "embed_lr", "llm_in_context"],
         "brier", "fig_brier.png", "Probability quality (Brier, lower is better): which fix is best at each k?")
    grid(S, labels, ["in_context", "fmt_compact", "fmt_json"], "auroc", "fig_formats.png",
         "Prompt-format robustness (AUROC)")
    grid(S, labels, ["in_context", "noisy", "shuffled", "self_feedback"], "brier", "fig_noise.png",
         "Label noise, shuffled labels, and self-feedback (Brier)")
    grid(S, labels, ["in_context", "both", "stack_lr", "stack_tabpfn", "tabpfn", "zero_shot"], "auroc", "fig_stack_auroc.png",
         "Stack, don't stuff (AUROC): Jev's zero-shot score as a feature vs examples in Jev's context")
    grid(S, labels, ["in_context", "both", "stack_lr", "stack_tabpfn", "tabpfn", "zero_shot"], "brier", "fig_stack_brier.png",
         "Stack, don't stuff (Brier)")
    noise_bars(S)
    when_plot(S, D)
    pd.set_option("display.width", 250)
    for m in ["brier", "ece", "auroc"]:
        print(f"\n=== {m}")
        print(S[S.metric == m].pivot_table(index=["task", "k"], columns="arm", values="value")
              .reindex(columns=[a for a in ARMS if a in S.arm.unique()]).round(3).to_string())
