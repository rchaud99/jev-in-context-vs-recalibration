"""Final replication table and headline figure (no API calls).

Reads results/v2/{e1,e1r}_posthoc_pooled.csv and {e1,e1r}_tests.csv (from baserate_analyze.py) and writes
  results/v2/replication_table.md   models x {E1, E1-R} pooled differences with 95% CIs, Holm, per-seed signs,
                                     per-task counts; plus the AUROC-band breakdown (exploratory)
  results/v2/fig_headline.png       the headline chart used in README
    cd v2 && python final_tables.py
"""
import os

import numpy as np
import pandas as pd

from common import OUT
from make_report import ORDER, pooled_chart

ELICIT = {"jev": "native", "qwen3.7-flash": "logprob", "deepseek-v4-flash": "logprob", "gpt-4o-mini": "logprob",
          "llama-3.3-70b": "logprob", "gpt-4o-mini-verbal": "stated", "deepseek-v4-flash-verbal": "stated",
          "gemini-3.8-flash": "stated", "claude-haiku-4.5": "stated"}
COST = {}  # $ per 1,000 decisions, from the ledger


def fmt(r):
    s = f"{r.est:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
    return f"**{s}**" if r.holm else s


def main():
    L = pd.read_csv(os.path.join(OUT, "spend_ledger.csv"))
    L = L[L.experiment.str.startswith("e1")]
    g = L.groupby("model").agg(usd=("usd", "sum"), n=("n_cases", "sum"))
    for m, r in g.iterrows():
        COST[m] = 1000 * r.usd / r.n
    P = {p: pd.read_csv(os.path.join(OUT, f"{p}_posthoc_pooled.csv")) for p in ["e1", "e1r"]}
    T = {p: pd.read_csv(os.path.join(OUT, f"{p}_tests.csv")) for p in ["e1", "e1r"]}
    out = ["# Replication table: base rate in the prompt vs in code", "",
           "Brier-score differences (positive = the first-named arm is worse), pooled per model as the mean over tasks "
           "of the per-task, seed-averaged difference; 95% bootstrap CI (test cases resampled within task, B=5000). "
           "**Bold** = significant after Holm across the 9 configurations. Seeds = sign of the pooled difference in "
           "each of the 3 seeds (each seed = a different 100-label estimate of the base rate). Tasks = number of "
           "tasks with a Holm-significant difference (within model) in the pooled direction / tasks. "
           "E1: 8 tasks (June–Aug 2026 records); E1-R: 5 fresh tasks (September 2026), preregistered.", ""]
    for fam, title in [("told-const", "Told the base rate − constant (predict the stated rate for every case)"),
                       ("H1 told-shift", "Told the base rate − same number applied in code (label-free logit shift)"),
                       ("P5 shift-const", "Code shift − constant"),
                       ("P6 platt-shift", "Platt on 100 labels − code shift"),
                       ("LR lr-shift", "Logistic regression on the same 100 labels − code shift")]:
        out += [f"## {title}", "", "| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |",
                "|---|---|---|---|---|---|---|---|---|"]
        for m in ORDER:
            cells = []
            for p in ["e1", "e1r"]:
                r = P[p][(P[p].model == m) & (P[p].family == fam)]
                if r.empty:
                    cells += ["–", "", ""]
                    continue
                r = r.iloc[0]
                seeds = "".join("+" if float(x) > 0 else "−" for x in str(r.per_seed).split()) if isinstance(r.per_seed, str) else ""
                t = T[p][(T[p].model == m) & (T[p].family == fam)]
                n_dir = int((t.holm_model & (np.sign(t.est) == np.sign(r.est))).sum())
                cells += [fmt(r), seeds, f"{n_dir}/{len(t)}"]
            out.append(f"| {m} | {ELICIT[m]} | {COST.get(m, np.nan):.3f} | " + " | ".join(cells) + " |")
        out.append("")
    # exploratory: AUROC bands
    A = pd.concat([T[p].assign(data=p) for p in T])
    out += ["## Exploratory: effect by the model's zero-shot AUROC on the task (all model × task cells, E1 + E1-R)", "",
            "| Comparison | AUROC band | cells | mean Δ | median Δ | CI > 0 | CI < 0 |", "|---|---|---|---|---|---|---|"]
    for fam in ["told-const", "H1 told-shift", "P5 shift-const"]:
        g = A[A.family == fam].copy()
        g["band"] = pd.cut(g.auc_zs, [0, 0.55, 0.65, 0.75, 1])
        for b, h in g.groupby("band", observed=True):
            out.append(f"| {fam} | {b} | {len(h)} | {h.est.mean():+.4f} | {h.est.median():+.4f} | "
                       f"{int((h.sig & (h.est > 0)).sum())} | {int((h.sig & (h.est < 0)).sum())} |")
    open(os.path.join(OUT, "replication_table.md"), "w").write("\n".join(out) + "\n")
    pooled_chart(os.path.join(OUT, "fig_headline.png"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
