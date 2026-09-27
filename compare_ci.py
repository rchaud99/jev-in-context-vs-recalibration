"""Jev vs Gemini 3.8 Flash with paired bootstrap CIs (no API calls: reads results/pred_*.csv and results/gem_*.csv).

For each task, k in Gemini's k grid and arm (zero_shot at k=0; in_context and platt at each k): Brier and AUROC for
both models on draws 0-1 and the same 300 test cases. Two-level paired bootstrap as in analyze2.py (resample test
cases and draws jointly; the same resamples for every model/arm, so differences are paired).

Differences: Gemini - Jev (same arm), and within each model in_context - platt.
Outputs: results/compare_ci.csv, results/compare_ci.md
"""
import os

import numpy as np
import pandas as pd

from analyze import auroc, brier
from tasks import ROOT, TEXT, load

OUT = os.path.join(ROOT, "results")
B = 2000
DRAWS = [0, 1]
TASK_ORDER = ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]
METRICS = {"brier": brier, "auroc": auroc}


def ks_for(name):
    return [5, 20, 100, 200 if name in TEXT else 400]


def matrix(P, arm, k):
    g = P[(P.arm == arm) & (P.k == k)]
    if arm == "zero_shot":
        v = g.sort_values("i").p.values
        return np.repeat(v[None], len(DRAWS), 0)
    M = g[g.draw.isin(DRAWS)].pivot(index="draw", columns="i", values="p")
    assert list(M.index) == DRAWS, (arm, k, list(M.index))
    return M.values


def tie_share(p):
    """Fraction of test-case pairs with identical predictions."""
    c = pd.Series(np.round(p, 6)).value_counts().values
    n = len(p)
    return (c * (c - 1) / 2).sum() / (n * (n - 1) / 2)


def main():
    rows, ties = [], []
    for task in TASK_ORDER:
        y = load(task)[1].label.values.astype(int)
        n = len(y)
        rng = np.random.default_rng(0)
        I = rng.integers(0, n, (B, n))
        J = rng.integers(0, len(DRAWS), (B, len(DRAWS)))
        yb = y[I]
        models = {"jev": pd.read_csv(f"{OUT}/pred_{task}.csv"), "gemini": pd.read_csv(f"{OUT}/gem_{task}.csv")}
        for k in [0] + ks_for(task):
            arms = ["zero_shot"] if k == 0 else ["in_context", "platt"]
            boot, point = {}, {}
            for mod, P in models.items():
                for arm in arms:
                    M = matrix(P, arm, k)
                    for d in range(len(DRAWS)):
                        ties.append(dict(task=task, model=mod, arm=arm, k=k, draw=DRAWS[d], n_unique=len(np.unique(M[d])),
                                         tie_share=tie_share(M[d])))
                    for m, f in METRICS.items():
                        per = f(M[:, I], yb[None])                      # draws x B
                        boot[(mod, arm, m)] = per[J.T, np.arange(B)].mean(0)
                        point[(mod, arm, m)] = f(M, y[None]).mean()
            for m in METRICS:
                for arm in arms:
                    for mod in models:
                        b = boot[(mod, arm, m)]
                        rows.append(dict(task=task, k=k, metric=m, kind="value", model=mod, arm=arm, a=f"{mod}:{arm}", b="",
                                         est=point[(mod, arm, m)], lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5)))
                    d = boot[("gemini", arm, m)] - boot[("jev", arm, m)]
                    rows.append(dict(task=task, k=k, metric=m, kind="gemini_minus_jev", model="", arm=arm,
                                     a=f"gemini:{arm}", b=f"jev:{arm}",
                                     est=point[("gemini", arm, m)] - point[("jev", arm, m)],
                                     lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
                if k > 0:
                    for mod in models:
                        d = boot[(mod, "in_context", m)] - boot[(mod, "platt", m)]
                        rows.append(dict(task=task, k=k, metric=m, kind="ic_minus_platt", model=mod, arm="",
                                         a=f"{mod}:in_context", b=f"{mod}:platt",
                                         est=point[(mod, "in_context", m)] - point[(mod, "platt", m)],
                                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
    R = pd.DataFrame(rows)
    R["sig"] = (R.lo > 0) | (R.hi < 0)
    R.loc[R.kind == "value", "sig"] = False
    R["n_draws"] = len(DRAWS)
    R.to_csv(f"{OUT}/compare_ci.csv", index=False)
    return R, pd.DataFrame(ties)


def fmt_diff(r, nd=3):
    s = f"{r.est:+.{nd}f} [{r.lo:+.{nd}f}, {r.hi:+.{nd}f}]"
    return f"**{s}**" if r.sig else s


def write_md(R, T):
    L = ["# Jev vs Gemini 3.8 Flash: Brier and AUROC with paired bootstrap 95% CIs", "",
         "Draws 0–1 for both models (paired), same 300 test cases; bootstrap resamples test cases and draws jointly "
         f"(B={B}; two-level scheme as analyze2.py, which uses B=1000). **Bold** = 95% CI excludes 0. Brier: lower is better, so ΔG−J < 0 means Gemini "
         "is better. AUROC: higher is better, so ΔG−J > 0 means Gemini is better. zero_shot is the k=0 column; "
         "in_context / platt use k labels. Max k = 400 (tabular) / 200 (text).", "",
         "Generated by `compare_ci.py` (full numbers: `results/compare_ci.csv`).", ""]
    V = R[R.kind == "value"].set_index(["task", "k", "metric", "model", "arm"]).est
    GJ = R[R.kind == "gemini_minus_jev"].set_index(["task", "k", "metric", "arm"])
    IP = R[R.kind == "ic_minus_platt"].set_index(["task", "k", "metric", "model"])
    for task in TASK_ORDER:
        ks = [0] + ks_for(task)
        y = load(task)[1].label.values
        L += [f"## {task} (base rate {y.mean():.2f})", ""]
        for m, lab in [("brier", "Brier J/G, ΔG−J [95% CI]"), ("auroc", "AUROC J/G, ΔG−J [95% CI]")]:
            L += [f"**{lab}**", "", "| arm | " + " | ".join(f"k={k}" for k in ks) + " |",
                  "|---|" + "---|" * len(ks)]
            for arm in ["zero_shot", "in_context", "platt"]:
                cells = []
                for k in ks:
                    if (arm == "zero_shot") != (k == 0):
                        cells.append("")
                        continue
                    r = GJ.loc[(task, k, m, arm)]
                    cells.append(f"{V[(task, k, m, 'jev', arm)]:.3f}/{V[(task, k, m, 'gemini', arm)]:.3f}, {fmt_diff(r)}")
                L.append(f"| {arm} | " + " | ".join(cells) + " |")
            L.append("")
        L += ["**in_context − platt, within model (Brier; AUROC)**", "", "| model | " + " | ".join(f"k={k}" for k in ks[1:]) + " |",
              "|---|" + "---|" * (len(ks) - 1)]
        for mod in ["jev", "gemini"]:
            cells = [f"{fmt_diff(IP.loc[(task, k, 'brier', mod)])}; {fmt_diff(IP.loc[(task, k, 'auroc', mod)], 2)}"
                     for k in ks[1:]]
            L.append(f"| {mod} | " + " | ".join(cells) + " |")
        L.append("")

    # summaries
    L += ["## Summary: where Gemini is significantly better / worse than Jev (tasks out of 8)", "",
          "| metric | arm | " + " | ".join(["k=0", "k=5", "k=20", "k=100", "k=max"]) + " |", "|---|---|" + "---|" * 5]
    G = R[R.kind == "gemini_minus_jev"].copy()
    G["kk"] = G.k.map(lambda k: k if k in (0, 5, 20, 100) else "max")
    for m in ["brier", "auroc"]:
        better = (G.hi < 0) if m == "brier" else (G.lo > 0)
        worse = (G.lo > 0) if m == "brier" else (G.hi < 0)
        G["b"], G["w"] = better, worse
        for arm in ["zero_shot", "in_context", "platt"]:
            cells = []
            for kk in [0, 5, 20, 100, "max"]:
                g = G[(G.metric == m) & (G.arm == arm) & (G.kk == kk)]
                if g.empty:
                    cells.append("")
                    continue
                bt = ",".join(g[g.b].task)
                wt = ",".join(g[g.w].task)
                cells.append(f"G better {g.b.sum()} ({bt or '–'}); worse {g.w.sum()} ({wt or '–'})")
            L.append(f"| {m} | {arm} | " + " | ".join(cells) + " |")
    L += ["", "## Summary: in_context vs platt on Brier, per model (tasks out of 8)", "",
          "Cells: in_context better (sig) / nominally better / platt better (sig).", "",
          "| model | k=5 | k=20 | k=100 | k=max |", "|---|---|---|---|---|"]
    Pm = R[(R.kind == "ic_minus_platt") & (R.metric == "brier")].copy()
    Pm["kk"] = Pm.k.map(lambda k: k if k in (5, 20, 100) else "max")
    for mod in ["jev", "gemini"]:
        cells = []
        for kk in [5, 20, 100, "max"]:
            g = Pm[(Pm.model == mod) & (Pm.kk == kk)]
            ic_sig = ",".join(g[g.hi < 0].task)
            pl_sig = ",".join(g[g.lo > 0].task)
            cells.append(f"{(g.hi < 0).sum()} ({ic_sig or '–'}) / {(g.est < 0).sum()} / {(g.lo > 0).sum()} ({pl_sig or '–'})")
        L.append(f"| {mod} | " + " | ".join(cells) + " |")

    # ties
    T2 = T[T.model == "gemini"].groupby(["task", "arm", "k"])[["n_unique", "tie_share"]].mean().reset_index()
    J2 = T[T.model == "jev"].groupby(["task", "arm", "k"])[["n_unique", "tie_share"]].mean().reset_index()
    L += ["", "## Ties in stated probabilities (mean over draws; 300 test cases)", "",
          "tie share = fraction of test-case pairs with identical predictions (AUROC counts ties as half). "
          "Gemini states integer percentages; Jev returns 2-decimal probabilities.", "",
          "| task | Jev zero-shot uniq / tie | Gemini zero-shot uniq / tie | Jev in_context k=max | Gemini in_context k=5 | Gemini in_context k=max |",
          "|---|---|---|---|---|---|"]
    for task in TASK_ORDER:
        km = max(ks_for(task))
        def c(D, arm, k):
            r = D[(D.task == task) & (D.arm == arm) & (D.k == k)].iloc[0]
            return f"{r.n_unique:.0f} / {r.tie_share:.3f}"
        L.append(f"| {task} | {c(J2, 'zero_shot', 0)} | {c(T2, 'zero_shot', 0)} | {c(J2, 'in_context', km)} | "
                 f"{c(T2, 'in_context', 5)} | {c(T2, 'in_context', km)} |")
    L.append("")
    open(f"{OUT}/compare_ci.md", "w").write("\n".join(L))


if __name__ == "__main__":
    R, T = main()
    write_md(R, T)
    T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "compare_ties.csv"), index=False)
    print("wrote results/compare_ci.csv, results/compare_ci.md, results/compare_ties.csv")
