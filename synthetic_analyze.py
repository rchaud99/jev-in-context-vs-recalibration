"""Score the synthetic study (no API calls). Merges results/synth_part_*_<task>.csv and results/synth_opus_<task>.csv
(Claude Opus 5.5, synthetic_opus.py; k <= 100) into results/synth_<task>.csv and writes
results/synth_summary.md and results/synth_ci.csv.

Metrics on the 300 test cases: mean prediction, MSE to the true per-case probabilities (primary), Brier, AUROC.
Paired two-level bootstrap (resample test cases and draws 0-1 jointly; the same resamples for every model) for
Jev - Gemini, Opus - {Gemini, Jev, logistic regression, base-rate-only} and each LLM - base-rate-only, at each k
(k=0: zero-shot, repeated over draws).
"""
import glob
import json
import os
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import synthetic_tasks as S

OUT = os.path.join(S.ROOT, "results")
B, DRAWS, KS = 2000, [0, 1], [0, 5, 20, 100, 400]
MODELS = [("jev", "Jev"), ("gemini", "Gemini 3.8 Flash"), ("opus", "Claude Opus 5.5"), ("logreg", "Logistic regression"),
          ("tabpfn", "TabPFN"), ("base_rate", "Base-rate only (mean of k labels)")]
OPUS_KS = [0, 5, 20, 100]
TITLES = {"kellan": "Task 1: Kellan fever (additive, base rate 0.3)",
          "interaction": "Task 2: interaction (morvax works only for a deep red rash)",
          "noise": "Task 3: pure noise (p = 0.3 for every case)",
          "kellan_br10": "Task 4a: Kellan fever, base rate 0.1", "kellan_br90": "Task 4b: Kellan fever, base rate 0.9",
          "neutral": "Task 5: neutral naming (task 1 with 'attribute A = level 3' codes)"}


def auroc(P, y):
    y = np.broadcast_to(y, P.shape)
    r = rankdata(P, axis=-1)
    n1 = y.sum(-1)
    n0 = y.shape[-1] - n1
    return ((r * y).sum(-1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def mse(P, t):
    return ((P - t) ** 2).mean(-1)


def matrix(P, model, k):
    g = P[(P.model == model) & (P.k == k)]
    if g.empty:
        return None
    if k == 0:
        return np.repeat(g.sort_values("i").p.values[None], len(DRAWS), 0)
    M = g[g.draw.isin(DRAWS)].pivot(index="draw", columns="i", values="p")
    return M.values if list(M.index) == DRAWS else None


def fmt_diff(boot, ci, task, a, b, m, k):
    if (a, k, m) not in boot or (b, k, m) not in boot:
        return "–"
    pe = boot[(a, k, m)][0] - boot[(b, k, m)][0]
    d = boot[(a, k, m)][1] - boot[(b, k, m)][1]
    lo, hi = np.percentile(d, [2.5, 97.5])
    ci.append(dict(task=task, k=k, model=f"{a}-{b}", kind=f"diff_{m}", est=pe, lo=lo, hi=hi))
    star = "*" if lo > 0 or hi < 0 else ""
    fmt = "{:+.3f}" if m == "auroc" else "{:+.4f}"
    return f"{fmt.format(pe)} [{fmt.format(lo)}, {fmt.format(hi)}]{star}"


def opus_spend():
    f = os.path.join(S.DATA, "opus_cache.sqlite")
    if not os.path.exists(f):
        return 0.0
    db = sqlite3.connect(f)
    return sum(json.loads(r[0]).get("cost", 0) or 0 for r in db.execute("SELECT response FROM calls"))


def main():
    md = ["# Synthetic tasks: Jev, Gemini 3.8 Flash and Claude Opus 5.5 on fictional content with known true "
          "probabilities", "",
          "300 fixed test cases per task; examples from nested draws 0-1 (same example rows for every task); "
          "k=0 is zero-shot. Cells: **mean prediction (AUROC) | MSE to the true probabilities**, averaged over the "
          "2 draws. Generator: `synthetic_tasks.py`; runs: `synthetic_study.py`, `synthetic_opus.py`, "
          "`synthetic_tabpfn.py`; this file: `synthetic_analyze.py`. CIs: paired bootstrap over test cases and draws "
          "(B=2000). Opus (same prompts, draws, examples and test cases as Gemini) was run at k = 0-100 only "
          "(k=400 too expensive).", ""]
    n_head = len(md)
    ci, Ps, opus100 = [], {}, []
    for task in S.ORDER:
        parts = sorted(glob.glob(f"{OUT}/synth_part_*_{task}.csv")) + glob.glob(f"{OUT}/synth_opus_{task}.csv")
        P = pd.concat([pd.read_csv(f) for f in parts])
        P = P[~((P.task != task))]
        P.to_csv(f"{OUT}/synth_{task}.csv", index=False)
        Ps[task] = P
        test = S.load(task)[1]
        y, t = test.label.values.astype(int), test.p_true.values
        bayes_auc = auroc(t[None], y)[0] if np.ptp(t) > 0 else float("nan")
        md += [f"## {TITLES[task]}", "",
               f"Test: true rate (mean p_true) {t.mean():.3f}, realised label rate {y.mean():.3f}; "
               f"Bayes-optimal AUROC {bayes_auc:.3f}, Bayes Brier {((t - y) ** 2).mean():.3f}. "
               f"MSE of predicting the true base rate for every case: {mse(np.full_like(t, t.mean()), t):.4f}.", "",
               "| model | " + " | ".join(f"k={k}" for k in KS) + " |", "|---" * (len(KS) + 1) + "|"]
        rng = np.random.default_rng(0)
        I = rng.integers(0, len(y), (B, len(y)))
        J = rng.integers(0, len(DRAWS), (B, len(DRAWS)))
        boot, briers = {}, {}
        for mod, label in MODELS:
            cells = []
            for k in KS:
                M = matrix(P, mod, k)
                if M is None:
                    cells.append("–")
                    continue
                a, m = auroc(M, y[None]).mean(), mse(M, t[None]).mean()
                cells.append(f"{M.mean():.2f} ({a:.2f}) \\| {m:.4f}")
                for name, f, ref in [("mse", mse, t), ("auroc", auroc, y), ("brier", mse, y)]:
                    per = f(M[:, I], ref[I][None])                      # draws x B
                    boot[(mod, k, name)] = (f(M, ref[None]).mean(), per[J.T, np.arange(B)].mean(0))
                briers[(mod, k)] = ((M - y) ** 2).mean()
                ci.append(dict(task=task, k=k, model=mod, mean=M.mean(), auroc=a, mse_truth=m,
                               brier=briers[(mod, k)], kind="value"))
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append(f"| Bayes-optimal (true p) | " + " | ".join([f"{t.mean():.2f} ({bayes_auc:.2f}) \\| 0"] * len(KS)) + " |")
        md += ["", f"Brier score (Bayes-optimal {((t - y) ** 2).mean():.3f}; lower is better):", "",
               "| model | " + " | ".join(f"k={k}" for k in KS) + " |", "|---" * (len(KS) + 1) + "|"]
        for mod, label in MODELS:
            md.append(f"| {label} | " + " | ".join(f"{briers[(mod, k)]:.4f}" if (mod, k) in briers else "–"
                                                    for k in KS) + " |")
        diff = lambda a, b, m, k: fmt_diff(boot, ci, task, a, b, m, k)  # noqa: E731
        md += ["", "Paired differences, point [95% CI]; negative MSE difference = first model closer to the truth:", "",
               "| k | Jev − Gemini, MSE | Jev − Gemini, AUROC | Jev − base-rate-only, MSE | Gemini − base-rate-only, MSE |",
               "|---|---|---|---|---|"]
        for k in KS:
            row = [diff("jev", "gemini", "mse", k), diff("jev", "gemini", "auroc", k)]
            row += [diff("jev", "base_rate", "mse", k), diff("gemini", "base_rate", "mse", k)] if k > 0 else ["–", "–"]
            md.append(f"| {k} | " + " | ".join(row) + " |")
        if any(m == "opus" for m, _, _ in boot):
            md += ["", "Claude Opus 5.5, paired differences (same resamples):", "",
                   "| k | Opus − Gemini, MSE | Opus − Jev, MSE | Opus − logreg, MSE | Opus − base-rate-only, MSE | "
                   "Opus − Gemini, AUROC | Opus − Jev, AUROC | Opus − Gemini, Brier | Opus − Jev, Brier |",
                   "|---|---|---|---|---|---|---|---|---|"]
            for k in OPUS_KS:
                row = [diff("opus", "gemini", "mse", k), diff("opus", "jev", "mse", k),
                       diff("opus", "logreg", "mse", k), diff("opus", "base_rate", "mse", k),
                       diff("opus", "gemini", "auroc", k), diff("opus", "jev", "auroc", k),
                       diff("opus", "gemini", "brier", k), diff("opus", "jev", "brier", k)]
                md.append(f"| {k} | " + " | ".join(row) + " |")
            if ("opus", 100, "mse") in boot:
                v = lambda m, s="mse": boot[(m, 100, s)][0] if (m, 100, s) in boot else float("nan")  # noqa: E731
                opus100.append([TITLES[task].split(":")[0]] +
                               [f"{v(m):.4f}" for m in ["opus", "gemini", "jev", "logreg", "base_rate"]] +
                               [f"{v(m, 'auroc'):.2f}" for m in ["opus", "gemini", "jev", "logreg"]] +
                               [diff("opus", "gemini", "mse", 100), diff("opus", "jev", "mse", 100),
                                diff("opus", "logreg", "mse", 100), diff("opus", "base_rate", "mse", 100)])
        md += ["", "(* = CI excludes 0)", ""]

    if opus100:
        md[n_head:n_head] = [
            "## Claude Opus 5.5 at k=100 (paired bootstrap, same method as the Jev − Gemini CIs)", "",
            "MSE to the true probabilities (lower is better), AUROC, and paired Opus − other MSE differences "
            "[95% CI] (* = CI excludes 0; negative = Opus closer to the truth).", "",
            "| task | MSE Opus | Gemini | Jev | logreg | base-rate | AUROC Opus | Gemini | Jev | logreg | "
            "Opus − Gemini | Opus − Jev | Opus − logreg | Opus − base-rate |", "|---" * 14 + "|"] + \
            [f"| {' | '.join(r)} |" for r in opus100] + [
            "",
            "Reading (numbers from the tables below):",
            "- Opus is the best of the three LLMs: lower MSE than Gemini and Jev at k=100 on every task (CI excludes 0 "
            "everywhere except Opus − Gemini on task 5), and it roughly ties logistic regression on the additive tasks "
            "1 and 4a/4b.",
            "- Most of that edge is its prior, not learning: on task 1 its zero-shot MSE (0.0232) already equals its "
            "k=100 MSE (0.0224), and k=5-20 are worse than zero-shot. Logistic regression goes on to 0.0054 at k=400.",
            "- It does learn the base rate from examples (task 4a: 0.32 → 0.09, task 4b: 0.32 → 0.87 at k=100, like "
            "Gemini), but more slowly than simply averaging the labels (at k=5 it is still at 0.29 / 0.60).",
            "- Pure noise (task 3): it spreads its predictions less than Gemini or Jev (SD 0.09 vs 0.19-0.21 at k=100), "
            "but still invents case-level signal, so it is significantly worse than base-rate-only (MSE 0.0114 vs 0.0004).",
            "- Interaction (task 2): AUROC 0.56 at k=100 (Bayes 0.73), the same as Gemini and Jev, and worse than "
            "base-rate-only on MSE. It does not pick up the rash × treatment interaction from 100 examples.",
            "- Task 1 tellin (which truly raises risk): at k=100 Opus predicts 0.34 for tellin cases vs a true 0.44 "
            "(Jev 0.43, Gemini 0.40), so its 'treatment helps' prior resists the examples more than the others' does.",
            ""]

    # zero-shot priors: realistic fake names vs neutral codes (same cases, same true probabilities)
    md += ["## Zero-shot priors: realistic fake names (task 1) vs neutral codes (task 5)", "",
           "| model | naming | mean | SD of predictions | AUROC | MSE to truth | corr(pred, true logit) |", "|---|---|---|---|---|---|---|"]
    for mod in ["jev", "gemini", "opus"]:
        for task in ["kellan", "neutral"]:
            P = Ps[task]
            test = S.load(task)[1]
            p = P[(P.model == mod) & (P.k == 0)].sort_values("i").p.values
            if len(p) == 0:
                continue
            tl = np.log(test.p_true / (1 - test.p_true))
            c = np.corrcoef(p, tl)[0, 1] if np.std(p) > 0 else float("nan")
            md.append(f"| {mod} | {task} | {p.mean():.3f} | {p.std():.3f} | {auroc(p[None], test.label.values)[0]:.3f} | "
                      f"{mse(p, test.p_true.values):.4f} | {c:.2f} |")
    md.append("")

    # does the model learn the base rate from the examples? (extreme-base-rate tasks)
    rows = []
    for mod, label in MODELS[:-1] + [("base_rate", "Base-rate only")]:
        r = {"model": label}
        for task in ["kellan_br10", "kellan_br90"]:
            for k in OPUS_KS:
                M = matrix(Ps[task], mod, k)
                r[f"{task[-4:]} k={k}"] = f"{M.mean():.2f}" if M is not None else "–"
        rows.append(r)
    rows.append({"model": "Truth (mean p_true)", **{f"{task[-4:]} k={k}": f"{S.load(task)[1].p_true.mean():.2f}"
                                                    for task in ["kellan_br10", "kellan_br90"] for k in OPUS_KS}})
    md += ["## Learning the base rate from examples (tasks 4a/4b)", "",
           "Mean prediction over the 300 test cases (averaged over draws). Zero-shot, every LLM gives its generic "
           "prior; the question is how far the examples move it to the true rate (0.10 / 0.89).", "",
           pd.DataFrame(rows).set_index("model").to_markdown(), ""]

    # diagnostics
    N, te = pd.read_csv(f"{OUT}/synth_neutral.csv"), S.load("neutral")[1]
    lv = sum(te[f].str.extract(r"(\d)")[0].astype(int) for f in S.FEATS)
    cors = {m: np.corrcoef(N[(N.model == m) & (N.k == 0)].sort_values("i").p.values, lv)[0, 1] for m in ["jev", "gemini", "opus"]}
    Z = Ps["noise"]
    sd = Z[Z.model.isin(["jev", "gemini", "opus", "logreg", "tabpfn"])].groupby(["model", "k", "draw"]).p.std() \
        .groupby(["model", "k"]).mean().unstack("k").round(3)
    K, tk = Ps["kellan"], S.load("kellan")[1]
    eff = {f"{m} k={k}": pd.Series(K[(K.model == m) & (K.k == k)].groupby("i").p.mean().sort_index().values)
           .groupby(tk.treatment.values).mean() for m, k in [("jev", 0), ("gemini", 0), ("opus", 0), ("jev", 100), ("gemini", 100), ("opus", 100),
                                                  ("jev", 400), ("gemini", 400)]
           if not K[(K.model == m) & (K.k == k)].empty}
    eff["true"] = tk.p_true.groupby(tk.treatment).mean()
    md += ["## Diagnostics", "",
           "- **Noise task, SD of predictions across the 300 cases** (truth: 0; mean over draws). Jev and Gemini spread "
           "their predictions as widely on pure noise as on the real tasks; Opus spreads them about half as much, but "
           "does not collapse to the base rate the way TabPFN does:", "",
           sd.to_markdown(), "",
           "- **Kellan, mean prediction by treatment** (tellin truly *raises* risk). With 400 examples Jev and Gemini learn "
           "the group-level effects, including the counter-intuitive one; their MSE is dominated by case-level spread. "
           "Opus (k=100 at most) learns morvax but under-predicts tellin:", "",
           pd.DataFrame(eff).round(2).to_markdown(), "",
           f"- **Neutral codes are not fully neutral:** level numbers were assigned in listing order, which for age and "
           f"rash is also increasing risk. Correlation of zero-shot predictions with the sum of level numbers: Jev "
           f"{cors['jev']:.2f}, Gemini {cors['gemini']:.2f}, Opus {cors['opus']:.2f}, so Gemini's and Opus's above-chance neutral zero-shot "
           f"AUROC is a 'higher level = more likely' heuristic, not knowledge.",
           "- **Shared prompts:** labels use common random numbers, so at small k some example sets carry identical "
           "labels across tasks (e.g. noise and kellan at k=5, draw 0), giving identical prompts and predictions.",
           f"- **Spend:** Jev $0.18, Gemini 3.8 Flash $3.59, Claude Opus 5.5 ${opus_spend():.2f} via OpenRouter "
           f"(logged per call in data/synthetic/*_cache.sqlite).", ""]
    pd.DataFrame(ci).drop_duplicates(["task", "k", "model", "kind"]).to_csv(f"{OUT}/synth_ci.csv", index=False)
    with open(f"{OUT}/synth_summary.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
