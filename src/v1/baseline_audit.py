"""Trivial-baseline audit (cache-only, no API calls, deterministic).

A. Bayesian-prior test (base-rate sentence, 8 tasks, 300 test cases, one run). Predictors:
   const        the stated rate (the number in the sentence: round(100 * pool rate) %) for every case
   jev_zs       Jev zero-shot
   jev_told     Jev told the base rate
   ideal_shift  logit(jev_zs) + b, b chosen so the mean prediction over the 300 (unlabeled) test inputs equals the
                stated rate. Label-free: the best a proper prior update of Jev's zero-shot scores could do.
   jev_shift100 the README's 1-parameter shift fitted on 100 labels (draw 0), baserate_ci.py
   platt100     study.fit_platt (2-parameter Platt, C=1) fitted on the same 100 labels' zero-shot scores
   gem_told     Gemini told the base rate
   Jev/Gemini predictions are rebuilt by baserate_ci.predictions (cache miss raises) and checked against
   results/v1/baserate_check.csv / gemini_baserate.csv. Platt slopes are reported on the raw logit scale: the fitted
   C=1 slope (the one used), the unregularised slope on the same 100 labels, and, as a diagnostic with more data,
   the unregularised calibration slope of zero-shot and told Jev on the 300 test labels (bootstrap CI).
B. Every model x arm x k of the real-task study vs a constant predictor, Brier difference (model - constant):
   zero-shot vs the pool rate; in-context, Platt (study.fit_platt, unconstrained), constrained Platt
   (results/v1/platt_constrained.csv) and, for Jev, examples+Platt ("both") vs the mean of the same k example labels
   (per draw, the base rate a user would know from those labels). All completed draws of each arm are used
   (Jev in-context 6-7, Jev Platt 10, Gemini/Opus 2); "max" = each model's largest k (Jev 400, 200 on GitHub and
   Stack Overflow; Gemini 400, 200 on text tasks; Opus has no k above 100). Extras for the README audit: Jev's
   zero-shot score stacked into TabPFN (results/v1/stack_*.csv) and TabPFN alone (results/v1/tabpfn_*.csv), same labels,
   10 draws; and the calibration slope of each model's in-context predictions at k=100 on the test labels.

Statistics follow verify_claims.py: paired bootstrap B=2000, seed 0; test cases (and draws, where there are any)
resampled jointly; p = 2 x smaller tail share (+1 correction); Holm (alpha 0.05) across tasks within each family.

Outputs: results/v1/baseline_audit_A.csv (+ _A_values, _A_slopes), results/v1/baseline_audit_B.csv (+ _B_slopes),
         results/v1/baseline_audit.md
    python baseline_audit.py
"""
import os

os.environ["JEV_OFFLINE"] = "1"

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

import baserate_ci as bc  # rebuilds the base-rate predictions from the caches
import jev
import platt_sweep as ps
from analyze import auroc, brier
from study import draw_order, logit
from tasks import ROOT, TASKS, load
from verify_claims import B, TASK_ORDER, holm, pval, two_level

assert jev.OFFLINE
OUT = os.path.join(ROOT, "results", "v1")
F = {"brier": brier, "auroc": auroc}


def sig(z):
    return 1 / (1 + np.exp(-z))


def summarize(R):
    R["sig"] = R.p.notna() & ((R.lo > 0) | (R.hi < 0))
    R["holm"] = False
    for _, g in R[R.p.notna()].groupby("family"):
        R.loc[g.index, "holm"] = holm(g.p.values)
    return R


# ---------------------------------------------------------------- A: Bayesian-prior test
A_NAMES = ["const", "jev_zs", "jev_told", "ideal_shift", "jev_shift100", "platt100", "gem_told"]
A_PAIRS = [("jev_told", "const"), ("jev_told", "ideal_shift"), ("ideal_shift", "const"), ("platt100", "const"),
           ("gem_told", "const"),
           # extra context
           ("jev_zs", "const"), ("jev_shift100", "const"), ("platt100", "jev_shift100"), ("platt100", "ideal_shift")]


def unreg_slope(s, y):
    return float(LogisticRegression(C=1e10, max_iter=1000).fit(s[:, None], y).coef_[0, 0])


def part_a():
    vals, rows, slopes = [], [], []
    for t in TASK_ORDER:
        task = TASKS[t]
        y, P = bc.predictions(t)
        bc.check(t, y, P)
        pool = load(t)[0]
        ypool = pool.label.values.astype(int)
        stated = round(100 * pool.label.mean()) / 100  # the number in the sentence (bc.note_for)
        n = len(y)
        P["const"] = np.full(n, stated)
        z = logit(P["jev_zs"])
        b = brentq(lambda b: sig(z + b).mean() - stated, -20, 20)
        P["ideal_shift"] = sig(z + b)
        zp = np.array(jev.noul_batch(task["context"], list(pool.text), task["question"]))
        idx = draw_order(t, 0, len(pool))[:100]
        P["platt100"], coef = ps.platt_c(zp[idx], ypool[idx], P["jev_zs"], 1.0)
        zi = logit(zp[idx])
        rng = np.random.default_rng(0)
        I = rng.integers(0, n, (B, n))
        yb = y[I]
        zt = logit(P["jev_told"])
        diag = {}
        for nm, s in [("zs", z), ("told", zt)]:
            bs = np.array([unreg_slope(s[ii], y[ii]) for ii in I])
            diag[nm] = (unreg_slope(s, y), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
        slopes.append(dict(task=t, stated=stated, test_rate=y.mean(), ideal_shift_b=b,
                           platt100_slope_C1=coef / (zi.std() + 1e-6), platt100_slope_unreg=unreg_slope(zi, ypool[idx]),
                           test_slope_zs=diag["zs"][0], test_slope_zs_lo=diag["zs"][1], test_slope_zs_hi=diag["zs"][2],
                           test_slope_told=diag["told"][0], test_slope_told_lo=diag["told"][1],
                           test_slope_told_hi=diag["told"][2]))
        boot = {(k, m): f(P[k][I], yb) for k in A_NAMES for m, f in F.items()}
        point = {(k, m): f(P[k], y) for k in A_NAMES for m, f in F.items()}
        for k in A_NAMES:
            vals.append(dict(task=t, predictor=k, mean=P[k].mean(), brier=point[(k, "brier")],
                             auroc=point[(k, "auroc")]))
        for a, c in A_PAIRS:
            for m in F:
                d = boot[(a, m)] - boot[(c, m)]
                degenerate = np.allclose(d, 0)  # e.g. AUROC of a monotone shift vs its source
                rows.append(dict(part="A", family=f"{a} - {c} {m}", task=t, comparison=f"{a} - {c}", metric=m,
                                 a_value=point[(a, m)], b_value=point[(c, m)], est=point[(a, m)] - point[(c, m)],
                                 lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                 p=np.nan if degenerate else pval(d)))
        print("A", t, "done", flush=True)
    return summarize(pd.DataFrame(rows)), pd.DataFrame(vals), pd.DataFrame(slopes)


# ---------------------------------------------------------------- B: every cell vs a constant
BASE_ARMS = ["zero_shot", "in_context", "platt", "platt_c"]
SOURCES = [("Jev", "pred", BASE_ARMS + ["both"]), ("Gemini", "gem", BASE_ARMS), ("Opus", "opus", BASE_ARMS),
           # extras for the README audit (recipe step 4): stacking and TabPFN on the same k labels
           ("Jev", "stack", ["stack_tabpfn"]), ("TabPFN", "tabpfn", ["tabpfn"])]
PC = pd.read_csv(f"{OUT}/platt_constrained.csv")


def cell_matrix(t, model, arm, k, P):
    if arm == "platt_c":
        g = PC[(PC.task == t) & (PC.model == model) & (PC.k == k)]
    else:
        g = P[(P.arm == arm) & (P.k == k)]
    if g.empty:
        return None, None
    M = g.pivot(index="draw", columns="i", values="p")
    assert M.shape[1] == 300
    return list(M.index), M.values


def part_b():
    rows, slopes = [], []
    for t in TASK_ORDER:
        pool, test, _ = load(t)
        y = test.label.values.astype(int)
        ly = pd.read_csv(f"{OUT}/labels_{t}.csv").sort_values("i").label.values
        assert (ly == y).all()
        ypool = pool.label.values.astype(int)
        npool, n = len(pool), len(y)
        for model, pre, arms in SOURCES:
            P = pd.read_csv(f"{OUT}/{pre}_{t}.csv")
            cells = [("zero_shot", 0, "0")] if "zero_shot" in arms else []
            for arm in [a for a in arms if a != "zero_shot"]:
                src = PC[(PC.task == t) & (PC.model == model)] if arm == "platt_c" else P[P.arm == arm]
                ks = sorted(src.k.unique())
                cells += [(arm, k, str(k)) for k in [5, 20, 100] if k in ks]
                if max(ks) > 100:
                    cells.append((arm, max(ks), "max"))
            for arm, k, klab in cells:
                if arm == "zero_shot":
                    M = P[P.arm == "zero_shot"].sort_values("i").p.values[None]
                    draws, C = [0], np.full((1, n), ypool.mean())
                else:
                    draws, M = cell_matrix(t, model, arm, k, P)
                    if M is None:
                        continue
                    C = np.array([np.full(n, ypool[draw_order(t, d, npool)[:k]].mean()) for d in draws])
                    if pre == "pred":  # the stored base_rate arm is the smoothed mean of the same k labels
                        br = P[(P.arm == "base_rate") & (P.k == k)].groupby("draw").p.first()
                        for d in draws:
                            yk = ypool[draw_order(t, d, npool)[:k]]
                            assert abs(br[d] - (yk.sum() + 0.5) / (k + 1)) < 1e-9, (t, k, d)
                bs = two_level({"m": M, "c": C}, y, "brier")
                d = bs["m"][1] - bs["c"][1]
                rows.append(dict(part="B", family=f"{model} {arm} k={klab}", model=model, arm=arm, k=k, k_label=klab,
                                 task=t, n_draws=len(draws), const_mean=C[:, 0].mean(), a_value=bs["m"][0],
                                 b_value=bs["c"][0], est=bs["m"][0] - bs["c"][0], lo=np.percentile(d, 2.5),
                                 hi=np.percentile(d, 97.5), p=pval(d), metric="brier",
                                 comparison=f"{model} {arm} - constant"))
            if pre in ("pred", "gem", "opus"):  # calibration slope of in-context predictions on the test labels
                for k in [100]:
                    draws, M = cell_matrix(t, model, "in_context", k, P)
                    slopes.append(dict(task=t, model=model, k=k, n_draws=len(draws),
                                       slope=np.mean([unreg_slope(logit(m), y) for m in M]),
                                       sd_pred=M.std(1).mean(), auroc=auroc(M, y[None]).mean()))
        print("B", t, "done", flush=True)
    return summarize(pd.DataFrame(rows)), pd.DataFrame(slopes)


# ---------------------------------------------------------------- report
FILES_M = ["Jev", "Gemini", "Opus"]
LAB = {"const": "(1) constant = stated rate", "jev_zs": "(2) Jev zero-shot", "jev_told": "(3) Jev told",
       "ideal_shift": "(4) ideal prior shift (label-free)", "jev_shift100": "(5) shift, 100 labels",
       "platt100": "(6) Platt, 100 labels", "gem_told": "(7) Gemini told"}


def cell(r, nd=3):
    s = f"{r.est:+.{nd}f} [{r.lo:+.{nd}f}, {r.hi:+.{nd}f}]"
    return f"**{s}**" + ("†" if r.holm else "") if r.sig else s


def count_line(g, lower_better=True):
    good = (g.est < 0) if lower_better else (g.est > 0)
    return (f"{int((g.sig & good).sum())} ({int((g.holm & good).sum())}) / "
            f"{int((g.sig & ~good).sum())} ({int((g.holm & ~good).sum())})")


def write_md(A, V, S, Bt, BS):
    L = ["# Trivial-baseline audit", "",
         f"Generated by `baseline_audit.py` (cache-only). Paired bootstrap, B={B}, seed 0; test cases (and draws where "
         "there are any) resampled jointly; p = 2 × min tail share; Holm across tasks within each family (as "
         "`verify_claims.py`). Cells: estimate [95% CI]; **bold** = CI excludes 0; † = survives Holm. Brier: negative "
         "= first-named is better. Full numbers: `results/v1/baseline_audit_A.csv`, `results/v1/baseline_audit_B.csv`.", "",
         "## A. Is the stated base rate used as a Bayesian prior?", "",
         "One run, 300 test cases (bootstrap over test cases only). The constant is the number stated in the sentence. "
         "The ideal prior shift moves Jev's zero-shot logits by one constant so the mean prediction over the unlabeled "
         "test inputs equals the stated rate (no labels used).", "",
         "### Brier (AUROC) of each predictor", "",
         "| Task | " + " | ".join(LAB[k] for k in A_NAMES) + " |", "|---|" + "---|" * len(A_NAMES)]
    for t in TASK_ORDER:
        v = V[V.task == t].set_index("predictor")
        L.append(f"| {t} | " + " | ".join(
            f"{v.loc[k, 'brier']:.3f}" + ("" if k == "const" else f" ({v.loc[k, 'auroc']:.2f})") for k in A_NAMES) + " |")
    L += ["", "### Brier differences", ""]
    heads = [("jev_told", "const"), ("jev_told", "ideal_shift"), ("ideal_shift", "const"), ("platt100", "const"),
             ("gem_told", "const"), ("jev_zs", "const"), ("jev_shift100", "const"), ("platt100", "jev_shift100")]
    L += ["| Task | " + " | ".join(f"({A_NAMES.index(a) + 1}) − ({A_NAMES.index(b) + 1})" for a, b in heads) + " |",
          "|---|" + "---|" * len(heads)]
    for t in TASK_ORDER:
        L.append(f"| {t} | " + " | ".join(
            cell(A[(A.task == t) & (A.family == f"{a} - {b} brier")].iloc[0]) for a, b in heads) + " |")
    L.append("| **sig better / worse (Holm)** | " + " | ".join(
        count_line(A[A.family == f"{a} - {b} brier"]) for a, b in heads) + " |")
    L += ["", "### AUROC differences (constant AUROC = 0.5; the ideal shift has Jev zero-shot's AUROC)", "",
          "| Task | (3) − (4) = told − zero-shot | (6) − (1) | (7) − (1) |", "|---|---|---|---|"]
    for t in TASK_ORDER:
        L.append(f"| {t} | " + " | ".join(
            cell(A[(A.task == t) & (A.family == f"{a} - {b} auroc")].iloc[0], 2)
            for a, b in [("jev_told", "ideal_shift"), ("platt100", "const"), ("gem_told", "const")]) + " |")
    L += ["", "### Slopes (raw logit scale; < 1 = Jev's per-case scores too extreme)", "",
          "| Task | stated / test rate | ideal shift b | Platt slope, 100 labels (C=1, used) | unregularised, 100 labels | "
          "test-set slope, zero-shot [95% CI] | test-set slope, told [95% CI] |", "|---|---|---|---|---|---|---|"]
    for _, s in S.iterrows():
        L.append(f"| {s.task} | {s.stated:.2f} / {s.test_rate:.2f} | {s.ideal_shift_b:+.2f} | {s.platt100_slope_C1:.2f} | "
                 f"{s.platt100_slope_unreg:.2f} | {s.test_slope_zs:.2f} [{s.test_slope_zs_lo:.2f}, {s.test_slope_zs_hi:.2f}] | "
                 f"{s.test_slope_told:.2f} [{s.test_slope_told_lo:.2f}, {s.test_slope_told_hi:.2f}] |")
    L += ["", "## B. Every model × arm × k vs a constant predictor (Brier, model − constant)", "",
          "Constant: pool rate for zero-shot; for k > 0 the mean of the same k example labels (per draw). Platt = "
          "study.fit_platt (unconstrained); constrained Platt = `platt_constrained.csv` (draws 0–1); both = Jev "
          "examples + Platt. Counts over tasks: significantly better (after Holm) / significantly worse (after Holm).", "",
          "| Model | Arm | k | tasks | draws | better than constant (Holm) / worse (Holm) | tasks sig. better | tasks sig. worse |",
          "|---|---|---|---|---|---|---|---|"]
    for fam, g in Bt.groupby("family", sort=False):
        r = g.iloc[0]
        L.append(f"| {r.model} | {r.arm} | {r.k_label} | {len(g)} | {g.n_draws.min()}–{g.n_draws.max()} | "
                 f"{count_line(g)} | {','.join(g[g.sig & (g.est < 0)].task) or '–'} | "
                 f"{','.join(g[g.sig & (g.est > 0)].task) or '–'} |")
    L += ["", "### Per task, k=100 (model − constant Brier)", "",
          "| Task | " + " | ".join(f"{m} {a}" for m, a in [("Jev", "in_context"), ("Jev", "platt"), ("Jev", "both"),
                                                         ("Gemini", "in_context"), ("Gemini", "platt"),
                                                         ("Opus", "in_context"), ("Opus", "platt_c")]) + " |",
          "|---|" + "---|" * 7]
    for t in TASK_ORDER:
        cs = []
        for m, a in [("Jev", "in_context"), ("Jev", "platt"), ("Jev", "both"), ("Gemini", "in_context"),
                     ("Gemini", "platt"), ("Opus", "in_context"), ("Opus", "platt_c")]:
            g = Bt[(Bt.task == t) & (Bt.family == f"{m} {a} k=100")]
            cs.append(cell(g.iloc[0]) if len(g) else "")
        L.append(f"| {t} | " + " | ".join(cs) + " |")
    L += ["", "### Calibration slope of in-context predictions at k=100 (test labels, raw logit, mean over draws)", "",
          "Slope > 1: per-case spread too small; < 1: too spread out (overconfident); < 0: inverted ranking.", "",
          "| Task | " + " | ".join(f"{m}: slope (AUROC)" for m in FILES_M) + " |", "|---|" + "---|" * len(FILES_M)]
    for t in TASK_ORDER:
        g = BS[BS.task == t].set_index("model")
        L.append(f"| {t} | " + " | ".join(f"{g.loc[m, 'slope']:.2f} ({g.loc[m, 'auroc']:.2f})" for m in FILES_M) + " |")
    L.append("")
    open(f"{OUT}/baseline_audit.md", "w").write("\n".join(L))


def main():
    A, V, S = part_a()
    Bt, BS = part_b()
    BS.to_csv(f"{OUT}/baseline_audit_B_slopes.csv", index=False)
    A["B"] = Bt["B"] = B
    A.to_csv(f"{OUT}/baseline_audit_A.csv", index=False)
    V.to_csv(f"{OUT}/baseline_audit_A_values.csv", index=False)
    S.to_csv(f"{OUT}/baseline_audit_A_slopes.csv", index=False)
    Bt.to_csv(f"{OUT}/baseline_audit_B.csv", index=False)
    write_md(A, V, S, Bt, BS)
    print("wrote results/v1/baseline_audit_A*.csv, results/v1/baseline_audit_B.csv, results/v1/baseline_audit.md")


if __name__ == "__main__":
    main()
