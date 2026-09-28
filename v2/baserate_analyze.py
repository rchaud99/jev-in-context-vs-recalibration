"""E1 analysis (preregistered in RESEARCH_AGENDA.md, section 2). No API calls.

Reads results/v2/e1_<model>.csv (from baserate_exp.py) and writes
  results/v2/e1_arms.csv       per model x task x seed x arm: Brier, log loss, AUROC, mean prediction
  results/v2/e1_tests.csv      every paired comparison: estimate, 95% CI, p, Holm (within model; all cells)
  results/v2/e1_summary.md     the preregistered counts and verdicts

    cd v2 && python baserate_analyze.py
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from baserate_exp import SEEDS, seed_rate
from common import MODELS, OUT, PREFIX, TASK_ORDER, load

B = 5000
EPS = 1e-3
CHEAP = {"jev", "qwen3.7-flash", "deepseek-v4-flash"}
PROVIDER = {"jev": "TypeSafe", "qwen3.7-flash": "Alibaba", "deepseek-v4-flash": "DeepSeek (StreamLake)",
            "gpt-4o-mini": "OpenAI", "llama-3.3-70b": "Meta (Parasail)", "gemini-3.8-flash": "Google",
            "claude-haiku-4.5": "Anthropic", "gpt-4o-mini-verbal": "OpenAI",
            "deepseek-v4-flash-verbal": "DeepSeek (StreamLake)"}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def shift_to(z_fit, rate, z_apply):
    """b such that mean(sig(z_fit + b)) = rate; returns sig(z_apply + b)."""
    b = brentq(lambda b: sig(z_fit + b).mean() - rate, -30, 30)
    return sig(z_apply + b)


def fit_platt(s_train, y, s_test):  # identical to study.fit_platt, with this file's clip
    if len(np.unique(y)) < 2:
        return np.full(len(s_test), (y.sum() + 0.5) / (len(y) + 1))
    z, zt = logit(s_train), logit(s_test)
    mu, sd = z.mean(), z.std() + 1e-6
    m = LogisticRegression(C=1.0).fit(((z - mu) / sd)[:, None], y)
    return m.predict_proba(((zt - mu) / sd)[:, None])[:, 1]


def fit_lr(Xtr, y, Xte):
    m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=1.0, max_iter=2000))
    return m.fit(Xtr, y).predict_proba(Xte)[:, 1]


def brier_rows(P, Y):  # P, Y: (..., n) -> per-row mean
    return ((P - Y) ** 2).mean(-1)


def logloss_rows(P, Y):
    P = np.clip(P, EPS, 1 - EPS)
    return -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean(-1)


def pval(d):
    n = len(d)
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1)))


def holm(ps, alpha=0.05):
    ps = np.asarray(ps, float)
    rej = np.zeros(len(ps), bool)
    for r, i in enumerate(np.argsort(ps)):
        if ps[i] > alpha / (len(ps) - r):
            break
        rej[i] = True
    return rej


def arms_for(model, task, D):
    """Returns y (n,), dict arm -> (n_seeds, n) predictions, stated rates, zs AUROC."""
    pool, test, feats = load(task)
    y = test.label.values.astype(int)
    g = D[D.task == task]
    get = lambda arm, st, seed=-1: g[(g.arm == arm) & (g.set == st) & (g.seed == seed)].sort_values("i").p.values
    zs, zp = get("zs", "test"), get("zs", "pool")
    if np.isnan(zs).any() or np.isnan(zp).any():
        print(f"  {model} {task}: {np.isnan(zs).sum()} / {np.isnan(zp).sum()} NaN zero-shot scores -> set to 0.5")
        zs, zp = np.nan_to_num(zs, nan=0.5), np.nan_to_num(zp, nan=0.5)
    A = {k: [] for k in ["zs", "told", "shift", "const", "platt", "lr", "shift_tr", "told_shift_tr"]}
    rates = []
    for s in SEEDS:
        idx, n = seed_rate(pool, s)
        r = n / 100
        rates.append(r)
        told = np.nan_to_num(get("told", "test", s), nan=0.5)
        A["zs"].append(zs)
        A["told"].append(told)
        A["shift"].append(shift_to(logit(zp), r, logit(zs)))
        A["shift_tr"].append(shift_to(logit(zs), r, logit(zs)))
        A["told_shift_tr"].append(shift_to(logit(told), r, logit(told)))
        A["const"].append(np.full(len(y), r))
        yl = pool.label.values[idx]
        A["platt"].append(fit_platt(zp[idx], yl, zs))
        A["lr"].append(fit_lr(pool[feats].iloc[idx], yl, test[feats]))
    A = {k: np.array(v) for k, v in A.items()}
    for w in ["w1", "w2"]:
        v = get(f"told_{w}", "test", 0)
        if len(v):
            A[f"told_{w}"] = np.nan_to_num(v, nan=0.5)[None]
            A[f"shift_s0"] = A["shift"][:1]
    return y, A, rates, roc_auc_score(y, zs)


COMPARISONS = [  # (family, a, b, metric): difference a - b; for Brier/log loss, > 0 means b is better
    ("H1 told-shift", "told", "shift", "brier"),
    ("H1 told-shift logloss", "told", "shift", "logloss"),
    ("P5 shift-const", "shift", "const", "brier"),
    ("P6 platt-shift", "platt", "shift", "brier"),
    ("LR lr-shift", "lr", "shift", "brier"),
    ("told-const", "told", "const", "brier"),
    ("told-zs", "told", "zs", "brier"),
    ("sens shift_tr-shift", "shift_tr", "shift", "brier"),
    ("expl told_shift_tr-shift", "told_shift_tr", "shift", "brier"),
    ("P4 told_w1-shift", "told_w1", "shift_s0", "brier"),
    ("P4 told_w2-shift", "told_w2", "shift_s0", "brier"),
    ("P4 told_w0-shift seed0", "told_s0", "shift_s0", "brier"),
]


POOL = {}  # (family, model) -> list of per-task bootstrap difference vectors (post hoc pooled test)


def main(models):
    tests, arms = [], []
    for model in models:
        path = os.path.join(OUT, f"{PREFIX}_{model}.csv")
        if not os.path.exists(path):
            print(f"{model}: no data")
            continue
        D = pd.read_csv(path)
        for task in TASK_ORDER:
            if task not in set(D.task):
                continue
            y, A, rates, auc_zs = arms_for(model, task, D)
            A["told_s0"] = A["told"][:1]
            n = len(y)
            I = np.random.default_rng(0).integers(0, n, (B, n))
            Yb = y[I]
            for arm, M in A.items():
                for s in range(M.shape[0]):
                    arms.append(dict(model=model, task=task, seed=SEEDS[s] if M.shape[0] > 1 else 0, arm=arm,
                                     rate=rates[s if M.shape[0] > 1 else 0], brier=brier_rows(M[s], y),
                                     logloss=logloss_rows(M[s], y), mean=M[s].mean(), test_rate=y.mean(),
                                     auroc=roc_auc_score(y, M[s]) if np.ptp(M[s]) > 0 else 0.5, auc_zs=auc_zs))
            for fam, a, b, met in COMPARISONS:
                if a not in A or b not in A:
                    continue
                f = brier_rows if met == "brier" else logloss_rows
                # per seed point estimates and bootstrap: (seeds, B)
                pa = np.array([f(A[a][s], y) for s in range(A[a].shape[0])])
                pb = np.array([f(A[b][s], y) for s in range(A[b].shape[0])])
                ba = np.array([f(A[a][s][I], Yb) for s in range(A[a].shape[0])])
                bb = np.array([f(A[b][s][I], Yb) for s in range(A[b].shape[0])])
                d = ba.mean(0) - bb.mean(0)
                per_seed = pa - pb
                seed_sig = [int(np.percentile(ba[s] - bb[s], 2.5) > 0) - int(np.percentile(ba[s] - bb[s], 97.5) < 0)
                            for s in range(len(per_seed))] if len(per_seed) > 1 else []
                POOL.setdefault((fam, model), []).append((d, pa.mean() - pb.mean(), per_seed))
                tests.append(dict(model=model, provider=PROVIDER.get(model, "?"), cheap=model in CHEAP, task=task,
                                  family=fam, a=a, b=b, metric=met, a_value=pa.mean(), b_value=pb.mean(),
                                  est=pa.mean() - pb.mean(), lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                                  p=pval(d), per_seed=" ".join(f"{v:+.4f}" for v in per_seed),
                                  seeds_same_sign=bool(np.all(np.sign(per_seed) == np.sign(per_seed.mean()))),
                                  seeds_sig=" ".join(str(int(v)) for v in seed_sig), auc_zs=auc_zs))
    T = pd.DataFrame(tests)
    T["sig"] = (T.lo > 0) | (T.hi < 0)
    T["holm_model"] = False
    for _, g in T.groupby(["model", "family"]):
        T.loc[g.index, "holm_model"] = holm(g.p.values)
    T["holm_all"] = False
    for _, g in T.groupby("family"):
        T.loc[g.index, "holm_all"] = holm(g.p.values)
    T.to_csv(os.path.join(OUT, f"{PREFIX}_tests.csv"), index=False)
    # POST HOC (not preregistered; added 2026-09-28 after seeing 2 of 9 models): per model, the mean over tasks of the
    # per-task differences; bootstrap resamples test cases within each task independently. Holm across models.
    P = []
    for (fam, model), lst in POOL.items():
        d = np.mean([v[0] for v in lst], 0)
        P.append(dict(family=fam, model=model, tasks=len(lst), est=np.mean([v[1] for v in lst]),
                      lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5), p=pval(d),
                      tasks_positive=sum(v[1] > 0 for v in lst),
                      per_seed=" ".join(f"{x:+.4f}" for x in np.mean([v[2] for v in lst if len(v[2]) == 3], 0))
                      if all(len(v[2]) == 3 for v in lst) else ""))
    P = pd.DataFrame(P)
    P["holm"] = False
    for _, g in P.groupby("family"):
        P.loc[g.index, "holm"] = holm(g.p.values)
    P.to_csv(os.path.join(OUT, f"{PREFIX}_posthoc_pooled.csv"), index=False)
    pd.DataFrame(arms).to_csv(os.path.join(OUT, f"{PREFIX}_arms.csv"), index=False)
    write_summary(T, pd.DataFrame(arms))
    return T


def verdict_rows(T, fam):
    rows = []
    for m, g in T[T.family == fam].groupby("model", sort=False):
        better = int((g.holm_model & (g.est > 0)).sum())
        worse = int((g.holm_model & (g.est < 0)).sum())
        rows.append(dict(model=m, provider=g.provider.iloc[0], tasks=len(g), shift_better_holm=better,
                         told_better_holm=worse, shift_better_holm_all=int((g.holm_all & (g.est > 0)).sum()),
                         mean_est=g.est.mean(), replicates=(better >= 5 and worse == 0) and len(g) == 8))
    return pd.DataFrame(rows)


def write_summary(T, Aa):
    L = ["# E1 summary: tell the model the base rate, or shift its scores in code?", "",
         "Generated by `v2/baserate_analyze.py`. Paired bootstrap over the 300 test cases (B=5000); estimates are "
         "means over 3 seeds. Brier difference a − b: positive = b is better. Holm within model across 8 tasks.", ""]
    V = verdict_rows(T, "H1 told-shift")
    L += ["## H1 (primary): Brier(told) − Brier(shift)", "", V.to_markdown(index=False, floatfmt=".4f"), ""]
    rep = V[V.replicates]
    provs = {p.split(" ")[0] for p in rep.provider}
    L.append(f"Replicates for {len(rep)} of {len(V)} configurations; providers: {sorted(provs)}; cheap model among "
             f"them: {bool(set(rep.model) & CHEAP)}. Robust (≥3 providers incl. a cheap model): "
             f"**{len(provs) >= 3 and bool(set(rep.model) & CHEAP)}**")
    g = T[(T.family == "H1 told-shift") & T.holm_model]
    L += ["", f"P3 seeds: Holm-significant H1 cells with the same sign in all 3 seeds: {int(g.seeds_same_sign.sum())} of "
          f"{len(g)}.", ""]
    for fam in ["H1 told-shift logloss", "P5 shift-const", "P6 platt-shift", "LR lr-shift", "told-const", "told-zs",
                "sens shift_tr-shift", "P4 told_w0-shift seed0", "P4 told_w1-shift", "P4 told_w2-shift"]:
        V = verdict_rows(T, fam)
        if len(V):
            V = V.rename(columns={"shift_better_holm": "b_better_holm", "told_better_holm": "a_better_holm",
                                  "shift_better_holm_all": "b_better_holm_all"}).drop(columns=["replicates"])
            L += [f"## {fam} (a − b; b_better = second-named arm significantly better)", "",
                  V.to_markdown(index=False, floatfmt=".4f"), ""]
    # P5 conditional on zero-shot AUROC
    g = T[T.family == "P5 shift-const"]
    hi = g[g.auc_zs >= 0.65]
    L += ["## P5 detail", "",
          f"Cells with zero-shot AUROC ≥ 0.65: {len(hi)}; shift lower Brier than const in {int((hi.est < 0).sum())} "
          f"({(hi.est < 0).mean():.0%}). Shift significantly worse than const: "
          + ", ".join(f"{r.model}/{r.task} (AUROC {r.auc_zs:.2f})" for r in g[g.sig & (g.est > 0)].itertuples()), ""]
    # mechanism
    M = Aa[Aa.arm.isin(["told", "zs", "shift"])].groupby(["model", "task", "arm"]).agg(
        mean=("mean", "mean"), rate=("rate", "mean"), auroc=("auroc", "mean")).reset_index()
    W = M.pivot_table(index=["model", "task"], columns="arm", values=["mean", "auroc"]).reset_index()
    W.columns = [" ".join(c).strip() for c in W.columns]
    W["rate"] = M[M.arm == "told"].set_index(["model", "task"]).rate.values
    W["told_miss"] = W["mean told"] - W["rate"]
    W["zs_miss"] = W["mean zs"] - W["rate"]
    L += ["## Mechanism (exploratory): mean prediction vs stated rate; ranking", "",
          W[["model", "task", "rate", "mean zs", "mean told", "zs_miss", "told_miss", "auroc zs", "auroc told"]]
          .to_markdown(index=False, floatfmt=".3f"), ""]
    open(os.path.join(OUT, f"{PREFIX}_summary.md"), "w").write("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:] or MODELS)
    print(open(os.path.join(OUT, f"{PREFIX}_summary.md")).read()[:6000])
