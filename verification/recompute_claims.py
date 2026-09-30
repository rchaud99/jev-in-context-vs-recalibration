"""Independent recompute of the headline claims from the saved per-item predictions (no API calls, no caches).

Written 2026-09-30 as an audit. It deliberately imports NOTHING from src/ and re-implements every step:
the 100-label seed draw, the label-free logit shift (bisection), Brier, the paired bootstrap and Holm. It differs from
src/v2/baserate_analyze.py in two ways on purpose: bootstrap indices are drawn independently for every task and model
(the repo reuses one index matrix), and a different RNG seed is used. Point estimates must match the repo exactly;
CIs and p-values should agree to within bootstrap noise.

Inputs:  data/v2/*.csv, data/v2_fresh/*.csv (labels; first 300 rows = test, next 300 = unlabelled pool U)
         results/v2/{e1,e1r}_<model>.csv (per-item probabilities: zero-shot on test and U; told for seeds 0-2)
         results/v1/pred_*.csv, results/v1/tabpfn_*.csv, results/v1/labels_*.csv, data/*.csv (v1 claims S3, S4)
Output:  printed report (saved copy: verification/recompute_output.txt)

    python verification/recompute_claims.py > verification/recompute_output.txt
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS, B = 1e-3, 5000
DS = {"e1": ("v2", ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]),
      "e1r": ("v2_fresh", ["nyc311", "crimes", "hn", "github_prs", "inspections"])}
NINE = ["jev", "qwen3.7-flash", "deepseek-v4-flash", "gpt-4o-mini", "llama-3.3-70b", "gpt-4o-mini-verbal",
        "deepseek-v4-flash-verbal", "gemini-3.8-flash", "claude-haiku-4.5"]
EXT = ["deepseek-v4-pro", "gpt-5-mini-think", "deepseek-v4-flash-think"]  # E2, E3 (single-model preregistered checks)


def lg(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def shift(z_pool, rate, z_test):
    """label-free shift: b with mean(sigmoid(z_pool + b)) = rate, found by bisection; returns sigmoid(z_test + b)"""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if (1 / (1 + np.exp(-(z_pool + mid)))).mean() < rate else (lo, mid)
    return 1 / (1 + np.exp(-(z_test + (lo + hi) / 2)))


def holm(p, a=0.05):
    p = np.asarray(p)
    rej = np.zeros(len(p), bool)
    for r, i in enumerate(np.argsort(p)):
        if p[i] > a / (len(p) - r):
            break
        rej[i] = True
    return rej


def boot_p(d):
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))


def cell(prefix, model, task):
    df = pd.read_csv(f"{ROOT}/data/{DS[prefix][0]}/{task}.csv")
    y, U = df.label.values[:300].astype(float), df.label.values[300:600].astype(float)
    P = pd.read_csv(f"{ROOT}/results/v2/{prefix}_{model}.csv")
    P = P[P.task == task]

    def get(arm, st, seed):
        g = P[(P.arm == arm) & (P.set == st) & (P.seed == seed)].sort_values("i")
        assert len(g) == 300 and (g.i.values == np.arange(300)).all()
        return np.nan_to_num(g.p.values.astype(float), nan=0.5)

    zs, zp = get("zs", "test", -1), get("zs", "pool", -1)
    c = dict(y=y, told=[], shift=[], const=[], rates=[], auc=roc_auc_score(y, zs))
    for s in (0, 1, 2):
        r = round(100 * U[np.random.default_rng(1000 + s).choice(300, 100, replace=False)].mean()) / 100
        c["rates"].append(r)
        c["told"].append(get("told", "test", s))
        c["shift"].append(shift(lg(zp), r, lg(zs)))
        c["const"].append(np.full(300, r))
    return c


def diff(c, a, b):  # per-case seed-averaged Brier difference a - b, and per-seed point estimates
    D = np.array([(c[a][s] - c["y"]) ** 2 - (c[b][s] - c["y"]) ** 2 for s in range(3)])
    return D.mean(0), D.mean(1)


def v2_claims():
    rng = np.random.default_rng(2026)
    cells = {(p, m, t): cell(p, m, t) for p, (_, ts) in DS.items() for m in NINE + EXT for t in ts}
    pooled, tasks = [], []
    for p, (_, ts) in DS.items():
        for m in NINE + EXT:
            for a, b in [("told", "const"), ("told", "shift")]:
                boots, ests, seeds = [], [], []
                for t in ts:
                    d, ps = diff(cells[(p, m, t)], a, b)
                    bt = d[rng.integers(0, 300, (B, 300))].mean(1)
                    boots.append(bt), ests.append(d.mean()), seeds.append(ps)
                    tasks.append(dict(ds=p, model=m, task=t, cmp=f"{a}-{b}", est=d.mean(), lo=np.percentile(bt, 2.5),
                                      hi=np.percentile(bt, 97.5), p=boot_p(bt), auc=cells[(p, m, t)]["auc"]))
                pb = np.mean(boots, 0)
                pooled.append(dict(ds=p, model=m, cmp=f"{a}-{b}", est=np.mean(ests), lo=np.percentile(pb, 2.5),
                                   hi=np.percentile(pb, 97.5), p=boot_p(pb),
                                   seeds=" ".join(f"{v:+.4f}" for v in np.mean(seeds, 0))))
    P, T = pd.DataFrame(pooled), pd.DataFrame(tasks)
    P["holm9"] = ""
    for _, g in P[P.model.isin(NINE)].groupby(["ds", "cmp"]):
        P.loc[g.index, "holm9"] = np.where(holm(g.p.values), "yes", "no")
    T["holm_model"] = False
    for _, g in T.groupby(["ds", "model", "cmp"]):
        T.loc[g.index, "holm_model"] = holm(g.p.values)
    T["sig"] = (T.lo > 0) | (T.hi < 0)

    print("## V1 / V2 / V11: pooled Brier differences (mean over tasks of seed-averaged per-task differences)")
    print("Positive = telling is worse. holm9 = Holm across the 9 main configurations (extensions: raw p, "
          "preregistered as single-model checks).\n")
    print(P.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    for p in DS:
        for cmp in ["told-const", "told-shift"]:
            g = P[(P.ds == p) & (P.cmp == cmp) & P.model.isin(NINE)]
            pos = int(((g.holm9 == "yes") & (g.est > 0)).sum())
            neg = int(((g.holm9 == "yes") & (g.est < 0)).sum())
            agree = int(sum(len({np.sign(float(x)) for x in s.split()}) == 1 for s in g.seeds))
            print(f"{p} {cmp}: Holm-significant > 0 in {pos}/9, < 0 in {neg}/9; all 3 seeds same sign in {agree}/9")

    print("\n## Borderline pooled cells: analytic z-test and 30 bootstrap seeds")
    for p, m, a, b in [("e1r", "gemini-3.8-flash", "told", "shift"), ("e1r", "jev", "told", "const")]:
        Ds = [diff(cells[(p, m, t)], a, b)[0] for t in DS[p][1]]
        est = np.mean([d.mean() for d in Ds])
        se = np.sqrt(sum(d.var(ddof=1) / len(d) for d in Ds)) / len(Ds)
        ps = [boot_p(np.mean([d[np.random.default_rng(100 + s).integers(0, 300, (B, 300))].mean(1) for d in Ds], 0))
              for s in range(30)]
        print(f"{p} {m} {a}-{b}: est {est:+.4f}; z-test p={2 * stats.norm.sf(abs(est / se)):.4f}; bootstrap p "
              f"median {np.median(ps):.4f} (range {min(ps):.4f}-{max(ps):.4f}). Largest p in its Holm family, so the "
              f"threshold is 0.05.")

    N = T[T.model.isin(NINE)]
    ts = N[N.cmp == "told-shift"]
    print(f"\n## 117-cell count (9 configurations x 13 task datasets), told - shift, Holm within model and dataset")
    print(f"cells {len(ts)}; shift better {int((ts.holm_model & (ts.est > 0)).sum())}; told better "
          f"{int((ts.holm_model & (ts.est < 0)).sum())}; (unadjusted CIs: shift better {int((ts.sig & (ts.est > 0)).sum())}, "
          f"told better {int((ts.sig & (ts.est < 0)).sum())}); Holm across all 117: shift better "
          f"{int((holm(ts.p.values) & (ts.est.values > 0)).sum())}")
    e1 = ts[ts.ds == "e1"]
    print("\n## V3: E1 per-task primary (shift better on >= 5/8 tasks after Holm, worse on 0)")
    for m in NINE:
        g = e1[e1.model == m]
        bt, wr = int((g.holm_model & (g.est > 0)).sum()), int((g.holm_model & (g.est < 0)).sum())
        print(f"  {m:26s} better {bt}/8 worse {wr}/8 -> {'PASS' if bt >= 5 and wr == 0 else 'fail'}")
    tc = N[N.cmp == "told-const"]
    print("\n## V8: told - const by the model's zero-shot AUROC on the cell")
    for lo_, hi_ in [(0, 0.65), (0.65, 0.75), (0.75, 1)]:
        g = tc[(tc.auc > lo_) & (tc.auc <= hi_)]
        print(f"  AUROC ({lo_}, {hi_}]: {len(g)} cells; told sig. worse {int((g.sig & (g.est > 0)).sum())} "
              f"(Holm {int((g.holm_model & (g.est > 0)).sum())}); told sig. better {int((g.sig & (g.est < 0)).sum())}")
    print(tc[tc.auc > 0.75][["ds", "model", "task", "auc", "est", "lo", "hi"]].to_string(
        index=False, float_format=lambda v: f"{v:+.3f}"))


def v1_claims():
    print("\n## S3 (v1, Jev only): in-context examples vs one-hot LR trained on the same k labels")
    rng = np.random.default_rng(99)
    rows = []
    for t in DS["e1"][1]:
        y = pd.read_csv(f"{ROOT}/results/v1/labels_{t}.csv").sort_values("i").label.values
        P = pd.read_csv(f"{ROOT}/results/v1/pred_{t}.csv")
        for k in (50, 100, 200):
            g = P[P.k == k]
            draws = sorted(set(g[g.arm == "in_context"].draw) & set(g[g.arm == "logreg"].draw))
            M = {a: np.array([g[(g.arm == a) & (g.draw == d)].sort_values("i").p.values for d in draws])
                 for a in ("in_context", "logreg")}
            d = ((M["in_context"] - y) ** 2).mean(0) - ((M["logreg"] - y) ** 2).mean(0)
            I = rng.integers(0, len(y), (2000, len(y)))
            bt = d[I].mean(1)
            auc = lambda A, ii: np.mean([roc_auc_score(y[ii], m[ii]) for m in A])
            ba = np.array([auc(M["in_context"], ii) - auc(M["logreg"], ii) for ii in I[:500]])
            rows.append(dict(task=t, k=k, draws=len(draws), d_brier=d.mean(), p_brier=boot_p(bt),
                             d_auroc=auc(M["in_context"], np.arange(len(y))) - auc(M["logreg"], np.arange(len(y))),
                             p_auroc=boot_p(ba)))
    R = pd.DataFrame(rows)
    for col in ("brier", "auroc"):
        R[f"holm_{col}"] = False
        for _, g in R.groupby("k"):
            R.loc[g.index, f"holm_{col}"] = holm(g[f"p_{col}"].values)
    print(R.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    for k, g in R.groupby("k"):
        print(f"  k={k}: in-context significantly better Brier (Holm) on {int((g.holm_brier & (g.d_brier < 0)).sum())}/8; "
              f"better AUROC (Holm) on {int((g.holm_auroc & (g.d_auroc > 0)).sum())}/8 "
              f"({', '.join(g[g.holm_auroc & (g.d_auroc > 0)].task) or '-'})")

    print("\n## S4 (v1, Jev only): Stack Overflow AUROC, in-context vs trained text baselines")
    t = "stackoverflow"
    df = pd.read_csv(f"{ROOT}/data/{t}.csv")
    test, pool = df.iloc[:300], df.iloc[300:].reset_index(drop=True)
    y = test.label.values

    def tfidf(k, d):  # same example rows as src/v1/study.py (draw_order); C and n-grams tuned by 5-fold CV
        ex = pool.iloc[np.random.default_rng([ord(c) for c in t] + [d]).permutation(len(pool))[:k]]
        gs = GridSearchCV(make_pipeline(TfidfVectorizer(sublinear_tf=True), LogisticRegression(max_iter=3000)),
                          {"logisticregression__C": [0.01, 0.1, 1, 10, 100],
                           "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)]},
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(ex.text, ex.label)
        return gs.predict_proba(test.text)[:, 1]

    P = pd.read_csv(f"{ROOT}/results/v1/pred_{t}.csv")
    TP = pd.read_csv(f"{ROOT}/results/v1/tabpfn_{t}.csv")
    I = np.random.default_rng(5).integers(0, 300, (2000, 300))
    auc = lambda A, ii: np.mean([roc_auc_score(y[ii], m[ii]) for m in A])
    for k in (100, 200):
        g = P[(P.k == k) & (P.arm == "in_context")]
        draws = sorted(g.draw.unique())
        IC = np.array([g[g.draw == d].sort_values("i").p.values for d in draws])
        TF = np.array([tfidf(k, d) for d in draws])
        bt = np.array([auc(IC, ii) - auc(TF, ii) for ii in I])
        base = {a: np.mean([roc_auc_score(y, h.sort_values("i").p.values) for _, h in src[(src.k == k) & (src.arm == a)].groupby("draw")])
                for a, src in [("logreg", P), ("embed_lr", TP), ("tabpfn", TP)]}
        print(f"  k={k}: Jev in-context {auc(IC, np.arange(300)):.3f} vs tuned TF-IDF+LR (this script, same draws "
              f"{draws}) {auc(TF, np.arange(300)):.3f}: diff {auc(IC, np.arange(300)) - auc(TF, np.arange(300)):+.3f} "
              f"[{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}], p={boot_p(bt):.4f}. Saved baselines: "
              + ", ".join(f"{a} {v:.3f}" for a, v in base.items()))
    TF400 = np.array([tfidf(400, d) for d in range(10)])
    print(f"  k=400 tuned TF-IDF+LR: {auc(TF400, np.arange(300)):.3f} (Jev in-context was not run at k=400 on this task)")
    g = P[(P.k == 200) & P.arm.isin(["in_context", "shuffled"])]
    print("  Note: k=200 in-context vs shuffled-label AUROC = "
          + " vs ".join(f"{np.mean([roc_auc_score(y, h.sort_values('i').p.values) for _, h in g[g.arm == a].groupby('draw')]):.4f}"
                        for a in ("in_context", "shuffled")))


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    v2_claims()
    v1_claims()
