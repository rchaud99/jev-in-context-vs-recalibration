"""Robustness of "examples beat Platt where labels carry local knowledge": post-hoc corrections with different
regularisation, and a 1-parameter base-rate shift, vs in-context examples. No API calls.

Zero-shot scores come from the caches only: Jev via jev.py with JEV_OFFLINE=1 (reproduces study.py's zs_pool / zs_test
exactly), Gemini by reading data/gemini_cache.sqlite directly with gemini_study's prompt/hash (reproduces
gemini_platt.py's zs_pool). The script checks that its C=1 Platt reproduces the stored `platt` arm.

Corrections, all fitted on the zero-shot scores of the k example cases of each draw and applied to zero-shot test scores:
  platt_C{0.1,1,10,100}  study.fit_platt (standardized logit, L2) with that C (C=1 is the study's `platt`)
  shift                  logit(p') = logit(p) + b, b fitted by penalized log loss with a N(0, 2^2) prior on b
Compared with in_context (Brier, and AUROC), paired two-level bootstrap over test cases and draws (as analyze2.py).
Jev: all draws where in_context completed, k in 5..400. Gemini: draws 0-1, k in 5, 20, 100, 400 (200 for text).
"best" = the correction with the lowest point-estimate Brier at that (model, task, k), chosen on the test set
(optimistic for post-hoc, so "in_context beats best" is a conservative statement).
Output: results/v1/platt_sweep.csv
"""
import hashlib
import json
import os
import sqlite3
import sys

os.environ["JEV_OFFLINE"] = "1"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression

import jev
from analyze import auroc, brier
from study import draw_order, fit_platt, logit, smoothed_rate
from tasks import ROOT, TASKS, TEXT, load

OUT = os.path.join(ROOT, "results", "v1")
B = 2000
CS = [0.1, 1.0, 10.0, 100.0]
PRIOR_SD = 2.0
JEV_KS = [5, 10, 20, 50, 100, 200, 400]
TASK_ORDER = ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]

GEM_MODEL, GEM_BATCH = "gemini-3.8-flash", 25
_gdb = sqlite3.connect(os.path.join(ROOT, "data", "gemini_cache.sqlite"))


def gem_prompt(state, question, cases):  # identical to gemini_study.prompt (copied to avoid importing google-genai)
    listing = "\n\n".join(f"[{j + 1}] {c}" for j, c in enumerate(cases))
    return (f"{state}\n\n---\nFor each of the {len(cases)} new cases below, estimate the probability (0 to 100) "
            f"that the answer to this question is yes: {question}\n"
            f"Judge each case on its own. Reply with JSON only: "
            f'{{"probabilities": [one number per case, in order]}}\n\n{listing}')


def gem_cached(text):
    h = hashlib.sha256(f"{GEM_MODEL}|{text}".encode()).hexdigest()
    row = _gdb.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    if row is None:
        raise KeyError("Gemini cache miss (would need a live call)")
    return json.loads(row[0])["p"]


def gem_ks(name):
    return [5, 20, 100, 200 if name in TEXT else 400]


def platt_c(s_train, y, s_test, C):
    """study.fit_platt with a variable C; also returns the fitted slope sign (on standardized logit)."""
    if len(np.unique(y)) < 2:
        return np.full(len(s_test), smoothed_rate(y)), 0.0
    z, zt = logit(s_train), logit(s_test)
    mu, sd = z.mean(), z.std() + 1e-6
    lr = LogisticRegression(C=C).fit(((z - mu) / sd)[:, None], y)
    return lr.predict_proba(((zt - mu) / sd)[:, None])[:, 1], float(lr.coef_[0, 0])


def shift(s_train, y, s_test):
    z, zt = logit(s_train), logit(s_test)

    def loss(b):
        q = 1 / (1 + np.exp(-(z + b)))
        q = np.clip(q, 1e-12, 1 - 1e-12)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).sum() + b * b / (2 * PRIOR_SD ** 2)

    b = minimize_scalar(loss, bounds=(-15, 15), method="bounded").x
    return 1 / (1 + np.exp(-(zt + b))), float(b)


def corrections(s_train, y, s_test):
    out = {}
    for C in CS:
        out[f"platt_C{C:g}"] = platt_c(s_train, y, s_test, C)
    out["shift"] = shift(s_train, y, s_test)
    return out


def model_inputs(name):
    t = TASKS[name]
    pool, test, _ = load(name)
    y, pool_y = test.label.values.astype(int), pool.label.values.astype(int)
    # Jev: exactly study.py's calls, cache only
    jz_test = np.array(jev.noul_batch(t["context"], list(test.text), t["question"]))
    jz_pool = np.array(jev.noul_batch(t["context"], list(pool.text), t["question"]))
    P = pd.read_csv(f"{OUT}/pred_{name}.csv")
    assert np.allclose(P[P.arm == "zero_shot"].sort_values("i").p.values, jz_test)
    # Gemini: exactly gemini_platt.py's calls, cache only
    G = pd.read_csv(f"{OUT}/gem_{name}.csv")
    gz_test = G[G.arm == "zero_shot"].sort_values("i").p.values
    need = sorted({int(j) for d in [0, 1] for j in draw_order(name, d, len(pool))[:max(gem_ks(name))]})
    texts = list(pool.text.iloc[need])
    res = [gem_cached(gem_prompt(t["context"], t["question"], texts[i:i + GEM_BATCH]))
           for i in range(0, len(texts), GEM_BATCH)]
    gz = dict(zip(need, [v for r in res for v in r]))
    gz_pool = np.full(len(pool), np.nan)
    gz_pool[list(gz)] = list(gz.values())
    return dict(y=y, pool_y=pool_y, npool=len(pool),
                jev=dict(P=P, z_test=jz_test, z_pool=jz_pool, ks=JEV_KS),
                gemini=dict(P=G, z_test=gz_test, z_pool=gz_pool, ks=gem_ks(name)))


def run():
    rows = []
    for name in TASK_ORDER:
        D = model_inputs(name)
        y, n = D["y"], len(D["y"])
        rng = np.random.default_rng(1)
        I = rng.integers(0, n, (B, n))
        yb = y[I]
        for mod in ["jev", "gemini"]:
            M = D[mod]
            P = M["P"]
            for k in M["ks"]:
                ic = P[(P.arm == "in_context") & (P.k == k)]
                if mod == "gemini":
                    ic = ic[ic.draw.isin([0, 1])]
                if ic.empty:
                    continue
                ICm = ic.pivot(index="draw", columns="i", values="p")
                draws = list(ICm.index)
                preds = {"in_context": ICm.values}
                slopes, shifts = {}, []
                for d in draws:
                    idx = draw_order(name, d, D["npool"])[:k]
                    cs = corrections(M["z_pool"][idx], D["pool_y"][idx], M["z_test"])
                    for c, (p, par) in cs.items():
                        preds.setdefault(c, []).append(p)
                        if c.startswith("platt"):
                            slopes.setdefault(c, []).append(par)
                        else:
                            shifts.append(par)
                    stored = P[(P.arm == "platt") & (P.k == k) & (P.draw == d)].sort_values("i").p.values
                    assert np.allclose(stored, cs["platt_C1"][0]), (mod, name, k, d)
                preds = {c: np.asarray(v) for c, v in preds.items()}
                J = rng.integers(0, len(draws), (B, len(draws)))
                boot, point = {}, {}
                for c, Mx in preds.items():
                    for m, f in (("brier", brier), ("auroc", auroc)):
                        per = f(Mx[:, I], yb[None])
                        boot[(c, m)] = per[J.T, np.arange(B)].mean(0)
                        point[(c, m)] = f(Mx, y[None]).mean()
                corr = [c for c in preds if c != "in_context"]
                best = min(corr, key=lambda c: point[(c, "brier")])
                for c in corr + ["best"]:
                    cc = best if c == "best" else c
                    d_b = boot[("in_context", "brier")] - boot[(cc, "brier")]
                    d_a = boot[("in_context", "auroc")] - boot[(cc, "auroc")]
                    lo, hi = np.percentile(d_b, [2.5, 97.5])
                    alo, ahi = np.percentile(d_a, [2.5, 97.5])
                    rows.append(dict(
                        model=mod, task=name, k=k, n_draws=len(draws), correction=c, best_correction=best,
                        brier_corr=point[(cc, "brier")], brier_ic=point[("in_context", "brier")],
                        diff_brier_ic_minus_corr=point[("in_context", "brier")] - point[(cc, "brier")],
                        lo=lo, hi=hi,
                        verdict="in_context" if hi < 0 else ("correction" if lo > 0 else "tie"),
                        auroc_corr=point[(cc, "auroc")], auroc_ic=point[("in_context", "auroc")],
                        diff_auroc_ic_minus_corr=point[("in_context", "auroc")] - point[(cc, "auroc")],
                        auroc_lo=alo, auroc_hi=ahi,
                        n_neg_slope=int(np.sum(np.array(slopes[cc]) < 0)) if cc in slopes else np.nan,
                        n_single_class=int(sum(len(np.unique(D["pool_y"][draw_order(name, d, D["npool"])[:k]])) < 2
                                               for d in draws)),
                        mean_shift_b=float(np.mean(shifts)) if cc == "shift" else np.nan))
        print(name, "done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(f"{OUT}/platt_sweep.csv", index=False)
    return R


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    R = run()
    for mod in ["jev", "gemini"]:
        print(f"\n=== {mod}: Brier in_context - correction [95% CI]")
        S = R[R.model == mod].copy()
        S["cell"] = S.apply(lambda r: f"{r.diff_brier_ic_minus_corr:+.3f}[{r.lo:+.3f},{r.hi:+.3f}]"
                            + ("*" if r.verdict != "tie" else ""), axis=1)
        print(S.pivot_table(index=["task", "k"], columns="correction", values="cell", aggfunc="first")
              .reindex(columns=["platt_C0.1", "platt_C1", "platt_C10", "platt_C100", "shift", "best"]).to_string())
        print(S[S.correction == "best"].pivot_table(index="task", columns="k", values="best_correction", aggfunc="first").to_string())
