"""Skeptical re-check of the README's headline claims (cache-only, no API calls, deterministic).

Every number the README quotes from the statistical review comes from this script's output:
  results/verify_claims.csv   one row per comparison: estimate, 95% CI, two-sided p, Holm decision
  results/verify_claims.md    the same, grouped by claim, with per-family counts (uncorrected and after Holm)

Method (all claims):
  * Paired bootstrap, B=2000, rng seed 0 per task. Where a comparison has repeated draws, test cases and draws are
    resampled jointly (two-level, as analyze2.py); zero-shot is one run repeated for every draw.
    The base-rate checks (claims 3, 4, 6) are one run with no draws, so only test cases are resampled.
  * CI = 2.5th / 97.5th percentiles of the bootstrap difference. "sig" = CI excludes 0.
  * p = 2 * min(share of bootstrap differences <= 0, share >= 0), with the +1 correction, capped at 1.
  * Holm (alpha 0.05) across the tasks within one claim cell (the "family" column); "holm" = rejected.

Claims (numbering follows the review):
  1  GitHub AUROC: shuffled vs zero-shot, in-context vs zero-shot / shuffled, per k
  2  in-context vs shuffled AUROC at k=200, 8 tasks
  3  Jev zero-shot mean prediction vs realised test rate, 8 tasks
  4  base-rate sentence: told vs zero-shot, told vs a 100-label shift, share of the gap closed
  5  Gemini vs Jev Brier per arm and k
  6  told Gemini vs told Jev (Brier, AUROC); share of the zero-shot Gemini-Jev Brier gap the sentence closes
  7  cost per 1,000 decisions from the logged per-call costs
  8-11 synthetic tasks: LLMs vs base-rate-only / logistic regression, Opus k vs zero-shot, tellin under-prediction
  12 recipe: examples+Platt vs Platt / in-context at every k, stacking vs in-context, Stack Overflow ranking
  O  Opus vs Jev / Gemini (Brier, AUROC) at k=0 and k=100
Plus a count of all difference CIs in results/*.csv (for the multiple-comparisons caveat).

    python verify_claims.py
"""
import glob
import hashlib
import json
import os
import sqlite3

os.environ["JEV_OFFLINE"] = "1"  # jev.py: a cache miss raises instead of calling the API

import numpy as np
import pandas as pd

from analyze import auroc, brier
from tasks import ROOT, TASKS, TEXT, load

OUT = os.path.join(ROOT, "results")
B = 2000
TASK_ORDER = ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]
F = {"brier": brier, "auroc": auroc}
ROWS = []


def pval(d):
    n = len(d)
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (n + 1), (np.sum(d >= 0) + 1) / (n + 1)))


def holm(ps, alpha=0.05):
    ps = np.asarray(ps)
    rej = np.zeros(len(ps), bool)
    for r, i in enumerate(np.argsort(ps)):
        if ps[i] > alpha / (len(ps) - r):
            break
        rej[i] = True
    return rej


def add(claim, family, task, cmp, metric, est, d, k=None, a_val=np.nan, b_val=np.nan, note="", test=True):
    ROWS.append(dict(claim=claim, family=family, task=task, k=k, comparison=cmp, metric=metric, a_value=a_val,
                     b_value=b_val, est=est, lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5),
                     p=pval(d) if test else np.nan, note=note))


def two_level(mats, y, metric, seed=0):
    """mats: {name: draws x n}. Returns {name: (point, bootstrap vector)} with shared resamples."""
    n, nd = len(y), next(iter(mats.values())).shape[0]
    rng = np.random.default_rng(seed)
    I = rng.integers(0, n, (B, n))
    J = rng.integers(0, nd, (B, nd))
    f, out = F[metric], {}
    for a, M in mats.items():
        per = f(M[:, I], y[I][None])
        out[a] = (f(M, y[None]).mean(), per[J.T, np.arange(B)].mean(0))
    return out


class Jev:
    """Jev predictions for one task; draws = those where in_context exists at k (as analyze2.py)."""

    def __init__(self, task, extra=None):
        self.y = pd.read_csv(f"{OUT}/labels_{task}.csv").sort_values("i").label.values.astype(int)
        self.P = pd.read_csv(f"{OUT}/pred_{task}.csv")
        if extra is not None:
            self.P = pd.concat([self.P, extra])

    def mats(self, k, arms, intersect=False):
        """intersect=True: use only draws present for every arm (else skip the task if an arm lacks a draw)."""
        P = self.P
        draws = sorted(P[(P.arm == "in_context") & (P.k == k)].draw.unique())
        if intersect:
            for a in arms:
                if a != "zero_shot":
                    draws = [d for d in draws if d in set(P[(P.arm == a) & (P.k == k)].draw)]
        out = {}
        for a in arms:
            if a == "zero_shot":
                out[a] = np.repeat(P[P.arm == "zero_shot"].sort_values("i").p.values[None], len(draws), 0)
            else:
                g = P[(P.arm == a) & (P.k == k)].pivot(index="draw", columns="i", values="p")
                if not set(draws) <= set(g.index):
                    return draws, None
                out[a] = g.loc[draws].values
        return draws, out


def diff(bs, a, b):
    return bs[a][0] - bs[b][0], bs[a][1] - bs[b][1]


# ---------------------------------------------------------------- claims 1-2: labels vs shuffled labels
def claims_1_2():
    T = Jev("github_prs")
    for k in [5, 10, 20, 50, 100, 200]:
        _, M = T.mats(k, ["zero_shot", "in_context", "shuffled"])
        bs = two_level(M, T.y, "auroc")
        for a, b in [("shuffled", "zero_shot"), ("in_context", "zero_shot"), ("in_context", "shuffled")]:
            add("1", f"github {a}-{b}", "github_prs", f"{a} - {b}", "auroc", *diff(bs, a, b), k=k,
                a_val=bs[a][0], b_val=bs[b][0])
    for t in TASK_ORDER:
        T = Jev(t)
        dr, M = T.mats(200, ["zero_shot", "in_context", "shuffled"])
        bs = two_level(M, T.y, "auroc")
        for a, b in [("in_context", "shuffled"), ("shuffled", "zero_shot")]:
            add("2", f"{a}-{b} k=200", t, f"{a} - {b}", "auroc", *diff(bs, a, b), k=200,
                a_val=bs[a][0], b_val=bs[b][0], note=f"{len(dr)} draws")


# ---------------------------------------------------------------- claim 3: zero-shot mean vs base rate
def claim_3():
    for t in TASK_ORDER:
        y = pd.read_csv(f"{OUT}/labels_{t}.csv").sort_values("i").label.values
        P = pd.read_csv(f"{OUT}/pred_{t}.csv")
        p = P[P.arm == "zero_shot"].sort_values("i").p.values
        I = np.random.default_rng(0).integers(0, len(y), (B, len(y)))
        add("3", "zs mean - rate", t, "Jev zero-shot mean - realised rate", "mean", p.mean() - y.mean(),
            p[I].mean(1) - y[I].mean(1), k=0, a_val=p.mean(), b_val=y.mean(), note="test cases only")


# ---------------------------------------------------------------- claims 4, 6: the base-rate sentence
OFF = ["nyc311", "crimes", "github_prs", "hn", "stackoverflow", "inspections"]  # Jev's zero-shot mean was off


def claims_4_6():
    import baserate_ci as bc  # rebuilds predictions from the caches (a cache miss raises)
    for t in bc.TASK_ORDER:
        y, P = bc.predictions(t)
        bc.check(t, y, P)
        n = len(y)
        I = np.random.default_rng(0).integers(0, n, (B, n))
        yb = y[I]
        b = {m: brier(p[I], yb) for m, p in P.items()}
        a = {m: auroc(p[I], yb) for m, p in P.items()}
        pb = {m: brier(p, y) for m, p in P.items()}
        pa = {m: auroc(p, y) for m, p in P.items()}
        fam = "off" if t in OFF else "on"
        add("4", f"told-zs ({fam})", t, "Jev told - Jev zero-shot", "brier", pb["jev_told"] - pb["jev_zs"],
            b["jev_told"] - b["jev_zs"], a_val=pb["jev_told"], b_val=pb["jev_zs"], note="test cases only")
        add("4", f"told-shift100 ({fam})", t, "Jev told - Jev shift(100 labels)", "brier",
            pb["jev_told"] - pb["jev_shift100"], b["jev_told"] - b["jev_shift100"], a_val=pb["jev_told"],
            b_val=pb["jev_shift100"], note="test cases only")
        if t in OFF:  # share of the zero-shot -> shift gap closed by the sentence (not a test against 0)
            fr = (b["jev_zs"] - b["jev_told"]) / (b["jev_zs"] - b["jev_shift100"])
            add("4", "share closed (zs->shift)", t, "share of Jev zs->shift Brier gap closed by told", "share",
                (pb["jev_zs"] - pb["jev_told"]) / (pb["jev_zs"] - pb["jev_shift100"]), fr, note="ratio, not a test", test=False)
            sh = 1 - (b["gem_told"] - b["jev_told"]) / (b["gem_zs"] - b["jev_zs"])
            add("6", "share of Gemini-Jev gap closed", t, "share of zero-shot Gemini-Jev Brier gap closed by telling Jev",
                "share", 1 - (pb["gem_told"] - pb["jev_told"]) / (pb["gem_zs"] - pb["jev_zs"]), sh,
                note="ratio, not a test", test=False)
        add("6", "gem_told-jev_told brier", t, "Gemini told - Jev told", "brier", pb["gem_told"] - pb["jev_told"],
            b["gem_told"] - b["jev_told"], a_val=pb["gem_told"], b_val=pb["jev_told"], note="test cases only")
        add("6", "gem_told-jev_told auroc", t, "Gemini told - Jev told", "auroc", pa["gem_told"] - pa["jev_told"],
            a["gem_told"] - a["jev_told"], a_val=pa["gem_told"], b_val=pa["jev_told"], note="test cases only")


# ---------------------------------------------------------------- claim 5: Gemini vs Jev Brier
def claim_5():
    for t in TASK_ORDER:
        y = load(t)[1].label.values.astype(int)
        n = len(y)
        rng = np.random.default_rng(0)
        I, J = rng.integers(0, n, (B, n)), rng.integers(0, 2, (B, 2))
        yb = y[I]
        mods = {"jev": pd.read_csv(f"{OUT}/pred_{t}.csv"), "gemini": pd.read_csv(f"{OUT}/gem_{t}.csv")}
        kmax = 200 if t in TEXT else 400
        for k in [0, 5, 20, 100, kmax]:
            for arm in (["zero_shot"] if k == 0 else ["in_context", "platt"]):
                bs = {}
                for m, P in mods.items():
                    g = P[(P.arm == arm) & (P.k == k)]
                    if arm == "zero_shot":
                        M = np.repeat(g.sort_values("i").p.values[None], 2, 0)
                    else:
                        M = g[g.draw.isin([0, 1])].pivot(index="draw", columns="i", values="p").values
                    per = brier(M[:, I], yb[None])
                    bs[m] = (brier(M, y[None]).mean(), per[J.T, np.arange(B)].mean(0))
                kl = k if k in (0, 5, 20, 100) else "max"
                add("5", f"{arm} k={kl}", t, "Gemini - Jev", "brier", *diff(bs, "gemini", "jev"), k=k,
                    a_val=bs["gemini"][0], b_val=bs["jev"][0], note="draws 0-1; platt = unconstrained")


# ---------------------------------------------------------------- claim 7: cost
def claim_7():
    import gemini_study as g
    import jev
    import opus_study as o
    from study import draw_order, state_examples
    assert jev.OFFLINE
    jdb, gdb, odb = (sqlite3.connect(f"{ROOT}/data/{m}_cache.sqlite") for m in ("jev", "gemini", "opus"))

    def jev_cost(t, state, cases):
        q_tok = max(jev.est_tokens(f"Case: {c}\nQuestion: {t['question']}") for c in cases) + 5
        bs = min(25, int((jev.MAX_TOKENS - 230 - jev.est_tokens(state)) // q_tok))
        tot = 0
        for i in range(0, len(cases), bs):
            qs = {f"c{j}": {"type": "noul", "instructions": f"Case: {cases[j]}\nQuestion: {t['question']}"}
                  for j in range(i, min(i + bs, len(cases)))}
            body = {"model": jev.MODEL, "state": state, "questions": qs}
            h = hashlib.sha256((json.dumps(body, sort_keys=True) + "|rep=0").encode()).hexdigest()
            r = jdb.execute("select response from calls where h=?", (h,)).fetchone()
            assert r is not None, "Jev cache miss"
            tot += (json.loads(r[0]).get("usage", {}) or {}).get("cost", 0) or 0
        return tot

    def gem_cost(t, state, cases):
        tot = 0
        for i in range(0, len(cases), 25):
            h = hashlib.sha256(f"{g.MODEL}|{g.prompt(state, t['question'], cases[i:i + 25])}".encode()).hexdigest()
            r = gdb.execute("select response from calls where h=?", (h,)).fetchone()
            assert r is not None, "Gemini cache miss"
            tot += json.loads(r[0]).get("cost", 0)
        return tot

    def opus_cost(t, state, cases):
        tot = 0
        for i in range(0, len(cases), 25):
            h = hashlib.sha256(f"{o.MODEL}|{state}|{o.tail(t['question'], cases[i:i + 25])}".encode()).hexdigest()
            r = odb.execute("select response from calls where h=?", (h,)).fetchone()
            assert r is not None, "Opus cache miss"
            tot += json.loads(r[0]).get("cost", 0)
        return tot

    rows = []
    for name in TASK_ORDER:
        t = TASKS[name]
        pool, test, _ = load(name)
        cases = list(test.text)
        states = [("zero_shot", 0, t["context"])]
        for d in [0, 1]:
            ex = pool.iloc[draw_order(name, d, len(pool))[:100]]
            states.append(("in_context_k100", d, state_examples(t, list(ex.text), ex.label.values)))
        for arm, d, st in states:
            for m, f in [("jev", jev_cost), ("gemini", gem_cost), ("opus", opus_cost)]:
                rows.append(dict(task=name, arm=arm, draw=d, model=m, per1000=f(t, st, cases) / len(cases) * 1000))
    A = pd.DataFrame(rows).groupby(["arm", "model"]).per1000.mean().unstack()
    for arm in A.index:
        for m in A.columns:
            ROWS.append(dict(claim="7", family=f"cost {arm}", task="mean of 8", k=100 if "100" in arm else 0,
                             comparison=f"{m} $ per 1,000 decisions (ratio = x Jev)", metric="usd", a_value=A.loc[arm, m],
                             b_value=A.loc[arm, "jev"], est=A.loc[arm, m] / A.loc[arm, "jev"], lo=np.nan, hi=np.nan,
                             p=np.nan, note="logged per-call cost, 300 test cases, draws 0-1"))


# ---------------------------------------------------------------- claims 8-11: synthetic tasks
def claims_8_11():
    import synthetic_tasks as S
    DR = [0, 1]

    def mat(P, m, k):
        g = P[(P.model == m) & (P.k == k)]
        if g.empty:
            return None
        if k == 0:
            return np.repeat(g.sort_values("i").p.values[None], 2, 0)
        return g[g.draw.isin(DR)].pivot(index="draw", columns="i", values="p").values

    for task in S.ORDER:
        P = pd.read_csv(f"{OUT}/synth_{task}.csv")
        test = S.load(task)[1]
        tp = test.p_true.values
        n = len(tp)
        rng = np.random.default_rng(0)
        I, J = rng.integers(0, n, (B, n)), rng.integers(0, 2, (B, 2))
        bs = {}
        for m in ["jev", "gemini", "opus", "logreg", "tabpfn", "base_rate"]:
            for k in [0, 5, 20, 100, 400]:
                M = mat(P, m, k)
                if M is not None:
                    per = ((M[:, I] - tp[I][None]) ** 2).mean(-1)
                    bs[(m, k)] = (((M - tp) ** 2).mean(), per[J.T, np.arange(B)].mean(0))

        def D(claim, a, b, fam):
            add(claim, fam, task, f"{a[0]}@{a[1]} - {b[0]}@{b[1]}", "mse_truth", bs[a][0] - bs[b][0],
                bs[a][1] - bs[b][1], k=a[1], a_val=bs[a][0], b_val=bs[b][0],
                note=f"ratio b/a = {bs[b][0] / bs[a][0]:.2f}")

        for m in ["jev", "gemini"]:
            D("8", (m, 400), ("base_rate", 400), f"{m}-base_rate k=400")
            D("8", ("logreg", 400), (m, 400), f"logreg-{m} k=400")
            D("9", (m, 100), ("base_rate", 100), f"{m}-base_rate k=100")
        D("9", ("opus", 100), ("base_rate", 100), "opus-base_rate k=100")
        D("9", ("opus", 100), ("logreg", 100), "opus-logreg k=100")
        D("9", ("opus", 100), ("tabpfn", 100), "opus-tabpfn k=100")
        for k in [5, 20, 100]:
            D("10", ("opus", k), ("opus", 0), f"opus k={k} - k=0")

        if task == "kellan":  # claim 11: tellin (raises risk) under-prediction
            tr = test.treatment.values
            idx, idn = np.where(tr == "treated with tellin")[0], np.where(tr == "no treatment")[0]
            for m, k in [("opus", 0), ("opus", 100), ("jev", 100), ("gemini", 100), ("jev", 400), ("gemini", 400)]:
                M = mat(P, m, k)
                r2 = np.random.default_rng(1)
                Ib, In, Jb = (r2.integers(0, len(idx), (B, len(idx))), r2.integers(0, len(idn), (B, len(idn))),
                              r2.integers(0, 2, (B, 2)))
                under = (M[:, idx][:, Ib] - tp[idx][Ib][None]).mean(-1)[Jb.T, np.arange(B)].mean(0)
                add("11", f"tellin under-prediction k={k}", "kellan", f"{m}: mean pred - mean true (tellin cases)",
                    "mean", M[:, idx].mean() - tp[idx].mean(), under, k=k, a_val=M[:, idx].mean(),
                    b_val=tp[idx].mean())
                g1 = M[:, idx][:, Ib].mean(-1) - M[:, idn][:, In].mean(-1)
                tg = tp[idx][Ib].mean(-1) - tp[idn][In].mean(-1)
                gap = (g1 - tg[None])[Jb.T, np.arange(B)].mean(0)
                pg, tgp = M[:, idx].mean() - M[:, idn].mean(), tp[idx].mean() - tp[idn].mean()
                add("11", f"tellin effect-size error k={k}", "kellan",
                    f"{m}: predicted (tellin - none) gap - true gap", "mean", pg - tgp, gap, k=k, a_val=pg, b_val=tgp)


# ---------------------------------------------------------------- claim 12: recipe / stacking
def claim_12():
    for t in TASK_ORDER:
        T = Jev(t, extra=pd.read_csv(f"{OUT}/stack_{t}.csv"))
        for k in [5, 10, 20, 50, 100, 200, 400]:
            dr, M = T.mats(k, ["platt", "both", "in_context"], intersect=True)
            if M is None or not dr:
                continue
            bs = two_level(M, T.y, "brier")
            add("12", f"both-platt k={k}", t, "examples+Platt - Platt", "brier", *diff(bs, "both", "platt"), k=k,
                a_val=bs["both"][0], b_val=bs["platt"][0])
            if k < 50:
                continue
            dr, M = T.mats(k, ["both", "in_context", "stack_tabpfn"])
            if M is None:
                continue
            bs = two_level(M, T.y, "brier")
            for a, b in [("stack_tabpfn", "in_context"), ("stack_tabpfn", "both")]:
                add("12", f"{a}-{b} k={k}", t, f"{a} - {b}", "brier", *diff(bs, a, b), k=k, a_val=bs[a][0],
                    b_val=bs[b][0])
    T = Jev("stackoverflow", extra=pd.read_csv(f"{OUT}/stack_stackoverflow.csv"))
    for k in [5, 10, 20, 50, 100, 200]:
        _, M = T.mats(k, ["in_context", "stack_tabpfn", "stack_lr"])
        bs = two_level(M, T.y, "auroc")
        for a in ["stack_tabpfn", "stack_lr"]:
            add("12", f"SO {a}-in_context auroc", "stackoverflow", f"{a} - in_context", "auroc",
                *diff(bs, a, "in_context"), k=k, a_val=bs[a][0], b_val=bs["in_context"][0])


# ---------------------------------------------------------------- Opus vs Jev / Gemini
def claim_opus():
    import compare_opus as co
    for task in co.TASK_ORDER:
        y = load(task)[1].label.values.astype(int)
        n = len(y)
        rng = np.random.default_rng(0)
        I, J = rng.integers(0, n, (B, n)), rng.integers(0, 2, (B, 2))
        yb = y[I]
        for arm, k in [("zero_shot", 0), ("in_context", 100), ("platt", 100)]:
            for met, f in co.METRICS.items():
                if met == "auroc" and arm == "platt":
                    continue
                bs = {}
                for m in co.MODELS:
                    M = co.matrix(task, m, arm, k)
                    bs[m] = (f(M, y[None]).mean(), f(M[:, I], yb[None])[J.T, np.arange(B)].mean(0))
                for o in ["Jev", "Gemini"]:
                    add("O", f"{arm} k={k} {met} Opus-{o}", task, f"Opus - {o}", met, *diff(bs, "Opus", o), k=k,
                        a_val=bs["Opus"][0], b_val=bs[o][0], note="platt = constrained")


# ---------------------------------------------------------------- CI count for the multiple-comparisons caveat
def ci_count():
    """Difference CIs (not single-value CIs) in the current results CSVs, plus this script's own."""
    n = 0
    for f, kinds in [("diffs2", None), ("compare_ci", ["gemini_minus_jev", "ic_minus_platt"]),
                     ("compare_opus_ci", ["opus_minus_jev", "opus_minus_gemini"]),
                     ("synth_ci", ["diff_mse", "diff_auroc", "diff_brier"]), ("baserate_ci", None),
                     ("platt_sweep", None)]:
        d = pd.read_csv(f"{OUT}/{f}.csv")
        if kinds:
            d = d[d.kind.isin(kinds)]
        n += int(d.lo.notna().sum())
    n += 2 * len(pd.read_csv(f"{OUT}/ensemble_ci.csv")) + 2 * len(pd.read_csv(f"{OUT}/qwen_vs_jev.csv"))
    return n


def summarize(R):
    R["sig"] = R.p.notna() & ((R.lo > 0) | (R.hi < 0))
    R["holm"] = False
    for _, g in R[R.p.notna()].groupby(["claim", "family"]):
        R.loc[g.index, "holm"] = holm(g.p.values)
    return R


def fmt(x, nd):
    return "" if pd.isna(x) else f"{x:+.{nd}f}"


def write_md(R, n_ci):
    L = ["# Skeptical re-check of the README's claims", "",
         f"Generated by `verify_claims.py` (cache-only). Paired bootstrap, B={B}, seed 0 per task; test cases and "
         "draws resampled jointly where there are draws (base-rate checks: test cases only). "
         "p = 2 × min tail share; Holm across tasks within each family. sig = 95% CI excludes 0.", "",
         f"Difference CIs across `results/*.csv` (diffs2, compare_ci, compare_opus_ci, synth_ci, baserate_ci, "
         f"platt_sweep, ensemble_ci, qwen_vs_jev) plus this file: **{n_ci + int(R.p.notna().sum())}**, "
         "none corrected for multiplicity except the Holm columns here.", "",
         "## Counts per family (tests only)", "",
         "| claim | family | n | sig, a < b | sig, a > b | after Holm, a < b | after Holm, a > b | sig tasks | Holm tasks |",
         "|---|---|---|---|---|---|---|---|---|"]
    T = R[R.p.notna()]
    for (c, fam), g in T.groupby(["claim", "family"], sort=False):
        neg, pos = g.sig & (g.est < 0), g.sig & (g.est > 0)
        hn, hp = g.holm & (g.est < 0), g.holm & (g.est > 0)
        lab = g.task if g.task.nunique() > 1 else "k=" + g.k.astype(int).astype(str)  # one task: list the k values
        L.append(f"| {c} | {fam} | {len(g)} | {neg.sum()} | {pos.sum()} | {hn.sum()} | {hp.sum()} | "
                 f"{','.join(lab[g.sig]) if len(g) > 1 else ''} | {','.join(lab[g.holm]) if len(g) > 1 else ''} |")
    L += ["", "## All rows", "",
          "| claim | family | task | k | comparison | a | b | est [95% CI] | p | sig | Holm |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        nd = 2 if r.metric in ("share", "usd") else 4
        ci = "" if pd.isna(r.lo) else f" [{fmt(r.lo, nd)}, {fmt(r.hi, nd)}]"
        p = "" if pd.isna(r.p) else f"{r.p:.4f}"
        k = "" if pd.isna(r.k) else int(r.k)
        L.append(f"| {r.claim} | {r.family} | {r.task} | {k} | {r.comparison} | {r.a_value:.4f} | {r.b_value:.4f} | "
                 f"{fmt(r.est, nd)}{ci} | {p} | {'yes' if r.sig else ''} | {'yes' if r.holm else ''} |")
    L.append("")
    open(f"{OUT}/verify_claims.md", "w").write("\n".join(L))


def main():
    claims_1_2()
    claim_3()
    claims_4_6()
    claim_5()
    claim_7()
    claims_8_11()
    claim_12()
    claim_opus()
    R = summarize(pd.DataFrame(ROWS))
    R["B"] = B
    R.to_csv(f"{OUT}/verify_claims.csv", index=False)
    write_md(R, ci_count())
    print(f"{len(R)} rows; wrote results/verify_claims.csv, results/verify_claims.md")


if __name__ == "__main__":
    main()
