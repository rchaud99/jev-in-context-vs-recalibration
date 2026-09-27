"""Synthetic binary tasks with fictional content and known per-case true probabilities.

All tasks share the same 1,500 generated cases (features), so zero-shot prompts are identical where the wording is
identical, and every task uses the same nested example draws: rows 0-299 are the fixed test set, rows 300+ the pool.
Labels are drawn with common random numbers (label = u < p_true, same u per case), so kellan / kellan_br10 /
kellan_br90 / neutral differ only through p_true (and neutral has exactly kellan's labels).

  kellan       additive logistic model on 5 fictional categorical features, base rate ~0.3
  interaction  outcome driven mostly by rash x treatment (morvax works only for a deep red rash)
  noise        p_true = 0.3 for every case (labels independent of features)
  kellan_br10  kellan's effects with the intercept shifted to a base rate ~0.1
  kellan_br90  ... ~0.9
  neutral      kellan's cases and labels, with every name replaced by "attribute A = level 3" style codes

    python synthetic_tasks.py      # writes data/synthetic/<task>.csv and prints base rates / true AUROC
Only numpy/pandas: importable from the TabPFN venv.
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "synthetic")
N, N_TEST, SEED = 1500, 300, 20260927
FEATS = ["age", "region", "rash", "onset", "treatment"]

LEVELS = {  # level: (sampling weight, additive logit effect in the kellan model)
    "age": {"under 18": (0.20, -0.5), "18-40": (0.30, -0.35), "41-65": (0.30, 0.25), "over 65": (0.20, 1.0)},
    "region": {"Varnholt": (0.30, 0.0), "Eskerby": (0.25, 0.7), "Lune Valley": (0.25, -0.5), "Corrimar": (0.20, 0.1)},
    "rash": {"no rash": (0.25, -0.7), "pale pink rash": (0.30, -0.25), "deep red rash": (0.25, 0.45),
             "violet rash": (0.20, 1.0)},
    "onset": {"sudden onset": (0.45, 0.35), "gradual onset": (0.55, -0.3)},
    # tellin (a treatment) raises risk: a realistic-sounding name whose true effect runs against the "treatment helps" prior
    "treatment": {"no treatment": (0.35, 0.0), "treated with morvax": (0.35, -0.9), "treated with tellin": (0.30, 0.55)},
}
# interaction model: logit = b0 + small age effect + RASH_TX[rash][treatment]
RASH_TX = {"no rash": [0.0, 1.6, 0.0], "pale pink rash": [0.0, 1.6, 0.0],
           "deep red rash": [1.6, -3.0, 1.6], "violet rash": [0.2, 1.6, 0.2]}  # cols: none, morvax, tellin

KELLAN = dict(
    context="Patients diagnosed with Kellan fever at clinics in the Varn Isles in 2026.",
    question="Will this patient be hospitalized within 14 days of diagnosis?",
    yes="was hospitalized within 14 days", no="was not hospitalized within 14 days",
)
NEUTRAL = dict(
    context="Records from an anonymized dataset. Each record has five coded attributes (A to E).",
    question="Does this record have outcome Y?",
    yes="the record has outcome Y", no="the record does not have outcome Y",
)
TASKS = {"kellan": KELLAN, "interaction": KELLAN, "noise": KELLAN, "kellan_br10": KELLAN, "kellan_br90": KELLAN,
         "neutral": NEUTRAL}
ORDER = ["kellan", "interaction", "noise", "kellan_br10", "kellan_br90", "neutral"]
DRAW_KEY = "kellan"  # every task uses the same example rows per draw (paired across tasks)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def features():
    rng = np.random.default_rng(SEED)
    X = pd.DataFrame({f: rng.choice(list(LEVELS[f]), N, p=[w for w, _ in LEVELS[f].values()]) for f in FEATS})
    u = rng.random(N)  # common random numbers for the labels
    return X, u


def intercept_for(z, rate):
    """b0 such that mean(sigmoid(b0 + z)) == rate over the generated cases (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigmoid(mid + z).mean() < rate else (lo, mid)
    return (lo + hi) / 2


def kellan_text(r):
    return f"Kellan fever patient aged {r.age}, from {r.region}; {r.rash}, {r.onset}; {r.treatment}."


NEUTRAL_CODE = {f: {lvl: j + 1 for j, lvl in enumerate(LEVELS[f])} for f in FEATS}


def neutral_text(r):
    parts = [f"attribute {'ABCDE'[j]} = level {NEUTRAL_CODE[f][r[f]]}" for j, f in enumerate(FEATS)]
    return "Record with " + ", ".join(parts) + "."


def build():
    X, u = features()
    z_add = sum(X[f].map({k: v[1] for k, v in LEVELS[f].items()}).values for f in FEATS)
    tx_col = X.treatment.map({"no treatment": 0, "treated with morvax": 1, "treated with tellin": 2}).values
    z_int = np.array([RASH_TX[r][c] for r, c in zip(X.rash, tx_col)]) \
        + 0.3 * X.age.map({k: v[1] for k, v in LEVELS["age"].items()}).values
    p = {
        "kellan": sigmoid(intercept_for(z_add, 0.3) + z_add),
        "interaction": sigmoid(intercept_for(z_int, 0.3) + z_int),
        "noise": np.full(N, 0.3),
        "kellan_br10": sigmoid(intercept_for(z_add, 0.1) + z_add),
        "kellan_br90": sigmoid(intercept_for(z_add, 0.9) + z_add),
    }
    p["neutral"] = p["kellan"]
    out = {}
    for name in ORDER:
        df = X.copy()
        if name == "neutral":
            for f in FEATS:
                df[f] = df[f].map(lambda v, f=f: f"level {NEUTRAL_CODE[f][v]}")
            df["text"] = X.apply(neutral_text, axis=1)
        else:
            df["text"] = X.apply(kellan_text, axis=1)
        df["p_true"] = p[name]
        df["label"] = (u < p[name]).astype(int)
        out[name] = df
    return out


def load(task):
    """Returns (pool, test, feats), like tasks.load."""
    df = pd.read_csv(os.path.join(DATA, f"{task}.csv"))
    return df.iloc[N_TEST:].reset_index(drop=True), df.iloc[:N_TEST].reset_index(drop=True), FEATS


def draw_order(draw, n):
    """Same scheme as study.draw_order, keyed on DRAW_KEY for every synthetic task."""
    seed = [ord(c) for c in DRAW_KEY] + [draw]
    return np.random.default_rng(seed).permutation(n)


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    os.makedirs(DATA, exist_ok=True)
    for name, df in build().items():
        df.to_csv(os.path.join(DATA, f"{name}.csv"), index=False)
        te = df.iloc[:N_TEST]
        auc = roc_auc_score(te.label, te.p_true) if te.p_true.nunique() > 1 else float("nan")
        auc_all = roc_auc_score(df.label, df.p_true) if df.p_true.nunique() > 1 else float("nan")
        print(f"{name:12} base rate {df.label.mean():.3f} (test {te.label.mean():.3f}, mean p_true {df.p_true.mean():.3f})"
              f"  true AUROC test {auc:.3f} / all {auc_all:.3f}  p_true range {df.p_true.min():.2f}-{df.p_true.max():.2f}"
              f"  text ~{df.text.str.len().mean():.0f} chars")
    df = build()["interaction"]
    print("\ninteraction: mean p_true by rash x treatment (marginals should be weak)")
    print(df.pivot_table(index="rash", columns="treatment", values="p_true", aggfunc="mean").round(2))
    print(df.groupby("treatment").p_true.mean().round(3), df.groupby("rash").p_true.mean().round(3))
    print("\ne.g.", build()["kellan"].text[0], "|", build()["neutral"].text[0])
