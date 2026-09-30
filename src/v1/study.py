"""Main study: generate held-out predictions for every arm.

For each task, k in KS and draw in DRAWS, the k labeled examples are the first k rows of a
fixed per-(task, draw) permutation of the pool (so draws are nested across k). Every arm uses
exactly those k labels and is scored on the same 300 held-out test cases.

Arms
  zero_shot     Jev, no labels
  in_context    Jev with the k labeled examples in its state
  shuffled      same examples, labels permuted among them (keeps the base rate)
  noisy         same examples, 20% of labels flipped (annotation noise)
  fmt_compact   in_context with a one-line "case => YES/NO" format
  fmt_json      in_context with the examples as a JSON list
  self_feedback examples + Jev's own zero-shot estimate for each + the outcome
  platt         2-parameter logistic fit of labels on logit(Jev zero-shot p)
  isotonic      isotonic fit of labels on Jev zero-shot p
  both          in_context, then Platt fit on leave-one-out (k<=100) / 10-fold (k>100) in-context
                predictions of the k examples
  base_rate     smoothed mean of the k labels
  logreg        one-hot logistic regression on the k labels (no Jev)
  (TabPFN is produced separately by baselines_tabpfn.py)

Output: results/v1/pred_<task>.csv (task, k, draw, arm, i, p) and results/v1/labels_<task>.csv
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

import jev
from tasks import ROOT, TASKS, load

KS = [5, 10, 20, 50, 100, 200, 400]
DRAWS = range(10)
NOISE = 0.2  # fraction of example labels flipped in the `noisy` arm
OUT = os.path.join(ROOT, "results", "v1")


def draw_order(task, draw, n):
    seed = [ord(c) for c in task] + [draw]
    return np.random.default_rng(seed).permutation(n)


def prob_words(p):
    for cut, w in [(0.1, "very unlikely"), (0.3, "unlikely"), (0.45, "somewhat unlikely"), (0.55, "toss-up"),
                   (0.7, "somewhat likely"), (0.9, "likely"), (1.01, "very likely")]:
        if p < cut:
            return f"{round(p * 100)}% ({w})"


def outcome(t, y):
    return f"yes, {t['yes']}" if y else f"no, {t['no']}"


def state_examples(t, texts, labels, estimates=None):
    if estimates is None:
        head = (f"Below are past cases from this same setting with their actual outcomes. "
                f"For each case the question was: {t['question']}")
        blocks = [f"Case: {x}\nActual outcome: {outcome(t, y)}" for x, y in zip(texts, labels)]
    else:
        head = (f"Below are past cases from this same setting. For each case the question was: {t['question']} "
                f"Each shows the probability of yes that you previously estimated, and the actual outcome. "
                f"Use this track record to judge how reliable your estimates are.")
        blocks = [f"Case: {x}\nYour earlier estimate: {prob_words(e)} chance of yes\nActual outcome: {outcome(t, y)}"
                  for x, y, e in zip(texts, labels, estimates)]
    return f"{t['context']}\n\n{head}\n\n" + "\n\n".join(blocks)


def state_compact(t, texts, labels):
    lines = [f"{x} => {'YES' if y else 'NO'}" for x, y in zip(texts, labels)]
    return (f"{t['context']}\n\nLabeled past cases. Question: {t['question']} "
            f"Each line is one case followed by its actual answer (=> YES or => NO).\n\n" + "\n".join(lines))


def state_json(t, texts, labels):
    import json
    rows = [{"case": x, "answer": "yes" if y else "no"} for x, y in zip(texts, labels)]
    return (f"{t['context']}\n\nLabeled past cases for the question \"{t['question']}\", as JSON:\n"
            + json.dumps(rows, ensure_ascii=False))


def ask(t, state, cases):
    """Jev Noul for each case under `state`, with the batch size shrunk so the request fits in context.
    Returns None if even one question does not fit."""
    q_tok = max(jev.est_tokens(f"Case: {c}\nQuestion: {t['question']}") for c in cases) + 5
    room = jev.MAX_TOKENS - 230 - jev.est_tokens(state)
    bs = min(25, int(room // q_tok))
    if bs < 1:
        return None
    try:
        return np.array(jev.noul_batch(state, cases, t["question"], batch_size=bs))
    except jev.CacheMiss:
        return "missing"
    except RuntimeError as e:  # e.g. the token estimate was slightly off and the request overflowed
        print("  request failed:", str(e)[:200], flush=True)
        return None


def logit(p):
    p = np.clip(np.asarray(p, float), 0.005, 0.995)
    return np.log(p / (1 - p))


def smoothed_rate(y):
    return (np.sum(y) + 0.5) / (len(y) + 1)


def fit_platt(score_train, y, score_test):
    """Platt scaling on logit scores (standardized on the k training points, mild L2)."""
    if len(np.unique(y)) < 2:
        return np.full(len(score_test), smoothed_rate(y))
    z, zt = logit(score_train), logit(score_test)
    mu, sd = z.mean(), z.std() + 1e-6
    lr = LogisticRegression(C=1.0).fit(((z - mu) / sd)[:, None], y)
    return lr.predict_proba(((zt - mu) / sd)[:, None])[:, 1]


def fit_isotonic(score_train, y, score_test):
    if len(np.unique(y)) < 2:
        return np.full(len(score_test), smoothed_rate(y))
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip", y_min=0, y_max=1)
    return iso.fit(score_train, y).predict(score_test)


def fit_logreg(Xtr, y, Xte):
    if len(np.unique(y)) < 2:
        return np.full(len(Xte), smoothed_rate(y))
    m = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(C=1.0, max_iter=2000))
    return m.fit(Xtr, y).predict_proba(Xte)[:, 1]


def loo_in_context(t, texts, labels):
    """Jev's in-context prediction for each example, with that example left out of the state
    (leave-one-out up to k=100; 10-fold cross-fitting above that, to keep the cost linear)."""
    k = len(texts)
    if k > 100:
        folds = np.arange(k) % 10
        out = np.zeros(k)
        for f in range(10):
            keep = np.where(folds != f)[0]
            held = np.where(folds == f)[0]
            st = state_examples(t, [texts[j] for j in keep], [labels[j] for j in keep])
            p = ask(t, st, [texts[j] for j in held])
            if p is None or isinstance(p, str):
                return None
            out[held] = p
        return out

    def one(i):
        keep = [j for j in range(len(texts)) if j != i]
        st = state_examples(t, [texts[j] for j in keep], [labels[j] for j in keep])
        try:
            return jev.noul_batch(st, [texts[i]], t["question"], batch_size=1, workers=1)[0]
        except jev.CacheMiss:
            return None
    with ThreadPoolExecutor(16) as ex:
        out = list(ex.map(one, range(len(texts))))
    return None if any(v is None for v in out) else np.array(out)


def run_task(name):
    t = TASKS[name]
    pool, test, feats = load(name)
    test_texts = list(test.text)
    rows = []

    def add(k, d, arm, p):
        rows.extend((name, k, d, arm, i, float(v)) for i, v in enumerate(p))

    # Jev zero-shot on every pool and test case (inputs to Platt / isotonic / self-feedback)
    zs_test = np.array(jev.noul_batch(t["context"], test_texts, t["question"]))
    zs_pool = np.array(jev.noul_batch(t["context"], list(pool.text), t["question"]))
    add(0, 0, "zero_shot", zs_test)

    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            if k > len(pool):
                continue
            ex = pool.iloc[order[:k]]
            texts, y = list(ex.text), ex.label.values
            zs_k = zs_pool[order[:k]]
            shuffled = np.random.default_rng([k, d, 7]).permutation(y)
            noisy = y.copy()
            flip = np.random.default_rng([k, d, 11]).choice(k, int(round(NOISE * k)), replace=False)
            noisy[flip] = 1 - noisy[flip]

            jev_arms = {
                "in_context": state_examples(t, texts, y),
                "shuffled": state_examples(t, texts, shuffled),
                "noisy": state_examples(t, texts, noisy),
                "self_feedback": state_examples(t, texts, y, zs_k),
                "fmt_compact": state_compact(t, texts, y),
                "fmt_json": state_json(t, texts, y),
            }
            ic, missing = None, []
            for arm, st in jev_arms.items():
                p = ask(t, st, test_texts)
                if isinstance(p, str):
                    missing.append(arm)
                    continue
                if p is None:
                    print(f"{name} d{d} k={k}: {arm} does not fit in context, skipped", flush=True)
                    continue
                add(k, d, arm, p)
                if arm == "in_context":
                    ic = p
            add(k, d, "platt", fit_platt(zs_k, y, zs_test))
            add(k, d, "isotonic", fit_isotonic(zs_k, y, zs_test))
            add(k, d, "base_rate", np.full(len(test), smoothed_rate(y)))
            add(k, d, "logreg", fit_logreg(ex[feats], y, test[feats]))
            if ic is not None:
                cf = loo_in_context(t, texts, y)
                if cf is not None:
                    add(k, d, "both", fit_platt(cf, y, ic))
                else:
                    missing.append("both")
            if missing:
                print(f"{name} d{d} k={k}: not yet run (cache miss): {missing}", flush=True)
            print(f"{name} draw {d} k={k} done  (spent ${jev.spent()['usd']:.3f})", flush=True)

    return rows, test.label.values


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", message="Confidence interval of the Spearman")
    os.makedirs(OUT, exist_ok=True)
    for name in sys.argv[1:] or list(TASKS):
        rows, y = run_task(name)
        pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(
            os.path.join(OUT, f"pred_{name}.csv"), index=False)
        pd.DataFrame({"task": name, "i": range(len(y)), "label": y}).to_csv(
            os.path.join(OUT, f"labels_{name}.csv"), index=False)
    print("total spent:", jev.spent())
