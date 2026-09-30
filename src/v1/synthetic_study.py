"""Synthetic-task study: Jev and Gemini 3.8 Flash (zero-shot / in-context), logistic regression, base-rate-only and
Bayes-optimal (true probabilities) on the tasks in synthetic_tasks.py. Same protocol as study.py / gemini_study.py:
state_examples prose examples, jev.noul_batch(state, cases, question) (batch shrunk to fit the 32k context, as
study.ask), Gemini via gemini_study.prompt/ask at 25 cases per request, nested draws 0-1, k = 5, 20, 100, 400.

Calls are cached in data/synthetic/{jev,gemini}_cache.sqlite (not the real-task caches). TabPFN: synthetic_tabpfn.py.

    python synthetic_study.py --estimate [tasks]
    python synthetic_study.py [tasks] [--gemini-ks 5 20 100] [--no-gemini] [--gemini-cap 5.0]
Output: results/v1/synth_part_llm_<task>.csv (task, k, draw, model, arm, i, p)
"""
import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import gemini_study as G
import jev
import synthetic_tasks as S
from study import fit_logreg, state_examples

KS = [5, 20, 100, 400]
DRAWS = [0, 1]
OUT = os.path.join(S.ROOT, "results", "v1")

# separate caches for the synthetic study
jev._db = sqlite3.connect(os.path.join(S.DATA, "jev_cache.sqlite"), check_same_thread=False, timeout=120)
jev._db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")
G._db = sqlite3.connect(os.path.join(S.DATA, "gemini_cache.sqlite"), check_same_thread=False, timeout=120)
G._db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")


def states(name):
    t = S.TASKS[name]
    pool, test, _ = S.load(name)
    out = [("zero_shot", 0, 0, t["context"], None)]
    for d in DRAWS:
        order = S.draw_order(d, len(pool))
        for k in KS:
            ex = pool.iloc[order[:k]]
            out.append(("in_context", k, d, state_examples(t, list(ex.text), ex.label.values), order[:k]))
    return t, pool, test, out


def jev_ask(t, state, cases):
    q_tok = max(jev.est_tokens(f"Case: {c}\nQuestion: {t['question']}") for c in cases) + 5
    room = jev.MAX_TOKENS - 230 - jev.est_tokens(state)
    bs = min(25, int(room // q_tok))
    if bs < 1:
        raise RuntimeError(f"state does not fit Jev's context (~{jev.est_tokens(state):.0f} tokens)")
    return np.array(jev.noul_batch(state, cases, t["question"], batch_size=bs))


def cache_spend():
    """Total logged cost of every call in the synthetic caches (so reruns report the true total)."""
    j = sum((json.loads(r[0]).get("usage", {}) or {}).get("cost", 0) or 0
            for r in jev._db.execute("SELECT response FROM calls"))
    g = sum(json.loads(r[0]).get("cost", 0) for r in G._db.execute("SELECT response FROM calls"))
    return j, g


def estimate(names, gemini_ks):
    tot_j = tot_g = 0.0
    seen = set()
    for name in names:
        t, pool, test, js = states(name)
        cases = list(test.text)
        j_tok = g_tok = 0.0
        n_req = 0
        for arm, k, d, st, _ in js:
            key = (st, name == "neutral")
            if key in seen:  # identical zero-shot prompt: cached
                continue
            seen.add(key)
            j_tok += 300 / 25 * 230 + jev.est_tokens(st) * 12 + sum(jev.est_tokens(f"Case: {c}\nQuestion: {t['question']}") for c in cases)
            if k == 0 or k in gemini_ks:
                g_tok += sum(len(G.prompt(st, t["question"], cases[i:i + 25])) / 3.5 for i in range(0, 300, 25))
                n_req += 12
        jusd = j_tok * 0.04 / 1e6  # Jev: ~$0.04/M input tokens (fitted on the real-study cache)
        gusd = g_tok * G.PRICE_IN / 1e6 + n_req * 800 * G.PRICE_OUT / 1e6
        tot_j += jusd
        tot_g += gusd
        print(f"{name:12} Jev ~{j_tok / 1e6:.2f}M tok ~${jusd:.2f} | Gemini {n_req} req ~{g_tok / 1e6:.2f}M in ~${gusd:.2f}")
    print(f"total ~${tot_j + tot_g:.2f} (Jev ~${tot_j:.2f}, Gemini ~${tot_g:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--gemini-ks", type=int, nargs="*", default=KS)
    ap.add_argument("--no-gemini", action="store_true")
    ap.add_argument("--gemini-cap", type=float, default=5.0, help="max Gemini $ in this process")
    args = ap.parse_args()
    names = args.tasks or S.ORDER
    if args.estimate:
        estimate(names, [] if args.no_gemini else args.gemini_ks)
        return
    G.MAX_DOLLARS = args.gemini_cap
    client = None
    if not args.no_gemini:
        from dotenv import load_dotenv
        from google import genai
        load_dotenv(G.ENV)
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    for name in names:
        t, pool, test, js = states(name)
        cases = list(test.text)
        rows = []

        def add(k, d, model, arm, p):
            rows.extend((name, k, d, model, arm, i, float(v)) for i, v in enumerate(p))

        add(0, 0, "bayes", "true_p", test.p_true.values)
        for arm, k, d, st, idx in js:
            add(k, d, "jev", arm, jev_ask(t, st, cases))
            if client is not None and (k == 0 or k in args.gemini_ks):
                try:
                    with ThreadPoolExecutor(8) as ex:
                        res = list(ex.map(lambda i: G.ask(client, G.prompt(st, t["question"], cases[i:i + 25]),
                                                          len(cases[i:i + 25])), range(0, len(cases), 25)))
                    add(k, d, "gemini", arm, [v for r in res for v in r])
                except G.Budget as e:
                    print(f"STOP (budget): {e}", flush=True)
                    pd.DataFrame(rows, columns=["task", "k", "draw", "model", "arm", "i", "p"]).to_csv(
                        f"{OUT}/synth_part_llm_{name}.csv", index=False)
                    sys.exit(1)
            if k > 0:
                ex_ = pool.iloc[idx]
                y = ex_.label.values
                add(k, d, "logreg", "trained", fit_logreg(ex_[S.FEATS], y, test[S.FEATS]))
                add(k, d, "base_rate", "trained", np.full(len(test), y.mean()))
            print(f"{name} {arm} k={k} d={d}  session: Jev ${jev.spent()['usd']:.3f}, Gemini ${G.spent['usd']:.3f}"
                  f" ({G.spent['calls']} calls)", flush=True)
        pd.DataFrame(rows, columns=["task", "k", "draw", "model", "arm", "i", "p"]).to_csv(
            f"{OUT}/synth_part_llm_{name}.csv", index=False)
    j, g = cache_spend()
    print(f"done. session: Jev ${jev.spent()['usd']:.3f}, Gemini ${G.spent['usd']:.3f}; "
          f"all synthetic calls ever: Jev ${j:.3f}, Gemini ${g:.3f}, total ${j + g:.3f}")


if __name__ == "__main__":
    main()
