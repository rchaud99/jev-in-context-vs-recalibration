"""Same protocol as study.py, with a general-purpose LLM (see llm.py) in place of Jev.

Arms: llm_zero_shot, llm_in_context (same prose examples as Jev's in_context), llm_platt
(Platt on the LLM's zero-shot scores), and llm_shuffled at k=100 only.
k in {5, 20, 100}, draws 0-4 (a subset of the Jev grid, to bound cost).
Output: results/v1/llm_<task>.csv
"""
import os
import sys

import numpy as np
import pandas as pd

import llm
from jev import CacheMiss
from study import OUT, draw_order, fit_platt, state_examples
from tasks import TASKS, load

KS = [5, 20, 100]
DRAWS = range(5)

for name in sys.argv[1:] or list(TASKS):
    t = TASKS[name]
    pool, test, _ = load(name)
    test_texts = list(test.text)
    rows = []

    def add(k, d, arm, p):
        rows.extend((name, k, d, arm, i, float(v)) for i, v in enumerate(p))

    try:
        zs_test = np.array(llm.ask_all(t["context"], test_texts, t["question"]))
    except CacheMiss:
        print(f"{name}: zero-shot not run yet, skipping task", flush=True)
        continue
    add(0, 0, "llm_zero_shot", zs_test)
    need = sorted({j for d in DRAWS for j in draw_order(name, d, len(pool))[:max(KS)]})
    try:
        zs_pool = dict(zip(need, llm.ask_all(t["context"], list(pool.text.iloc[need]), t["question"])))
    except CacheMiss:
        zs_pool = None

    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            ex = pool.iloc[order[:k]]
            texts, y = list(ex.text), ex.label.values
            try:
                add(k, d, "llm_in_context", llm.ask_all(state_examples(t, texts, y), test_texts, t["question"]))
            except CacheMiss:
                print(f"{name} d{d} k={k}: llm_in_context not run yet", flush=True)
            if zs_pool is not None:
                add(k, d, "llm_platt", fit_platt(np.array([zs_pool[j] for j in order[:k]]), y, zs_test))
            if k == 100:
                shuffled = np.random.default_rng([k, d, 7]).permutation(y)  # same permutation as study.py
                try:
                    add(k, d, "llm_shuffled", llm.ask_all(state_examples(t, texts, shuffled), test_texts, t["question"]))
                except CacheMiss:
                    print(f"{name} d{d} k={k}: llm_shuffled not run yet", flush=True)
            print(f"{name} draw {d} k={k} done  {llm.stats()}", flush=True)

    pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(
        os.path.join(OUT, f"llm_{name}.csv"), index=False)
print("done", llm.stats())
