"""Label-sensitivity profile of a local general-purpose LLM (Qwen2.5-7B-Instruct, MLX), same protocol
and same examples/label permutations/flips as study.py:
  qwen_zero_shot, qwen_in_context, qwen_shuffled, qwen_noisy   at k in {20, 100}, draws 0-2.
Question: does a general LLM use the *labels* in its examples the way Jev does (Min et al. 2022 found
LLM in-context learning to be largely insensitive to label correctness)?
Output: results/qwen_<task>.csv. Run with .venv_mlx/bin/python.
"""
import os
import sys

import numpy as np
import pandas as pd

import local_llm
from study import NOISE, OUT, draw_order, state_examples
from tasks import TASKS, load

KS = [20, 100]
DRAWS = range(3)

for name in sys.argv[1:] or ["nyc311", "github_prs", "shelter", "crimes", "stackoverflow", "flights", "hn", "inspections"]:
    t = TASKS[name]
    pool, test, _ = load(name)
    cases = list(test.text)
    rows = []

    def add(k, d, arm, p):
        rows.extend((name, k, d, arm, i, float(v)) for i, v in enumerate(p))

    add(0, 0, "qwen_zero_shot", local_llm.ask_all(t["context"], cases, t["question"]))
    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in KS:
            ex = pool.iloc[order[:k]]
            texts, y = list(ex.text), ex.label.values
            shuffled = np.random.default_rng([k, d, 7]).permutation(y)
            noisy = y.copy()
            flip = np.random.default_rng([k, d, 11]).choice(k, int(round(NOISE * k)), replace=False)
            noisy[flip] = 1 - noisy[flip]
            for arm, labels in [("qwen_in_context", y), ("qwen_shuffled", shuffled), ("qwen_noisy", noisy)]:
                add(k, d, arm, local_llm.ask_all(state_examples(t, texts, labels), cases, t["question"]))
            print(f"{name} draw {d} k={k} done", flush=True)
    pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(os.path.join(OUT, f"qwen_{name}.csv"), index=False)
