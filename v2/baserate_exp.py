"""E1 (preregistered, see RESEARCH_AGENDA.md): tell the model the base rate, or shift its scores in code?

For every model x task, on the 300 fixed test cases and 300 unlabeled pool cases (U = the first 300 pool rows):
  zs_test, zs_pool    zero-shot probabilities (state = task context only)
  told_s  (s=0,1,2)   state = context + one sentence stating the base rate r_s, test cases only
  told_w1, told_w2    two alternative wordings of that sentence, seed 0's rate, test cases only
Seed s: L_s = 100 cases drawn (numpy default_rng(1000 + s)) from U without replacement; r_s = round(100 * mean label
of L_s) / 100, i.e. the base rate a team would estimate from 100 labelled cases. Everything that uses a rate (told,
constant, label-free shift) uses the same rounded r_s. Label-using arms (Platt, logistic regression) use L_s.
Label-free arms never see labels beyond r_s. All derived arms are computed in baserate_analyze.py.

Output: results/v2/e1_<model>.csv with columns task, seed, arm, set, i, p   (set: test | pool)
    cd v2 && python baserate_exp.py <model> [task ...]
"""
import os
import sys

import numpy as np
import pandas as pd

from common import OUT, PREFIX, TASK_ORDER, TASKS, OutOfCredit, OverCap, ledger_total, load, predict

SEEDS = [0, 1, 2]
N_U, N_L = 300, 100
WORDINGS = {
    "w0": 'Base rate: across all cases like these, the answer to "{q}" is yes about {n}% of the time.',
    "w1": "For reference: historically, about {n}% of cases like these turned out yes ({yes}).",
    "w2": ("Important calibration information: in this population, {n} out of every 100 cases have the answer yes. "
           "Take this into account when giving your answer."),
}


def seed_rate(pool, s):
    idx = np.random.default_rng(1000 + s).choice(N_U, N_L, replace=False)
    return idx, round(100 * pool.label.values[idx].mean())


def told_state(t, n, w="w0"):
    return f"{t['context']}\n\n" + WORDINGS[w].format(q=t["question"], n=n, yes=t["yes"])


def run(model, tasks):
    path = os.path.join(OUT, f"{PREFIX}_{model}.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["task", "seed", "arm", "set", "i", "p"])
    for task in tasks:
        if task in set(done.task):
            print(f"{model} {task}: already done", flush=True)
            continue
        t = TASKS[task]
        pool, test, _ = load(task)
        cases, ucases = list(test.text), list(pool.text[:N_U])
        rows = []

        def add(seed, arm, st, p):
            rows.extend((task, seed, arm, st, i, float(v)) for i, v in enumerate(p))

        add(-1, "zs", "test", predict(model, t["context"], cases, t["question"], f"{PREFIX}:{task}")[0])
        add(-1, "zs", "pool", predict(model, t["context"], ucases, t["question"], f"{PREFIX}:{task}")[0])
        for s in SEEDS:
            _, n = seed_rate(pool, s)
            add(s, "told", "test", predict(model, told_state(t, n), cases, t["question"], f"{PREFIX}:{task}")[0])
        _, n0 = seed_rate(pool, 0)
        for w in ([] if PREFIX == "e1r" else ["w1", "w2"]):  # E1-R: no wording variants
            add(0, f"told_{w}", "test", predict(model, told_state(t, n0, w), cases, t["question"], f"{PREFIX}:{task}")[0])
        done = pd.concat([done, pd.DataFrame(rows, columns=done.columns)], ignore_index=True)
        done.to_csv(path, index=False)
        print(f"{model} {task}: done  (v2 ledger ${ledger_total():.3f})", flush=True)


if __name__ == "__main__":
    model, tasks = sys.argv[1], sys.argv[2:] or TASK_ORDER
    try:
        run(model, tasks)
    except (OutOfCredit, OverCap) as e:
        print(f"STOPPED ({type(e).__name__}): {e}", flush=True)
        sys.exit(2)
