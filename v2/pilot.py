"""Pilot: does each client return usable probabilities? 20 test cases x 2 tasks, zero-shot and told.
Every call's cost goes to results/v2/spend_ledger.csv under experiment "pilot".

    cd v2 && python pilot.py <model> [<model> ...]
"""
import sys

import numpy as np

from common import TASKS, ledger_total, load, predict

models = sys.argv[1:]
for task in ["nyc311", "stackoverflow"]:
    t = TASKS[task]
    pool, test, _ = load(task)
    cases = list(test.text[:20])
    y = test.label.values[:20]
    rate = round(100 * pool.label.mean())
    told = t["context"] + (f"\n\nBase rate: across all cases like these, the answer to \"{t['question']}\" is yes "
                           f"about {rate}% of the time.")
    for m in models:
        for arm, st in [("zs", t["context"]), ("told", told)]:
            p, d = predict(m, st, cases, t["question"], "pilot")
            extra = f" mass {np.nanmean(d['yesno_mass']):.3f} prov {set(d['provider'])}" if d else ""
            print(f"{task:13s} {m:18s} {arm:4s} mean {np.nanmean(p):.2f} nan {np.isnan(p).sum()} "
                  f"min {np.nanmin(p):.3f} max {np.nanmax(p):.3f} distinct {len(np.unique(np.round(p, 3)))}{extra}"
                  f"  | y rate {y.mean():.2f}", flush=True)
print("ledger total $", round(ledger_total(), 4))
