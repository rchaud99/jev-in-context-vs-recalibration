"""Does telling Jev the base rate fix its probabilities? Zero-shot Jev on the 300 test cases with one added sentence
stating the population base rate (the pool's rate: a number a deployer would know from a dashboard). Compared with
plain zero-shot and with a post-hoc base-rate shift fitted on 100 labels (draw 0). Cost: well under $0.05."""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import jev
from study import draw_order
from tasks import ROOT, TASKS, load
import platt_sweep as ps

rows = []
for name in TASKS:
    t = TASKS[name]; pool, test, _ = load(name); y = test.label.values.astype(int)
    rate = pool.label.mean()
    note = (f"Base rate: across all cases like these, the answer to \"{t['question']}\" is yes about "
            f"{round(100 * rate)}% of the time.")
    zs = np.array(jev.noul_batch(t["context"], list(test.text), t["question"]))
    br = np.array(jev.noul_batch(f"{t['context']}\n\n{note}", list(test.text), t["question"]))
    zp = np.array(jev.noul_batch(t["context"], list(pool.text), t["question"]))
    idx = draw_order(name, 0, len(pool))[:100]
    shifted, _ = ps.shift(zp[idx], pool.label.values[idx], zs)
    for arm, p in [("zero_shot", zs), ("told_base_rate", br), ("shift_100_labels", shifted)]:
        rows.append(dict(task=name, true=y.mean(), told=rate, arm=arm, mean=p.mean(), auroc=roc_auc_score(y, p),
                         brier=np.mean((p - y) ** 2)))
    print(name, "done", flush=True)
R = pd.DataFrame(rows); R.to_csv("results/v1/baserate_check.csv", index=False)
for name, g in R.groupby("task", sort=False):
    print(f"{name:14} true {g.true.iloc[0]:.2f} told {g.told.iloc[0]:.2f} | " +
          " | ".join(f"{r.arm}: mean {r['mean']:.2f} AUC {r.auroc:.2f} Brier {r.brier:.3f}" for _, r in g.iterrows()))
