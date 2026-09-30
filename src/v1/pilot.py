"""Gate check before the main study.

Per task, on 200 held-out cases:
  - zero-shot AUROC / Brier (does Jev rank at all? is it better than the base rate?)
  - batched (25 questions / request) vs one question per request
  - run-to-run repeat of the batched call
"""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

import jev
from tasks import TASKS, load

N = 200

for name, t in TASKS.items():
    _, test, _ = load(name)
    test = test.iloc[:N]
    y, cases = test.label.values, list(test.text)
    b25 = np.array(jev.noul_batch(t["context"], cases, t["question"], batch_size=25))
    b25r = np.array(jev.noul_batch(t["context"], cases, t["question"], batch_size=25, rep=1))
    b1 = np.array(jev.noul_batch(t["context"], cases, t["question"], batch_size=1, workers=16))
    base = np.full(N, y.mean())
    print(f"\n== {name}  (n={N}, base rate {y.mean():.3f})")
    for label, p in [("batch25", b25), ("batch25 repeat", b25r), ("one-per-request", b1)]:
        print(f"  {label:16s} AUROC {roc_auc_score(y, p):.3f}  Brier {brier_score_loss(y, p):.4f}  "
              f"mean p {p.mean():.3f}  range [{p.min():.2f}, {p.max():.2f}]  distinct {len(np.unique(p))}")
    print(f"  {'base rate':16s} Brier {brier_score_loss(y, base):.4f}")
    print(f"  repeat: mean |diff| {np.abs(b25 - b25r).mean():.4f}, max {np.abs(b25 - b25r).max():.2f}, "
          f"changed {np.mean(b25 != b25r):.1%}")
    print(f"  batching: mean |diff| {np.abs(b25 - b1).mean():.4f}, max {np.abs(b25 - b1).max():.2f}, "
          f"corr {np.corrcoef(b25, b1)[0, 1]:.3f}")

print("\nspent:", jev.spent())
