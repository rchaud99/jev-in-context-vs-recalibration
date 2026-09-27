"""Table: true rate, and mean predicted probability (AUROC) at each k, for Jev and Gemini (and Opus when present).
Paired: Gemini uses draws 0-1; Jev is shown on the same draws where it has them."""
import sys, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from tasks import load, ROOT

def cell(P, y, arm, k, draws=None):
    d = P[(P.arm == arm) & (P.k == k)]
    if draws is not None and k > 0: d = d[d.draw.isin(draws)]
    if d.empty: return "–"
    aucs = [roc_auc_score(y, g.set_index("i").p.reindex(range(len(y))).values) for _, g in d.groupby("draw")]
    return f"{d.p.mean():.2f} ({np.mean(aucs):.2f})"

tasks = sys.argv[1:] or ["nyc311","crimes","flights","shelter","stackoverflow","hn","github_prs","inspections"]
for t in tasks:
    _, test, _ = load(t); y = test.label.values.astype(int)
    print(f"\n{t}  true rate {y.mean():.2f}   cells: in-context | Platt, each as mean prediction (AUROC)")
    srcs = [("Jev", f"results/pred_{t}.csv"), ("Gemini 3.8 Flash", f"results/gem_{t}.csv"), ("Opus 5.5", f"results/opus_{t}.csv")]
    ks = [0, 5, 20, 100, 200, 400]
    print(f"  {'model':17}" + "".join(f"{'k='+str(k):>26}" for k in ks))
    for name, f in srcs:
        if not os.path.exists(os.path.join(ROOT, f)): continue
        P = pd.read_csv(os.path.join(ROOT, f))
        cells = []
        for k in ks:
            if k == 0:
                cells.append(cell(P, y, "zero_shot", 0))
            else:
                ic, pl = cell(P, y, "in_context", k, [0, 1]), cell(P, y, "platt", k, [0, 1])
                cells.append("–" if ic == pl == "–" else f"{ic} | {pl}")
        print(f"  {name:17}" + "".join(f"{c:>26}" for c in cells))
