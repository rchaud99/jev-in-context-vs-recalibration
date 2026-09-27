"""Gemini 3.8 Flash zero-shot with the same base-rate sentence as baserate_check.py (the pool's rate)."""
import os, numpy as np, pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor
import gemini_study as g
from tasks import TASKS, load
load_dotenv(g.ENV)
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
rows = []
for name in TASKS:
    t = TASKS[name]; pool, test, _ = load(name); y = test.label.values.astype(int); cases = list(test.text)
    note = (f"Base rate: across all cases like these, the answer to \"{t['question']}\" is yes about "
            f"{round(100 * pool.label.mean())}% of the time.")
    state = f"{t['context']}\n\n{note}"
    chunks = [cases[i:i + g.BATCH] for i in range(0, len(cases), g.BATCH)]
    with ThreadPoolExecutor(8) as ex:
        p = np.array([v for r in ex.map(lambda c: g.ask(client, g.prompt(state, t["question"], c), len(c)), chunks) for v in r])
    G = pd.read_csv(f"results/gem_{name}.csv"); z = G[G.arm == "zero_shot"].sort_values("i").p.values
    for arm, q in [("zero_shot", z), ("told_base_rate", p)]:
        rows.append(dict(task=name, true=y.mean(), arm=arm, mean=q.mean(), auroc=roc_auc_score(y, q), brier=np.mean((q - y) ** 2)))
    print(name, f"${g.spent['usd']:.2f}", flush=True)
pd.DataFrame(rows).to_csv("results/gemini_baserate.csv", index=False)
