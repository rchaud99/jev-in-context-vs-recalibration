"""Platt arm for Gemini, as study.py does it for Jev: Gemini's zero-shot probability on the k example cases (the
same cases, draws 0-1), Platt fitted on those against their labels (study.fit_platt), applied to its zero-shot
probabilities on the 300 test cases. Needs Gemini zero-shot on the pooled example cases (cached, ~$0.05/task).
Appends arm "platt" rows to results/v1/gem_<task>.csv."""
import os, sys
import numpy as np, pandas as pd
from dotenv import load_dotenv
import gemini_study as g
from study import draw_order, fit_platt
from tasks import ROOT, TASKS, load

load_dotenv(g.ENV)
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
from concurrent.futures import ThreadPoolExecutor

for name in sys.argv[1:] or list(TASKS):
    f = f"{ROOT}/results/v1/gem_{name}.csv"
    if not os.path.exists(f):
        print(f"{name}: no gemini results yet"); continue
    G = pd.read_csv(f); G = G[G.arm != "platt"]
    t = TASKS[name]; pool, test, _ = load(name)
    zs_test = G[G.arm == "zero_shot"].sort_values("i").p.values
    kmax = max(g.ks_for(name))
    need = sorted({int(j) for d in g.DRAWS for j in draw_order(name, d, len(pool))[:kmax]})
    texts = list(pool.text.iloc[need])
    chunks = [texts[i:i + g.BATCH] for i in range(0, len(texts), g.BATCH)]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda c: g.ask(client, g.prompt(t["context"], t["question"], c), len(c)), chunks))
    zs_pool = dict(zip(need, [v for r in res for v in r]))
    rows = []
    for d in g.DRAWS:
        order = draw_order(name, d, len(pool))
        for k in g.ks_for(name):
            idx = order[:k]
            p = fit_platt(np.array([zs_pool[int(j)] for j in idx]), pool.label.values[idx], zs_test)
            rows += [(name, k, d, "platt", i, float(v)) for i, v in enumerate(p)]
    pd.concat([G, pd.DataFrame(rows, columns=G.columns)]).to_csv(f, index=False)
    print(f"{name}: platt added  ${g.spent['usd']:.2f}", flush=True)
