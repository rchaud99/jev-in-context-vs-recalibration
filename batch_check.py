"""Does asking 25 cases per request change Gemini's answers? Asks the first 100 test cases one per request
(k=0 and k=20 draw 0) and compares with the batched answers in results/gem_<task>.csv. Also times requests."""
import os, sys, time
import numpy as np, pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
import gemini_study as g
from study import draw_order, state_examples
from tasks import ROOT, TASKS, load
from concurrent.futures import ThreadPoolExecutor
load_dotenv(g.ENV)
from google import genai
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
N = 100
out = []
for name in sys.argv[1:]:
    t = TASKS[name]; pool, test, _ = load(name); cases = list(test.text)[:N]; y = test.label.values[:N]
    G = pd.read_csv(f"{ROOT}/results/gem_{name}.csv")
    o = draw_order(name, 0, len(pool)); ex = pool.iloc[o[:20]]
    for k, state in [(0, t["context"]), (20, state_examples(t, list(ex.text), ex.label.values))]:
        arm = "zero_shot" if k == 0 else "in_context"
        b = G[(G.arm == arm) & (G.k == k) & (G.draw == 0)].sort_values("i").p.values[:N]
        times = []
        def one(c):
            t0 = time.time(); p = g.ask(client, g.prompt(state, t["question"], [c]), 1)[0]; times.append(time.time() - t0); return p
        with ThreadPoolExecutor(8) as ex_:
            s = np.array(list(ex_.map(one, cases)))
        out.append(dict(task=name, k=k, mean_batched=b.mean(), mean_single=s.mean(), true=y.mean(),
                        auc_batched=roc_auc_score(y, b), auc_single=roc_auc_score(y, s),
                        corr=np.corrcoef(b, s)[0, 1], mean_abs_diff=np.abs(b - s).mean(),
                        brier_batched=np.mean((b - y) ** 2), brier_single=np.mean((s - y) ** 2),
                        median_latency_s=float(np.median(times))))
        print(out[-1], f"${g.spent['usd']:.2f}", flush=True)
pd.DataFrame(out).to_csv(f"{ROOT}/results/batch_check.csv", index=False)
