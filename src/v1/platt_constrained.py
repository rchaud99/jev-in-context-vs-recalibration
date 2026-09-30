"""Platt with the degenerate fits removed: fit study.py's Platt (C=1); if the slope is <= 0 (flat or inverted ranking)
or the k labels are all one class, fall back to a 1-parameter base-rate shift (logit(p) + b). Cache-only (no API
calls) for Jev, Gemini and Opus; draws 0-1. Writes results/v1/platt_c_<task>.csv (task, model, k, draw, i, p, fallback)."""
import hashlib, json, sqlite3, sys
import numpy as np, pandas as pd
import platt_sweep as ps
from study import draw_order
from tasks import ROOT, TASKS, load
import opus_study as o

def opus_zero(name, pool, test):
    t = TASKS[name]; db = sqlite3.connect(f"{ROOT}/data/opus_cache.sqlite")
    def cached(state, cases):
        out = []
        for i in range(0, len(cases), o.BATCH):
            c = cases[i:i + o.BATCH]
            h = hashlib.sha256(f"{o.MODEL}|{state}|{o.tail(t['question'], c)}".encode()).hexdigest()
            r = db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
            if r is None: return None
            out += json.loads(r[0])["p"]
        return out
    zt = cached(t["context"], list(test.text))
    need = sorted({int(j) for d in o.DRAWS for j in draw_order(name, d, len(pool))[:max(o.KS)]})
    zp = cached(t["context"], list(pool.text.iloc[need]))
    if zt is None or zp is None: return None
    pool_z = np.full(len(pool), np.nan); pool_z[need] = zp
    return np.array(zt), pool_z

rows = []
for name in sys.argv[1:] or list(TASKS):
    inp = ps.model_inputs(name); pool, test, _ = load(name)
    models = {"Jev": (inp["jev"]["z_test"], inp["jev"]["z_pool"]), "Gemini": (inp["gemini"]["z_test"], inp["gemini"]["z_pool"])}
    oz = opus_zero(name, pool, test)
    if oz: models["Opus"] = oz
    for m, (zt, zp) in models.items():
        for d in (0, 1):
            order = draw_order(name, d, len(pool))
            for k in (5, 20, 100):
                idx = order[:k]; y = inp["pool_y"][idx]; s = zp[idx]
                if np.isnan(s).any(): continue
                p, slope = ps.platt_c(s, y, zt, 1.0)
                fb = len(np.unique(y)) < 2 or slope <= 0
                if fb: p, _ = ps.shift(s, y, zt)
                rows += [(name, m, k, d, i, float(v), fb) for i, v in enumerate(p)]
    print(name, "done", flush=True)
pd.DataFrame(rows, columns=["task", "model", "k", "draw", "i", "p", "fallback"]).to_csv(f"{ROOT}/results/v1/platt_constrained.csv", index=False)
