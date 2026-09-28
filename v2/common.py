"""Shared infrastructure for the v2 (confirmatory) experiments.

* v2 task data (label leaks repaired; built by build_data.py into data/v2/)
* an append-only spend ledger with a hard cap (results/v2/spend_ledger.csv)
* one on-disk cache for every API call (data/v2/cache.sqlite), keyed by a hash of the request
* three OpenRouter clients, each pinned to one provider:
    - jev_noul:        TypeSafe Jev decisions endpoint (native probabilities, 25 questions per request)
    - chat_logprob:    chat model, P(yes) from the first answer token's log-probabilities (one case per request)
    - chat_verbalized: chat model states a probability 0-100 per case, 25 cases per request, JSON reply

Nothing here prints or logs the API key.
"""
import csv
import hashlib
import json
import math
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2DATA = os.path.join(ROOT, "data", "v2")
OUT = os.path.join(ROOT, "results", "v2")
os.makedirs(OUT, exist_ok=True)
os.makedirs(V2DATA, exist_ok=True)
LEDGER = os.path.join(OUT, "spend_ledger.csv")
CAP_USD = float(os.environ.get("V2_CAP_USD", "8.0"))  # current authorised cap for v2 spend (coordinator: <= $8)
OFFLINE = os.environ.get("V2_OFFLINE") == "1"
N_TEST = 300


def _key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    for line in open(os.path.join(ROOT, ".env")):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.strip().split("=", 1)[1]
    raise RuntimeError("OPENROUTER_API_KEY not set")


# ------------------------------------------------------------------ tasks
import sys  # noqa: E402
sys.path.insert(0, ROOT)
from tasks import TASKS  # noqa: E402  (context / question / yes / no wording is unchanged from v1)

TASK_ORDER = ["nyc311", "crimes", "shelter", "flights", "stackoverflow", "hn", "github_prs", "inspections"]


def load(task):
    """(pool, test, feats) from data/v2/<task>.csv: first 300 rows = fixed test set, the rest = pool."""
    df = pd.read_csv(os.path.join(V2DATA, f"{task}.csv"))
    feats = [c for c in df.columns if c not in ("text", "label")]
    df[feats] = df[feats].astype(str)
    return df.iloc[N_TEST:].reset_index(drop=True), df.iloc[:N_TEST].reset_index(drop=True), feats


# ------------------------------------------------------------------ ledger
_lock = threading.Lock()


class OutOfCredit(Exception):
    pass


class OverCap(Exception):
    pass


def ledger_total():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(pd.read_csv(LEDGER).usd.sum())


_running = {"usd": None}


def _charge(experiment, model, usd, n):
    with _lock:
        if _running["usd"] is None:
            _running["usd"] = ledger_total()
        _running["usd"] += usd
        new = not os.path.exists(LEDGER)
        with open(LEDGER, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts", "experiment", "model", "usd", "n_cases"])
            w.writerow([f"{time.time():.0f}", experiment, model, f"{usd:.8f}", n])


def _check_cap():
    with _lock:
        if _running["usd"] is None:
            _running["usd"] = ledger_total()
        if _running["usd"] >= CAP_USD:
            raise OverCap(f"v2 spend ${_running['usd']:.3f} reached cap ${CAP_USD}")


# ------------------------------------------------------------------ cache
_db = sqlite3.connect(os.path.join(V2DATA, "cache.sqlite"), check_same_thread=False, timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, model TEXT, request TEXT, response TEXT, ts REAL)")


class CacheMiss(Exception):
    pass


def _cached(h):
    with _lock:
        r = _db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    return None if r is None else json.loads(r[0])


def _store(h, model, req, resp):
    with _lock:
        _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?)", (h, model, json.dumps(req), json.dumps(resp), time.time()))
        _db.commit()


_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))


def _post(url, body, timeout=180):
    """POST with retries on transient errors. 402 -> OutOfCredit (never retried)."""
    last = None
    for attempt in range(7):
        _check_cap()
        try:
            r = _session.post(url, json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
            if r.status_code == 402:
                raise OutOfCredit(r.text[:300])
            if r.status_code == 200:
                j = r.json()
                if "error" not in j:
                    return j
                last = str(j["error"])[:300]
                if "credit" in last.lower() or "402" in last:
                    raise OutOfCredit(last)
            elif r.status_code in (400, 401, 403, 404):
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            else:
                last = f"HTTP {r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = repr(e)[:200]
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed after retries: {last}")


# ------------------------------------------------------------------ clients
def jev_noul(state, cases, question, experiment, batch=25, workers=8):
    """Jev (typesafe/jev-1.13) Noul probability per case, 25 questions per request under one state."""
    from concurrent.futures import ThreadPoolExecutor
    chunks = [list(range(i, min(i + batch, len(cases)))) for i in range(0, len(cases), batch)]

    def run(idx):
        qs = {f"c{j}": {"type": "noul", "instructions": f"Case: {cases[j]}\nQuestion: {question}"} for j in idx}
        body = {"model": "typesafe/jev-1.13", "state": state, "questions": qs}
        h = hashlib.sha256(("v2|" + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        resp = _cached(h)
        if resp is None:
            if OFFLINE:
                raise CacheMiss()
            resp = _post("https://openrouter.ai/api/alpha/decisions", body)
            if "answers" not in resp:
                raise RuntimeError(f"no answers: {str(resp)[:200]}")
            _store(h, "jev", body, resp)
            _charge(experiment, "jev", float((resp.get("usage") or {}).get("cost") or 0), len(idx))
        return [(j, float(resp["answers"][f"c{j}"]["noul"])) for j in idx]

    out = [None] * len(cases)
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(run, chunks):
            for j, p in res:
                out[j] = p
    return np.array(out)


LOGPROB_MODELS = {  # name: (OpenRouter id, pinned provider, extra body)
    "deepseek-v4-flash": ("deepseek/deepseek-v4-flash", "streamlake/fp8", {"reasoning": {"enabled": False}}),
    "llama-3.3-70b": ("meta-llama/llama-3.3-70b-instruct", "parasail/fp8", {}),
    "gpt-4o-mini": ("openai/gpt-4o-mini", "openai", {}),
    "qwen3.7-flash": ("qwen/qwen3.7-flash", "alibaba", {"reasoning": {"enabled": False}}),
}
SYSTEM = "You predict outcomes of real-world cases. Answer with just Yes or No."


def lp_prompt(state, case, question):
    return f"{state}\n\nCase: {case}\nQuestion: {question}\nAnswer with just Yes or No."


def _p_yes(resp):
    """P(yes) renormalised over yes/no tokens among the top first-token alternatives; also the yes+no mass."""
    c = resp["choices"][0]
    lp = (c.get("logprobs") or {}).get("content") or []
    if not lp:
        return None, 0.0
    top = lp[0].get("top_logprobs") or []
    y = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().strip(".") == "yes")
    n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().strip(".") == "no")
    if y + n == 0:
        return None, 0.0
    return y / (y + n), y + n


def chat_logprob(model, state, cases, question, experiment, workers=16):
    from concurrent.futures import ThreadPoolExecutor
    mid, prov, extra = LOGPROB_MODELS[model]

    def one(case):
        body = {"model": mid, "messages": [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": lp_prompt(state, case, question)}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "usage": {"include": True},
                "provider": {"order": [prov], "allow_fallbacks": False, "require_parameters": True}, **extra}
        h = hashlib.sha256(("v2|" + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        resp = _cached(h)
        if resp is None:
            if OFFLINE:
                raise CacheMiss()
            for _ in range(3):
                resp = _post("https://openrouter.ai/api/v1/chat/completions", body)
                _charge(experiment, model, float((resp.get("usage") or {}).get("cost") or 0), 1)
                if _p_yes(resp)[0] is not None:
                    break
            _store(h, model, body, resp)
        p, mass = _p_yes(resp)
        return (np.nan if p is None else p), mass, resp.get("provider")

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, cases))
    return np.array([r[0] for r in res]), np.array([r[1] for r in res]), [r[2] for r in res]


VERBAL_MODELS = {
    "gemini-3.8-flash": ("google/gemini-3.8-flash", "google-ai-studio/flex", {"reasoning": {"effort": "low"}}),
    "claude-haiku-4.5": ("anthropic/claude-haiku-4.5", "anthropic", {}),
    # the same two models as their logprob configurations, asked to state probabilities instead
    "gpt-4o-mini-verbal": ("openai/gpt-4o-mini", "openai", {}),
    "deepseek-v4-flash-verbal": ("deepseek/deepseek-v4-flash", "streamlake/fp8", {"reasoning": {"enabled": False}}),
}
SCHEMA = {"type": "object", "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}},
          "required": ["probabilities"], "additionalProperties": False}


def verbal_prompt(state, question, cases):  # identical to v1 gemini_study.prompt
    listing = "\n\n".join(f"[{j + 1}] {c}" for j, c in enumerate(cases))
    return (f"{state}\n\n---\nFor each of the {len(cases)} new cases below, estimate the probability (0 to 100) "
            f"that the answer to this question is yes: {question}\n"
            f"Judge each case on its own. Reply with JSON only: "
            f'{{"probabilities": [one number per case, in order]}}\n\n{listing}')


def chat_verbalized(model, state, cases, question, experiment, batch=25, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    mid, prov, extra = VERBAL_MODELS[model]
    chunks = [cases[i:i + batch] for i in range(0, len(cases), batch)]

    def run(ch):
        body = {"model": mid, "messages": [{"role": "user", "content": verbal_prompt(state, question, ch)}],
                "max_tokens": 8000, "usage": {"include": True},
                "response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": SCHEMA}},
                "provider": {"order": [prov], "allow_fallbacks": False, "require_parameters": True}, **extra}
        h = hashlib.sha256(("v2|" + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        resp = _cached(h)
        if resp is None:
            if OFFLINE:
                raise CacheMiss()
            for attempt in range(10):
                resp = _post("https://openrouter.ai/api/v1/chat/completions", body, timeout=300)
                _charge(experiment, model, float((resp.get("usage") or {}).get("cost") or 0), len(ch))
                try:
                    p = json.loads(resp["choices"][0]["message"]["content"])["probabilities"]
                    if len(p) == len(ch):
                        break
                except Exception:  # noqa: BLE001 - malformed reply: retry
                    pass
            else:
                raise RuntimeError("verbalized: no valid reply after 10 attempts")
            _store(h, model, body, resp)
        p = json.loads(resp["choices"][0]["message"]["content"])["probabilities"]
        return [min(max(float(v) / 100.0, 0.0), 1.0) for v in p]

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, chunks))
    return np.array([v for r in res for v in r])


MODELS = ["jev", "qwen3.7-flash", "deepseek-v4-flash", "gpt-4o-mini", "llama-3.3-70b",
          "gpt-4o-mini-verbal", "deepseek-v4-flash-verbal", "gemini-3.8-flash", "claude-haiku-4.5"]


def predict(model, state, cases, question, experiment):
    """Returns (p, diagnostics dict)."""
    if model == "jev":
        return jev_noul(state, cases, question, experiment), {}
    if model in LOGPROB_MODELS:
        p, mass, prov = chat_logprob(model, state, cases, question, experiment)
        return p, {"yesno_mass": mass, "provider": prov}
    return chat_verbalized(model, state, cases, question, experiment), {}
