"""General-purpose LLM comparison arm: P(yes) from the first answer token's logprobs.

Model: DeepSeek V4 Flash via OpenRouter ($0.047/M input tokens, vs Jev's $0.042/M), reasoning off,
pinned to one provider (StreamLake).
P(yes) = P("Yes"-like tokens) / (P("Yes"-like) + P("No"-like)) among the top-10 first tokens.
Cached on disk like jev.py.
"""
import hashlib
import json
import math
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from jev import OFFLINE, ROOT, _KEY, CacheMiss, session as _session

MODEL = "deepseek/deepseek-v4-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = "You predict outcomes of real-world cases. Answer with just Yes or No."

_lock = threading.Lock()
_db = sqlite3.connect(f"{ROOT}/data/llm_cache.sqlite", check_same_thread=False, timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")
_stats = {"usd": 0.0, "calls": 0, "no_yes_no": 0}
def _p_yes(resp):
    top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    y = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
    n = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
    if y + n == 0:
        return None
    return y / (y + n)


def yes_prob(prompt):
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
            "reasoning": {"enabled": False}, "usage": {"include": True},
            # one provider, so every probability comes from the same serving stack
            "provider": {"order": ["StreamLake"], "allow_fallbacks": False, "require_parameters": True}}
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    with _lock:
        row = _db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    if row:
        resp = json.loads(row[0])
    elif OFFLINE:
        raise CacheMiss()
    else:
        for attempt in range(8):
            try:
                r = _session().post(URL, json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"})
                if r.status_code == 200:
                    resp = r.json()
                    if resp.get("choices") and resp["choices"][0].get("logprobs"):
                        break
            except requests.RequestException:
                pass
            time.sleep(min(60, 2 ** attempt))
        else:
            raise RuntimeError(f"LLM failed after retries: {r.status_code} {r.text[:300]}")
        with _lock:
            _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?)", (h, json.dumps(body), json.dumps(resp), time.time()))
            _db.commit()
            _stats["usd"] += (resp.get("usage") or {}).get("cost", 0) or 0
            _stats["calls"] += 1
    p = _p_yes(resp)
    if p is None:
        with _lock:
            _stats["no_yes_no"] += 1
        return 0.5
    return p


def ask_all(state, cases, question, workers=32):
    prompts = [f"{state}\n\nNew case: {c}\nQuestion: {question}\nAnswer with just Yes or No." for c in cases]
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(yes_prob, prompts))


def stats():
    return dict(_stats)
