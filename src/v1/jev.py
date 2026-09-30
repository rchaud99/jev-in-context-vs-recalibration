"""Minimal Jev client (OpenRouter decisions endpoint) with an on-disk cache.

Every request/response pair is cached by a hash of the request body, so reruns of the
analysis are free and deterministic. Use `rep` to force a fresh, separately cached call
when measuring run-to-run noise.
"""
import hashlib
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests
import urllib3.util.connection

# openrouter.ai's IPv6 routes intermittently fail from this network ("No route to host"): use IPv4 only
urllib3.util.connection.allowed_gai_family = lambda: __import__("socket").AF_INET

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (file is in src/v1/)
URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"


def _load_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    with open(os.path.join(ROOT, ".env")) as f:
        for line in f:
            if line.startswith("OPENROUTER_API_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("OPENROUTER_API_KEY not set")


_KEY = _load_key()
_db_lock = threading.Lock()
_db = sqlite3.connect(os.path.join(ROOT, "data", "jev_cache.sqlite"), check_same_thread=False, timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")
_cost = {"usd": 0.0, "calls": 0}
_session = requests.Session()
_session.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=64))


def session():
    """One shared keep-alive connection pool: worker threads are short-lived, and opening a new TLS
    connection per call overwhelmed the local network."""
    return _session


OFFLINE = os.environ.get("JEV_OFFLINE") == "1"  # cache only: never call the API


class CacheMiss(Exception):
    pass


def decide(state, questions, rep=0):
    """questions: {name: {"type": "noul", "instructions": ...}} -> {name: answer dict}."""
    body = {"model": MODEL, "state": state, "questions": questions}
    h = hashlib.sha256((json.dumps(body, sort_keys=True) + f"|rep={rep}").encode()).hexdigest()
    with _db_lock:
        row = _db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    if row:
        return json.loads(row[0])["answers"]
    if OFFLINE:
        raise CacheMiss()

    for attempt in range(8):
        try:
            r = session().post(URL, json=body, timeout=120,
                              headers={"Authorization": f"Bearer {_KEY}", "Content-Type": "application/json"})
            if r.status_code == 200:
                resp = r.json()
                if "answers" in resp:
                    break
            elif r.status_code not in (408, 429, 500, 502, 503, 504):
                raise RuntimeError(f"Jev error {r.status_code}: {r.text[:500]}")
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** attempt))
    else:
        raise RuntimeError(f"Jev failed after retries: {r.status_code} {r.text[:300]}")

    with _db_lock:
        _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?)",
                    (h, json.dumps(body), json.dumps(resp), time.time()))
        _db.commit()
        _cost["usd"] += resp.get("usage", {}).get("cost", 0) or 0
        _cost["calls"] += 1
    return resp["answers"]


def noul_batch(state, cases, question, batch_size=25, rep=0, workers=8):
    """Ask `question` about each case (as separate Noul questions) under one shared state.

    Returns a list of probabilities aligned with `cases`.
    """
    chunks = [list(range(i, min(i + batch_size, len(cases)))) for i in range(0, len(cases), batch_size)]

    def run(idx):
        qs = {f"c{j}": {"type": "noul", "instructions": f"Case: {cases[j]}\nQuestion: {question}"} for j in idx}
        ans = decide(state, qs, rep=rep)
        return [(j, ans[f"c{j}"]["noul"]) for j in idx]

    out = [None] * len(cases)
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(run, chunks):
            for j, p in res:
                out[j] = p
    return out


def est_tokens(text):
    """Jev input tokens, fitted on cached calls: ~0.276 tokens/char (+ ~230 per request, ~4 per question)."""
    return 0.28 * len(text) + 5


MAX_TOKENS = 31500  # Jev context is 32k


def spent():
    return dict(_cost)
