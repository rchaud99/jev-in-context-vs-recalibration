"""Same protocol as gemini_study.py with Claude Opus 5.5: zero-shot, in-context (k = 5, 20, 100; draws 0-1) and
Platt on its zero-shot probabilities for the k example cases.

Claude exposes no token logprobs, so Opus states a probability (0-100) per case, 25 cases per request, as a
JSON-schema reply (the same prompt as Gemini's). Thinking can't be turned off (adaptive); effort "low" keeps it
short. The shared state (context + examples) is sent as a cached prefix. Calls are cached in
data/opus_cache.sqlite; a hard dollar cap stops new requests.

    python opus_study.py --estimate
    python opus_study.py [task ...]
Output: results/opus_<task>.csv (task, k, draw, arm, i, p) with arms zero_shot, in_context, platt.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv

import gemini_study as g
from study import draw_order, fit_platt, state_examples
from tasks import ROOT, TASKS, load

MODEL = "claude-opus-5-5"
IN, CACHE_WRITE, CACHE_READ, OUT = 4.0, 5.0, 0.4, 20.0  # $/M tokens
KS = [5, 20, 100]
DRAWS = [0, 1]
BATCH = 25
MAX_DOLLARS = 25.0

_lock = threading.Lock()
_db = sqlite3.connect(f"{ROOT}/data/opus_cache.sqlite", check_same_thread=False, timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, response TEXT, ts REAL)")
spent = {"usd": 0.0, "calls": 0}
SCHEMA = {"type": "object", "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}},
          "required": ["probabilities"], "additionalProperties": False}


class Budget(Exception):
    pass


def tail(question, cases):
    listing = "\n\n".join(f"[{j + 1}] {c}" for j, c in enumerate(cases))
    return (f"---\nFor each of the {len(cases)} new cases below, estimate the probability (0 to 100) that the answer "
            f"to this question is yes: {question}\nJudge each case on its own. Reply with JSON only: "
            f'{{"probabilities": [one number per case, in order]}}\n\n{listing}')


VIA_OPENROUTER = os.environ.get("OPUS_VIA") == "openrouter"
"""Anthropic rejected the card, so the same model runs through OpenRouter (anthropic/claude-opus-5.5, same list price).
The cache key is unchanged, so answers already fetched from Anthropic are reused."""


def ask_openrouter(state, text, n):
    import requests
    key = [l.split("=", 1)[1].strip() for l in open(f"{ROOT}/.env") if l.startswith("OPENROUTER_API_KEY=")][0]
    body = {"model": "anthropic/claude-opus-5.5", "max_tokens": 16000, "reasoning": {"effort": "low"},
            "usage": {"include": True},
            "response_format": {"type": "json_schema", "json_schema": {"name": "p", "strict": True, "schema": SCHEMA}},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": state, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": text}]}]}
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=600,
                      headers={"Authorization": f"Bearer {key}"})
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"openrouter: {j['error']}")
    return j["choices"][0]["message"]["content"], float(j.get("usage", {}).get("cost") or 0.0)


def ask(client, state, question, cases):
    import anthropic
    text = tail(question, cases)
    h = hashlib.sha256(f"{MODEL}|{state}|{text}".encode()).hexdigest()
    with _lock:
        row = _db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    if row:
        return json.loads(row[0])["p"]
    last = None
    for attempt in range(12):
        with _lock:
            if spent["usd"] > MAX_DOLLARS:
                raise Budget(f"${spent['usd']:.2f} spent, cap ${MAX_DOLLARS}")
        try:
            if VIA_OPENROUTER:
                reply, cost = ask_openrouter(state, text, len(cases))
                with _lock:
                    spent["usd"] += cost
                    spent["calls"] += 1
                p = json.loads(reply)["probabilities"]
                if len(p) != len(cases):
                    raise ValueError(f"got {len(p)} probabilities for {len(cases)} cases")
                p = [min(max(float(v) / 100.0, 0.0), 1.0) for v in p]
                with _lock:
                    _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?)", (h, json.dumps({"p": p, "cost": cost, "via": "openrouter"}), time.time()))
                    _db.commit()
                return p
            with client.messages.stream(
                model=MODEL, max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": "low", "format": {"type": "json_schema",
                                                           "schema": anthropic.transform_schema(SCHEMA)}},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": state, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": text}]}],
            ) as stream:
                m = stream.get_final_message()
            u = m.usage
            cost = ((u.input_tokens or 0) * IN + (u.cache_creation_input_tokens or 0) * CACHE_WRITE
                    + (u.cache_read_input_tokens or 0) * CACHE_READ + (u.output_tokens or 0) * OUT) / 1e6
            with _lock:
                spent["usd"] += cost
                spent["calls"] += 1
            reply = "".join(b.text for b in m.content if b.type == "text")
            p = json.loads(reply)["probabilities"]
            if len(p) != len(cases):
                raise ValueError(f"got {len(p)} probabilities for {len(cases)} cases")
            p = [min(max(float(v) / 100.0, 0.0), 1.0) for v in p]
            with _lock:
                _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?)", (h, json.dumps({"p": p, "cost": cost}), time.time()))
                _db.commit()
            return p
        except Budget:
            raise
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403) or "credit" in str(e).lower():
                raise
            last = e
        except Exception as e:  # noqa: BLE001 - malformed replies and transient network errors are retried
            last = e
            if "429" in str(e) or "Rate limit" in str(e):
                time.sleep(20)  # OpenRouter new-account limit: 20 requests/minute for this model
                continue
        time.sleep(2 ** attempt)
    raise RuntimeError(f"failed after retries: {last}")


def run_all(client, state, question, cases, workers):
    chunks = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
    # the first chunk writes the cache; the rest read it
    first = ask(client, state, question, chunks[0])
    with ThreadPoolExecutor(workers) as ex:
        rest = list(ex.map(lambda c: ask(client, state, question, c), chunks[1:]))
    return first + [v for r in rest for v in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    names = args.tasks or list(TASKS)
    if args.estimate:
        tot = 0.0
        for name in names:
            t = TASKS[name]; pool, test, _ = load(name); cases = list(test.text)
            states = [t["context"]] + [state_examples(t, list(pool.iloc[draw_order(name, d, len(pool))[:k]].text),
                                                      pool.iloc[draw_order(name, d, len(pool))[:k]].label.values)
                                       for d in DRAWS for k in KS]
            n_req = len(states) * 12 + 8
            st_tok = sum(len(s) / 3.5 for s in states)
            case_tok = sum(len(c) for c in cases) / 3.5
            usd = (st_tok * (CACHE_WRITE + 11 * CACHE_READ) + len(states) * case_tok * IN + n_req * 900 * OUT) / 1e6
            tot += usd
            print(f"{name:14} {n_req} requests, ~${usd:.2f}")
        print(f"total ~${tot:.2f} (cap ${MAX_DOLLARS}); output tokens assumed ~900/request incl. thinking")
        return
    load_dotenv(g.ENV)
    import anthropic
    client = None if VIA_OPENROUTER else anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)
    for name in names:
        t = TASKS[name]; pool, test, _ = load(name); cases = list(test.text); rows = []
        try:
            zs = run_all(client, t["context"], t["question"], cases, args.workers)
            rows += [(name, 0, 0, "zero_shot", i, v) for i, v in enumerate(zs)]
            need = sorted({int(j) for d in DRAWS for j in draw_order(name, d, len(pool))[:max(KS)]})
            zp = dict(zip(need, run_all(client, t["context"], t["question"], list(pool.text.iloc[need]), args.workers)))
            for d in DRAWS:
                order = draw_order(name, d, len(pool))
                for k in KS:
                    ex = pool.iloc[order[:k]]
                    p = run_all(client, state_examples(t, list(ex.text), ex.label.values), t["question"], cases, args.workers)
                    rows += [(name, k, d, "in_context", i, v) for i, v in enumerate(p)]
                    pl = fit_platt(np.array([zp[int(j)] for j in order[:k]]), ex.label.values, np.array(zs))
                    rows += [(name, k, d, "platt", i, float(v)) for i, v in enumerate(pl)]
                    print(f"{name} k={k} d={d}  ${spent['usd']:.2f} ({spent['calls']} calls)", flush=True)
        except Budget as e:
            print(f"stopping: {e}", flush=True)
            pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(f"{ROOT}/results/opus_{name}.csv", index=False)
            sys.exit(1)
        pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(f"{ROOT}/results/opus_{name}.csv", index=False)
        print(f"{name} done  ${spent['usd']:.2f}", flush=True)
    print(f"all done  ${spent['usd']:.2f}")


if __name__ == "__main__":
    main()
