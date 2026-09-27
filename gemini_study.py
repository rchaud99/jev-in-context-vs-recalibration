"""Same protocol as study.py (zero-shot and in-context arms), with Gemini 3.8 Flash in place of Jev.

Gemini 3.8 Flash exposes no token logprobs ("Logprobs is not enabled for this model") and can't turn thinking off
(lowest level: "low"), so it states a probability (0-100) for each case. Test cases are asked 25 per request (as Jev
is batched), in a JSON-schema reply. Examples, draws and test cases are exactly Jev's: the same draw_order and
state_examples, draws 0-1 (a subset of Jev's, so comparisons stay paired), k = 5, 20, 100 and 400 (200 for the text
tasks, which Jev ran to 200). Calls are cached in data/gemini_cache.sqlite; a hard dollar cap stops new requests.

    python gemini_study.py --estimate            # tokens and dollars, no calls
    python gemini_study.py [task ...]            # run (default: all tasks)
Output: results/gem_<task>.csv (task, k, draw, arm, i, p), like results/pred_<task>.csv.
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

from study import draw_order, state_examples
from tasks import ROOT, TASKS, TEXT, load

MODEL = "gemini-3.8-flash"
PRICE_IN, PRICE_OUT = 0.75, 3.75  # $/M tokens (thinking billed as output)
BATCH = 25
DRAWS = [0, 1]
MAX_DOLLARS = 15.0
ENV = os.path.expanduser("~/Desktop/how_much_is_a_look_worth/.env")


def ks_for(name):
    return [5, 20, 100, 200 if name in TEXT else 400]


def prompt(state, question, cases):
    listing = "\n\n".join(f"[{j + 1}] {c}" for j, c in enumerate(cases))
    return (f"{state}\n\n---\nFor each of the {len(cases)} new cases below, estimate the probability (0 to 100) "
            f"that the answer to this question is yes: {question}\n"
            f"Judge each case on its own. Reply with JSON only: "
            f'{{"probabilities": [one number per case, in order]}}\n\n{listing}')


_lock = threading.Lock()
_db = sqlite3.connect(f"{ROOT}/data/gemini_cache.sqlite", check_same_thread=False, timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, request TEXT, response TEXT, ts REAL)")
spent = {"usd": 0.0, "calls": 0}


class Budget(Exception):
    pass


def ask(client, text, n):
    from google.genai import types
    h = hashlib.sha256(f"{MODEL}|{text}".encode()).hexdigest()
    with _lock:
        row = _db.execute("SELECT response FROM calls WHERE h=?", (h,)).fetchone()
    if row:
        return json.loads(row[0])["p"]
    schema = {"type": "object", "properties": {"probabilities": {"type": "array", "items": {"type": "number"}}},
              "required": ["probabilities"]}
    for attempt in range(4):
        with _lock:
            if spent["usd"] > MAX_DOLLARS:
                raise Budget(f"${spent['usd']:.2f} spent, cap ${MAX_DOLLARS}")
        try:
            r = client.models.generate_content(model=MODEL, contents=text, config=types.GenerateContentConfig(
                response_mime_type="application/json", response_json_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="low")))
            u = r.usage_metadata
            cost = ((u.prompt_token_count or 0) * PRICE_IN
                    + ((u.candidates_token_count or 0) + (u.thoughts_token_count or 0)) * PRICE_OUT) / 1e6
            with _lock:
                spent["usd"] += cost
                spent["calls"] += 1
            p = json.loads(r.text)["probabilities"]
            if len(p) != n:
                raise ValueError(f"got {len(p)} probabilities for {n} cases")
            p = [min(max(float(v) / 100.0, 0.0), 1.0) for v in p]
            with _lock:
                _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?,?,?)",
                            (h, text[-2000:], json.dumps({"p": p, "cost": cost}), time.time()))
                _db.commit()
            return p
        except Budget:
            raise
        except Exception as e:  # noqa: BLE001 - transient errors and malformed replies are retried
            if "credit" in str(e).lower() or "billing" in str(e).lower() or "API key" in str(e):
                raise
            time.sleep(2 ** attempt)
            last = e
    raise RuntimeError(f"failed after retries: {last}")


def jobs(name):
    t = TASKS[name]
    pool, test, _ = load(name)
    cases = list(test.text)
    out = [("zero_shot", 0, 0, t["context"])]
    for d in DRAWS:
        order = draw_order(name, d, len(pool))
        for k in ks_for(name):
            ex = pool.iloc[order[:k]]
            out.append(("in_context", k, d, state_examples(t, list(ex.text), ex.label.values)))
    return t, cases, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    names = args.tasks or list(TASKS)
    if args.estimate:
        total = 0.0
        for name in names:
            t, cases, js = jobs(name)
            toks = sum(len(prompt(s, t["question"], cases[i:i + BATCH])) / 3.5
                       for _, _, _, s in js for i in range(0, len(cases), BATCH))
            usd = toks * PRICE_IN / 1e6 + len(js) * (len(cases) / BATCH) * 800 * PRICE_OUT / 1e6
            total += usd
            print(f"{name:14} {len(js)} states, ~{toks / 1e6:.2f}M input tokens, ~${usd:.2f}")
        print(f"total ~${total:.2f} (cap ${MAX_DOLLARS})")
        return
    load_dotenv(ENV)
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for name in names:
        t, cases, js = jobs(name)
        rows = []
        try:
            for arm, k, d, state in js:
                chunks = [cases[i:i + BATCH] for i in range(0, len(cases), BATCH)]
                with ThreadPoolExecutor(args.workers) as ex:
                    res = list(ex.map(lambda c: ask(client, prompt(state, t["question"], c), len(c)), chunks))
                p = [v for r in res for v in r]
                rows += [(name, k, d, arm, i, v) for i, v in enumerate(p)]
                print(f"{name} {arm} k={k} d={d}  ${spent['usd']:.2f} ({spent['calls']} calls)", flush=True)
        except Budget as e:
            print(f"stopping: {e}", flush=True)
            pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(f"{ROOT}/results/gem_{name}.csv", index=False)
            sys.exit(1)
        pd.DataFrame(rows, columns=["task", "k", "draw", "arm", "i", "p"]).to_csv(f"{ROOT}/results/gem_{name}.csv", index=False)
    print(f"done  ${spent['usd']:.2f}")


if __name__ == "__main__":
    main()
