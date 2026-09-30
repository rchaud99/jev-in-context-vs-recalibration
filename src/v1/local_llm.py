"""Local general-purpose LLM arm (MLX): P(yes) from next-token logits after a chat prompt.

The shared prefix (context + in-context examples) is prefilled once into a KV cache, then test cases
are scored in batches on top of it (right-padded; causal attention means padding cannot affect the
real tokens). P(yes) = softmax over {Yes, yes, No, no} first-token logits -> P(Yes-like).
Results are cached in data/local_llm_cache.sqlite. Run with .venv_mlx/bin/python.
"""
import hashlib
import json
import os
import sqlite3

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import KVCache, make_prompt_cache

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (file is in src/v1/)
MODEL = os.environ.get("LOCAL_LLM", "mlx-community/Qwen2.5-7B-Instruct-4bit")
SYSTEM = "You predict outcomes of real-world cases. Answer with just Yes or No."

_model, _tok = None, None
_db = sqlite3.connect(os.path.join(ROOT, "data", "local_llm_cache.sqlite"), timeout=120)
_db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, p REAL)")


def _load():
    global _model, _tok, YES, NO
    if _model is None:
        _model, _tok = load(MODEL)
        ids = lambda words: sorted({_tok.encode(w, add_special_tokens=False)[0] for w in words})
        YES, NO = ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"])
    return _model, _tok


def _prompt_ids(state, case, question):
    _, tok = _load()
    user = f"{state}\n\nNew case: {case}\nQuestion: {question}\nAnswer with just Yes or No."
    text = tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def _key(ids):
    return hashlib.sha256((MODEL + json.dumps(ids)).encode()).hexdigest()


def ask_all(state, cases, question, kv_budget_gb=24):
    model, _ = _load()
    all_ids = [_prompt_ids(state, c, question) for c in cases]
    keys = [_key(ids) for ids in all_ids]
    out = [None] * len(cases)
    for j, h in enumerate(keys):
        r = _db.execute("SELECT p FROM calls WHERE h=?", (h,)).fetchone()
        if r:
            out[j] = r[0]
    todo = [j for j in range(len(cases)) if out[j] is None]
    if not todo:
        return out

    # longest common token prefix among the prompts still to score
    first = all_ids[todo[0]]
    L = len(first)
    for j in todo[1:]:
        ids = all_ids[j]
        m = min(L, len(ids))
        i = 0
        while i < m and ids[i] == first[i]:
            i += 1
        L = i
    L = max(0, L - 1)  # keep >=1 suffix token per case
    prefix = first[:L]

    cache = make_prompt_cache(model)
    for s in range(0, len(prefix), 2048):  # chunked prefill
        model(mx.array(prefix[s:s + 2048])[None], cache=cache)
        mx.eval([c.state for c in cache])

    per_tok = sum(c.keys.nbytes + c.values.nbytes for c in cache) / max(1, L)
    bs = int(max(1, min(64, kv_budget_gb * 1e9 / max(1.0, per_tok * (L + 300)))))

    for s in range(0, len(todo), bs):
        batch = todo[s:s + bs]
        sufs = [all_ids[j][L:] for j in batch]
        S = max(len(x) for x in sufs)
        arr = np.zeros((len(batch), S), dtype=np.int32)
        for b, x in enumerate(sufs):
            arr[b, :len(x)] = x
        bcache = []
        for c in cache:
            n = KVCache()
            if L > 0:
                n.keys = mx.repeat(c.keys[..., :c.offset, :], len(batch), axis=0)
                n.values = mx.repeat(c.values[..., :c.offset, :], len(batch), axis=0)
                n.offset = c.offset
            bcache.append(n)
        logits = model(mx.array(arr), cache=bcache)
        last = mx.array([len(x) - 1 for x in sufs])
        sel = logits[mx.arange(len(batch)), last].astype(mx.float32)
        ly = mx.logsumexp(sel[:, YES], axis=-1)
        ln = mx.logsumexp(sel[:, NO], axis=-1)
        p = np.array(mx.sigmoid(ly - ln))
        for b, j in enumerate(batch):
            out[j] = float(p[b])
            _db.execute("INSERT OR REPLACE INTO calls VALUES (?,?)", (keys[j], out[j]))
        _db.commit()
    return out
