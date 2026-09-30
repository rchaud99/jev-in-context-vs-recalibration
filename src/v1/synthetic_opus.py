"""Synthetic-task study, Claude Opus 5.5 arm (paired with the Jev / Gemini runs of synthetic_study.py).

Same states (context + prose examples from synthetic_study.states: same nested draws 0-1, same example rows), same
300 test cases, 25 cases per request, same question wording as Gemini (opus_study.tail = gemini_study.prompt minus
the state, which is sent as a cached prefix). k = 0, 5, 20, 100 only (k = 400 is too expensive for Opus). The
Gemini synthetic runs had no Platt arm, so neither does this one.

Calls go through opus_study.ask (OPUS_VIA=openrouter -> anthropic/claude-opus-5.5), cached in
data/synthetic/opus_cache.sqlite (kept apart from the real-task cache, like the Jev / Gemini synthetic caches).
The dollar cap counts everything already logged in that cache, so restarts cannot exceed it.

    python synthetic_opus.py --estimate [tasks]
    python synthetic_opus.py [tasks] [--workers 2] [--cap 8]
Output: results/v1/synth_opus_<task>.csv (task, k, draw, model, arm, i, p), model = "opus".
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys

import pandas as pd

import opus_study as O
import synthetic_tasks as S
from synthetic_study import states

KS = [0, 5, 20, 100]
OUT = os.path.join(S.ROOT, "results", "v1")
O._db = sqlite3.connect(os.path.join(S.DATA, "opus_cache.sqlite"), check_same_thread=False, timeout=120)
O._db.execute("CREATE TABLE IF NOT EXISTS calls (h TEXT PRIMARY KEY, response TEXT, ts REAL)")


def cache_spend():
    return sum(json.loads(r[0]).get("cost", 0) or 0 for r in O._db.execute("SELECT response FROM calls"))


def jobs(name):
    t, pool, test, js = states(name)
    return t, list(test.text), [(arm, k, d, st) for arm, k, d, st, _ in js if k in KS]


def estimate(names, out_tok):
    """Conservative: every input token at the full $4/M (no prompt-cache credit), `out_tok` output tokens per
    request (thinking included; the real-task OpenRouter calls averaged $0.0137/request in total)."""
    seen, tot, n_tot = set(), 0.0, 0
    for name in names:
        t, cases, js = jobs(name)
        usd, n = 0.0, 0
        for arm, k, d, st in js:
            for i in range(0, len(cases), O.BATCH):
                chunk = cases[i:i + O.BATCH]
                text = O.tail(t["question"], chunk)
                h = hashlib.sha256(f"{O.MODEL}|{st}|{text}".encode()).hexdigest()
                if h in seen or O._db.execute("SELECT 1 FROM calls WHERE h=?", (h,)).fetchone():
                    continue
                seen.add(h)
                n += 1
                usd += (len(st) + len(text)) / 3.5 * O.IN / 1e6 + out_tok * O.OUT / 1e6
        tot += usd
        n_tot += n
        print(f"{name:12} {n:4d} new requests  ~${usd:.2f}")
    print(f"total {n_tot} new requests ~${tot:.2f} (assuming {out_tok} output tokens/request, no cache discount); "
          f"already logged in the synthetic Opus cache: ${cache_spend():.2f}")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--out-tok", type=int, default=600)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cap", type=float, default=8.0, help="max $ in the synthetic Opus cache, all runs combined")
    args = ap.parse_args()
    names = args.tasks or S.ORDER
    if args.estimate:
        estimate(names, args.out_tok)
        return
    if not O.VIA_OPENROUTER:
        sys.exit("set OPUS_VIA=openrouter")
    prior = cache_spend()
    O.MAX_DOLLARS = args.cap
    O.spent["usd"] = prior  # the cap covers earlier runs too
    for name in names:
        t, cases, js = jobs(name)
        rows = []
        try:
            for arm, k, d, st in js:
                p = O.run_all(None, st, t["question"], cases, args.workers)
                rows += [(name, k, d, "opus", arm, i, float(v)) for i, v in enumerate(p)]
                print(f"{name} {arm} k={k} d={d}  total ${O.spent['usd']:.2f} ({O.spent['calls']} new calls)", flush=True)
        except O.Budget as e:
            print(f"STOP (budget): {e}", flush=True)
            pd.DataFrame(rows, columns=["task", "k", "draw", "model", "arm", "i", "p"]).to_csv(
                f"{OUT}/synth_opus_{name}.csv", index=False)
            sys.exit(1)
        pd.DataFrame(rows, columns=["task", "k", "draw", "model", "arm", "i", "p"]).to_csv(
            f"{OUT}/synth_opus_{name}.csv", index=False)
    print(f"done. this run ${O.spent['usd'] - prior:.3f}; all synthetic Opus calls ever ${cache_spend():.3f}")


if __name__ == "__main__":
    main()
