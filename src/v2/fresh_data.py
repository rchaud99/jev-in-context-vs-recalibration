"""Fresh, later replication data: the same sources and text templates as data/v2, with the same leak repairs, but drawn
from September 2026 (after every model's training cutoff, and after E1's data was fixed). No API spend (public APIs).

Windows chosen so labels have matured by 2026-09-28:
  nyc311       complaints created 2026-09-01..09-20 (7-day closure outcome known)
  crimes       reports 2026-09-01..09-20, police-initiated types dropped
  inspections  inspections 2026-09-01..
  shelter      outcomes 2026-09-01..; sex and intake_condition dropped
  hn           stories 2026-09-01..09-10
  github_prs   PRs created 2026-09-01..09-15 and closed; draft marker dropped
  stackoverflow questions 2026-09-01..09-10 (accepted answers have had ~3 weeks: the base rate is lower than June's)
  (flights: BTS data for September is not yet published, so the task is left out)
Output: data/v2_fresh/<task>.csv, up to 700 rows sampled with seed 0 (first 300 = test, rest = pool).
"""
import html
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (file is in src/v2/)
sys.path.insert(0, os.path.join(ROOT, "src", "v1"))  # tasks.py, prepare_data.py
import prepare_data as pdx  # noqa: E402

OUT = os.path.join(ROOT, "data", "v2_fresh")
os.makedirs(OUT, exist_ok=True)
N = 700
POLICE_INITIATED = {"narcotics", "other narcotic violation", "weapons violation", "concealed carry license violation",
                    "interference with public officer", "liquor law violation", "public peace violation",
                    "gambling", "prostitution", "public indecency", "obscenity"}  # as v2/build_data.py


def shelter():
    pdx.SHELTER_FROM = "2026-09-01"
    df = pdx.shelter().drop(columns=["sex", "intake_condition"])
    # rebuild text without the dropped fields (same template as v2/build_data.py)
    raw = df
    raw["text"] = raw.apply(lambda r: f"A {r['size']} {r.type}, {r.breed}, {r.color}, {r.age}. "
                            f"Intake: {r.intake_type} ({r.intake_subtype}), from {r.intake_jurisdiction}.", axis=1)
    return raw


def inspections():
    pdx.INSPECT_FROM = "2026-09-01"
    return pdx.inspections()


def nyc311():
    pdx.NYC_DAYS = [f"2026-09-{d:02d}" for d in range(1, 21)]
    return pdx.nyc311()


def crimes():
    df = pdx.socrata("https://data.cityofchicago.org/resource/ijzp-q8t2.json",
                     "date >= '2026-09-01' AND date < '2026-09-21'", limit=50000,
                     select="date,primary_type,description,location_description,domestic,arrest,district")
    for c in ["primary_type", "description", "location_description"]:
        df[c] = df[c].fillna("unknown").str.lower()
    t = pd.to_datetime(df.date)
    df["time"] = t.dt.hour.map(pdx.hour_word)
    df["weekday"] = t.dt.day_name()
    df["domestic"] = df.domestic.map({True: "domestic", False: "not domestic"}).fillna("not domestic")
    df["district"] = "police district " + df.district.fillna("unknown").astype(str).str.lstrip("0")
    df["label"] = df.arrest.astype(bool).astype(int)
    df = df[~df.primary_type.isin(POLICE_INITIATED)]
    feats = ["primary_type", "description", "location_description", "domestic", "district", "time", "weekday"]
    df["text"] = df.apply(lambda r: f"Reported {r.primary_type} ({r.description}) at a {r.location_description}, "
                          f"{r.domestic}, {r.district}, on a {r.weekday}, {r.time}.", axis=1)
    return df[feats + ["text", "label"]]


def hn():
    rows, start = [], int(pd.Timestamp("2026-09-01", tz="UTC").timestamp())
    for h in range(0, 24 * 10, 6):
        a = start + h * 3600
        r = requests.get("https://hn.algolia.com/api/v1/search_by_date", timeout=120,
                         params={"tags": "story", "numericFilters": f"created_at_i>={a},created_at_i<{a + 21600}",
                                 "hitsPerPage": 1000}).json()
        rows += r["hits"]
    df = pd.DataFrame(rows).drop_duplicates("objectID")
    created = pd.to_datetime(df.created_at_i, unit="s", utc=True).dt.tz_convert("US/Eastern")
    df["title"] = df.title.fillna("").map(html.unescape)
    df["domain"] = df.url.map(lambda u: urlparse(u).netloc.removeprefix("www.") if isinstance(u, str) else "self post (no link)")
    df["kind"] = np.where(df.title.str.startswith("Show HN"), "Show HN",
                          np.where(df.title.str.startswith("Ask HN"), "Ask HN", "link"))
    df["weekday"] = created.dt.day_name()
    df["time"] = created.dt.hour.map(pdx.hour_word) + " US Eastern"
    df["label"] = (df.points >= 10).astype(int)
    feats = ["domain", "kind", "weekday", "time"]
    df["text"] = df.apply(lambda r: f'Hacker News submission: "{r.title}" ({r.domain}). '
                          f"Posted on a {r.weekday}, {r.time}.", axis=1)
    return df[feats + ["text", "label"]]


def github_prs():
    items = []
    for repo in pdx.REPOS:
        for page in (1, 2):
            out = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f",
                                  f"q=repo:{repo} is:pr is:closed created:2026-09-01..2026-09-15",
                                  "-f", "per_page=100", "-f", f"page={page}"], capture_output=True, text=True)
            try:
                res = json.loads(out.stdout).get("items", [])
            except json.JSONDecodeError:
                print(f"  gh error for {repo}: {out.stderr[:120]}", flush=True)
                res = []
            items += [dict(it, repo=repo) for it in res]
            time.sleep(2.5)
            if len(res) < 100:
                break
    df = pd.DataFrame(items)
    df = df[(df.user.map(lambda u: u.get("type")) != "Bot") & ~df.user.map(lambda u: u["login"].endswith("[bot]"))]
    df["label"] = df.pull_request.map(lambda p: p.get("merged_at") is not None).astype(int)
    df["description"] = df.body.fillna("").map(lambda b: pdx.length_word(len(pdx.clean_text(b, 10 ** 6))))
    feats = ["repo", "description"]
    df["text"] = df.apply(lambda r: f'Pull request to {r.repo} titled "{r.title}". '
                          f"Description ({r.description}): {pdx.clean_text(r.body, 250) or '(none)'}", axis=1)
    return df[feats + ["text", "label"]]


def stackoverflow():
    rows = []
    for d in range(1, 11):
        a = int(pd.Timestamp(f"2026-09-{d:02d}", tz="UTC").timestamp())
        for page in (1, 2):
            r = requests.get("https://api.stackexchange.com/2.3/questions", timeout=120, params={
                "site": "stackoverflow", "pagesize": 100, "page": page, "fromdate": a, "todate": a + 86400,
                "sort": "creation", "order": "desc", "filter": "withbody"}).json()
            rows += r.get("items", [])
            time.sleep(1)
            if not r.get("has_more"):
                break
    df = pd.DataFrame(rows)
    df["title"] = df.title.map(html.unescape)
    df["tags_str"] = df.tags.map(lambda t: ", ".join(t[:4]))
    df["main_tag"] = df.tags.map(lambda t: t[0] if t else "none")
    df["has_code"] = df.body.str.contains("<code>").map({True: "includes code", False: "no code"})
    df["length"] = df.body.map(lambda b: pdx.length_word(len(pdx.clean_text(b, 10 ** 6))))
    df["label"] = df.get("accepted_answer_id").notna().astype(int) if "accepted_answer_id" in df else 0
    feats = ["main_tag", "has_code", "length"]
    df["text"] = df.apply(lambda r: f'Stack Overflow question "{r.title}" tagged [{r.tags_str}]; '
                          f"{r.length}, {r.has_code}. Body: {pdx.clean_text(r.body, 250)}", axis=1)
    return df[feats + ["text", "label"]]


FNS = {"nyc311": nyc311, "crimes": crimes, "shelter": shelter, "stackoverflow": stackoverflow, "hn": hn,
       "github_prs": github_prs, "inspections": inspections}

if __name__ == "__main__":
    for name in sys.argv[1:] or FNS:
        if os.path.exists(os.path.join(OUT, f"{name}.csv")):
            print(f"{name}: exists, skipped")
            continue
        df = FNS[name]().dropna(subset=["label"])
        full = len(df)
        df = df.sample(n=min(N, len(df)), random_state=0).reset_index(drop=True)
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"{name}: {full} rows -> kept {len(df)}, base rate {df.label.mean():.3f}; e.g. {df.text.iloc[0][:120]}",
              flush=True)
