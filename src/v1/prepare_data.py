"""Download and prepare the four study tasks.

Each task becomes data/<task>.csv with:
  - categorical feature columns (used by the LR / TabPFN baselines)
  - `text`: the same features written as words (what Jev sees)
  - `label`: 0/1
Numbers are binned into words because Jev is weak with numbers.
"""
import glob
import io
import os

import numpy as np
import pandas as pd
import time

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (file is in src/v1/)
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data")
N_PER_TASK = 1500
SEED = 0
# Date windows (overridable, e.g. by contamination_check.py for post-release records)
SHELTER_FROM, INSPECT_FROM = "2026-01-01", "2026-01-01"
NYC_DAYS = [f"2026-06-{d:02d}" for d in range(1, 31)]

QUESTIONS = {
    "flights": "Will this flight arrive more than 15 minutes late?",
    "inspections": "Will this establishment fail the food inspection?",
    "shelter": "Will this animal be adopted or returned to its owner?",
    "nyc311": "Will this complaint be closed within 7 days of being filed?",
}


def socrata(url, where, limit=50000, select=None):
    params = {"$where": where, "$limit": limit}
    if select:
        params["$select"] = select
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def hour_word(h):
    if h < 6:
        return "early morning (before 6am)"
    if h < 9:
        return "morning (6-9am)"
    if h < 12:
        return "late morning (9am-noon)"
    if h < 15:
        return "early afternoon (noon-3pm)"
    if h < 18:
        return "late afternoon (3-6pm)"
    if h < 21:
        return "evening (6-9pm)"
    return "late night (after 9pm)"


AIRLINES = {
    "AA": "American Airlines", "DL": "Delta Air Lines", "UA": "United Airlines",
    "WN": "Southwest Airlines", "B6": "JetBlue", "AS": "Alaska Airlines",
    "NK": "Spirit Airlines", "F9": "Frontier Airlines", "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines", "OO": "SkyWest (regional)", "YX": "Republic Airways (regional)",
    "MQ": "Envoy Air (regional)", "OH": "PSA Airlines (regional)", "9E": "Endeavor Air (regional)",
    "ZW": "Air Wisconsin (regional)", "C5": "CommutAir (regional)", "G7": "GoJet (regional)",
    "PT": "Piedmont Airlines (regional)", "QX": "Horizon Air (regional)", "MX": "Breeze Airways",
}


def flights():
    csv = glob.glob(os.path.join(RAW, "On_Time_Reporting*2026_6.csv"))[0]
    df = pd.read_csv(csv, low_memory=False)
    df = df[(df.Origin == "PIT") & (df.Cancelled == 0) & (df.Diverted == 0)].copy()
    df["airline"] = df.Reporting_Airline.map(AIRLINES).fillna(df.Reporting_Airline)
    df["date"] = pd.to_datetime(df.FlightDate).dt.strftime("%A, June %-d")
    df["dep_time"] = (df.CRSDepTime // 100).map(hour_word)
    df["destination"] = df.DestCityName
    df["length"] = pd.cut(df.Distance, [0, 500, 1000, 2000, 99999],
                          labels=["short (under 500 miles)", "medium (500-1000 miles)",
                                  "long (1000-2000 miles)", "very long (over 2000 miles)"]).astype(str)
    df["label"] = df.ArrDel15.astype(int)
    feats = ["airline", "date", "dep_time", "destination", "length"]
    df["text"] = df.apply(lambda r: f"{r.airline} flight departing Pittsburgh (PIT) on {r.date}, "
                          f"scheduled {r.dep_time}, to {r.destination}; {r.length}.", axis=1)
    return df[feats + ["text", "label"]]


def age_word(days):
    if pd.isna(days):
        return "unknown age"
    y = days / 365.25
    if y < 0.5:
        return "baby (under 6 months)"
    if y < 1:
        return "juvenile (6-12 months)"
    if y < 3:
        return "young adult (1-3 years)"
    if y < 7:
        return "adult (3-7 years)"
    if y < 10:
        return "older adult (7-10 years)"
    return "senior (over 10 years)"


def shelter():
    df = socrata("https://data.sonomacounty.ca.gov/resource/924a-vesw.json",
                 f"outcome_date >= '{SHELTER_FROM}' AND outcome_type IS NOT NULL")
    dob = pd.to_datetime(df.get("date_of_birth"), errors="coerce")
    intake = pd.to_datetime(df.intake_date, errors="coerce")
    df["age"] = [age_word(d) for d in (intake - dob).dt.days]
    for c in ["type", "breed", "color", "sex", "size", "intake_type", "intake_subtype",
              "intake_condition", "intake_jurisdiction"]:
        df[c] = df[c].fillna("unknown").str.lower()
    df["label"] = df.outcome_type.isin(["ADOPTION", "RETURN TO OWNER"]).astype(int)
    feats = ["type", "breed", "color", "sex", "size", "age", "intake_type", "intake_subtype",
             "intake_condition", "intake_jurisdiction"]
    df["text"] = df.apply(lambda r: f"A {r['size']} {r.sex} {r.type}, {r.breed}, {r.color}, {r.age}. "
                          f"Intake: {r.intake_type} ({r.intake_subtype}), condition {r.intake_condition}, "
                          f"from {r.intake_jurisdiction}.", axis=1)
    return df[feats + ["text", "label"]]


def inspections():
    df = socrata("https://data.cityofchicago.org/resource/4ijn-s7e5.json",
                 f"inspection_date >= '{INSPECT_FROM}' AND results in ('Pass', 'Pass w/ Conditions', 'Fail')")
    for c in ["dba_name", "facility_type", "risk", "inspection_type", "zip"]:
        df[c] = df[c].fillna("unknown").astype(str)
    df["business"] = df.dba_name.str.title()
    df["facility_type"] = df.facility_type.str.lower()
    df["risk"] = df.risk.str.replace("Risk ", "").str.lower()
    df["inspection_type"] = df.inspection_type.str.lower()
    df["label"] = (df.results == "Fail").astype(int)
    feats = ["business", "facility_type", "risk", "inspection_type", "zip"]
    df["text"] = df.apply(lambda r: f"{r.business}, a {r.facility_type} in Chicago ZIP {r.zip}. "
                          f"Risk category: {r.risk}. Inspection type: {r.inspection_type}.", axis=1)
    return df[feats + ["text", "label"]]


def nyc311():
    # Fetch every June complaint (one query per day, all rows) then sample; an unordered
    # $limit returns a time-clustered slice, not a random one.
    from concurrent.futures import ThreadPoolExecutor
    cols = "created_date,closed_date,agency,complaint_type,descriptor,location_type,borough,open_data_channel_type"
    day = lambda d: socrata("https://data.cityofnewyork.us/resource/erm2-nwe9.json",
                            f"created_date between '{d}T00:00:00' and '{d}T23:59:59'",
                            limit=50000, select=cols)
    with ThreadPoolExecutor(6) as ex:
        df = pd.concat(list(ex.map(day, NYC_DAYS)), ignore_index=True)
    created = pd.to_datetime(df.created_date)
    closed = pd.to_datetime(df.get("closed_date"), errors="coerce")
    df["label"] = ((closed - created) <= pd.Timedelta(days=7)).astype(int)
    df["weekday"] = created.dt.day_name()
    df["time"] = created.dt.hour.map(hour_word)
    for c in ["agency", "complaint_type", "descriptor", "location_type", "borough", "open_data_channel_type"]:
        df[c] = df.get(c, pd.Series(index=df.index, dtype=str)).fillna("unspecified").astype(str)
    df["channel"] = df.open_data_channel_type.str.lower()
    feats = ["agency", "complaint_type", "descriptor", "location_type", "borough", "channel", "weekday", "time"]
    df["text"] = df.apply(lambda r: f"311 complaint to {r.agency}: {r.complaint_type} ({r.descriptor}), "
                          f"location type {r.location_type}, {r.borough.title()}. Filed by {r.channel} "
                          f"on a {r.weekday}, {r.time}.", axis=1)
    return df[feats + ["text", "label"]]


def length_word(n, unit="characters"):
    for cut, w in [(1, "empty"), (200, "very short"), (800, "short"), (2500, "medium-length"), (1e12, "long")]:
        if n < cut:
            return w


def clean_text(s, n=300):
    import re
    s = re.sub(r"<!--.*?-->", " ", s if isinstance(s, str) else "", flags=re.S)   # PR templates hide instructions in comments
    s = re.sub(r"<[^>]+>", " ", s)
    import html
    s = re.sub(r"\s+", " ", html.unescape(s)).strip()
    return (s[:n] + "...") if len(s) > n else s


def hn():
    """Hacker News stories from 1-10 June 2026; label: reached 10+ points."""
    import html
    from urllib.parse import urlparse
    rows, start = [], 1780272000  # 2026-06-01 UTC
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
    df["time"] = created.dt.hour.map(hour_word) + " US Eastern"
    df["label"] = (df.points >= 10).astype(int)
    feats = ["domain", "kind", "weekday", "time"]
    df["text"] = df.apply(lambda r: f'Hacker News submission: "{r.title}" ({r.domain}). '
                          f"Posted on a {r.weekday}, {r.time}.", axis=1)
    return df[feats + ["text", "label"]]


ASSOC = {"MEMBER": "a member of the project's organization", "OWNER": "the repository owner",
         "COLLABORATOR": "a repository collaborator", "CONTRIBUTOR": "a previous contributor",
         "FIRST_TIME_CONTRIBUTOR": "a first-time contributor", "FIRST_TIMER": "a first-time GitHub contributor",
         "NONE": "someone with no prior association with the project"}
REPOS = ["microsoft/vscode", "huggingface/transformers", "rust-lang/rust", "kubernetes/kubernetes",
         "home-assistant/core", "vercel/next.js", "godotengine/godot", "ggml-org/llama.cpp", "facebook/react",
         "python/cpython", "langchain-ai/langchain", "ollama/ollama", "microsoft/TypeScript", "grafana/grafana",
         "vllm-project/vllm", "denoland/deno"]


def github_prs():
    """Closed (merged or not) human pull requests opened June 2026 in popular repos; label: merged."""
    import json
    import subprocess
    items = []
    for repo in REPOS:
        for page in (1, 2):
            out = subprocess.run(["gh", "api", "-X", "GET", "search/issues", "-f",
                                  f"q=repo:{repo} is:pr is:closed created:2026-06-01..2026-06-30",
                                  "-f", "per_page=100", "-f", f"page={page}"], capture_output=True, text=True)
            res = json.loads(out.stdout).get("items", [])
            items += [dict(it, repo=repo) for it in res]
            time.sleep(2.5)  # search API: 30 requests/minute
            if len(res) < 100:
                break
    df = pd.DataFrame(items)
    df = df[(df.user.map(lambda u: u.get("type")) != "Bot") & ~df.user.map(lambda u: u["login"].endswith("[bot]"))]
    df["label"] = df.pull_request.map(lambda p: p.get("merged_at") is not None).astype(int)
    # NB: author_association is reported as of *now*, so a merged PR's author has become a CONTRIBUTOR:
    # it leaks the label and is deliberately not used.
    df["draft"] = df.draft.fillna(False).map({True: "draft", False: "ready for review"})
    df["description"] = df.body.fillna("").map(lambda b: length_word(len(clean_text(b, 10 ** 6))))
    feats = ["repo", "draft", "description"]
    df["text"] = df.apply(lambda r: f'Pull request to {r.repo} titled "{r.title}" '
                          f"({r.draft}). Description ({r.description}): {clean_text(r.body, 250) or '(none)'}", axis=1)
    return df[feats + ["text", "label"]]


def rep_word(r):
    for cut, w in [(2, "brand-new account (1 reputation)"), (50, "new user (under 50 reputation)"),
                   (500, "occasional user (50-500 reputation)"), (5000, "established user (500-5k reputation)"),
                   (1e12, "very experienced user (over 5k reputation)")]:
        if r < cut:
            return w


def stackoverflow():
    """Stack Overflow questions asked in June 2026; label: has an accepted answer (as of late Sept)."""
    rows = []
    for d in range(1, 31):
        a = int(pd.Timestamp(f"2026-06-{d:02d}", tz="UTC").timestamp())
        for page in (1, 2):
            r = requests.get("https://api.stackexchange.com/2.3/questions", timeout=120, params={
                "site": "stackoverflow", "pagesize": 100, "page": page, "fromdate": a, "todate": a + 86400,
                "sort": "creation", "order": "desc", "filter": "withbody"}).json()
            rows += r.get("items", [])
            time.sleep(1)
            if not r.get("has_more"):
                break
    df = pd.DataFrame(rows)
    import html
    df["title"] = df.title.map(html.unescape)
    df["tags_str"] = df.tags.map(lambda t: ", ".join(t[:4]))
    df["main_tag"] = df.tags.map(lambda t: t[0] if t else "none")
    # NB: owner reputation is current, and accepting an answer gives the asker +2 -> leaks the label; not used.
    df["has_code"] = df.body.str.contains("<code>").map({True: "includes code", False: "no code"})
    df["length"] = df.body.map(lambda b: length_word(len(clean_text(b, 10 ** 6))))
    df["label"] = df.get("accepted_answer_id").notna().astype(int)
    feats = ["main_tag", "has_code", "length"]
    df["text"] = df.apply(lambda r: f'Stack Overflow question "{r.title}" tagged [{r.tags_str}]; '
                          f"{r.length}, {r.has_code}. Body: {clean_text(r.body, 250)}", axis=1)
    return df[feats + ["text", "label"]]


def crimes():
    """Chicago reported crimes, 2026; label: an arrest was made."""
    # one query per month so the sample spans the year (an unordered $limit is time-clustered)
    df = pd.concat([socrata("https://data.cityofchicago.org/resource/ijzp-q8t2.json",
                            f"date >= '2026-{m:02d}-01' AND date < '2026-{m + 1:02d}-01'", limit=50000,
                            select="date,primary_type,description,location_description,domestic,arrest,district")
                    for m in range(1, 9)], ignore_index=True)
    for c in ["primary_type", "description", "location_description"]:
        df[c] = df[c].fillna("unknown").str.lower()
    t = pd.to_datetime(df.date)
    df["time"] = t.dt.hour.map(hour_word)
    df["weekday"] = t.dt.day_name()
    df["domestic"] = df.domestic.map({True: "domestic", False: "not domestic"}).fillna("not domestic")
    df["district"] = "police district " + df.district.fillna("unknown").astype(str).str.lstrip("0")
    df["label"] = df.arrest.astype(bool).astype(int)
    feats = ["primary_type", "description", "location_description", "domestic", "district", "time", "weekday"]
    df["text"] = df.apply(lambda r: f"Reported {r.primary_type} ({r.description}) at a {r.location_description}, "
                          f"{r.domestic}, {r.district}, on a {r.weekday}, {r.time}.", axis=1)
    return df[feats + ["text", "label"]]


if __name__ == "__main__":
    import sys
    tasks = {"flights": flights, "shelter": shelter, "inspections": inspections, "nyc311": nyc311,
             "hn": hn, "github_prs": github_prs, "stackoverflow": stackoverflow, "crimes": crimes}
    for name, fn in [(t, tasks[t]) for t in (sys.argv[1:] or tasks)]:
        df = fn().dropna(subset=["label"])
        full = len(df)
        df = df.sample(n=min(N_PER_TASK, len(df)), random_state=SEED).reset_index(drop=True)
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"{name}: {full} rows -> kept {len(df)}, base rate {df.label.mean():.3f}")
        print("   e.g.", df.text.iloc[0])
