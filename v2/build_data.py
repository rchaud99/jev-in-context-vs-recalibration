"""Build data/v2/<task>.csv: the v1 task data with the known label leaks repaired (no downloads, no API calls).

Repairs (see REVIEW_2026-09-27.md and results/v2/leak_audit.csv):
  shelter      drop `sex` (spay/neuter status is largely recorded after the outcome; alone it gives CV AUROC 0.78)
               and `intake_condition` (looks post-outcome) from both the features and the text.
  github_prs   drop `draft` ("draft" is the state at closing; 0/48 drafts merged) from features and text.
  crimes       drop police-initiated offence types, whose records exist because of an arrest
               (narcotics 98% arrested, weapons violation 76%, ...).
Other tasks are copied unchanged. Row order is preserved, so the first 300 rows (the test set) are the same cases as
v1 except for crimes, where the filtered rows shift up.
"""
import os
import re

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "v2")
os.makedirs(OUT, exist_ok=True)
POLICE_INITIATED = {"narcotics", "other narcotic violation", "weapons violation", "concealed carry license violation",
                    "interference with public officer", "liquor law violation", "public peace violation",
                    "gambling", "prostitution", "public indecency", "obscenity"}

for t in ["flights", "inspections", "shelter", "nyc311", "hn", "github_prs", "stackoverflow", "crimes"]:
    df = pd.read_csv(os.path.join(ROOT, "data", f"{t}.csv"))
    n0 = len(df)
    if t == "shelter":
        df = df.drop(columns=["sex", "intake_condition"])
        df["text"] = df.apply(lambda r: f"A {r['size']} {r.type}, {r.breed}, {r.color}, {r.age}. "
                              f"Intake: {r.intake_type} ({r.intake_subtype}), from {r.intake_jurisdiction}.", axis=1)
    if t == "github_prs":
        df = df.drop(columns=["draft"])
        df["text"] = df.text.str.replace(r" \((?:draft|ready for review)\)\. Description", ". Description", n=1, regex=True)
        assert not df.text.str.contains(r"\((?:draft|ready for review)\)").any()
    if t == "crimes":
        df = df[~df.primary_type.isin(POLICE_INITIATED)].reset_index(drop=True)
    df.to_csv(os.path.join(OUT, f"{t}.csv"), index=False)
    print(f"{t:14s} rows {n0} -> {len(df)}  base rate {df.label.mean():.3f}  test rate {df.label.iloc[:300].mean():.3f}")
    print("    e.g.", df.text.iloc[0][:160])
