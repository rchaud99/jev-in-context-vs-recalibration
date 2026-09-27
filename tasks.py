"""Task definitions shared by the pilot and the main study."""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

TASKS = {
    "flights": dict(
        context="Flights departing Pittsburgh International Airport (PIT) in June 2026.",
        question="Will this flight arrive more than 15 minutes late?",
        yes="arrived more than 15 minutes late", no="arrived on time (within 15 minutes)",
    ),
    "inspections": dict(
        context="City of Chicago food establishment inspections in 2026.",
        question="Will this establishment fail the food inspection?",
        yes="failed the inspection", no="passed the inspection",
    ),
    "shelter": dict(
        context="Animals leaving the Sonoma County (California) animal shelter in 2026.",
        question="Will this animal be adopted or returned to its owner?",
        yes="was adopted or returned to its owner", no="had another outcome (transfer, euthanasia, etc.)",
    ),
    "nyc311": dict(
        context="New York City 311 service requests filed in June 2026.",
        question="Will this complaint be closed within 7 days of being filed?",
        yes="was closed within 7 days", no="was not closed within 7 days",
    ),
    "hn": dict(
        context="Stories submitted to Hacker News (news.ycombinator.com) in early June 2026.",
        question="Will this submission receive at least 10 points (upvotes)?",
        yes="received 10 or more points", no="received fewer than 10 points",
    ),
    "github_prs": dict(
        context="Pull requests opened on popular open-source GitHub repositories in June 2026 and since closed.",
        question="Was this pull request merged (rather than closed without merging)?",
        yes="was merged", no="was closed without merging",
    ),
    "stackoverflow": dict(
        context="Questions asked on Stack Overflow in June 2026.",
        question="Will this question end up with an accepted answer?",
        yes="got an accepted answer", no="never got an accepted answer",
    ),
    "crimes": dict(
        context="Crimes reported to the Chicago Police Department in 2026.",
        question="Was an arrest made for this reported crime?",
        yes="an arrest was made", no="no arrest was made",
    ),
}

TABULAR = ["flights", "inspections", "shelter", "nyc311", "crimes"]
TEXT = ["hn", "github_prs", "stackoverflow"]

N_TEST = 300


def load(task):
    """Returns (pool, test): a fixed held-out test set and the pool that labeled examples are drawn from."""
    df = pd.read_csv(os.path.join(ROOT, "data", f"{task}.csv"))
    feats = [c for c in df.columns if c not in ("text", "label")]
    df[feats] = df[feats].astype(str)
    return df.iloc[N_TEST:].reset_index(drop=True), df.iloc[:N_TEST].reset_index(drop=True), feats
