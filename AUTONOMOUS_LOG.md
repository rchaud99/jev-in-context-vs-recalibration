# Autonomous lab notebook (v2), started 2026-09-28

Times are local (EDT). Costs come from the v2 spend ledger (`results/v2/spend_ledger.csv`), which records the per-call cost OpenRouter returns.

## 2026-09-28 ~18:00 — orientation
- Read in full:
  - README, STUDY_PLAN, REVIEW_2026-09-27, NOVELTY_CHECK;
  - verify_claims.py, baseline_audit.py (+ its output), baserate_ci.py, contamination_check.py;
  - the main pipeline: study.py, jev.py, llm.py, gemini_study.py, opus_study.py, prepare_data.py, tasks.py.
- Git: one commit, plus untracked review files. There is an `origin` remote (GitHub). I will **never push**.
- **Credit:** the OpenRouter account had only about $16 left and another process was spending on the same key. I told main. The user then added $100, so the $30 cap from the brief stands. The coordinator had briefly set $8; `CAP_USD` in `v2/common.py` stays at 8 until a larger run needs it.
- Goal clarification from the user (via coordinator): one non-trivial, significant, robust finding, replicated across seeds × ≥3 models (multiple providers, one cheap) × tasks. Depth over breadth. Website write-up + résumé bullet, not a paper.
- Committed the pre-existing untracked review files (291966f).

## 18:20 — leak screen (`v2/leak_audit.py`, $0)
- Single-feature 5-fold CV AUROC over all 1,500 rows per task, with one-hot LR.
- Flags:
  - shelter `sex` 0.78: the known leak;
  - NYC 311 `agency=NYPD` 100% closed: kept, since it is known at filing and is a real operational pattern;
  - crimes `narcotics` 98% arrested: police-initiated;
  - GitHub `draft` 0/48 merged: the known leak.
- `v2/build_data.py` writes `data/v2/`:
  - shelter drops `sex` and `intake_condition`;
  - GitHub drops the draft marker from features and text;
  - crimes drops police-initiated types (1,500 → 1,402 rows, base rate 0.14 → 0.10).
  - v1 data is untouched.

## 18:30 — choosing the single finding
Candidates:
- (a) telling the base rate vs a code-side shift;
- (b) ICL ≈ LR on structured data / an edge on free text;
- (c) Platt vs ICL.

Picked **(a)**. It is cheap to replicate widely (zero-shot-sized prompts) and gives a practical rule. It already had exploratory support for Jev: shift beat telling on 5/8 tasks (Holm).

Exploratory check on the v1 Gemini cache: told ≈ shift (|Δ| ≤ 0.008). So the finding may be model- or elicitation-dependent. That is why the preregistration predicts a null for Gemini and adds two same-model "verbal" configurations as a control.

## 18:40–18:55 — clients and pilot (`v2/pilot.py`, $0.024)
- Clients live in `v2/common.py`: pinned providers, an sqlite cache, an append-only ledger with a hard cap, and 402 → clean stop.
- Pilot: 20 cases × {NYC 311, Stack Overflow} × {zero-shot, told} for each of 9 configurations.
- Issues and fixes:
  - Alibaba caps `top_logprobs` at 5, so all configurations use 5.
  - Llama-3.3-70B on Novita often returned no logprobs, so it is pinned to Parasail fp8 (0 missing).
- Observations:
  - Logprob chat models overshoot when told: GPT-4o-mini went to mean 1.00 on NYC 311 when told 80%; DeepSeek 0.96.
  - Verbalising models move only partway.
- Cost per 1,000 decisions: Jev $0.005, Qwen $0.005, DeepSeek $0.007, GPT-4o-mini $0.021, Llama $0.026, Gemini flex $0.038.

## 18:57 — preregistration
- Wrote RESEARCH_AGENDA.md (ranked agenda + PREREGISTRATION for E1), `v2/baserate_exp.py` and `v2/baserate_analyze.py`. All were committed before any E1 data was collected.
- Estimated E1 cost for all 9 configurations: about $4.

## 19:05–19:40 — E1 data collection (running)
- Order: jev → qwen → deepseek (chained), then gpt-4o-mini, llama, and the verbalising configurations in parallel background processes. Each model uses a different provider. Logs are in `logs/v2/`.
- **Bug fix 1 (analysis, no effect on the statistics):** `baserate_analyze.py` subtracted numpy booleans in the per-seed significance flag, which raised a TypeError. Cast to int.
- **Bug fix 2 (collection):** GPT-4o-mini-verbal sometimes returns 24 or 26 probabilities for 25 cases, and 4 retries were not enough on crimes. Raised to 10 retries. Replies are only ever accepted when the count matches, so this adds no selection on content. One debug call ($0.0003) is logged in the ledger under `debug`.
- **First look, Jev (all 8 tasks):** shift beats told significantly after Holm on 3/8 (NYC 311, flights, inspections), told never significantly better; mean Δ +0.0097.
  - **P1 fails for Jev** (the preregistered criterion needs ≥5/8).
  - The repaired data lowers Jev's zero-shot AUROC:
    - shelter 0.64 → 0.53: the leak also inflated zero-shot ranking;
    - crimes 0.78 → 0.68;
    - GitHub 0.53 → 0.49.
  - Differences between arms shrink where the model can't rank.
- **Qwen (7/8 so far):** 3/7 significant, including HN +0.20 Brier (telling makes Qwen's HN scores much worse).
