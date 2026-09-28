# Autonomous lab notebook (v2), started 2026-09-28

Times are local (EDT). Costs come from the v2 spend ledger (`results/v2/spend_ledger.csv`), which records the per-call cost OpenRouter returns.

## 2026-09-28 ~18:30–18:44 — orientation (early times approximate)
- Read in full:
  - README, STUDY_PLAN, REVIEW_2026-09-27, NOVELTY_CHECK;
  - verify_claims.py, baseline_audit.py (+ its output), baserate_ci.py, contamination_check.py;
  - the main pipeline: study.py, jev.py, llm.py, gemini_study.py, opus_study.py, prepare_data.py, tasks.py.
- Git: one commit, plus untracked review files. There is an `origin` remote (GitHub). I will **never push**.
- **Credit:** the OpenRouter account had only about $16 left and another process was spending on the same key. I told main. The user then added $100, so the $30 cap from the brief stands. The coordinator had briefly set $8; `CAP_USD` in `v2/common.py` stays at 8 until a larger run needs it.
- Goal clarification from the user (via coordinator): one non-trivial, significant, robust finding, replicated across seeds × ≥3 models (multiple providers, one cheap) × tasks. Depth over breadth. Website write-up + résumé bullet, not a paper.
- Committed the pre-existing untracked review files (291966f).

## ~18:45 — leak screen (`v2/leak_audit.py`, $0)
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

## ~18:47 — choosing the single finding
Candidates:
- (a) telling the base rate vs a code-side shift;
- (b) ICL ≈ LR on structured data / an edge on free text;
- (c) Platt vs ICL.

Picked **(a)**. It is cheap to replicate widely (zero-shot-sized prompts) and gives a practical rule. It already had exploratory support for Jev: shift beat telling on 5/8 tasks (Holm).

Exploratory check on the v1 Gemini cache: told ≈ shift (|Δ| ≤ 0.008). So the finding may be model- or elicitation-dependent. That is why the preregistration predicts a null for Gemini and adds two same-model "verbal" configurations as a control.

## ~18:48–18:56 — clients and pilot (`v2/pilot.py`, $0.024)
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

## 18:58–19:09 — E1 data collection (running)
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

## 19:09–19:13 — collection issues (times corrected from the system clock)
- **Claude Haiku 4.5:** hit OpenRouter's new-account limit of 20 requests/minute and crashed after retries. `_post` now waits 20 s on HTTP 429 without using up an attempt (up to 60 waits). Haiku restarted with 2 workers.
- **DeepSeek V4 Flash:** StreamLake latency spikes of up to 27 s per call left it at about 70 calls/min. Restarted with 48 workers (`V2_WORKERS`). Everything already fetched is cached.
- **GPT-4o-mini-verbal:** still failed on a chunk after 10 retries (it miscounts 25 cases). Added a logged fallback: re-ask that chunk in sub-batches of 5 (`results/v2/verbal_fallbacks.log`). This is a protocol deviation for those cases only.
- Three untracked diagnostic DeepSeek calls (about $0.00003) are not in the ledger.
- **Gemini complete (8/8):**
  - H1: 0/8 tasks significant in either direction. The pooled Δ (post hoc) is −0.0007 [−0.0035, +0.0020]. **P2 (null for Gemini) confirmed.**
  - Shift beats const on 2/8 (Holm) and told beats const on 2/8.
- **Post hoc test added before seeing 7 of the 9 configurations:** a per-model mean of Δ over the 8 tasks. The bootstrap resamples test cases within each task, with Holm across models. It is labelled post hoc everywhere.
- **Qwen complete (8/8):** H1 is significant on 3/8. Post hoc pooled Δ +0.032 [+0.024, +0.040]. **But shift is *worse* than the constant for Qwen** (pooled +0.012): its zero-shot ranking is too weak and its scores too extreme.

## 19:13–19:25 — E1 interim (7/9) → E1-R preregistered and launched
- **E1 H1 (preregistered per-task criterion):**
  - Replicates only for GPT-4o-mini (5/8) and Llama-3.3-70B (6/8).
  - Jev 3/8, Qwen 1/8, GPT-4o-mini-verbal 2/8, DeepSeek-verbal 3/8, Gemini 0/8.
  - Told was never significantly better than shift in any cell.
  - **P1 is refuted** for Jev and Qwen (and was not tested as P1 for the verbal configurations). **P2 (Gemini null) is confirmed.**
- **Post hoc pooled** (mean over 8 tasks; bootstrap within task; Holm over models):
  - told − shift > 0 for 6/7 (not Gemini);
  - **told − const > 0 for 6/7**: telling the rate is worse than outputting the rate. Only Gemini's told beats the constant.
- On the repaired data, only Gemini (and LR(100) on 4/8 tasks) clearly beats the constant. The cheaper models' zero-shot scores carry little signal beyond the base rate here.
- **Fresh data (`v2/fresh_data.py`, $0):** September 2026 pulls, same templates and repairs.
  - 5 tasks with ≥600 rows: nyc311, crimes, hn, github_prs, inspections.
  - Shelter (225 rows) and SO (347) are too small; flights is not published yet.
- **E1-R preregistered** (RESEARCH_AGENDA §3, commit cfc2884) as a confirmatory test of the two post hoc patterns on untouched, post-cutoff data. Launched for 8 configurations at 19:25; Haiku follows when its E1 run ends (20 rpm limit).
