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

## 19:25–20:05 — E1 and E1-R complete for all 9 configurations
- **E1 (8 tasks):**
  - The preregistered per-task H1 criterion was met by 2/9 configurations (GPT-4o-mini, Llama). Haiku reached 3/8.
  - P3 (seeds): 27/27 significant cells have the same sign in all seeds.
  - P4 (wording): every configuration keeps its sign under both alternative wordings (seed 0; `told_w1/w2`). w2 ("Important calibration information…") is usually the worst.
- **E1-R (preregistered confirmatory, 5 fresh tasks):**
  - **R1 (told − const) is confirmed for all 9 configurations** in the predicted direction after Holm (8 positive; Gemini negative), and all seeds agree. The weakest is Jev, +0.006 [+0.001, +0.012], p=0.025, which just passes Holm as the largest p.
  - **R2 (told − shift) is > 0 for 9/9.** Gemini was predicted null and came out +0.004 [+0.000, +0.008], so that prediction was wrong, in the direction of the general claim.
- Messages sent to main: REFUTED (v1 Jev 5/8), NEW (R1/R2), and a correction (the "0 of 117 cells" count holds after Holm; before correction, 3 cells favour told).
- **Red team so far:**
  - **AUROC moderation.** told − const shrinks with the model's zero-shot AUROC. At AUROC > 0.75 (5 cells: Gemini nyc311 and crimes, Jev nyc311), told beats or ties the constant. told − shift stays > 0 in every band. So "worse than the constant" is a claim about the typical hard, low-signal task, while "code beats prompt" is general.
  - **Leave-one-task-out (E1-R):** every configuration keeps its sign whichever task is dropped.
  - **Elicitation:** the stated-probability versions of GPT-4o-mini and DeepSeek show the same sign as their logprob versions, with smaller effects. So the result is not only a one-token-logprob artefact.
- **E2 preregistered** (RESEARCH_AGENDA §4) and launched: deepseek-v4-pro (logprob), a boundary check at about $1.1.
- Spend so far: about $4.4.

## 20:00–20:27 — E2 boundary check; write-ups
- **E2 (DeepSeek-V4-Pro, logprob, preregistered §4), about $0.91:**
  - Mean zero-shot AUROC is 0.59 (E1) and 0.61 (E1-R), so E2b predicts told − const > 0. **Confirmed:** +0.051 and +0.042 [+0.033, +0.051].
  - **E2a (told − shift > 0) confirmed:** +0.030 and +0.029 [+0.019, +0.038]. All seeds agree.
  - Notable exploratory counterexample: on NYC 311 the model ranks well (AUROC 0.80), yet told scores Brier 0.29 against the constant's 0.16. Told it barely moves its mean (0.57 against a stated 0.76). So "ranks well ⇒ telling is fine" is false. The only escapee is a strong model that states its probabilities (Gemini), and that rests on n=1.
  - Analysis files for E2 are tagged `_e2`. They include 10 models, so their Holm family differs from the 9-configuration primary files.
- **Red-team numbers checked before they went into the README:**
  - AUROC bands with Haiku included: 95 cells ≤ 0.65 → 61 significantly worse, 1 better.
  - Platt vs shift: 38 better, 2 worse (Holm).
  - NYC 311 told means: 0.95–0.97 for the logprob GPT-4o-mini, DeepSeek and Llama.
- **P5 refuted:**
  - 64% (E1) / 73% (E1-R) of cells with AUROC ≥ 0.65 have shift < const, against a predicted ≥ 70%.
  - Shift is significantly worse than the constant at AUROC up to 0.70.
- **P4 holds for 8/9.** Jev fails in the direction of the claim: the wording "Important calibration information…" makes telling worse.
- Wrote:
  - README (neutral guide; one table, one chart; no vendor criticism);
  - WEBSITE_DRAFT.md;
  - DIGEST.md;
  - CLAIMS.md verdicts V1–V10.
- The v1 README is archived in `docs/`, with two evaluative vendor passages removed. The numbers are unchanged.
- v2 spend total: $5.23. Scan: no API key in any committed file.

## 20:28 — E3 (reasoning models) preregistered (§5) and launched
- The obvious objection: most configurations answered without reasoning. Pilot costs: deepseek-v4-flash-think $0.014/1k decisions, gpt-5-mini-think $0.082/1k.
- E3 runs on E1 and E1-R, with an estimated cost of about $2.4.

## 20:28–21:20 — E3 (reasoning) complete
- GPT-5-mini-think:
  - told − const +0.033 [+0.026, +0.041] (E1) and +0.015 [+0.008, +0.023] (E1-R);
  - told − shift +0.028 and +0.009 [+0.003, +0.014].
- DeepSeek-V4-Flash-think:
  - told − const +0.022 and +0.019;
  - told − shift +0.007 [+0.002, +0.012] and +0.011 [+0.006, +0.017].
- All seeds agree. **E3a and E3b are confirmed** for both configurations on both datasets.
- E3c (exploratory): DeepSeek with thinking ≈ without (E1-R told − shift +0.011 vs +0.010).
- Cost: GPT-5-mini-think $1.65 (about 2× the pilot estimate: more reasoning tokens per 25-case request); DeepSeek-think $0.46.
- Sent CONFIRMED (E3) to main. v2 total spend: $7.35.
- Extensions table: `results/v2/extensions_e2_e3.csv`. README, DIGEST, CLAIMS (V11) and the website draft are updated.
- Protocol deviation count: GPT-4o-mini-verbal needed the 5-case sub-batch fallback for 5 chunks (`results/v2/verbal_fallbacks.log`).

## 21:21 — final report and wrap-up
- `reports/progress/progress_02_2026-09-28_2121.pdf` (final, 5 pages) contains:
  - the replication tables (9 configurations × E1/E1-R, with seeds and task counts);
  - the extensions (E2, E3);
  - the chart;
  - the red-team summary;
  - the website draft;
  - survived vs killed claims;
  - next steps;
  - spend.
- Final v2 API spend: $7.35 (cap $30). v1 had spent $33.56 before this session.
- Nothing was pushed, and no remote was changed.
