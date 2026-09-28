# Research agenda (v2): in-context information vs fixing the scores in code

Written 2026-09-28. This replaces the v1 framing, which compared one decision model against frontier LLMs. The guide
is now generic: *you have an LLM that returns probabilities and a little knowledge about your workflow (a base rate,
or k labelled cases). Should you put that knowledge in the prompt, or apply it to the scores in code?*

Goal (clarified by the user on 2026-09-28): produce one non-trivial finding that is
- statistically significant,
- replicated across seeds, ≥3 models from different providers (at least one cheap), and several tasks,
- beats the obvious baselines (a constant/prior predictor, and logistic regression trained on the same labels).

The finding should support a personal-website write-up and a résumé bullet. It is not paper material.
Depth on one finding beats breadth.

## 0. Validity fixes done before any new run
| Issue (REVIEW_2026-09-27.md) | Fix in v2 |
|---|---|
| Shelter label leak (`sex` = spay/neuter status, CV AUROC 0.78 alone; `intake_condition`) | Both dropped from features and text (`v2/build_data.py`) |
| GitHub `draft` is the state at closing (0/48 drafts merged) | Dropped from features and text |
| Crimes: police-initiated offence types exist because of an arrest (narcotics 98%) | Those types dropped (1,500 → 1,402 rows) |
| Other single-feature leaks? | `v2/leak_audit.py` screens every feature (CV AUROC alone). Also flagged: NYC 311 `agency=NYPD` (100% closed within 7 days). That is real operational knowledge available when the complaint is filed, not a leak, so it is kept. |
| LLM contamination unchecked | Two of the new models have 2023 training cutoffs (GPT-4o-mini, Llama-3.3-70B), so they cannot have seen 2026 outcomes. The finding is also a *contrast between two uses of the same model*, which memorised labels would not obviously favour. |
| Opus provider route varied by task | v2 uses no Opus. Every v2 model is pinned to one OpenRouter provider (`allow_fallbacks: false`), and the provider is logged per call. The v1 Opus claims are marked unverified in the killed-claims list. |
| Missing constant / k-matched LR baselines | Every v2 comparison includes a constant at the stated rate, and Platt scaling and one-hot logistic regression trained on the same 100 labels that produced the stated rate. |

## 1. Candidate questions, ranked by value ÷ cost
| # | Question | Value for the guide | Cost | Status |
|---|---|---|---|---|
| **E1** | **Tell the model the base rate in the prompt, or shift its scores in code (1 parameter, no labels)?** Replicate across 9 model configurations × 8 tasks × 3 seeds, with constant, Platt(100) and LR(100) baselines. | High: one concrete, actionable rule. v1 had exploratory support for one model (5/8 tasks). | ≈ $4 | **Chosen: preregistered below** |
| E2 | Same question with k labelled examples: do in-context examples move the mean only partway, while a code-side shift to the examples' rate fixes it? | Medium: extends E1 to the ICL setting | ≈ $3–6 (examples make prompts long) | If budget allows after E1 |
| E3 | Controlled free text vs structured fields (same task, LLM and LR each with and without the text) | Medium: tests "the LLM edge is free text" | ≈ $2 | Deferred (user: depth over breadth) |
| E4 | ICL saturation curve (k where gains stop) | Low–medium; v1 Jev data already reaches k=400 | ≈ $3+ | Deferred |
| E5 | Cost-matched comparison (tokens per Brier point) | Medium, mostly re-analysis | ≈ $0 | Folded into the write-up where the data allows |
| E6 | TabPFN baselines on the repaired data | Low for E1 | $0 (local CPU; no GPU/MLX needed) | Optional |

## 2. PREREGISTRATION — E1 (committed before any E1 data was collected)

**Hypothesis (one sentence).** Stating a known base rate to an LLM in its prompt gives worse probabilities (higher
Brier) than leaving the prompt alone and applying that same number in code as a one-parameter logit shift of the
model's zero-shot scores.

**What was seen before this registration (disclosed).**
- v1 exploratory data for Jev: the label-free shift beat telling on 5/8 tasks (`results/baseline_audit.md`).
- v1 exploratory data for Gemini 3.8 Flash (Google API): the two were about equal (differences ≤ 0.008 Brier; computed 2026-09-28 from cache).
- A pilot of 20 test cases × 2 tasks per model configuration (`v2/pilot.py`). In it, logprob-elicited chat models overshot the stated rate when told (e.g. GPT-4o-mini told 80% gave mean 1.00 on NYC 311), and verbalising models moved only partway.
- The pilot cases are part of the E1 test set. They are 40 of 2,400 cases per model; I do not exclude them.

**Models (9 configurations, 7 providers, each pinned to one provider).**

| Configuration | Provider | Elicitation |
|---|---|---|
| jev | TypeSafe | native Noul probabilities, 25 cases per request |
| qwen3.7-flash | Alibaba | P(yes) from first-token logprobs, one case per request, temperature 0, reasoning off |
| deepseek-v4-flash | StreamLake | as qwen3.7-flash |
| gpt-4o-mini | OpenAI | as qwen3.7-flash |
| llama-3.3-70b | Parasail fp8 | as qwen3.7-flash |
| gemini-3.8-flash | Google AI Studio flex | stated probability, 25 cases per request, JSON |
| claude-haiku-4.5 | Anthropic | stated probability, as gemini-3.8-flash |
| gpt-4o-mini-verbal | OpenAI | stated probability, as gemini-3.8-flash |
| deepseek-v4-flash-verbal | StreamLake | stated probability, as gemini-3.8-flash |

The two `-verbal` configurations are the same models as two of the logprob ones. They separate "which model" from "how the probability is elicited".

"Cheap" means ≤ $0.01 per 1,000 decisions in the pilot: jev, qwen3.7-flash and deepseek-v4-flash.

**Tasks.** The 8 v2 tasks (repaired data; 300 fixed test cases each).

**Seeds.**
- Seed s ∈ {0, 1, 2} draws 100 labelled cases L_s from the first 300 pool rows U. The generator is `default_rng(1000+s)`.
- The stated rate is r_s = round(100 · mean label of L_s)%.
- Seeds therefore vary the number a team would have estimated from 100 labels.

**Arms (per seed, scored on the 300 test cases).**
- **told:** state = task context + "Base rate: across all cases like these, the answer to "<question>" is yes about r_s% of the time."
- **shift (label-free):** logit(zero-shot p) + b_s, where b_s makes the mean shifted prediction over the 300 *unlabelled* pool inputs U equal r_s.
- **const:** r_s for every case.
- **platt:** 2-parameter Platt on the zero-shot scores of L_s. It uses `study.fit_platt`: logit standardised on L_s, L2 with C=1.
- **lr:** one-hot logistic regression (C=1) on L_s's structured features.
- Probabilities are clipped to [0.001, 0.999] before any logit.
- Exploratory arms:
  - shift fitted on the test inputs (transductive);
  - told followed by a transductive shift;
  - told with wordings w1 and w2 (seed 0 only).

**Primary test.**
- Statistic: for each model × task, Δ = mean over seeds of Brier(told_s) − Brier(shift_s). Δ > 0 means shift is better.
- CI: paired bootstrap over the 300 test cases, B = 5000, `default_rng(0)`, percentile 95% CI, p = 2 × the smaller tail share (+1 correction).
- Correction: Holm at α = 0.05 across the 8 tasks within each model. I also report Holm across all 72 model × task cells.
- **Replicates for a model** if shift is significantly better after Holm on ≥ 5/8 tasks and significantly worse on 0/8.
- **The finding counts as robust** if it replicates for models from ≥ 3 different providers, including ≥ 1 cheap model, and passes the seed and baseline checks below.

**Predictions.**
- **P1.** Replicates for all five logprob/native configurations: jev, qwen3.7-flash, deepseek-v4-flash, gpt-4o-mini and llama-3.3-70b.
- **P2.** Does *not* replicate for gemini-3.8-flash (≤ 2/8 tasks significant in either direction). There is no firm prediction for the other three verbalised configurations. The weak expectation is Δ > 0 on most tasks, with fewer significant cells than the logprob configurations.
- **P3 (seeds).** In every Holm-significant cell, the per-seed Δ has the same sign for all 3 seeds.
- **P4 (wording).** For each model, the number of significant tasks with wordings w1 and w2 is within ±2 of w0's.
- **P5 (non-trivial vs constant).** Among model × task cells where the model's zero-shot test AUROC ≥ 0.65, shift has lower Brier than const in ≥ 70%. Shift is significantly worse than const only where AUROC < 0.6.
- **P6 (labels).**
  - Logprob chat models: Platt(100) beats shift on ≥ half the tasks (their scores are too extreme, which one parameter cannot fix).
  - Jev and the verbalised models: no significant Platt − shift difference on most tasks.
  - LR(100) vs shift: exploratory, no prediction.

**Secondary metric.** Log loss (clipped 0.001), same test.

**Exploratory mechanism analysis.**
- How far told misses the stated rate: mean(told) − r_s.
- Whether telling changes ranking: AUROC(told) − AUROC(zs).

Anything not listed above is labelled exploratory in the write-up.

**Stopping rules.**
- On HTTP 402 or when the v2 ledger reaches the cap (`v2/common.py`: `CAP_USD`), stop cleanly. Keep whole models complete rather than a ragged grid.
- Run order is cheapest first: jev, qwen, deepseek, gpt-4o-mini, llama, gpt-4o-mini-verbal, deepseek-verbal, gemini, haiku.

**Code.**
- `v2/common.py`: clients, ledger, cache.
- `v2/baserate_exp.py`: data collection.
- `v2/baserate_analyze.py`: all arms and statistics. Written before any E1 data exists; only bug fixes may change it afterwards, and each fix is logged.

## 3. PREREGISTRATION — E1-R: confirmatory replication on fresh September 2026 data
Written 2026-09-28 at 19:23 (commit cfc2884), before any model was run on the fresh data. At that point E1 was complete for 7 of 9
configurations (DeepSeek-logprob and Haiku still running).

**Why.** In E1, the preregistered per-task criterion (≥ 5/8 tasks after Holm) was met by only 2 configurations
(GPT-4o-mini and Llama-3.3-70B, both logprob). Jev reached 3/8, so the v1 "5/8" did not replicate on the repaired
data. Two patterns that were *not* the preregistered primary test stood out:
- (a) told − shift, averaged over tasks, was significantly > 0 for 6 of 7 configurations; Gemini was null.
- (b) told − const was significantly > 0 for 6 of 7: telling the model the base rate gave probabilities *worse than
  outputting the base rate itself*. The exception was Gemini, where told beat the constant.

These are post hoc. E1-R tests them on data no model or analysis has touched.

**Data.** `data/v2_fresh/` (`v2/fresh_data.py`): the same sources, templates and leak repairs.
- Windows: September 2026, after every model's training cutoff. See the script for each task's window.
- Tasks: nyc311, crimes, hn, github_prs and inspections, the tasks with ≥ 600 fresh rows.
- Excluded:
  - shelter (225 rows) and Stack Overflow (347 rows): too few rows;
  - flights: September BTS data is not published yet.
- Split: first 300 rows = test; U = next 300 = unlabelled pool.
- Protocol: identical to E1 (`V2_DATASET=v2_fresh`) — same prompts, seeds (100-label rate estimate), arms and clipping. No wording variants.

**Models.** All 9 E1 configurations, same pinned providers.

**Primary tests.** Per model: the mean over the 5 tasks of the per-task, seed-averaged Brier difference. Bootstrap resamples the 300 test cases within each task (B=5000); p = 2 × the smaller tail; Holm across the 9 configurations.
- **R1:** Brier(told) − Brier(const) > 0.
  - Predicted > 0 for: jev, qwen3.7-flash, gpt-4o-mini, llama-3.3-70b, gpt-4o-mini-verbal, deepseek-v4-flash-verbal.
  - Predicted < 0 for gemini-3.8-flash.
  - For deepseek-v4-flash and claude-haiku-4.5: predicted to have the sign of their E1 pooled estimate. This is fixed mechanically when their E1 finishes, before their E1-R analysis is looked at.
- **R2:** Brier(told) − Brier(shift) > 0.
  - Predicted > 0 for every configuration except gemini-3.8-flash (predicted null: CI includes 0).
  - deepseek-v4-flash and claude-haiku-4.5: as for R1.

**Criterion for calling the headline claim (R1) robust.**
- Holm-significant in the predicted direction for ≥ 5 configurations;
- those configurations span ≥ 3 providers and include ≥ 1 cheap model (jev, qwen or deepseek);
- the per-seed estimates of the pooled difference have the same sign in all 3 seeds for each of them;
- no configuration is Holm-significant in the direction opposite to its prediction.

The same criterion applies to R2.

**Secondary (reported, no criterion).**
- Per-task Holm counts for R1 and R2.
- Shift − const, Platt(100) − const, LR(100) − const, Platt(100) − LR(100).
- Everything combined over E1 + E1-R (13 task datasets).

**Mechanical prediction fix, 19:39 (before any E1-R analysis of this model):** deepseek-v4-flash's E1 pooled estimates
are told − const +0.073 and told − shift +0.047, both > 0. So its R1 and R2 predictions are > 0.

**Mechanical prediction fix, 19:44 (before any E1-R analysis of this model):** claude-haiku-4.5's E1 pooled estimates
are told − const +0.020 and told − shift +0.013, both > 0. So its R1 and R2 predictions are > 0.

## 4. PREREGISTRATION — E2: boundary check with a stronger logprob model
Written 2026-09-28, about 19:47, before any E2 data. The only pilot data is 20 cases × 2 tasks (`v2/pilot.py`).

**Why.**
- In E1/E1-R, the one configuration whose told probabilities beat the constant was also the best ranker (Gemini 3.8 Flash).
- Across all cells, told − const shrinks as the model's zero-shot AUROC rises (exploratory table in `results/v2/replication_table.md`).
- Is "telling doesn't beat the constant" about model strength, or about elicitation (Gemini states its probabilities)?

**Model.** deepseek-v4-pro, pinned to StreamLake fp8, first-token logprobs, reasoning off (about $0.045 per 1,000 decisions).

**Data and protocol.** Identical to E1 (8 tasks, wordings included) and E1-R (5 fresh tasks), with the same seeds, arms and statistics.

**Predictions.**
- **E2a:** told − shift > 0 (pooled) on both datasets. Every logprob configuration showed this.
- **E2b:** told − const is conditional on ranking. If deepseek-v4-pro's mean zero-shot AUROC over the tasks of a dataset is ≥ 0.70, told − const ≤ 0 on that dataset (the CI includes 0 or lies below it). Otherwise told − const > 0.

This is a single-model check. It is reported as such and cannot by itself establish the boundary.
