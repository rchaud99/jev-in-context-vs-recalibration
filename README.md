# Put the base rate in code, not in the prompt

**A small, preregistered study of a common LLM-probability habit.** You ask an LLM for the probability that a case
is a "yes" (will this complaint close within a week, will this PR be merged). You also know roughly how often the
answer is yes, say 20%. Should you tell the model that number in its prompt, or leave the prompt alone and apply the
number to its scores in code?

**Answer, from 9 model configurations × 13 task datasets × 3 seeds.**
- **Applying the number in code was at least as good as telling the model in every configuration, and better in 8 of 9.** The code fix is one line of arithmetic: shift the model's log-odds so its average prediction equals the base rate.
- **For 8 of the 9 configurations, telling the model the number did worse than ignoring the model and predicting 20% for every case.**
- **The exception was the model that ranked these tasks best (Gemini 3.8 Flash).** Where a model ranks well, telling it the rate beats the constant, but it still doesn't beat doing it in code.

![Brier difference per model: told minus constant (left) and told minus code shift (right)](results/v2/fig_headline.png)

*Positive = telling the model is worse. Filled circles: the 8 original tasks (E1). Open squares: the preregistered
replication on 5 fresh September 2026 samples (E1-R). Bars: 95% paired-bootstrap CIs.*

## Setup
- **Tasks.** Binary outcomes from public 2026 records, each written as a one-line case description:
  - NYC 311 complaint closed within 7 days
  - Chicago crime report leads to an arrest
  - Sonoma shelter animal adopted or returned
  - Pittsburgh flight arrives more than 15 minutes late
  - Stack Overflow question gets an accepted answer
  - Hacker News story reaches 10 points
  - GitHub PR merged
  - Chicago food inspection fails

  Each task has 300 test cases and 300 unlabelled "pool" cases.
  - E1 used records from June–August 2026.
  - E1-R used 5 of the tasks re-drawn from **September 2026**, after every model's training cutoff. It was preregistered before any model saw that data.
- **Models: 9 configurations from 7 providers.** Each is pinned to one provider via OpenRouter.

  | Probability read from | Configurations |
  |---|---|
  | First-token log-probabilities | Qwen3.7-Flash, DeepSeek-V4-Flash, GPT-4o-mini, Llama-3.3-70B |
  | Stated in text | Gemini 3.8 Flash, Claude Haiku 4.5, and GPT-4o-mini and DeepSeek-V4-Flash asked this way too |
  | Native output | Jev 1.13, a decision model that returns probabilities directly |

  GPT-4o-mini and Llama-3.3 have 2023 training cutoffs, so they cannot have memorised these outcomes.
- **The base rate a team would know.** Each seed draws 100 labelled pool cases and rounds their yes-rate (e.g. "22%"). Three seeds give three slightly different numbers. Every arm gets the same number.
- **Arms:**
  - **told:** the prompt gains one sentence: *Base rate: across all cases like these, the answer to "…" is yes about 22% of the time.* Two other wordings gave the same result.
  - **code shift:** the prompt is unchanged. Add one constant b to the logit of each zero-shot probability, chosen so the average prediction over the 300 *unlabelled* pool cases equals 22%. It uses no labels beyond the number itself.
  - **constant:** predict 22% for every case.
  - **Reference arms,** trained on the same 100 labels:
    - Platt scaling of the zero-shot scores;
    - logistic regression on the cases' structured fields.
- **Statistics:**
  - Brier score, with log loss as a check.
  - Paired bootstrap over test cases (B = 5000), resampling within each task when pooling tasks.
  - Holm correction across models.
  - Preregistrations, predictions and every deviation are in [RESEARCH_AGENDA.md](RESEARCH_AGENDA.md) and [AUTONOMOUS_LOG.md](AUTONOMOUS_LOG.md).

## Results
Mean Brier difference over tasks. Positive means telling is worse.
- E1-R is the preregistered confirmatory replication. E1 is shown as a point estimate.
- Every E1-R difference below is significant after Holm across the 9 configurations.
- All 3 seeds have the same sign in every row.

| Model | Probability from | $ / 1k decisions | Zero-shot AUROC (E1 / E1-R) | Told − constant, E1 | **Told − constant, E1-R** [95% CI] | Told − code shift, E1 | **Told − code shift, E1-R** [95% CI] |
|---|---|---|---|---|---|---|---|
| Jev 1.13 | native | 0.004 | 0.59 / 0.62 | +0.008 | +0.006 [+0.001, +0.012] | +0.010 | +0.010 [+0.005, +0.015] |
| Qwen3.7-Flash | logprob | 0.005 | 0.58 / 0.60 | +0.044 | +0.061 [+0.049, +0.073] | +0.032 | +0.052 [+0.041, +0.064] |
| DeepSeek-V4-Flash | logprob | 0.006 | 0.55 / 0.57 | +0.073 | +0.060 [+0.048, +0.072] | +0.047 | +0.039 [+0.025, +0.052] |
| GPT-4o-mini | logprob | 0.020 | 0.55 / 0.57 | +0.141 | +0.127 [+0.110, +0.144] | +0.102 | +0.098 [+0.081, +0.117] |
| Llama-3.3-70B | logprob | 0.027 | 0.58 / 0.63 | +0.276 | +0.265 [+0.241, +0.289] | +0.232 | +0.232 [+0.209, +0.256] |
| GPT-4o-mini | stated | 0.009 | 0.52 / 0.54 | +0.052 | +0.048 [+0.037, +0.059] | +0.034 | +0.038 [+0.027, +0.050] |
| DeepSeek-V4-Flash | stated | 0.003 | 0.55 / 0.56 | +0.023 | +0.020 [+0.014, +0.027] | +0.012 | +0.010 [+0.004, +0.017] |
| Claude Haiku 4.5 | stated | 0.084 | 0.55 / 0.58 | +0.020 | +0.016 [+0.010, +0.023] | +0.013 | +0.010 [+0.005, +0.016] |
| Gemini 3.8 Flash | stated | 0.027 | 0.69 / 0.71 | −0.012 | −0.012 [−0.019, −0.006] | −0.001 | +0.004 [+0.000, +0.008] |

For scale: the constant's Brier is about 0.15 on a task where 20% of cases are yes, so +0.05 is about a third worse
than "always say 20%". Full tables, per-task counts and seeds are in
[results/v2/replication_table.md](results/v2/replication_table.md).

### What held up
**The code shift was never worse than telling.**
- Across all 117 model × task cells, telling was never significantly better than the code shift after Holm correction within each model. The code shift was significantly better in 47.
- This held whether the probability came from logprobs or was stated in text, for all three wordings of the sentence, and at every level of ranking skill.

**Why telling loses.**
- It moves the model's average prediction only partway toward the stated rate, and sometimes past it. On NYC 311, where the stated rate was 80–88%, the logprob versions of GPT-4o-mini, DeepSeek and Llama averaged 0.95–0.97.
- It doesn't improve ranking: AUROC is about the same told and zero-shot.
- The code shift sets the average exactly and keeps whatever ranking the model has.

**Whether telling beats the constant depends on ranking skill.**
- In the 5 cells where a model's zero-shot AUROC exceeded 0.75 (Gemini on NYC 311 and crimes, Jev on NYC 311), telling beat or tied the constant.
- In the 95 cells at or below 0.65, telling was significantly worse than the constant in 61 (57 after Holm within model) and better in 1.

### What didn't hold up
**My first preregistered test failed its own bar.**
- It required significance on at least 5 of 8 tasks *per model*. Only 2 of 9 configurations passed: GPT-4o-mini and Llama-3.3 (logprob).
- Per-task differences are often small, and the pooled pattern was found only after looking at E1. That is why it was re-tested on fresh data (E1-R), where it held.

**No calibration trick rescues a model that can't rank the task.** For most of these models, on these hard tasks, even the best fix doesn't beat the constant.
- Only Gemini's code-shifted scores clearly beat it.
- With 100 labels, Platt scaling brings the other models back to about the constant's level (within ±0.003 Brier), and rarely below it.
- Logistic regression on the same 100 labels beats the cheaper models' shifted scores.

## Practical recipe
1. **Don't put the base rate in the prompt to "calibrate" the model.** Keep the prompt about the case.
2. **Shift in code.** Compute `b` so that `mean(sigmoid(logit(p) + b)) = base_rate` over a batch of *unlabelled* recent inputs, then use `sigmoid(logit(p) + b)`. It needs no labels, just the number.
3. **Always compare against the constant** (predict the base rate for every case) and against logistic regression on whatever labels you have. Here, models beat the constant mainly where their AUROC was above about 0.7.
4. **With about 100 labels, fit Platt scaling** (2 parameters). It beat the shift in 38 of 117 cells and lost in 2. It helps most for logprob-read models, whose scores are too extreme.

## Limits
- **Tasks:** 8 task types from public records, with short, mostly structured case descriptions. Most models rank them only modestly (mean AUROC 0.52–0.71), and richer inputs may behave differently.
- **Prompt coverage:** one prompt template per elicitation style and three wordings of the base-rate sentence. Not tested: few-shot prompts, placing the rate in the system prompt, or chain-of-thought.
- **Uncertainty:** the CIs resample test cases, and the seeds vary the 100-label rate estimate. Model run-to-run noise is not resampled separately (most calls are at temperature 0).
- **Leak repairs:** an earlier version of this study had label leaks in the shelter, GitHub and crimes data. They were repaired before any of these runs (`v2/leak_audit.py`, `v2/build_data.py`). The GitHub PRs are the top search results per repo, not a random sample.
- **Versions:** results are for the model versions and providers listed, as of September 2026.
- **Earlier results:** an earlier, exploratory version of this project compared in-context examples with recalibration for one model. [CLAIMS.md](CLAIMS.md) lists which of its claims survived and which were killed, and the archived write-up is in [docs/](docs/README_v1_2026-09-27.md). Those results are not part of the claims above.

## Reproduce
```
python v2/build_data.py                 # repaired task data -> data/v2/   (from data/*.csv)
python v2/fresh_data.py                 # September 2026 replication data -> data/v2_fresh/  (public APIs)
cd v2
python baserate_exp.py <model>                            # E1 (calls cached in data/v2/cache.sqlite; needs OPENROUTER_API_KEY)
V2_DATASET=v2_fresh python baserate_exp.py <model>        # E1-R
python baserate_analyze.py && V2_DATASET=v2_fresh python baserate_analyze.py
python final_tables.py                  # results/v2/replication_table.md, results/v2/fig_headline.png
```
Every API call's cost is logged to `results/v2/spend_ledger.csv`, and the client stops at a hard cap. The API
response cache is not in the repo (see `.gitignore`).
