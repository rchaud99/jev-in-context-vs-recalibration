> **Note added 2026-09-30:** the repository was reorganised after this was written; the text below is unchanged apart from this note. `v2/` is now `src/v2/`, the top-level v1 scripts are in `src/v1/`, top-level `results/*` files are in `results/v1/`, and Markdown documents other than README.md and CLAIMS.md are in `docs/`. The original layout is on the `old` branch.

# Don't tell the model your base rate. Do the math yourself.

*Draft for the personal website, about 500 words. Use `results/v2/fig_headline.png` as the chart.*

When you use an LLM to score cases ("will this ticket be resolved this week?", "will this pull request be
merged?"), its probabilities often run too high or too low. You know the real rate, say 20% of cases are yes. So
you add a sentence to the prompt: *"For context, about 20% of cases like these are yes."*

I tested whether that works, against a one-line alternative. Leave the prompt alone, and shift the model's scores
in code so that their average equals 20% (add the same constant to every score's log-odds). It needs no labelled
data, only the number you were going to put in the prompt.

**Setup.**
- **Tasks:** eight prediction tasks from public 2026 records, 300 test cases each:
  - NYC 311 complaints;
  - Chicago crime reports and food inspections;
  - Hacker News posts;
  - GitHub pull requests;
  - Stack Overflow questions;
  - shelter outcomes;
  - flight delays.
- **Models:** nine setups from seven providers. They include cheap models, and two whose training data ends in 2023,
  so they can't have memorised these outcomes.
- **Seeds:** three different estimates of the base rate, each from 100 labelled cases.
- **Scoring:** Brier score, with paired bootstrap confidence intervals, corrected for testing many models at once.

**What I found.**
1. **Doing it in code was never worse than telling the model, and usually better.** Across 117 model-task
   combinations, the prompt version never won after correction. The code version won 47.
2. **For eight of the nine setups, telling the model the base rate did worse than ignoring the model and answering
   "20%" for every case.** The damage ranged from +0.006 Brier to +0.27, where one model's error became nearly three
   times the constant's.
   - Telling moves the model's average only partway toward the stated number, or past it: several models jumped
     to about 95% when told 80–88%.
   - It doesn't help the model rank cases at all.
3. **The one exception was the model that ranked these tasks best** (Gemini 3.8 Flash). Good ranking alone wasn't
   enough, though. A stronger DeepSeek model ranked NYC 311 complaints well, yet telling it the rate still left it
   far worse than the constant.

**How I tried to break it.**
- **Preregistration.** I wrote down the analysis and predictions and committed them to git before running
  anything.
- **A failed first test.** My first criterion demanded a significant result on at least 5 of 8 tasks for each model.
  Only 2 of 9 met it.
- **A replication on untouched data.** The overall pattern could have been a fluke of those tasks. So I pulled
  fresh September 2026 data that no model or analysis had seen, and preregistered the test again. It held for all
  nine setups, in all three seeds.
- **Other checks.** It also held:
  - with three wordings of the sentence;
  - with probabilities read from the model's internals or stated in text;
  - when any one task was dropped;
  - for two models allowed to reason before answering.

**Honest limits.**
- These tasks are hard. Most models ranked them only modestly (AUROC 0.52–0.71).
- On tasks like these, no calibration trick makes a weak model beat the constant. A small logistic regression on
  100 labelled cases often did better.
- I tested short case descriptions and one prompt template per style, on September 2026 model versions.

**Takeaways.**
- Keep the prompt about the case.
- Put the base rate in code.
- Always check the model against the dumbest baseline, predicting the base rate for everyone. It is surprisingly
  hard to beat.

*Code, preregistrations and every result, including the tests that failed:* [repo link]
