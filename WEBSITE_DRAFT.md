# Don't tell the model your base rate. Do the math yourself.

*Draft for the personal website, about 500 words. Use `results/v2/fig_headline.png` as the chart.*

A common pattern when you use an LLM to score cases, like "will this ticket be resolved this week?" or "will this
pull request be merged?", goes like this. You notice the model's probabilities run too high or too low, and you
know the real rate: about 20% of cases are "yes". So you add a sentence to the prompt: *"For context, about 20% of
cases like these are yes."*

I tested whether that works, and compared it with a one-line alternative. Leave the prompt alone, and nudge the
model's scores in code so that their average equals 20%. Technically, you add the same constant to every score's
log-odds. It needs no labelled data, only the number you were going to put in the prompt.

**Setup.**
- **Tasks:** eight prediction tasks built from public 2026 records (NYC 311 complaints, Chicago crime reports and
  food inspections, Hacker News posts, GitHub pull requests, Stack Overflow questions, animal-shelter outcomes,
  flight delays), 300 test cases each.
- **Models:** nine model setups from seven providers, including cheap ones like Qwen3.7-Flash and DeepSeek-V4-Flash,
  and two whose training data ends in 2023, so they can't have memorised these outcomes.
- **Seeds:** three, each a different estimate of the base rate from 100 labelled cases.
- **Scoring:** Brier score, the standard error measure for probabilities, with paired bootstrap confidence intervals
  and a correction for testing many models at once.

**What I found.**
1. **Doing it in code was never worse than telling the model, and usually better.** Across 117 model-task
   combinations, the prompt version was never significantly better after correction. The code version was
   significantly better in 47.
2. **For eight of the nine setups, telling the model the base rate did worse than ignoring the model entirely and
   answering "20%" for every case.** The damage ranged from small (+0.006 Brier) to large (+0.27 Brier, leaving an error nearly three times the
   constant's error). Telling moves the model's average only partway toward the stated number, and sometimes past
   it: several models jumped to around 95% when told 80–88%. It doesn't help the model rank cases at all.
3. **The one exception was the model that ranked these tasks best** (Gemini 3.8 Flash). When a model can genuinely
   tell cases apart, telling it the base rate beats the constant. It still doesn't beat doing the math in code.

**How I tried to break it.**
- **Preregistration:** before running anything, I wrote down the analysis and predictions and committed them to
  git.
- **A failed first test:** my first preregistered criterion was strict, requiring a significant result on at least
  5 of 8 tasks for each model. Only 2 of 9 models met it.
- **Fresh data:** the overall pattern I then saw could have been a fluke of those tasks. So I pulled fresh
  September 2026 data that no model or analysis had touched, and preregistered the test again. It held for all
  nine setups, in all three seeds.
- **Other checks:** the result also held with three different wordings of the sentence, with probabilities read
  from the model's internals or stated in text, and when I dropped any one task.

**Honest limits.**
- These tasks are hard for most models: their ranking accuracy (AUROC) was only 0.52–0.71.
- On such tasks, no calibration trick makes a weak model beat the simple constant. A small logistic regression
  trained on 100 labelled cases often did better.
- I tested short case descriptions, one prompt template per elicitation style, and September-2026 model versions.
  Richer inputs or chain-of-thought might behave differently.

**Takeaways.**
- Keep your prompt about the case.
- Put what you know about base rates into a line of code.
- Always check your model against the dumbest possible baseline: predicting the base rate for everyone. It is
  surprisingly hard to beat.

*Code, preregistrations and every result, including the tests that failed:* [repo link]
