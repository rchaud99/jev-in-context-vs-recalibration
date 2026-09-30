> **Note added 2026-09-30:** the repository was reorganised after this was written; the text below is unchanged apart from this note and one correction marked [corrected]. `v2/` is now `src/v2/`, the top-level v1 scripts are in `src/v1/`, top-level `results/*` files are in `results/v1/`, and Markdown documents other than README.md and CLAIMS.md are in `docs/`. The original layout is on the `old` branch.

# DIGEST (2026-09-28)

**One finding, preregistered and replicated.** If you know a task's base rate, putting it in the LLM's prompt gives
worse probabilities than applying the same number in code (a one-parameter, label-free logit shift). For most models
on these tasks, telling them is even worse than predicting the base rate for every case.

**Surviving findings (numbers are Brier differences; + = telling is worse)**
1. **Told − constant > 0 for 8 of 9 configurations** in the preregistered fresh-data replication (E1-R: 5 September 2026 tasks × 3 seeds × 300 cases).
   - All are Holm-significant, and all seeds agree.
   - Range +0.006 (Jev) to +0.265 (Llama-3.3-70B logprob); GPT-4o-mini +0.127 [+0.110, +0.144], Qwen3.7-Flash +0.061 [+0.049, +0.073], Haiku 4.5 +0.016 [+0.010, +0.023].
   - The exception, as predicted, is Gemini 3.8 Flash: −0.012 [−0.019, −0.006].
   - The same pattern held in E1 (8 tasks).
2. **Told − code shift > 0 for 9/9 in E1-R** (Gemini tiny: +0.004).
   - Across 117 model × task cells, telling is never better after Holm; the code shift is better in 47.
   - It holds for logprob and stated-probability elicitation, and for three wordings.
3. **Boundary:** told beat or tied the constant only where the model ranked well (zero-shot AUROC > 0.75; 5 cells, Gemini and Jev). But ranking isn't sufficient.
   - E2 was preregistered with a stronger logprob model, DeepSeek-V4-Pro.
   - It confirmed told − constant +0.042 [+0.033, +0.051] and told − shift +0.029 [+0.019, +0.038] on the fresh data.
   - On NYC 311, at AUROC 0.80, telling was still far worse than the constant (Brier 0.29 vs 0.16).
4. **Reasoning doesn't rescue telling** (E3, preregistered).
   - GPT-5-mini at medium effort: told − shift +0.028 / +0.009, told − constant +0.033 / +0.015 (E1 / E1-R).
   - DeepSeek-V4-Flash with thinking behaves like its non-thinking twin.
5. **With about 100 labels, Platt scaling is the safest fix** (better than the shift in 38/117 cells, worse in 2). Logistic regression on those 100 labels beats the cheap models' shifted scores.

**Killed or failed (full list: CLAIMS.md)**
- My own E1 primary criterion (≥ 5/8 tasks per model) passed for only 2/9.
- The v1 "shift beats telling on 5/8 for Jev" → 3/8.
- The v1 "one sentence fixes most of it" had no constant baseline.
- Shelter, crimes and GitHub v1 results were inflated by label leaks, now repaired.
- All v1 Opus results are withdrawn (provider-route confound, contamination unchecked).
- P5 ("the shift beats the constant when AUROC ≥ 0.65") is refuted.
- The v1 "Jev is better" is false in general.

**Experiments run (all API, about $7.35 of the $30 cap; ledger: results/v2/spend_ledger.csv)**
- Leak screen and repairs.
- Pilot.
- E1: 9 configurations × 8 tasks × 3 seeds + 2 wordings.
- E1-R: 9 × 5 fresh tasks × 3 seeds.
- E2: a DeepSeek-V4-Pro boundary check.
- E3: two reasoning models.

**Résumé bullet.** Designed and preregistered a 9-model, 7-provider LLM calibration study on 13 public-record
datasets. Stating a base rate in the prompt gave worse probabilities than predicting the base rate itself (8/9
models). A one-line code-side logit shift was never worse (0/117 cells after Holm). The result was confirmed on a
fresh post-cutoff replication set. Along the way I found and repaired 3 label leaks and withdrew 11 earlier claims [corrected 2026-09-30: CLAIMS.md lists 12 entries, but K12 is an editorial removal, not a withdrawn finding].
Cost: about $7 of API calls.

**Next 3 experiments**
1. **High-signal tasks** (AUROC > 0.75 for most models, e.g. richer free-text inputs). Does "code beats prompt" persist where models rank well? Only 5 such cells exist now.
2. **Labelled examples instead of a base rate.** k in-context examples vs a zero-shot + shift/Platt fit on the same k, same 9 configurations. Does "put it in code" extend to examples?
3. **Placement and format.** The rate in the system prompt, or conveyed via a few labelled examples. Can any prompt make the model use the number as well as arithmetic does? (Reasoning is already tested in E3; it didn't help.)

**Website draft:** WEBSITE_DRAFT.md · **Replication table:** results/v2/replication_table.md · **PDFs:** reports/progress/
