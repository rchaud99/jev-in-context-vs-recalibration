> **Note added 2026-09-30:** the repository was reorganised after this was written; the text below is unchanged apart from this note. `v2/` is now `src/v2/`, the top-level v1 scripts are in `src/v1/`, top-level `results/*` files are in `results/v1/`, and Markdown documents other than README.md and CLAIMS.md are in `docs/`. The original layout is on the `old` branch.

# Jev few-shot calibration study — plan (agreed 2026-09-27)

## Goal
A real experiment (uncertain outcome, reportable either way) that can replace the
"Visual grounding" bullet in the resume's Agent Reliability Studies. Also publish on
GitHub/LinkedIn Featured and submit to the Jev robustness list. Move fast: ~110
independent Jev studies already exist and the space fills up daily.

## Question
Jev ranks cases well but its probabilities are miscalibrated for any given workflow.
With only k labeled examples from your own workflow, what's the cheapest fix:
**put the k examples in Jev's context**, or **leave Jev unchanged and fit a 1–2 parameter
correction** (Platt / temperature) to its output probabilities?
Recalibration can only reshape Jev's numbers (ranking unchanged); in-context examples
can also improve ranking, but only if Jev genuinely learns from them. So the result also
answers "does Jev learn from examples at all?"

## Design
- Data: 3–4 real labeled datasets, binary outcome, not famous (avoid memorized
  datasets like Titanic/UCI Adult). Candidate: flight delays at a mid-size airport,
  a public loan-default set. Bin numeric features into words (Jev is weak with numbers).
- Budgets: k = 5 … 100 labels; multiple random draws per k; shuffle row order.
- Arms (same k labels each):
  1. Jev zero-shot (no fix)
  2. Jev + k examples in context
  3. Jev + Platt/temperature correction fit on k labels
  4. Jev + examples in context + correction (both)
  5. No-Jev baselines trained on the k labels: logistic regression, TabPFN
- Scoring on the same held-out cases: Brier, ECE (calibration) + AUROC (ranking).
- Optional sanity task: one semi-synthetic dataset (fit a model with interactions /
  small tree ensemble to real data, sample outcomes) so "distance to true probabilities"
  is measurable.

## Jev API specifics
- On OpenRouter: model `typesafe/jev-1.13` (also `jev-latest`, `jev-router`);
  $0.042 / M input tokens, output free. Needs OPENROUTER_API_KEY (not set yet).
- Use **Noul** (yes/no probability) per case, not Choice — Choice is badly overconfident
  (82.9% on its pick at 19% accuracy on a hidden fair die; Noul stayed near 1/6).
- Put examples in the state; ask each test case as its own question in the same request
  (questions are evaluated independently) → whole test set per request, study costs ~$ few.
- Many rows in one state reportedly degrades answers (40 rows broke a ranking gate):
  measure degradation as rows grow; if it breaks at large k, that's a finding.
- Identical requests change 1–3% of decisions run-to-run: repeat calls, bootstrap CIs.

## Checks before building
1. 15-min novelty check: in-context examples vs Platt recalibration for Jev at small k.
   Known nearby: jev-certify, evaljev (recalibration tooling); alexmolas.com
   "Jev can't be calibrated" (recommends Platt on a few hundred labels);
   a multi-armed bandit Jev study (read first).
2. Verify the premise "Jev ranks well" (AUROC) per dataset.

## Novelty check result (2026-09-27): core comparison looks OPEN
Both halves have been studied, but never against each other.
- Recalibration learning curves over label count:
  - AnthusAI/Jev-Calibration (sentiment text, n = 20…5,280): Platt ECE went 0.100 → 0.052; isotonic went 0.089 → 0.008 and matched or beat Platt at every n.
  - gkastanis/d3code-calibration: isotonic fixed about half the error with 25 labels; calibration didn't transfer across datasets.
- Few-shot, accuracy only: leepokai/llm-prompt-techniques-on-jev (LegalBench 81 → 89 at 5 shots, 95 at 50). No ECE/Brier/AUROC and no recalibration arm; examples hurt MMLU-Pro.
- Tabular, zero-shot only: rubinagentagi-tech/jev-heart-risk-bench (BRFSS, AUC 0.773, Brier skill −1.3). No few-shot and no small-k baselines.
- Gap: nobody compares in-context examples with recalibration at the same small k, reports AUROC as examples are added, or tests tabular few-shot against LR/TabPFN on the same k.
- Design changes from the check:
  - Add an isotonic arm.
  - Check batched against one-case-per-request answers (evaljev found batching shifts outputs).
  - Don't assume the direction of miscalibration (studies disagree).

## Additions agreed in design review
- Base-rate arm: predict the mean of the k labels for every case.
- Shuffled-label control: same examples, labels scrambled. It separates learning from features from learning only the base rate.
- Tasks:
  1. BTS flights (June 2026, mid-size airport; late > 15 min)
  2. Chicago food inspections (2026; fail)
  3. Sonoma animal shelter (2026; adopted or returned to owner)
  4. NYC 311 complaints (2026; closed within 7 days)
- Gate: zero-shot AUROC ≥ ~0.6 on 200 cases per task.
- API: POST https://openrouter.ai/api/alpha/decisions with {model, state, questions}. Noul values are rounded to 2 decimals. Context is 32k. Key is in .env.

## Arm H (added 2026-09-27): self-feedback
In-context examples annotated with Jev's own zero-shot estimate and the outcome
("Your earlier estimate: 82% (likely) chance of yes / Actual outcome: no ..."). It has the same
information as the Platt arm, so it asks: can Jev correct itself in context as well as a
2-parameter fit can?

## Pilot results (results_pilot.txt, 200 test cases/task, zero-shot, $0.016)
| Task | AUROC | Result |
|---|---|---|
| flights | 0.61–0.66 | Weak; Brier ≈ base rate |
| shelter | 0.66 | |
| nyc311 | 0.78 | Jev's mean p is 0.54 against a base rate of 0.81, so its Brier is worse than the base rate |
| inspections | 0.37 (INVERTED) | Jev expects re-inspections and high-risk places to fail; in the data, re-inspections almost always pass and medium risk fails most. Kept on purpose, as the test of whether examples can override a wrong prior. |

- Batched vs one-per-request: differences are about 0.01, the same size as repeat noise, so batching (25 questions per request) is fine.
- Repeat noise: 65–74% of probabilities change between identical calls, by about 0.01 on average (max 0.08).
- Jev accepts at least 256 questions per request.

## Code
- `prepare_data.py`: builds data/*.csv
- `tasks.py`
- `jev.py`: client with a sqlite cache
- `pilot.py`
- `study.py`: every Jev arm plus LR and base rate
- `baselines_tabpfn.py`: runs in `.venv_tabpfn2` with TabPFN 2.0.9, because newer versions gate weights behind a license
- `analyze.py`: bootstrap over draws and test cases, writes curves.png

## Expected result (honest)
Correction probably wins on calibration at every k (very data-efficient); in-context
examples may nudge ranking slightly or hurt. "Meh but actionable" → bullet ends in a
recommendation ("put examples in the prompt, or fit two numbers"). Surprise chance ~25%.

## Next steps
1. User sets key: `export OPENROUTER_API_KEY=...`
2. Novelty check + propose 3–4 datasets for approval
3. Build harness in this folder (Python; user has anaconda, M4 Max, MLX)
4. Run, compare final bullet against the visual-grounding bullet, keep the stronger

## Rejected ideas (don't re-pitch)
- Alien NPC game video (pure "cool"; games/NPCs saturated: Doom, StarCraft, Pokémon…)
- Jev as calibrated guard/checkpoint in τ²-bench ("not a real experiment")
- Jev as RAG reranker/search (done many times)
- Freehand silent-drift monitor study (user: "forget Freehand")
- Semi-synthetic "how fast does Jev grok true probabilities vs LLMs" as the main study
  (superseded by the design above; survives only as the optional sanity task)
- Guardrails, linting, routing, standalone benchmarks, System1→System2 escalation: crowded

## Overnight expansion (2026-09-27, ~01:00–04:00)
### New tasks
There are 4 new real 2026 tasks:
- hn: HN story reaches ≥10 points
- github_prs: PR merged
- stackoverflow: gets an accepted answer
- crimes: Chicago arrest made

Two label leaks were found and removed:
- GitHub author_association is reported as of *now*, so merged authors become CONTRIBUTORs.
- SO asker reputation is current, and accepting an answer gives the asker +2.

### Study v2
- k goes up to 400 (text tasks top out at 200 because of the 32k context).
- 10 planned draws.
- New arms:
  - noisy (20% of labels flipped)
  - fmt_compact and fmt_json (format robustness)
  - embed_lr (bge-small embeddings + logistic regression)

### Credits
- The OpenRouter account had only **$10 of credits**; the key's $100 is just a cap. Credits ran out at about $10.58.
- Jev arms therefore have 4–7 of the 10 draws (see coverage in logs).
- The DeepSeek V4 Flash comparison (llm.py and llm_study.py, price-matched to Jev) is incomplete. `JEV_OFFLINE=1` rebuilds every file from cache.

### Local additions (no API)
- `baselines_noisy.py`: noisy labels for Platt, LR and TabPFN.
- `baselines_stack.py`: "stack, don't stuff", i.e. LR or TabPFN with the Jev zero-shot score as a feature.
- `mechanism.py`: context ensembling, plus the "prior + group-rate" regression.
- `local_llm.py` and `local_llm_study.py`: Qwen2.5-7B on MLX, with correct, shuffled and noisy labels.
- `qwen_compare.py`
- `analyze2.py`: all v2 figures.

### Resume with credits
Rerunning `python study.py <task>` fills in only the missing draws, because everything else is cached. Estimated cost is about $10–15 to reach 10 draws.

### Ideas for later (need credits)
- **Summary table instead of examples:** put per-feature outcome rates in the state rather than raw examples (about $0.20). This tests the "group-mean smoother" mechanism.
- **Designed single-example interventions.**
- **Coverage-based example selection:** pick examples for coverage, e.g. one per repo.
- **Finish the DeepSeek comparison:** about $10.
