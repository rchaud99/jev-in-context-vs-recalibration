# Replication table: base rate in the prompt vs in code

Brier-score differences (positive = the first-named arm is worse), pooled per model as the mean over tasks of the per-task, seed-averaged difference; 95% bootstrap CI (test cases resampled within task, B=5000). **Bold** = significant after Holm across the 9 configurations. Seeds = sign of the pooled difference in each of the 3 seeds (each seed = a different 100-label estimate of the base rate). Tasks = number of tasks with a Holm-significant difference (within model) in the pooled direction / tasks. E1: 8 tasks (June–Aug 2026 records); E1-R: 5 fresh tasks (September 2026), preregistered.

## Told the base rate − constant (predict the stated rate for every case)

| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |
|---|---|---|---|---|---|---|---|---|
| jev | native | 0.004 | **+0.008 [+0.004, +0.012]** | +++ | 1/8 | **+0.006 [+0.001, +0.012]** | +++ | 0/5 |
| qwen3.7-flash | logprob | 0.005 | **+0.044 [+0.035, +0.052]** | +++ | 5/8 | **+0.061 [+0.049, +0.073]** | +++ | 3/5 |
| deepseek-v4-flash | logprob | 0.006 | **+0.073 [+0.063, +0.083]** | +++ | 7/8 | **+0.060 [+0.048, +0.072]** | +++ | 4/5 |
| gpt-4o-mini | logprob | 0.020 | **+0.141 [+0.128, +0.154]** | +++ | 7/8 | **+0.127 [+0.110, +0.144]** | +++ | 4/5 |
| llama-3.3-70b | logprob | 0.027 | **+0.276 [+0.258, +0.294]** | +++ | 8/8 | **+0.265 [+0.241, +0.289]** | +++ | 5/5 |
| gpt-4o-mini-verbal | stated | 0.009 | **+0.052 [+0.044, +0.059]** | +++ | 4/8 | **+0.048 [+0.037, +0.059]** | +++ | 3/5 |
| deepseek-v4-flash-verbal | stated | 0.003 | **+0.023 [+0.017, +0.029]** | +++ | 5/8 | **+0.020 [+0.014, +0.027]** | +++ | 1/5 |
| gemini-3.8-flash | stated | 0.027 | **-0.012 [-0.018, -0.007]** | −−− | 2/8 | **-0.012 [-0.019, -0.006]** | −−− | 1/5 |
| claude-haiku-4.5 | stated | 0.085 | – |  |  | – |  |  |

## Told the base rate − same number applied in code (label-free logit shift)

| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |
|---|---|---|---|---|---|---|---|---|
| jev | native | 0.004 | **+0.010 [+0.007, +0.013]** | +++ | 3/8 | **+0.010 [+0.005, +0.015]** | +++ | 2/5 |
| qwen3.7-flash | logprob | 0.005 | **+0.032 [+0.024, +0.040]** | +++ | 1/8 | **+0.052 [+0.041, +0.064]** | +++ | 3/5 |
| deepseek-v4-flash | logprob | 0.006 | **+0.047 [+0.036, +0.057]** | +++ | 4/8 | **+0.039 [+0.025, +0.052]** | +++ | 2/5 |
| gpt-4o-mini | logprob | 0.020 | **+0.102 [+0.089, +0.115]** | +++ | 5/8 | **+0.098 [+0.081, +0.117]** | +++ | 2/5 |
| llama-3.3-70b | logprob | 0.027 | **+0.232 [+0.213, +0.250]** | +++ | 6/8 | **+0.232 [+0.209, +0.256]** | +++ | 5/5 |
| gpt-4o-mini-verbal | stated | 0.009 | **+0.034 [+0.026, +0.042]** | +++ | 2/8 | **+0.038 [+0.027, +0.050]** | +++ | 2/5 |
| deepseek-v4-flash-verbal | stated | 0.003 | **+0.012 [+0.006, +0.017]** | +++ | 3/8 | **+0.010 [+0.004, +0.017]** | +++ | 1/5 |
| gemini-3.8-flash | stated | 0.027 | -0.001 [-0.004, +0.002] | −+− | 0/8 | **+0.004 [+0.000, +0.008]** | +++ | 0/5 |
| claude-haiku-4.5 | stated | 0.085 | – |  |  | – |  |  |

## Code shift − constant

| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |
|---|---|---|---|---|---|---|---|---|
| jev | native | 0.004 | -0.001 [-0.004, +0.001] | −−− | 1/8 | **-0.004 [-0.006, -0.001]** | −−− | 1/5 |
| qwen3.7-flash | logprob | 0.005 | **+0.012 [+0.007, +0.018]** | +++ | 2/8 | **+0.009 [+0.003, +0.015]** | +++ | 1/5 |
| deepseek-v4-flash | logprob | 0.006 | **+0.026 [+0.019, +0.033]** | +++ | 4/8 | **+0.021 [+0.014, +0.029]** | +++ | 2/5 |
| gpt-4o-mini | logprob | 0.020 | **+0.039 [+0.031, +0.047]** | +++ | 6/8 | **+0.028 [+0.019, +0.038]** | +++ | 3/5 |
| llama-3.3-70b | logprob | 0.027 | **+0.044 [+0.035, +0.054]** | +++ | 4/8 | **+0.032 [+0.022, +0.042]** | +++ | 2/5 |
| gpt-4o-mini-verbal | stated | 0.009 | **+0.018 [+0.012, +0.023]** | +++ | 3/8 | **+0.010 [+0.005, +0.015]** | +++ | 1/5 |
| deepseek-v4-flash-verbal | stated | 0.003 | **+0.011 [+0.006, +0.016]** | +++ | 2/8 | **+0.010 [+0.004, +0.016]** | +++ | 1/5 |
| gemini-3.8-flash | stated | 0.027 | **-0.012 [-0.017, -0.006]** | −−− | 2/8 | **-0.017 [-0.024, -0.010]** | −−− | 1/5 |
| claude-haiku-4.5 | stated | 0.085 | – |  |  | – |  |  |

## Platt on 100 labels − code shift

| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |
|---|---|---|---|---|---|---|---|---|
| jev | native | 0.004 | -0.000 [-0.003, +0.002] | −++ | 1/8 | +0.000 [-0.003, +0.003] | +−+ | 0/5 |
| qwen3.7-flash | logprob | 0.005 | **-0.014 [-0.018, -0.010]** | −−− | 4/8 | **-0.011 [-0.016, -0.006]** | −−− | 1/5 |
| deepseek-v4-flash | logprob | 0.006 | **-0.026 [-0.032, -0.019]** | −−− | 6/8 | **-0.023 [-0.030, -0.016]** | −−− | 3/5 |
| gpt-4o-mini | logprob | 0.020 | **-0.036 [-0.044, -0.028]** | −−− | 5/8 | **-0.028 [-0.038, -0.019]** | −−− | 4/5 |
| llama-3.3-70b | logprob | 0.027 | **-0.041 [-0.049, -0.034]** | −−− | 4/8 | **-0.032 [-0.041, -0.022]** | −−− | 1/5 |
| gpt-4o-mini-verbal | stated | 0.009 | **-0.016 [-0.022, -0.011]** | −−− | 3/8 | **-0.009 [-0.014, -0.004]** | −−− | 0/5 |
| deepseek-v4-flash-verbal | stated | 0.003 | **-0.011 [-0.014, -0.007]** | −−− | 3/8 | **-0.008 [-0.014, -0.002]** | −−− | 0/5 |
| gemini-3.8-flash | stated | 0.027 | -0.001 [-0.003, +0.000] | −−− | 0/8 | -0.001 [-0.003, +0.001] | −−− | 0/5 |
| claude-haiku-4.5 | stated | 0.085 | – |  |  | – |  |  |

## Logistic regression on the same 100 labels − code shift

| Model | Elicitation | $/1k decisions | E1 (8 tasks) | seeds | tasks | E1-R (5 fresh tasks) | seeds | tasks |
|---|---|---|---|---|---|---|---|---|
| jev | native | 0.004 | -0.006 [-0.011, -0.000] | −−− | 0/8 | **-0.011 [-0.016, -0.005]** | −−− | 2/5 |
| qwen3.7-flash | logprob | 0.005 | **-0.019 [-0.026, -0.012]** | −−− | 1/8 | **-0.023 [-0.031, -0.016]** | −−− | 3/5 |
| deepseek-v4-flash | logprob | 0.006 | **-0.033 [-0.041, -0.025]** | −−− | 3/8 | **-0.036 [-0.045, -0.027]** | −−− | 2/5 |
| gpt-4o-mini | logprob | 0.020 | **-0.046 [-0.055, -0.037]** | −−− | 6/8 | **-0.043 [-0.053, -0.032]** | −−− | 4/5 |
| llama-3.3-70b | logprob | 0.027 | **-0.051 [-0.061, -0.042]** | −−− | 6/8 | **-0.047 [-0.058, -0.036]** | −−− | 3/5 |
| gpt-4o-mini-verbal | stated | 0.009 | **-0.025 [-0.032, -0.017]** | −−− | 4/8 | **-0.024 [-0.032, -0.016]** | −−− | 3/5 |
| deepseek-v4-flash-verbal | stated | 0.003 | **-0.018 [-0.025, -0.011]** | −−− | 2/8 | **-0.024 [-0.033, -0.016]** | −−− | 3/5 |
| gemini-3.8-flash | stated | 0.027 | +0.005 [-0.002, +0.011] | +++ | 0/8 | +0.002 [-0.005, +0.009] | −++ | 0/5 |
| claude-haiku-4.5 | stated | 0.085 | – |  |  | – |  |  |

## Exploratory: effect by the model's zero-shot AUROC on the task (all model × task cells, E1 + E1-R)

| Comparison | AUROC band | cells | mean Δ | median Δ | CI > 0 | CI < 0 |
|---|---|---|---|---|---|---|
| told-const | (0.0, 0.55] | 34 | +0.0912 | +0.0494 | 31 | 0 |
| told-const | (0.55, 0.65] | 49 | +0.0864 | +0.0119 | 24 | 1 |
| told-const | (0.65, 0.75] | 16 | +0.0310 | +0.0088 | 7 | 1 |
| told-const | (0.75, 1.0] | 5 | -0.0235 | -0.0074 | 0 | 2 |
| H1 told-shift | (0.0, 0.55] | 34 | +0.0597 | +0.0169 | 16 | 0 |
| H1 told-shift | (0.55, 0.65] | 49 | +0.0753 | +0.0087 | 23 | 2 |
| H1 told-shift | (0.65, 0.75] | 16 | +0.0259 | +0.0188 | 9 | 1 |
| H1 told-shift | (0.75, 1.0] | 5 | +0.0054 | +0.0042 | 2 | 0 |
| P5 shift-const | (0.0, 0.55] | 34 | +0.0315 | +0.0252 | 27 | 0 |
| P5 shift-const | (0.55, 0.65] | 49 | +0.0111 | +0.0018 | 12 | 1 |
| P5 shift-const | (0.65, 0.75] | 16 | +0.0051 | -0.0016 | 4 | 1 |
| P5 shift-const | (0.75, 1.0] | 5 | -0.0289 | -0.0167 | 0 | 4 |
