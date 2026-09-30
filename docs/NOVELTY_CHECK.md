> **Note added 2026-09-30:** the repository was reorganised after this was written; the text below is unchanged apart from this note. `v2/` is now `src/v2/`, the top-level v1 scripts are in `src/v1/`, top-level `results/*` files are in `results/v1/`, and Markdown documents other than README.md and CLAIMS.md are in `docs/`. The original layout is on the `old` branch.

# Novelty check: in-context labels vs post-hoc recalibration for LLM probabilities (as of 2026-09-27)

**Closest prior work**
1. **Calibrating ICL.** Zhao'21 [1] names majority-label, recency and common-token bias and fixes them with a content-free ("N/A") affine correction. Follow-ups include domain-context calibration [2] (domain-label bias), prototypical calibration [3] (GMM decision boundary, robust to class imbalance) and Batch Calibration [4] (estimates the contextual prior from the test batch). LinC [5] and Supervised Calibration [6] fit an affine/logistic map on ICL logits from a few labels, which is essentially Platt on top of ICL. Zhang'24 [7] finds calibration error is worst at low k and recalibration (scaling-binning) helps. Li et al. [8] find instruction-tuned ICL is overconfident. The logit prior shift you apply is the classic Saerens'02 prior-shift correction [9].
2. **Verbalised vs token probabilities.** Kadavath [10]: large models are calibrated on MC/TF in the right format. Tian [11]: verbalised beats token probabilities for RLHF models. Xiong [12]: verbalised confidence is overconfident. Wang'24 [13]: Platt/temperature scaling on verbalised probabilities via inverse softmax. The **closest single paper** is Cruz, Hardt & Mendler-Dünner, NeurIPS'24 [14] (folktexts, US Census outcomes, 17 LLMs). Their zero-shot risk scores carry signal but are badly miscalibrated, instruction-tuned models are overconfident, and verbalised queries calibrate better. It is zero-shot only, with no k-sweep or recalibration comparison.
3. **Tabular/few-shot vs classical models.** TabLLM [15] is competitive with GBDTs at very low k. LIFT [16] fine-tunes on serialised rows. TabuLa-8B [17] beats XGBoost/TabPFN at 1–32 shots. **LLMTabBench (May 2026) [18]** is the closest on this axis: LR/XGB/TabPFN-3 baselines, k = 0–64, 8 post-cutoff datasets. It finds that zero-shot is competitive and gains plateau by about 16 shots, but it reports ROC-AUC only, with no calibration, base rates or text-vs-structured split. Buckmann & Hill [19]: LR on small-LLM embeddings matches large LLMs "tens-of-shot" on text. LLM-Boost/PFN-Boost [20] already stack LLM outputs into GBDTs. AutoElicit [21]: ICL loses to LR with LLM-elicited priors, and ICL's implicit prior is poorly used. Bordt [22]: LLMs memorise public tables, which motivates post-cutoff data. On label sensitivity, Min [23] finds random labels barely hurt, while Yoo [24], Wei [25] (large models follow flipped labels), Kossen [26] (ICL uses labels but is not conventional learning) and many-shot ICL [27] push back.
4. **Forecasting and base rates.** Halawi [28], ForecastBench [29], KalshiBench [30] (frontier models overconfident on post-cutoff events), Ren et al. [31] (zero-shot probabilistic tabular prediction is highly variable across tasks). On base-rate neglect: CogBias [32] finds it at human-like rates, partly reduced by CoT. Hobor'26 [33] finds population-statistic intervals cover only 9–44% of the time, fixable by conformal recalibration.
5. **Everything combined.** I found **no** paper running ICL vs Platt vs base-rate-in-prompt vs logit prior shift head-to-head across a k-sweep, on post-cutoff real tasks, with shuffled-label controls and LR/TabPFN baselines. The nearest are [7] (calibration vs k on NLU benchmarks), [14] (zero-shot census) and [18] (AUC-only tabular). My search was not exhaustive, so a workshop paper I missed is possible.

**(a) Already known; don't claim as new.** ICL carries label-prior bias that post-hoc affine/logit corrections fix [1–6]. Recalibration with a few labels works [6,7,13]. Instruction-tuned probabilities are overconfident and off base rate [8,12,14,30]. Models use in-context labels [24–27]. LLMs are near classical baselines on few-shot tabular data [15,17,18]. Stacking LLM scores into tree or PFN models [20]. Prior-shift correction itself [9]. LLM base-rate neglect [32,33].

**(b) Plausibly new or useful.** (i) A controlled head-to-head of **ICL vs Platt vs stated base rate vs code-side logit shift across k**, scored on Brier/ECE rather than AUC. (ii) The finding that *telling* the model the base rate moves it only partway while *shifting logits* works, measured on real tasks; I found this nowhere as a quantified result. (iii) "In-context ≈ LR on the same k labels for structured data, LLM edge on free text" as a within-study split. The direction echoes [18,19], but no one frames it this way for calibration. (iv) A rule of thumb for when to prefer Platt vs ICL, based on whether the labels carry local signal. (v) Fresh 2026 public-records tasks. (vi) Frontier API models (Gemini 3.8 Flash, Opus 5.5) alongside Qwen2.5-7B logits. Each piece is incremental; the value is in the combination and the practical recipe.

**(c) Worth expanding?** *For FDE/applied-AI roles: yes, strongly.* It answers a question clients actually ask ("can I trust these probabilities, and what do I do with 50 labels?"), and it is reproducible on public data. A 2–3 page write-up plus repo is high signal, and more models help less than a clear recipe with cost and latency. *For a workshop paper: yes, with care.* Targets like NeurIPS/ICML workshops on uncertainty, tabular data, or reliable ML are plausible. It needs: multiple seeds with CIs; Brier/log-loss/ECE; contextual/batch calibration [1,4] and Supervised Calibration [6] as baselines, not just Platt; a contamination check [22]; and honest positioning against [14,18]. It is not main-track novel.

**(d) Framing and what to cite.**
- Title it as a practitioner study, e.g. "In-context labels vs post-hoc recalibration for LLM probabilities on post-cutoff public-records tasks", not as a new method.
- Present the logit shift as Saerens-style prior correction [9] and the Platt arm as the LinC/SC family [5,6]. The contribution is *when each wins across k*, not the methods themselves.
- Lead with the base-rate result (prompted rate moves the model only partly; code-side shift works), tying it to [1,2,14,32].
- Position "≈ LR on structured, edge on text" against [15,17,18,19], and the TabPFN stacking against [20].
- Must-cite core: [1,4,6,7,9,11,14,18,20,23,25,29].

## References
1. Zhao et al. 2021, Calibrate Before Use: https://arxiv.org/abs/2102.09690
2. Fei et al. 2023, Mitigating Label Biases for ICL (ACL): https://arxiv.org/abs/2305.19148
3. Han et al. 2022, Prototypical Calibration: https://arxiv.org/abs/2205.10183
4. Zhou et al. 2024, Batch Calibration (ICLR): https://arxiv.org/abs/2309.17249
5. Abbas et al. 2024, Linear Probe Calibration (AISTATS): https://arxiv.org/abs/2401.12406
6. Gundem et al. 2025/26, Boosting ICL through Classical Supervised Learning (Supervised Calibration): https://arxiv.org/abs/2505.23783
7. Zhang et al. 2024, A Study on the Calibration of ICL (NAACL): https://arxiv.org/abs/2312.04021
8. Li et al. 2023/25, LLMs are Miscalibrated In-Context Learners: https://arxiv.org/abs/2312.13772
9. Saerens, Latinne, Decaestecker 2002, Adjusting outputs to new a priori probabilities: https://doi.org/10.1162/089976602753284446
10. Kadavath et al. 2022, LMs (Mostly) Know What They Know: https://arxiv.org/abs/2207.05221
11. Tian et al. 2023, Just Ask for Calibration (EMNLP): https://arxiv.org/abs/2305.14975
12. Xiong et al. 2024, Can LLMs Express Their Uncertainty? (ICLR): https://arxiv.org/abs/2306.13063
13. Wang et al. 2024, Calibrating Verbalized Probabilities for LLMs: https://arxiv.org/abs/2410.06707
14. Cruz, Hardt, Mendler-Dünner 2024, Evaluating Language Models as Risk Scores (NeurIPS): https://arxiv.org/abs/2407.14614
15. Hegselmann et al. 2023, TabLLM (AISTATS): https://arxiv.org/abs/2210.10723
16. Dinh et al. 2022, LIFT (NeurIPS): https://arxiv.org/abs/2206.06565
17. Gardner et al. 2024, TabuLa-8B (NeurIPS): https://arxiv.org/abs/2406.12031
18. Grushina et al. 2026, LLMTabBench: https://arxiv.org/abs/2605.24417
19. Buckmann & Hill 2024/25, LR makes small LLMs strong tens-of-shot classifiers: https://arxiv.org/abs/2408.03414
20. Jayawardhana et al. 2025, Transformers Boost Decision Trees (LLM-Boost, PFN-Boost): https://arxiv.org/abs/2502.02672
21. Capstick et al. 2025, AutoElicit (ICML): https://arxiv.org/abs/2411.17284
22. Bordt et al. 2024, Elephants Never Forget (COLM): https://arxiv.org/abs/2404.06209
23. Min et al. 2022, Rethinking the Role of Demonstrations (EMNLP): https://arxiv.org/abs/2202.12837
24. Yoo et al. 2022, Ground-Truth Labels Matter (EMNLP): https://arxiv.org/abs/2205.12685
25. Wei et al. 2023, Larger LMs do ICL Differently: https://arxiv.org/abs/2303.03846
26. Kossen, Gal, Rainforth 2024, ICL Learns Label Relationships but Is Not Conventional Learning (ICLR): https://arxiv.org/abs/2307.12375
27. Agarwal et al. 2024, Many-Shot ICL (NeurIPS): https://arxiv.org/abs/2404.11018
28. Halawi et al. 2024, Approaching Human-Level Forecasting (NeurIPS): https://arxiv.org/abs/2402.18563
29. Karger et al. 2025, ForecastBench (ICLR): https://arxiv.org/abs/2409.19839
30. Nel 2025, KalshiBench: https://arxiv.org/abs/2512.16030
31. Ren et al. 2025, Predicting LMs' Success at Zero-Shot Probabilistic Prediction (EMNLP Findings): https://arxiv.org/abs/2509.15356
32. CogBias 2026, Measuring and Mitigating Cognitive Bias in LLMs: https://arxiv.org/abs/2604.01366
33. Hobor et al. 2026, Bayesian Elicitation with LLMs: https://arxiv.org/abs/2604.01896

*Caveats:* Details for [18], [30], [32] and [33] come from abstract/HTML pages via WebFetch, not full reads. I did not verify the full author list for [32]. Coverage of 2026 workshop papers is incomplete.
