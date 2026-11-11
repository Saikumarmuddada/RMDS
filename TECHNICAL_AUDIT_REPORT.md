# RMDS Hackathon Dataset C — Technical Audit & Verification Report

## 1. Executive Summary & Audit Outcome

A Principal ML / Code Audit of the RMDS Hackathon Dataset C pipeline (`4,756` labeled training respondents; `4,749` test respondents; vaccine cost $C_v = \$1{,}500$; hospitalisation cost $C_h = \$41{,}000$; 100% infection assumption) identified, fixed, and verified every structural leakage vector, effect-measure inconsistency, CV threshold risk, feature-ablation gap, SHAP distortion, and test bottleneck across the codebase.

* **Automated Test Suite:** `py_compile` passed with exit code `0`; `pytest -v` executed all **18 tests in 77.91s (`18 passed, 0 failed, 0 skipped, 0 warnings`)** after hardening fast-mode training concurrency and adding nested calibration, `cloglog` / Zhang-Yu, CV fallback policy, and feature-ablation coverage.
* **Complete End-to-End Execution (`main.py --group 0`):** Completed stages `[1/6]` through `[6/6]` cleanly with exit code `0`, regenerating all 16 files in `outputs/`, all 10 charts in `outputs/figures/`, and all 5 diverse submissions + manifest in `submissions/`.
* **Winning Honest Out-of-Sample Solution:** [`Calibrated_Stack`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) (`Uncalibrated_Stack`, nested L2 logit meta-learner `C = 0.05` with inner-CV identity calibration) achieves **#1 OOF ROC-AUC (`0.8670`)**, **#1 LogLoss (`0.4173`)**, **#1 Brier (`0.1313`)**, **#1 ECE (`0.0088`)**, and **#1 Bayes-Exact Town Cost (`$14,281,676` | `$3,002.88/person`)** — saving **`$458,727` over fixed `0.50` (`$14,740,402`)**, **`$3,318,874` over Treat-None (`$17,600,550`)**, and capturing **`68.7%` of Oracle savings (`$12,778,366`)**.
* **Critical Empirical Caveat Maintained:** Dataset C contains **no empirical hospitalisation outcome column**. Individual `(q0, q1)` hospitalisation probabilities remain **modelled epidemiological estimates** anchored to peer-reviewed literature (`PARTIAL` scientific status — never claimed as clinically validated risk predictions on Dataset C).

---

## 2. Calibrated Stacking Leakage Audit (`src/modeling.py`)

### Root Leakage Risk Resolved
Previously, `best_c` regularization selection evaluated LogLoss across the full inner OOF matrix `Z_inner` **before** calling `select_and_fit_posthoc_calibration` across folds of `Z_inner`. Although minor (5 grid values over ~3,804 rows), `cal_va` labels participated in selecting the meta-learner regularization parameter `C`, introducing subtle selection bias into calibration comparison.

### Strictly Nested Architecture Implemented
[`fit_outer_fold_models`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) never accepts `y_va` in its signature and now invokes [`evaluate_calibration_nested`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) alongside [`_select_best_meta_c`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py):
1. Within each outer training fold `tr_idx` (`3,804–3,805` rows), an inner 4-fold Stratified CV (`3/4` inner train with strict `85%/15%` boosting early stopping; `0` overlap with outer `va_idx`) generates strictly clean base logits `Z_inner`.
2. Inside [`evaluate_calibration_nested`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py), for each calibration fold $j \in \{0,1,2,3\}$:
   * [`_select_best_meta_c`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) tunes $c_j \in \{0.05, 0.1, 0.25, 0.5, 1.0\}$ strictly out-of-fold **inside `cal_tr` (`inner_fold_ids != j`)** and returns strictly sub-OOF training logits `sub_meta_oof_tr`.
   * `meta_j` fits on `(Z_inner[cal_tr], y_tr[cal_tr])` using $c_j$ and predicts untouched `Z_inner[cal_va]` into `nested_raw_oof[cal_va]`.
   * Candidate calibrators (`none`, `platt`, `isotonic`) fit strictly on `(sub_meta_oof_tr, y_tr[cal_tr])` and score strictly untouched `cal_va`.
3. Only after choosing `best_cal_mode` strictly out-of-sample does [`_select_best_meta_c`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) pick full-`Z_inner` `best_c` and fit `meta_full` + `final_calibrator` to predict outer `va_idx` and `X_test`.

### Per-Fold Nested Calibration Audit (`outputs/stacking_calibration_audit.csv`)

| Outer Fold | Final Meta `C` | Per-Cal-Fold `nested_cal_c_per_fold` | Inner-OOF LogLoss (`none`) | Inner-OOF LogLoss (`platt`) | Inner-OOF LogLoss (`isotonic`) | Selected Mode | ES Fit / Val Rows |
| ---: | ---: | :--- | ---: | ---: | ---: | :--- | ---: |
| 0 | 0.05 | `0.05, 0.1, 0.05, 0.05` | 0.41922 | **0.41916** | 0.42872 | `none` *(min Δ < 1e-4)* | 3,233 / 571 |
| 1 | 0.05 | `0.05, 0.05, 0.05, 0.05` | **0.41712** | 0.41718 | 0.42105 | `none` | 3,234 / 571 |
| 2 | 0.05 | `0.05, 0.1, 0.1, 0.05` | **0.42292** | 0.42294 | 0.43238 | `none` | 3,234 / 571 |
| 3 | 0.05 | `0.05, 0.05, 0.05, 0.1` | 0.41701 | **0.41700** | 0.42172 | `none` *(min Δ < 1e-4)* | 3,234 / 571 |
| 4 | 0.05 | `0.05, 0.05, 0.05, 0.05` | **0.41848** | 0.41852 | 0.42174 | `none` | 3,234 / 571 |

*Notice:* Even with fully nested `c_j` tuning inside each calibration fold, regularized L2 logit meta-stacking sits cleanly at optimum calibration (`none` beats Platt on 3/5 folds and ties within `6e-5` LogLoss on folds 0 & 3; **outer OOF ECE = `0.00884`**). Synthetic stress-testing (`test_stacking_respondent_level_isolation_and_active_calibration`) confirms both respondent-level label-flip invariance inside `evaluate_calibration_nested` and active Platt selection when logits are miscalibrated.

---

## 3. Scientific Validation & Effect-Measure Harmonisation (`src/cost_model.py`)

### Effect-Measure Mathematical Correction
Previously, Ko et al. (2021) chronic-condition adjusted Rate Ratios (`aRR = 5.0` for `>=3` conditions) were added directly onto a logistic `log_odds` equation (`+= log(chronic_rr)`), mathematically conflating Rate Ratios with Odds Ratios (`OR > RR` when baseline risk is non-rare).

**Corrected Multi-Stage Formulation in [`estimate_hospitalisation_probabilities`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py):**
1. **Stage 1 (Odds-Ratio Scale):** Combines age-stratum baseline IHR (`BASE_AGE_IHR`), scenario odds scaling (`0.50` / `1.00` / `1.50`), and centered multivariable male/female odds ratio (`SEX_ODDS_MULTIPLIER`, $OR = 1.45$) on the logit scale to obtain $p_{\text{age\_sex}} \in [0.002, 0.850]$.
2. **Stage 2 (Rate-Ratio Complementary Log-Log Link):** Applies Ko et al. chronic condition rate ratios (`CHRONIC_CONDITION_RR_MAP`: `0 -> 1.00`, `1 -> 1.70`, `2 -> 2.90`, `>=3 -> 5.00`) on the bounded proportional-rate (`cloglog`) survival scale:
   $$p_{u,i} = \text{clip}\!\left(1 - (1 - p_{\text{age\_sex},i})^{\text{aRR}_i},\ 0.002,\ 0.850\right)$$
   At low baseline risk ($p \approx 0.006$), $1 - (1-p)^{\text{aRR}} \approx \text{aRR} \cdot p$, matching [`zhang_yu_rr_to_or`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) while preventing `>1.0` overflow in high-risk elderly severe strata (`65+`, `>=3`, `severe` stays bounded $\le 0.850$).
3. **Stage 3 (Hospitalisation VE):** Applies literature-grounded VE (`base`: `0.86` standard, `0.80` frail; `waned_omicron`: `0.65 / 0.55`; `boosted`: `0.91 / 0.86`) yielding `mean q0 = 0.0783`, `mean q1 = 0.0140` under `base`.

### Verified Literature Anchors, Bracket Mapping & Limitations

| Parameter Group | Exact Code Values | Verified Peer-Reviewed Citation & Statistic | Explicit Assumptions & Limitations |
| :--- | :--- | :--- | :--- |
| **Baseline Age IHR** (`BASE_AGE_IHR`, `AGE_BRACKET_MAPPING_DOC`) | `18-34`: `0.006`, `35-44`: `0.015`, `45-54`: `0.031`, `55-64`: `0.066`, `65+`: `0.145` | **Herrera-Esposito & de los Campos (2022)**, *BMC Infect Dis* `22:311`, `doi:10.1186/s12879-022-07262-0` (`20-29: 0.47%`, `30-39: 0.99%`, `40-49: 2.11%`, `50-59: 4.49%`, `60-69: 9.54%`); **Verity et al. (2020)**, *Lancet Infect Dis* `20(6):669-677`, `doi:10.1016/S1473-3099(20)30243-7` (`1.0%–18.4%` up to `80+`); **Salje et al. (2020)**, *Science* `369:208-211`. | Dataset C survey brackets merge 10-yr published bands; `65+ Years` pools 65–69, 70–79, and 80+ community-dwelling respondents. Pre-vaccination ancestral cohorts; uncertainty bracketed via `0.5x`/`1.5x` scenario odds. |
| **Chronic Conditions** (`CHRONIC_CONDITION_RR_MAP`) | `0: 1.00`, `1: 1.70`, `2: 2.90`, `>=3: 5.00` *(cloglog aRR)* | **Ko et al. (2021)**, *Clin Infect Dis* `72(11):e695-e703`, `doi:10.1093/cid/ciaa1419`: COVID-NET/BRFSS `>=3` conditions **aRR = 5.0 (95% CI: 3.9–6.3)**; **Zhang & Yu (1998)**, *JAMA* `280(19):1690-1691`. | Levels `1` (`1.70`) and `2` (`2.90`) are concave geometric interpolations ($5^{1/3}, 5^{2/3}$) since Ko et al. pooled `>=3`. COVID-NET population rate ratio assumes balanced exposure conditional on infection. |
| **Sex Adjustment** (`SEX_ODDS_MULTIPLIER`) | Centered $OR = 1.45$ (`Male: 1.20`, `Female: 0.8276`) | **Petrilli et al. (2020)**, *BMJ* `369:m1966`, `doi:10.1136/bmj.m1966` (admission `aOR = 2.8 [2.4–3.2]`, critical `aOR = 1.5 [1.3–1.8]`); **Peckham et al. (2020)**, *Nat Commun* `11:6317`; **Williamson et al. (2020)**, *Nature* `584:430-436`. | Toggleable via `include_sex_adjustment=False` in sensitivity grid (`$14.28M` with sex vs `$14.71M` sex-neutral, still saving `$447,791` over `0.50`). |
| **Hospitalisation VE** (`VE_REGIMES`) | `base: (0.86, 0.80)`, `waned_omicron: (0.65, 0.55)`, `boosted: (0.91, 0.86)` | **Tenforde et al. (2021)**, *JAMA* `326(20):2043-2054`, `doi:10.1001/jama.2021.19499` (`PMID 34734975`: Delta VE = `86%`); **Ferdinands et al. (2022)**, *MMWR* `71(7):255-263`, `doi:10.15585/mmwr.mm7107e2` (Omicron 3-dose `91% <2m` / `78% >=4m`; 2-dose `71% <2m` / `54% >=5m`). | High-frailty (`65+` or `>=3`) shift (`86% -> 80%`) is a sensitivity assumption motivated by weaker immunocompromised VE; all regimes spanned in 9-cell grid. |

---

## 4. Cost Optimisation & Decision Policy Verification (`src/cost_model.py`, `main.py`)

With exact town expenditure rules ($C_v = \$1{,}500$, $C_h = \$41{,}000$, 100% infection):
$$\tau^*(x_i) = \frac{C_v}{C_v + C_h (q_{0,i} - q_{1,i})} \quad (q_{0,i} > q_{1,i}), \qquad \tau^*(x_i) = 1.0 \quad (q_{0,i} \le q_{1,i})$$

* **Out-of-Sample Tuning Enforcement:** [`tune_threshold_on_oof`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) now guarantees 5-fold Stratified CV threshold search (`evaluation == "strictly_out_of_fold_cv"`) even when `folds=None`.
* **Policy & Component Breakdown for Winning `Calibrated_Stack` (`outputs/cost_policy_comparison.csv`):**

| Policy | Total Cost ($) | Vaccine Send ($) | Wasted ($) | Law Comp ($) | Exp. Hosp ($) | Allocated | Accepted | Wasted | Unmet | Exp. Hosp | Savings vs 0.50 ($) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Fixed Threshold (`0.50`)** | $14,740,402 | $1,944,000 | $450,000 | $837,000 | $11,959,402 | 1,296 | 996 | 300 | 558 | 291.7 | $0 |
| **CV-Tuned Global Threshold (`0.36`, OOF)** | $14,713,139 | $2,721,000 | $933,000 | $543,000 | $11,449,139 | 1,814 | 1,192 | 622 | 362 | 279.2 | $27,264 |
| **Bayes-Exact Threshold (`tau*(x_i)`, Analytical)** | **$14,281,676** | $2,230,500 | $736,500 | $837,000 | $11,214,176 | 1,487 | 996 | 491 | 558 | **273.5** | **$458,727** |
| **CV-Scaled Bayes Threshold (`1.035`, OOF)** | $14,283,408 | $2,158,500 | $708,000 | $880,500 | $11,244,408 | 1,439 | 967 | 472 | 587 | 274.3 | $456,995 |

*(Baselines over `n=4,756`: Treat-None `$17,600,550`; Treat-All `$17,581,366`; Oracle `$12,778,366`. Across all 9 sensitivity grid cells in `outputs/ve_and_risk_sensitivity_grid.csv`, Bayes-Exact beats Fixed `0.50` by **`$146,707` [`conservative + waned_omicron`] to `$824,266` [`severe + boosted`]**.)*

---

## 5. Complete Feature-Group Ablation & Preprocessing Audit (`src/data_prep.py`)

Upgraded [`run_feature_ablation`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/data_prep.py) to share the exact 5-fold Stratified CV partition (`seed=42`) used by modeling and report all 4 metrics (**ROC-AUC, LogLoss, Brier, ECE**) across additive and leave-one-group-out tests (`outputs/feature_ablation_report.csv`):

| Variant | Type | Features | LR AUC | LR LogLoss | LR Brier | LR ECE | LGB AUC | LGB LogLoss | LGB Brier | LGB ECE | Decision |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| `raw_baseline_0_untouched` | Baseline | 29 | 0.8306 | 0.4662 | 0.1486 | 0.0187 | 0.8598 | 0.4265 | 0.1348 | 0.0126 | `REFERENCE` |
| `raw_plus_zero_as_nan` | Additive (`+zero`) | 31 | 0.8384 | 0.4562 | 0.1458 | 0.0132 | 0.8611 | 0.4252 | 0.1344 | 0.0147 | `RETAIN` |
| `add_missing_flags` | Additive (`+miss`) | 36 | **0.8658** | **0.4210** | **0.1318** | 0.0157 | 0.8613 | 0.4248 | 0.1345 | 0.0143 | `RETAIN` |
| `add_doctor_interactions` | Additive (`+doc`) | 34 | 0.8380 | 0.4567 | 0.1460 | 0.0160 | 0.8614 | 0.4248 | 0.1343 | 0.0120 | `RETAIN` |
| `add_vulnerability_flags` | Additive (`+vuln`) | 36 | 0.8384 | 0.4563 | 0.1458 | 0.0138 | 0.8617 | 0.4239 | 0.1342 | 0.0167 | `RETAIN` |
| `validated_engineered_full` | Full Validated | 49 | 0.8654 | 0.4218 | 0.1321 | 0.0147 | **0.8624** | **0.4239** | **0.1340** | 0.0151 | `REFERENCE` |
| `plus_pruned_opinion_composites` | Pruned Check | 54 | 0.8649 | 0.4226 | 0.1323 | 0.0161 | 0.8609 | 0.4256 | 0.1347 | 0.0141 | `PRUNE` |
| `ablate_without_missing_flags` | Leave-One-Out | 44 | 0.8379 | 0.4570 | 0.1461 | 0.0183 | 0.8620 | 0.4243 | 0.1343 | 0.0111 | `RETAIN` |
| `ablate_without_ordinal` | Leave-One-Out | 46 | 0.8654 | 0.4218 | 0.1321 | 0.0146 | 0.8602 | 0.4262 | 0.1347 | 0.0150 | `RETAIN` |
| `ablate_without_doctor_interactions` | Leave-One-Out | 46 | 0.8656 | 0.4215 | 0.1319 | 0.0144 | 0.8609 | 0.4255 | 0.1347 | 0.0118 | `RETAIN` |

*Empirical Preprocessing Transparency:* Mapping `opinion_covid_vacc_effective == 0` (25.0% acceptance vs rank-1 3.2%) and `opinion_covid_risk == 0` (27.1% vs rank-1 12.5%) to `NaN` with `_is_zero` indicators is an empirically justified distribution decision documented explicitly in [`outputs/preprocessing_validation.json`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/outputs/preprocessing_validation.json) because official PDF/Image docs omit code `0`.

---

## 6. Model Comparison: Distinction Between Nested Inner-CV vs. Outer 5-Fold OOF

* **Nested Inner-CV LogLoss (`n = 3,804–3,805` per outer fold):** Used strictly inside training partitions to select `best_meta_c = 0.05` and verify calibration mode `none` (`0.4170–0.4229` across folds `0..4`).
* **Outer 5-Fold OOF Comparison (`n = 4,756` strictly untouched outer validation predictions) — Before vs. After Audit** *(both pre-audit and post-audit probability tables evaluated identically under the final `cloglog` base risk model)*:

| Model | Version | ROC-AUC | LogLoss | Brier | ECE | Bayes Cost ($) | Fixed 0.50 Cost ($) | Allocated / Wasted / Law | Exp. Hosp | Oracle % |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| **LogisticRegression** | Before | 0.86517 | 0.42267 | 0.13223 | 0.01662 | $14,348,719 | $14,775,554 | 1529/533/558 | 273.62 | 67.3% |
| **LogisticRegression** | **After** | **0.86537** | **0.42184** | **0.13206** | **0.01474** | **$14,336,991** | **$14,759,858** | 1536/538/556 | 273.15 | **67.5%** |
| **RandomForest** | Before | 0.85431 | 0.43573 | 0.13829 | **0.01915** | **$14,307,387** | **$14,964,813** | 1463/515/606 | 273.27 | **68.2%** |
| **RandomForest** | **After** | **0.85861** | **0.43508** | **0.13785** | 0.03417 | $14,308,264 | $15,031,856 | 1448/524/630 | 272.96 | 68.1% |
| **HistGradientBoosting** | Before | **0.85416** | **0.44403** | **0.13904** | **0.03716** | **$14,355,847** | **$14,929,666** | 1479/476/551 | 275.87 | **67.2%** |
| **HistGradientBoosting** | **After** | 0.85258 | 0.44855 | 0.14000 | 0.04076 | $14,390,153 | $14,947,879 | 1471/474/557 | 276.78 | 66.4% |
| **LightGBM** | Before (Leaky ES) | 0.86350 | 0.42187 | 0.13321 | 0.01369 | $14,307,981 | $14,811,864 | 1458/486/582 | 274.34 | 68.2% |
| **LightGBM** | **After (Inner ES)** | 0.86221 | 0.42380 | 0.13405 | **0.01279** | $14,343,198 | $14,845,302 | 1480/497/571 | 274.80 | 67.4% |
| **XGBoost** | Before (Leaky ES) | 0.86542 | 0.41990 | 0.13258 | 0.01364 | $14,289,232 | $14,814,953 | 1470/487/571 | 273.85 | 68.5% |
| **XGBoost** | **After (Inner ES)** | 0.86175 | 0.42495 | 0.13423 | 0.01368 | $14,347,335 | $14,816,648 | 1456/488/586 | 275.23 | 67.3% |
| **CatBoost** | Before (Leaky ES) | 0.86565 | 0.41879 | 0.13200 | 0.01680 | $14,280,310 | $14,768,270 | 1440/465/579 | 274.43 | 68.7% |
| **CatBoost** | **After (Inner ES)** | 0.86404 | 0.42037 | 0.13263 | **0.01357** | $14,364,624 | $14,831,290 | 1444/482/592 | 275.87 | 67.0% |
| **Calibrated_Stack** | Before (Leaky OOF) | 0.86702 | 0.41734 | 0.13128 | 0.01430 | $14,289,087 | **$14,735,936** | 1479/490/565 | 273.73 | 68.5% |
| **Calibrated_Stack / Uncalibrated_Stack** | **After (Nested 5x4)** | **0.86697** | **0.41725** | **0.13135** | **0.00884** | **$14,281,676** | $14,740,402 | 1487/491/558 | **273.52** | **68.7%** |

---

## 7. Multi-Fold OOF Explainability (`src/explainability.py`) & Top 5 Canvas Submissions

* **SHAP (`explainer(X).values` across all 5 outer LightGBM fold models, 0 warnings):** Top drivers remain consistent across folds (`health_insurance` `0.533 ± 0.060` rank `1.0`, `opinion_covid_risk` `0.455 ± 0.069` rank `2.2`, `doctor_x_effective` `0.381 ± 0.040` rank `3.0`, `opinion_covid_vacc_effective` `0.292 ± 0.013` rank `4.4`, `doctor_x_risk` `0.256 ± 0.075` rank `4.4`, `employment_sector` categorical `0.164 ± 0.034`), with separate numeric beeswarm (`08`) and categorical subgroup bars (`08b`) plus log-odds meta plot (`09`).
* **Verified Top 5 Submissions (`submissions/submission_manifest.csv`, pairwise distinct `2.74%–7.50%`, `4,749` unique IDs aligned with `example_submission`):**
  1. `challenge_submission_group_0_order_1.csv` — `Calibrated_Stack` (`bayes_exact`), OOF `$14,281,676`, allocates `1,555` (`32.7%`)
  2. `challenge_submission_group_0_order_2.csv` — `CatBoost` (`bayes_scaled=0.895`), OOF `$14,300,969`, allocates `1,663` (`35.0%`)
  3. `challenge_submission_group_0_order_3.csv` — `RandomForest` (`bayes_exact`), OOF `$14,308,264`, allocates `1,505` (`31.7%`)
  4. `challenge_submission_group_0_order_4.csv` — `LightGBM` (`bayes_scaled=1.16`), OOF `$14,321,612`, allocates `1,315` (`27.7%`)
  5. `challenge_submission_group_0_order_5.csv` — `LogisticRegression` (`bayes_exact`), OOF `$14,336,991`, allocates `1,589` (`33.5%`)

---

## 8. Final Issue-by-Issue Audit Matrix

| # | Issue Audited | Root Cause Found | File | Correction Implemented | Verification Run | Result |
|---:|---|---|---|---|---|:---:|
| 1 | LightGBM execution & API compat | Potential keyword mismatch across LightGBM 4.5–4.7+ & ES isolation | [`src/modeling.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) | [`fit_lgbm_with_early_stopping`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) (`eval_set` + fallback) & strict 85/15 inner ES | `pytest` LightGBM + legacy wrapper test & full CV | **PASS** |
| 2 | XGBoost & CatBoost ES leakage | Early stopping previously peeked at outer `y_va` | [`src/modeling.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) | Structural exclusion of `y_va`; strict internal 85/15 split & vocab fit | `test_outer_validation_isolation_and_no_leakage` | **PASS** |
| 3 | Calibrated stacking `C` selection bias | `best_c` tuned over full `Z_inner` prior to cal-fold split | [`src/modeling.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) | [`_select_best_meta_c`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) + [`evaluate_calibration_nested`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) nesting `c_j` inside `cal_tr` | Nested cal label-flip test & 5-fold audit CSV | **PASS** |
| 4 | Hospitalisation effect-measure inconsistency | Chronic Ko et al. `aRR` added in `log_odds` alongside `OR`s; `65+` unmapped | [`src/cost_model.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) | `CHRONIC_CONDITION_RR_MAP` on `cloglog` link + [`zhang_yu_rr_to_or`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) + `AGE_BRACKET_MAPPING_DOC` + verified DOIs | Effect-measure & `[0.002, 0.85]` bound tests; zero clinical data caveat | **PARTIAL** *(no empirical admission labels in Dataset C)* |
| 5 | Cost policy CV fallback & component reporting | `folds=None` fallback tuned in-sample; `hackathon_report.md` lacked 4-policy table | [`src/cost_model.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py), [`main.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/main.py) | Forced 5-fold CV when `folds=None`; [`summarize_cost_policies_table`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) exported to CSV & MD | Cost accounting identity & fallback CV tests + `cost_policy_comparison.csv` | **PASS** |
| 6 | Feature ablation completeness & fold parity | Missing additive runs, Brier/ECE metrics, and decision table | [`src/data_prep.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/data_prep.py) | Shared 5-fold Stratified CV (`seed=42`), additive + drop runs, all 4 metrics, `decision` & `reason` | `test_feature_ablation_metrics_and_structure` & `feature_ablation_report.csv` | **PASS** |
| 7 | Preprocessing & zero-opinion audit | Undocumented `0`s in 1–5 opinion scales not proven zero-leakage | [`src/data_prep.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/data_prep.py) | Empirical `0 -> NaN` + `_is_zero` flags logged to `preprocessing_validation.json` | 4 preprocessing/determinism/duplicate tests | **PASS** |
| 8 | Multi-fold SHAP warnings & categorical plot | Older single-fold list warning and `.cat.codes` beeswarm distortion | [`src/explainability.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/explainability.py) | 5-fold OOF `explainer(X).values`, separated numeric `08` & categorical `08b`, log-odds `09` | `test_multifold_shap_stability_and_separation` (0 warnings) | **PASS** |
| 9 | Pytest suite runtime timeout (>120s) | Heavy fixture CSV reads & unrestricted fast-mode threads/trees | [`src/modeling.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py), [`tests/test_pipeline.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/tests/test_pipeline.py) | Module-scoped `raw_data` fixture, single-thread `fast_mode` (`80/80/60`), added nested/cloglog/CV tests | `pytest -v`: **18 passed in 77.91s** | **PASS** |
| 10 | End-to-end pipeline & artifacts | Needed fresh run after `cloglog`, nested-cal, ablation, and policy updates | [`main.py`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/main.py) | Full execution `main.py --group 0` (`[1/6]..[6/6]`) | Exit code `0`; all 16 outputs & 10 PNGs regenerated | **PASS** |
| 11 | Top-5 submissions parity & distinctness | Required schema verification + proof of non-degenerate distinctness | [`submissions/`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/submissions/submission_manifest.csv) | Ranked distinct model families (`2.74%–7.50%` pairwise diffs, unique IDs `4757..9505`) | `test_generated_submissions_match_example_schema` + manifest inspection | **PASS** |
