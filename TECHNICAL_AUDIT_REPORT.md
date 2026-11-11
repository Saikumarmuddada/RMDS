# RMDS Hackathon Dataset C — Technical Audit & Correction Report

## 1. Audit Summary & Key Outcomes
An end-to-end technical audit of the RMDS Hackathon Dataset C pipeline (4,756 labeled training respondents; 4,749 testing respondents; vaccine cost $C_v = \$1{,}500$; COVID-19 hospitalisation cost $C_h = \$41{,}000$; 100% infection assumption) identified **four major methodological flaws** in the original implementation and corrected all of them in working code, verified by an expanded 11-test automated `pytest` suite and full pipeline re-execution (`main.py --group 0`).

### Root Problems Found & Fixed
1. **Outer Validation Leakage via Boosting Early Stopping (`src/modeling.py`):** LightGBM, XGBoost, and CatBoost previously passed the outer validation fold `(X_va, y_va)` directly to `eval_set` for early stopping, leaking validation labels into model complexity selection. **Fix:** Structural removal of `y_va` from [`fit_outer_fold_models`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py) and [`_fit_base_models_on_train_only`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/modeling.py); boosting early stopping now uses an internal 85%/15% stratified split carved strictly from the active training fold (`3,233–3,234` fit / `571` inner-ES rows per outer fold).
2. **Cross-Fitting & Calibration Leakage in Stacking (`src/modeling.py`):** `Calibrated_Stack` previously trained per-fold meta-learners over full outer `oof_dict` predictions whose underlying base models had seen the current outer fold's validation observations in complementary folds, without out-of-sample post-hoc calibration or ECE tracking. **Fix:** Implemented a **nested 5x4 cross-fitting architecture** inside each outer training partition to produce strictly uncontaminated inner OOF base logits `Z_inner`, selected regularization `C` via inner CV, generated strictly out-of-fold inner meta predictions, and evaluated `none` vs `platt` vs `isotonic` calibration on held-out inner folds. Across all 5 outer folds (`0..4`), isolated inner-CV LogLoss confirmed that uncalibrated `C = 0.05` L2 logit stacking (`0.4168–0.4228`) already outperformed Platt (`0.4176–0.4243`) and Isotonic (`0.4219–0.4303`), cutting out-of-fold **ECE from `0.0143` down to `0.0088`**.
3. **Undocumented / Heuristic Hospitalisation Parameters (`src/cost_model.py`):** `BASE_AGE_IHR`, exponential `chronic * log(1.55)` compounding, and uncalibrated socioeconomic multipliers (`health_insurance == 0 -> 1.15`, `income_poverty == "Below Poverty" -> 1.12`) lacked literature anchors. **Fix:** Replaced `BASE_AGE_IHR`, discrete chronic condition multipliers, sex scaling, and VE estimates with documented peer-reviewed anchors (`Herrera-Esposito & de los Campos 2022`, `Verity et al. 2020`, `Ko et al. 2021`, `Peckham et al. 2020`, `Williamson et al. 2020`, `Tenforde et al. 2021`, `Ferdinands et al. 2022`), removed the unsupported socioeconomic risk multipliers (noting 40.1% missingness on `health_insurance`), added edge-case protection (`VE <= 0`, `C_v = 0`, `NaN`/out-of-range inputs), and expanded multi-axis sensitivity evaluation across `IHR Scenario x VE Regime x Sex Adjustment`.
4. **SHAP Single-Fold Bias, Misleading Category Codes & Feature Bloat (`src/explainability.py`, `src/data_prep.py`):** SHAP previously analyzed only Fold 0 and mapped unordered categorical strings onto `.cat.codes` for the continuous beeswarm colorbar; the meta-learner chart mislabeled log-odds coefficients as "Blend Weights"; and 5 collinear opinion composites inflated variance. **Fix:** Computed out-of-fold SHAP across all 5 outer folds (`outputs/shap_feature_importance.csv` with fold mean/SD and rank stability), restricted the beeswarm plot to numeric/ordinal/binary features (`08_explainability_shap_beeswarm.png`), added a dedicated categorical category-level bar chart (`08b_explainability_categorical_shap.png`), renamed the meta-learner plot accurately (`Meta-Learner Logistic Regression Coefficients (Log-Odds)`), and pruned the 5 noisy opinion composites (`54 -> 49` features) after leakage-free CV ablation improved both Logistic Regression and LightGBM LogLoss/ROC-AUC.

### Remaining Limitations Explicitly Noted
* **No Ground-Truth Hospitalisation Outcomes in Dataset C:** `dataset_C_training.csv` contains vaccine acceptance (`covid_vaccine`), not COVID-19 hospitalisations. All hospitalisation probabilities `(q0, q1)` remain **modelled epidemiological estimates**, not empirical clinical predictions learned from Dataset C.
* **Epidemiological Bracket & Variant Extrapolation:** Cited seroprevalence IHR studies (`Herrera-Esposito 2022`, `Verity 2020`) use pre-vaccination ancestral cohorts and 10-year age brackets (`20-29`, `30-39`, ..., `70-79`, `80+`) mapped onto Dataset C's aggregated `18 - 34 Years` and `65+ Years` brackets.
* **Population Rate Ratios vs Conditional Severity:** Ko et al. (2021) reported population-rate `aRR = 5.0 (95% CI: 3.9–6.3)` for `>=3` chronic conditions; applying this as an odds multiplier on bracket IHR assumes comparable infection exposure across chronic strata (consistent with the assignment's 100% winter-party infection rule).

---

## 2. Leakage Audit & Nested Cross-Validation (`src/modeling.py`)

### Before (Leaky) vs. After (Audited) Validation Architecture

| Stage | Original Implementation (Leaky) | Corrected Implementation (Leakage-Free) |
| :--- | :--- | :--- |
| **Preprocessing & Categories** | Category vocabs fitted on `X_tr` inside outer loop (acceptable), but boosting early stopping looked directly at `(X_va, y_va)`. | `_prepare_category_frames` fits category vocabularies strictly on `X_bfit` (for boosting) and `X_tr` (for sklearn), mapping unseen categories safely to `"Missing"`. |
| **Boosting Early Stopping** | `LGBMClassifier`, `XGBClassifier`, and `CatBoostClassifier` passed outer `(X_va, y_va)` to `eval_set`. | `y_va` is absent from `fit_outer_fold_models` and `_fit_base_models_on_train_only` signatures; an internal 85%/15% stratified ES split (`3,233–3,234` fit / `571` ES rows) inside active `tr_idx` governs stopping. |
| **Stacking Meta-Features** | Meta-learner fit across outer `oof_dict` whose base folds overlapped the active outer fold. | Inside each outer training fold (`n = 3,804–3,805`), an inner 4-fold Stratified CV trains all 6 base models (with inner ES inside each 3/4 slice) to yield strictly clean `Z_inner` (`logit(inner_oof_probs)`). |
| **Hyperparameter & Calibration** | Fixed `C=0.3`; no post-hoc calibration comparison or ECE evaluation. | Inner CV selects `C in {0.05, 0.1, 0.25, 0.5, 1.0}` on `Z_inner` (`C=0.05` across all 5 outer folds); out-of-fold inner meta predictions (`meta_inner_oof`) evaluate `none` vs `platt` vs `isotonic` strictly out-of-sample before transforming outer `Z_va` and `Z_te`. |

### Isolated Calibration Audit (`outputs/stacking_calibration_audit.csv`)
| Outer Fold | Best Meta `C` | Inner-OOF LogLoss (`none`) | Inner-OOF LogLoss (`platt`) | Inner-OOF LogLoss (`isotonic`) | Selected Calibrator | ES Fit / Val Rows |
| ---: | ---: | ---: | ---: | ---: | :--- | ---: |
| 0 | 0.05 | **0.4191** | 0.4206 | 0.4303 | `none` | 3,233 / 571 |
| 1 | 0.05 | **0.4171** | 0.4193 | 0.4222 | `none` | 3,234 / 571 |
| 2 | 0.05 | **0.4228** | 0.4243 | 0.4303 | `none` | 3,234 / 571 |
| 3 | 0.05 | **0.4168** | 0.4176 | 0.4219 | `none` | 3,234 / 571 |
| 4 | 0.05 | **0.4185** | 0.4197 | 0.4257 | `none` | 3,234 / 571 |

Because L2 Logistic Regression meta-stacking (`C=0.05`) achieved the best inner-OOF LogLoss and an outer OOF **ECE of `0.0088`** without Platt/Isotonic warping, `Calibrated_Stack` and `Uncalibrated_Stack` agree identically while outperforming every base model on **ROC-AUC (`0.8670`), LogLoss (`0.4173`), Brier (`0.1313`), ECE (`0.0088`), and Bayes Town Cost (`$13,970,808`)**.

---

## 3. Scientific Validation of Hospitalisation Risk (`src/cost_model.py`)

| Component | Parameterization Used | Scientific Source & Empirical Estimate | Removed / Refined vs. Original |
| :--- | :--- | :--- | :--- |
| **Baseline Age IHR (`BASE_AGE_IHR`)** | `18-34`: `0.006`, `35-44`: `0.015`, `45-54`: `0.031`, `55-64`: `0.066`, `65+`: `0.145` (unknown fallback: `0.035`) | **Herrera-Esposito & de los Campos (2022)** *BMC Infect Dis* 22:311 (`10.1186/s12879-022-07262-0`: `20-29: 0.47%`, `30-39: 0.99%`, `40-49: 2.11%`, `50-59: 4.49%`, `60-69: 9.54%`); **Verity et al. (2020)** *Lancet Infect Dis* (`10.1016/S1473-3099(20)30243-7`: `1.0%–18.4%` through `80+`); **Salje et al. (2020)** *Science*. | Anchored lower/middle adult brackets directly to seroprevalence denominators rather than unreferenced `[0.01, 0.02, 0.04, 0.075, 0.16]`. |
| **Chronic Conditions (`CHRONIC_CONDITION_OR_MAP`)** | Discrete concave ORs: `0 -> 1.00`, `1 -> 1.70`, `2 -> 2.90`, `>=3 -> 5.00` (capped at `3`) | **Ko et al. (2021)** *Clin Infect Dis* 72(11):e695-e703 (`10.1093/cid/ciaa1419`: COVID-NET/BRFSS US adults with `>=3` conditions **aRR = 5.0 [95% CI: 3.9–6.3]** vs 0); concave marginal progression corroborated by **Kompaniyets et al. (2021)** *Prev Chronic Dis*. | Replaced uncited `1.55^chronic` (which reached `8.95x` at 5 conditions). |
| **Sex Adjustment (`SEX_ODDS_MULTIPLIER`)** | Centered OR = `1.45` (`Male: 1.20`, `Female: 0.8276`, missing: `1.00`; optional toggle `include_sex_adjustment`) | **Williamson et al. (2020, OpenSAFELY)** *Nature* (`HR = 1.59`); **Peckham et al. (2020)** *Nat Commun* (ITU `OR = 2.84`, death `OR = 1.39`); **Petrilli et al. (2020)** *BMJ* (NYC Wave-1 admission `aOR = 2.8`). | Documented conservative `1.45` Male/Female synthesis plus explicit `include_sex_adjustment=False` sensitivity row. |
| **Socioeconomic Factors** | **Removed** from hospitalisation IHR | Insufficiently calibrated independent biological admission ORs on top of age/chronic/sex, plus `health_insurance` has **40.1%** missingness in Dataset C (kept as features for *acceptance* modeling only). | Eliminated arbitrary `health_insurance == 0 -> 1.15` and `income_poverty == Below Poverty -> 1.12` multipliers. |
| **Vaccine Effectiveness (`VE_REGIMES`)** | `base`: `0.86` standard / `0.80` high-frailty (`65+` or `>=3` chronic). Sensitivity: `waned_omicron` (`0.65 / 0.55`), `boosted` (`0.91 / 0.86`). | **Tenforde et al. (2021)** *JAMA* 326(20):2043-2054 (`10.1001/jama.2021.19499`, PMID `34734975`: 2-dose mRNA `86%` [95% CI `82–88%`], `90%` immunocompetent, lower immunocompromised); **Ferdinands et al. (2022)** *MMWR* 71(7):255-263 (waned 2-dose Omicron `54–79%`, 3-dose boosted `91%`). | Updated standard VE `0.85 -> 0.86`, aligned frailty cutoff to `>=3` chronic / `65+`, and added Omicron waning & boosted sensitivity grid. |

All output hospitalisation probabilities are strictly verified finite and bounded in $[0.002, 0.850]$ (`mean q0 = 0.0760`, `mean q1 = 0.0135` under `base`).

---

## 4. Expected Town Cost Derivation & Policy Validation (`src/cost_model.py`)

Under 100% infection risk, $C_v = \$1{,}500$, $C_h = \$41{,}000$, acceptance probability $p_i = P(Y_i = 1 \mid X_i)$, unvaccinated hospitalisation risk $q_{0,i}$, and vaccinated hospitalisation risk $q_{1,i}$:

| Action $a_i$ | Outcome $Y_i = 1$ (Accepts) | Outcome $Y_i = 0$ (Refuses) |
| :--- | :--- | :--- |
| **Send ($a_i = 1$)** | $C_v + C_h q_{1,i}$ | $C_v + C_h q_{0,i}$ *(dose wasted)* |
| **Don't Send ($a_i = 0$)** | $C_v + C_h q_{0,i}$ *(local-law payout)* | $C_h q_{0,i}$ |

Taking expectations over $Y_i \sim \text{Bernoulli}(p_i)$:
$$E[\text{Cost}(a_i = 1)] = C_v + C_h q_{0,i} - p_i C_h (q_{0,i} - q_{1,i}), \qquad E[\text{Cost}(a_i = 0)] = p_i C_v + C_h q_{0,i}$$
Subtracting yields the exact Bayes decision rule:
$$E[\text{Cost}(1)] - E[\text{Cost}(0)] = (1 - p_i) C_v - p_i C_h (q_{0,i} - q_{1,i}) \le 0 \iff p_i \ge \tau^*(x_i) = \frac{C_v}{C_v + C_h (q_{0,i} - q_{1,i})} \quad (\text{for } q_{0,i} > q_{1,i})$$

* **Edge Case Guarantees Verified in Code & Tests:**
  * Zero or adverse VE ($q_{1,i} \ge q_{0,i}$): $\Delta q_i \le 0 \implies \tau^* = 1.0$ AND [`predict_with_cost_policy`](file:///c:/Users/syamp/Downloads/RMDS/RMDS/src/cost_model.py) enforces `beneficial = (q0 - q1) > 0`, ensuring zero doses are allocated even at `p_i = 1.0` or `threshold_param < 1.0`. Also corrected `oracle_min_cost` to allocate on `(y_t == 1) & (q0 > q1)`.
  * Free vaccine ($C_v = 0$): yields $\tau^* = 0.0$ when $q_{0,i} > q_{1,i}$ and $\tau^* = 1.0$ when $q_{0,i} \le q_{1,i}$.
  * Invalid / `NaN` probabilities: safe mode (`strict=False`) emits a `RuntimeWarning`, clips float overshoot, and defaults missing probabilities to conservative non-allocation (`tau* = 1.0`, `pred = 0`); `strict=True` raises `ValueError`.
* **Cost Policy & Multi-Axis Sensitivity (`outputs/ve_and_risk_sensitivity_grid.csv`, evaluated on `Calibrated_Stack` OOF):**
  * Fixed `0.50` cutoff (`$14,415,518`, `1,296` allocated, `283.8` exp. hosp) vs **Bayes-Exact** (`$13,970,808`, `1,476` allocated, `481` wasted [`$721,500`], `559` law payouts [`$838,500`], `266.3` exp. hosp [`$10,918,308`]): **saves `$444,710` (`68.2%` of Oracle savings)** without test-set tuning.
  * Across all **9 sensitivity cells** (`conservative/base/severe` $\times$ `waned_omicron/base/boosted` + sex-neutral check), person-specific Bayes thresholding consistently outperforms the fixed `0.50` cutoff by **`$139,559` (`conservative + waned_omicron`) to `$766,843` (`severe + boosted`)**. Removing sex adjustment shifts base Bayes town cost from `$13,970,808` to `$14,394,472` while still saving `$424,928` over fixed `0.50`.

---

## 5. Multi-Fold OOF Explainability (`src/explainability.py`)
* Computed `shap.TreeExplainer` strictly **out-of-fold across all 5 outer LightGBM fold models** (`n = 4,756` combined OOF rows, `outputs/shap_feature_importance.csv`), verifying rank stability:
  1. `health_insurance` (`numeric`): `0.5331 ± 0.0595` (`rank 1.0 ± 0.0`)
  2. `opinion_covid_risk` (`numeric`): `0.4552 ± 0.0685` (`rank 2.2 ± 0.45`)
  3. `doctor_x_effective` (`numeric`): `0.3811 ± 0.0398` (`rank 3.0 ± 0.71`)
  4. `opinion_covid_vacc_effective` (`numeric`): `0.2923 ± 0.0128` (`rank 4.4 ± 0.55`)
  5. `doctor_x_risk` (`numeric`): `0.2565 ± 0.0747` (`rank 4.4 ± 0.89`)
  6. `employment_sector` (`categorical`): `0.1644 ± 0.0345` (`rank 6.6 ± 1.34`)
* **Separated Visualizations:** `06_explainability_shap_importance.png` plots fold-SD error bars and distinguishes numeric vs categorical types; `08_explainability_shap_beeswarm.png` displays **strictly numeric/ordinal/binary features** (removing `.cat.codes` distortion); `08b_explainability_categorical_shap.png` reports exact signed mean SHAP by string category (e.g., `healthcare (n=527)` vs `finance (n=95)` in `employment_sector`; `Male` vs `Female`; `White` vs `Black`; `Married` vs `Not Married`); `09_meta_weights_and_sensitivity.png` plots `Meta-Learner Logistic Regression Coefficients (Log-Odds)` (`LogisticRegression: 0.3685`, `CatBoost: 0.2150`, `XGBoost: 0.1793`, `LightGBM: 0.1325`, `RandomForest: 0.0944`, `HistGradientBoosting: 0.0512`).

---

## 6. Preprocessing Audit & Feature-Group Ablation (`src/data_prep.py`)
* **Integrity Checks (`outputs/preprocessing_validation.json`):** `0` exact/feature duplicates in train (`4,756`) or test (`4,749`); `0` `respondent_id` overlap (`1..4756` vs `4757..9505`); `0` unseen nominal test categories; `respondent_id` and `covid_vaccine` excluded from all 49 feature columns; undocumented `0`s in 1–5 opinion scales verified empirically (`opinion_covid_vacc_effective == 0`: `25.0%` target rate vs rank `1`: `3.2%`; `opinion_covid_risk == 0`: `27.1%` vs rank `1`: `12.5%`).
* **CV Ablation (`outputs/feature_ablation_report.csv`):** Pruning the 5 noisy collinear opinion composites (`opinion_net_benefit`, `opinion_risk_minus_worry`, `opinion_pro_vaccine_index`, `opinion_eff_x_risk`, `concern_x_knowledge`) reduced variance and improved baseline-to-validated CV scores:
  * `raw_baseline_0_untouched` (29 cols): LR AUC `0.8306` / LogLoss `0.4662` | LGBM AUC `0.8598` / LogLoss `0.4265`
  * `raw_plus_zero_as_nan` (31 cols): LR AUC `0.8384` / LogLoss `0.4562` | LGBM AUC `0.8611` / LogLoss `0.4252`
  * `validated_engineered_full` (49 cols): LR AUC **`0.8654`** / LogLoss **`0.4218`** | LGBM AUC **`0.8624`** / LogLoss **`0.4239`**
  * Dropping `missing_flags` degraded LR drastically (`0.8654 -> 0.8379` AUC), while dropping `ordinal` (`0.8624 -> 0.8602`) or `doctor_interactions` (`0.8624 -> 0.8609`) hurt LightGBM.

---

## 7. Automated Testing (`tests/test_pipeline.py`)
Expanded from 5 basic checks to **11 comprehensive automated tests** covering:
* **Category A & B:** Whitespace cleanup, zero-opinion sentinels, pruned collinear composites, determinism, zero target-label leakage, and synthetic unseen-category safety.
* **Category C & D:** Structural verification that `fit_outer_fold_models` never accepts `y_va`, validation-row feature poisoning test proving zero fit/encoder/scaler/stack leakage onto other validation or test predictions, `es_fit + es_val == len(X_tr)`, inner OOF sizing, calibrator selection bounds, and exact `compute_ece` math.
* **Category E & F:** Literature-aligned monotonicity & `>=3` chronic cap (`4 == 3 == 5`), absence of unsupported socioeconomic IHR multipliers, VE regime ordering, exact expected-cost conservation identities, Bayes break-even equality `E[Cost(1)] == E[Cost(0)]`, `VE <= 0` dose refusal even at `p = 1.0`, `C_v = 0` boundary, and `NaN`/out-of-range handling (`strict=False` vs `strict=True`).
* **Category G & H:** Multi-fold SHAP stability/output files (`06`, `08`, `08b`) and strict validation of all 5 Canvas submission CSVs (`4,749` rows, exact `respondent_id` sequence `4757..9505`, binary `{0, 1}`).

---

## 8. Before-vs-After Benchmark Under Identical Evaluation
*(Both pre-audit and post-audit OOF probability tables evaluated identically under the scientifically grounded `base` hospitalisation risk model and `compute_ece`.)*

| Model | Version | ROC-AUC | LogLoss | Brier | ECE | Bayes Cost ($) | Fixed 0.50 Cost ($) | Allocated / Wasted / Law | Exp. Hosp | Oracle % |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: |
| **LogisticRegression** | Before (Leaky) | 0.8652 | 0.4227 | 0.1322 | 0.0166 | $14,034,514 | $14,446,926 | 1520 / 524 / 558 | 266.28 | 66.8% |
| **LogisticRegression** | After (Audited) | **0.8654** | **0.4218** | **0.1321** | **0.0147** | **$14,033,902** | **$14,431,588** | 1531 / 534 / 557 | 265.90 | **66.8%** |
| **RandomForest** | Before (Leaky) | 0.8543 | 0.4357 | 0.1383 | **0.0192** | $14,023,797 | **$14,628,416** | 1452 / 507 / 609 | 266.64 | 67.1% |
| **RandomForest** | After (Audited) | **0.8586** | **0.4351** | **0.1378** | 0.0342 | **$14,004,588** | $14,688,404 | 1432 / 510 / 632 | 266.06 | **67.5%** |
| **HistGradientBoosting** | Before (Leaky) | **0.8542** | **0.4440** | **0.1390** | **0.0372** | **$14,053,049** | **$14,591,799** | 1474 / 472 / 552 | 268.64 | **66.4%** |
| **HistGradientBoosting** | After (Audited) | 0.8526 | 0.4485 | 0.1400 | 0.0408 | $14,073,918 | $14,610,676 | 1464 / 467 / 557 | 269.33 | 66.0% |
| **LightGBM** | Before (Leaky ES) | 0.8635 | 0.4219 | 0.1332 | 0.0137 | $13,992,498 | $14,485,338 | 1446 / 475 / 583 | 267.05 | 67.7% |
| **LightGBM** | After (Inner ES) | 0.8622 | 0.4238 | 0.1341 | **0.0128** | $14,037,696 | $14,515,615 | 1471 / 489 / 572 | 267.64 | 66.8% |
| **XGBoost** | Before (Leaky ES) | 0.8654 | 0.4199 | 0.1326 | 0.0136 | $13,987,014 | $14,485,848 | 1461 / 479 / 572 | 266.77 | 67.9% |
| **XGBoost** | After (Inner ES) | 0.8618 | 0.4249 | 0.1342 | 0.0137 | $14,030,042 | $14,487,463 | 1442 / 475 / 587 | 267.96 | 66.9% |
| **CatBoost** | Before (Leaky ES) | 0.8656 | 0.4188 | 0.1320 | 0.0168 | $13,974,499 | $14,442,737 | 1433 / 459 / 580 | 267.20 | 68.1% |
| **CatBoost** | After (Inner ES) | 0.8640 | 0.4204 | 0.1326 | **0.0136** | $14,072,497 | $14,502,418 | 1437 / 477 / 594 | 268.93 | 66.0% |
| **Calibrated_Stack** | Before (Leaky OOF) | 0.86702 | 0.41734 | 0.13128 | 0.01430 | $13,984,743 | $14,410,808 | 1475 / 486 / 565 | 266.46 | 67.9% |
| **Calibrated_Stack / Uncalibrated_Stack** | After (Nested 5x4) | **0.86697** | **0.41725** | **0.13135** | **0.00884** | **$13,970,808** | $14,415,518 | 1476 / 481 / 559 | 266.30 | **68.2%** |

*Key Diagnostic Takeaway:* Removing outer-validation early stopping eliminated the optimistic bias on CatBoost (`0.8656 -> 0.8640` AUC, `$13.97M -> $14.07M`), XGBoost (`0.8654 -> 0.8618` AUC), and LightGBM (`0.8635 -> 0.8622` AUC), while nested 5x4 stacking over the pruned 49-feature space **lowered LogLoss (`0.41734 -> 0.41725`), cut ECE by 38% (`0.01430 -> 0.00884`), and reduced expected town cost to `$13,970,808` (`68.2%` of oracle savings)**.

---

## 9. Top 5 Verified Submissions & Reproducibility
All 5 submission files (`submissions/challenge_submission_group_0_order_1.csv` .. `order_5.csv`) match `DOCs/dataset_C_example_submission.csv` (`4,749` rows, `respondent_id` `4757..9505`, binary `{0, 1}`, zero NaNs):
1. `challenge_submission_group_0_order_1.csv` — `Uncalibrated_Stack` (`bayes_exact`, identical to `Calibrated_Stack`), OOF cost `$13,970,808`, allocates `1,546` (`32.6%`)
2. `challenge_submission_group_0_order_2.csv` — `CatBoost` (`bayes_scaled`, `param=0.88`), OOF cost `$13,992,147`, allocates `1,682` (`35.4%`)
3. `challenge_submission_group_0_order_3.csv` — `RandomForest` (`bayes_exact`), OOF cost `$14,004,588`, allocates `1,495` (`31.5%`)
4. `challenge_submission_group_0_order_4.csv` — `LightGBM` (`bayes_scaled`, `param=1.17`), OOF cost `$14,021,900`, allocates `1,296` (`27.3%`)
5. `challenge_submission_group_0_order_5.csv` — `XGBoost` (`bayes_exact`), OOF cost `$14,030,042`, allocates `1,503` (`31.6%`)

