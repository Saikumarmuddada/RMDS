# RMDS Hackathon (Dataset C — COVID-19) Complete Project Guide & Compliance Audit

This document serves two purposes:
1. **Compliance Audit** against every instruction in `Images/` and `DOCs/` (confirming what is implemented in code and what remains for your team to finalize for submission).
2. **Task-by-Task & File-by-File Explanation** detailing everything implemented and exactly what every file in this repository does.

---

## Part 1: Compliance Audit Against Docs & Images

### A. `Images/RDMS.jpeg` — Assignment Rules & Advice

| Requirement / Guideline | Status | Implementation / Notes |
| :--- | :--- | :--- |
| **Data exploration** (by 16:00) | Implemented | `src/eda.py` computes target rates, train/test missingness, Pearson correlations, subgroup rates, and Figures `01`–`03`. |
| **Pre-processing** (by 16:00) | Implemented | `src/data_prep.py` cleans `"MSA, Not Principle  City"` whitespace, converts undocumented `0`s in 1–5 opinion scales to `NaN` + `_is_zero` flags, creates `_is_missing` flags, and `src/modeling.py` runs fold-safe imputation/scaling/encoding. |
| **Modelling** (by 16:00) | Implemented | `src/modeling.py` trains 6 diverse models (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `LightGBM`, `XGBoost`, `CatBoost`) + leakage-free nested 5x4 cross-fitted `Calibrated_Stack` & `Uncalibrated_Stack`. |
| **Performance analysis** (by 16:00) | Implemented | Evaluates ROC-AUC, LogLoss, Brier, ECE, Accuracy, F1, naive `0.50` cost, global-tuned cost, Bayes-exact cost, Bayes-scaled cost, and 9 IHR x VE x Sex sensitivity cells (`outputs/model_comparison.csv`, `outputs/sensitivity_matrix.csv`, `outputs/ve_and_risk_sensitivity_grid.csv`). |
| **Submission of results** (best 5 files matching `dataset_X_example_submission.csv`, named `challenge_submission_group_X_order_Y`) | Implemented (Placeholder Group `0`) | `submissions/challenge_submission_group_0_order_1.csv` to `order_5.csv` match exact `respondent_id` ordering and columns. **Team action:** Re-run `.venv\Scripts\python.exe main.py --group <YOUR_GROUP_NUMBER>` before uploading to Canvas by 16:00. |
| **Code presentation and Documentation** | Implemented | `README.md`, `outputs/hackathon_report.md`, `TECHNICAL_AUDIT_REPORT.md`, `PROJECT_GUIDE.md`, and `notebooks/RMDS_Hackathon_Dataset_C.ipynb`. |
| **Explainable AI** (after hackathon / started here) | Implemented | `src/explainability.py` computes 5-fold OOF LightGBM TreeExplainer SHAP values with stability SD (`outputs/shap_feature_importance.csv`), bar chart (`06`), numeric-only beeswarm (`08`), categorical drivers (`08b`), and meta-learner log-odds coefficients (`09`). |
| **Figure embellishment** (after hackathon / started here) | Implemented | 9 publication-style seaborn/matplotlib charts saved in `outputs/figures/` (`01` through `09`). |
| **Presentation** (after hackathon) | Ready to build | Figures `01`–`09` and `outputs/hackathon_report.md` provide ready-to-paste metrics and charts. **Team action:** Put together your slide deck. |
| **Colab & GitHub commits** (visible individual coding/analysis contributions) | Ready / User Action | Colab-ready notebook created at `notebooks/RMDS_Hackathon_Dataset_C.ipynb`. Git repo is initialized on `main` with no commits yet — **Team action:** commit/push your team's work to GitHub and share code links on Canvas. |

### B. `Images/Target matrix info.jpeg` — Target Metric Information

| Requirement / Rule | Status | Implementation / Notes |
| :--- | :--- | :--- |
| **Goal: minimize town spend = vaccine costs + hospitalisation costs** | Implemented | `compute_town_cost()` in `src/cost_model.py` sums vaccine spend, local-law payout, and expected hospitalisation cost. |
| **100% infection assumption (winter indoor parties)** | Implemented | `estimate_hospitalisation_probabilities()` models hospitalisation probability conditional on infection (`P(hosp | infected, features, vacc_status)`). |
| **Local law: if someone wants a vaccine and none is available, town pays vaccine cost** | Implemented | `law_penalty_cost = sum((y_pred == 0) & (y_true == 1)) * vaccine_cost`. |
| **Spoilage: vaccines mailed home are ruined if unused** | Implemented | `vaccine_send_cost = sum(y_pred == 1) * vaccine_cost`; models predict individual `0`/`1` acceptance rather than just aggregate counts. |
| **Estimate hospitalisation probability from dataset factors + vaccination status** | Implemented | Incorporates peer-reviewed age IHR (`Herrera-Esposito 2022`, `Verity 2020`), chronic conditions (`Ko 2021`, capped `>=3` aRR `5.0`), sex (`OR=1.45`), and vaccine effectiveness (`Tenforde 2021`, `Ferdinands 2022`: `86%` standard, `80%` high-frailty); excludes uncalibrated socioeconomic multipliers. |
| **Use Generative AI for hospitalisation probability estimation & check robustness across AIs/perspectives** | Implemented + Optional Team Check | Tested 3 risk calibration profiles (`conservative` `0.5x`, `base` `1.0x`, `severe` `1.5x` odds) x 3 VE regimes (`waned_omicron`, `base`, `boosted`) + sex-neutral sensitivity in `outputs/sensitivity_matrix.csv` and `outputs/ve_and_risk_sensitivity_grid.csv`. |

### C. Dataset Specs (`Images/Dataset C.jpeg`, `Images/Dataset C2.jpeg`, `DOCs/Dataset B Information.pdf`)

| Detail | Status | Notes |
| :--- | :--- | :--- |
| **Dataset C (COVID-19) active data** | Implemented | `DATASET_C_CONFIG` uses `covid_*` features, target `covid_vaccine`, `$1,500` vaccine cost, `$41,000` hospitalisation cost (`DOCs/dataset_C_*.csv`). |
| **Dataset B (H1N1) PDF present in `DOCs/`** | Configured | `DATASET_B_CONFIG` (`$500` / `$25,000`, `h1n1_vaccine`) is also defined in `src/config.py`, though only Dataset C CSVs are present in `DOCs/`. |
| **31 columns wording vs 29 features** | Handled | Verified training has 31 columns (`respondent_id` + 29 features + `covid_vaccine`) and testing has 30 columns (`respondent_id` + 29 features). |
| **Undocumented `0`s in 1–5 opinion scales & double whitespace** | Handled | Cleaned cleanly in `src/data_prep.py` and verified by unit tests in `tests/test_pipeline.py`. |

### D. `Images/RDMS BEhaviour rules.jpeg` — CDIO Room Regulations
Operational/conduct rules for participants (own laptops only, silent phones, no food, drinks only on door shelves, staff pass for breaks/re-entry, stay with assigned team, step outside for calls). Noted for your team on-site; not a software feature.

---

## Part 2: Task-by-Task Summary (What Was Done)

1. **Exploratory Data Analysis (`src/eda.py`)**
   * Loaded `4,756` training rows (`32.67%` accepted `covid_vaccine = 1`) and `4,749` testing rows.
   * Audited missingness parity between train and test (`employment_sector` ~49–51% missing, `health_insurance` ~40–42% missing, `income_poverty` ~16–17% missing).
   * Discovered that missing `health_insurance` is strongly predictive of *lower* acceptance (`17.3%` vs `45.8%` for insured) and that `doctor_recc_covid = 1` raises acceptance from `21.7%` to `69.9%`.

2. **Cleaning & Feature Engineering (`src/data_prep.py`)**
   * Normalized whitespace in categorical strings (`"MSA, Not Principle  City"` -> `"MSA, Not Principle City"`).
   * Flagged undocumented `0`s in `opinion_covid_vacc_effective` and `opinion_covid_risk` (`*_is_zero`) and replaced `0` with `NaN` so ordinal calculations are undistorted.
   * Created missingness indicators (`health_insurance_is_missing`, `employment_sector_is_missing`, `income_poverty_is_missing`, `rent_or_own_is_missing`, `row_missing_count`).
   * Engineered ordinal mappings (`age_ordinal`, `education_ordinal`, `income_ordinal`), behavioral totals (`behavioral_sum`, `behavioral_mean`), doctor interactions (`doctor_x_effective`, `doctor_x_risk`, `doctor_x_not_worried`), and vulnerability flags (`household_total`, `has_chronic_condition`, `high_risk_elderly`, `is_healthcare_sector`, `uninsured_flag`) — validated via CV ablation (`outputs/feature_ablation_report.csv`, pruning 5 collinear opinion composites to `49` features).

3. **Hospitalisation Cost Model & Bayes Thresholding (`src/cost_model.py`)**
   * Formulated the exact town cost matrix combining `$1,500` vaccine spend/penalty and `$41,000` expected hospitalisation cost with edge-case guards (`VE <= 0`, `NaN`/out-of-range inputs, `C_v = 0`).
   * Derived the per-person break-even threshold:
     $$\tau^*(x_i) = \frac{C_v}{C_v + C_h \cdot \big(p_{\text{hosp, unvacc}}(x_i) - p_{\text{hosp, vacc}}(x_i)\big)}$$
   * Compared 4 thresholding strategies using leak-free 5-fold CV: fixed `0.50`, nested-CV tuned global threshold, analytical Bayes-exact (`tau*(x_i)`), and nested-CV scaled Bayes.

4. **Multi-Model Training & Nested Cross-Fitted Stacking (`src/modeling.py`)**
   * Trained 6 distinct model families inside a leakage-free nested 5-fold outer / 4-fold inner Stratified CV (`RANDOM_SEED = 42`) with internal `85%/15%` early stopping inside active training folds only.
   * Built strictly uncontaminated inner OOF logit predictions for `Uncalibrated_Stack` and `Calibrated_Stack` (selecting `C = 0.05` and verifying calibration out-of-sample across `none`, `platt`, `isotonic` — selecting `none` across all 5 folds).
   * Ranked models by Bayes-exact OOF town cost (`Calibrated_Stack` / `Uncalibrated_Stack` achieved #1 ROC-AUC `0.8670`, lowest LogLoss `0.4173`, lowest Brier `0.1313`, lowest ECE `0.0088`, and lowest town cost `\$14,281,676`, capturing `68.7%` of theoretical oracle savings).

5. **Explainable AI & Embellished Visuals (`src/explainability.py`)**
   * Computed out-of-fold SHAP values across all 5 outer LightGBM fold models (`outputs/shap_feature_importance.csv` with fold SD stability).
   * Generated 10 figures (`outputs/figures/01_*.png` through `09_*.png` + `08b`) separating numeric beeswarm (`08`) from true categorical subgroup SHAP (`08b`) and labeling meta-learner log-odds coefficients (`09`).

6. **Submission Export & Automated Testing (`src/modeling.py`, `tests/test_pipeline.py`)**
   * Exported the top 5 diverse model families (`Calibrated_Stack`, `CatBoost`, `RandomForest`, `LightGBM`, `LogisticRegression`) into `submissions/challenge_submission_group_<X>_order_<1..5>.csv` verified against `dataset_C_example_submission.csv`.
   * Built 18 `pytest` tests covering Categories A–L: data cleaning & duplicate/dtype checks, zero-opinion handling & feature bounds, version-safe LightGBM/XGBoost/CatBoost isolation, respondent-level nested stacking & calibration isolation, `cloglog` / Zhang-Yu effect-measure consistency & risk constraints, exact cost math & edge cases (`VE <= 0`, `C_v = 0`, `NaN`, CV fallback), additive + drop feature ablation, multi-fold SHAP stability, and distinct submission schema alignment.

---

## Part 3: File-by-File Reference (Which File Does What)

### Input Data & Reference Materials (Original Files)
* `DOCs/dataset_C_training.csv` — Labeled COVID-19 survey training dataset (4,756 rows, 31 columns including `covid_vaccine`).
* `DOCs/dataset_C_testing.csv` — Unlabeled COVID-19 test dataset for submission predictions (4,749 rows, 30 columns).
* `DOCs/dataset_C_example_submission.csv` — Submission template showing exact column names (`respondent_id,covid_vaccine`) and ID order.
* `DOCs/Dataset B Information.pdf` — Feature/cost reference sheet for Dataset B (H1N1 flu: `$500` vaccine, `$25,000` hospitalisation).
* `Images/RDMS.jpeg` — Hackathon task timeline, Canvas submission naming/format rules (`challenge_submission_group_X_order_Y`), and teamwork/Colab/GitHub advice.
* `Images/RDMS BEhaviour rules.jpeg` — CDIO room conduct and logistics regulations.
* `Images/Target matrix info.jpeg` — Town cost minimization objective, local-law payout rule, spoilage rule, and Generative AI hospitalisation risk instructions.
* `Images/Dataset C.jpeg` & `Images/Dataset C2.jpeg` — Feature definitions (items 1–29), target (`covid_vaccine`), and COVID costs (`$1,500` / `$41,000`).

### Source Code (`src/` and Root)
* `main.py` — Master CLI orchestrator (`--group X`) that runs Steps 1–5 end-to-end (data prep -> EDA -> model training -> SHAP/plots -> top 5 submission export -> `outputs/hackathon_report.md`).
* `src/__init__.py` — Package marker for `src`.
* `src/config.py` — Central configuration (`DatasetConfig`, `DATASET_C_CONFIG`, `DATASET_B_CONFIG`, feature groupings, ordinal maps, `RANDOM_SEED=42`, `N_FOLDS=5`, directory paths).
* `src/data_prep.py` — `load_raw_datasets()`, `engineer_features()` (whitespace cleanup, `0->NaN` flags, missingness flags, domain interaction features), and `get_feature_columns()`.
* `src/cost_model.py` — `estimate_hospitalisation_probabilities()` (3 scenarios: `conservative`, `base`, `severe`), `compute_town_cost()` (`CostBreakdown`), `bayes_optimal_thresholds()`, `predict_with_cost_policy()`, and `tune_threshold_on_oof()` (with nested-CV threshold verification).
* `src/eda.py` — `run_eda()` computing descriptive/missingness/correlation/subgroup statistics and generating Figures `01`–`03`.
* `src/modeling.py` — `train_and_evaluate_all_models()` (5-fold CV for all 6 base models + logit `Calibrated_Stack`), OOF/test CSV exports, and `export_top5_submissions()` (ranks top 5 models and validates submission schema).
* `src/explainability.py` — `run_explainability_and_plots()` generating SHAP feature importances (`outputs/shap_feature_importance.csv`), sensitivity matrix (`outputs/sensitivity_matrix.csv`), and Figures `04`–`09`.

### Notebook, Tests & Documentation
* `notebooks/RMDS_Hackathon_Dataset_C.ipynb` — Google Colab & Jupyter-ready 7-cell interactive walkthrough invoking the `src/` modules.
* `tests/test_pipeline.py` — 11 `pytest` automated tests validating preprocessing, determinism, zero target/outer-validation leakage, calibration/ECE, literature-grounded risk constraints, exact cost math & edge cases, multi-fold SHAP, and generated submission schemas.
* `requirements.txt` — Pinned minimum Python package dependencies (`numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `matplotlib`, `seaborn`, `shap`, `nbformat`, `pytest`).
* `.gitignore` — Excludes `.venv/`, `__pycache__/`, `.pytest_cache/`, checkpoints, and OS/IDE artifacts from Git.
* `README.md` — Concise quick-start and structure overview.
* `PRESENTATION.md` — 6-slide presentation deck outline embedding Figures `01`–`09` and verified metrics.
* `PROJECT_GUIDE.md` — This full compliance audit, task summary, and file-by-file reference.

### Generated Outputs (`outputs/` & `submissions/`)
* `outputs/eda_summary.json`, `outputs/eda_missingness.csv`, `outputs/eda_correlations.csv` — Quantitative EDA tables and subgroup vaccine acceptance rates.
* `outputs/model_comparison.csv` — Full benchmark table of all 7 models across AUC, LogLoss, Brier, Accuracy, F1, cost policies, and risk scenarios.
* `outputs/sensitivity_matrix.csv` — Multi-scenario cost & oracle-savings matrix (`conservative` / `base` / `severe`).
* `outputs/oof_predictions.csv` — Out-of-fold predicted probabilities (`n=4,756`) for every model alongside `covid_vaccine`.
* `outputs/test_probabilities.csv` — Test-set predicted acceptance probabilities (`n=4,749`) for every model.
* `outputs/shap_feature_importance.csv` — Ranked features by LightGBM `mean |SHAP|`.
* `outputs/meta_summary.json` — Average meta-learner log-odds blend weights for `Calibrated_Stack`.
* `outputs/hackathon_report.md` — Auto-generated markdown summary of run results.
* `outputs/figures/01_target_and_missingness.png` — Target class balance and train/test missingness comparison.
* `outputs/figures/02_key_drivers_acceptance.png` — Acceptance rates by doctor recommendation, effectiveness opinion, risk opinion, and age group.
* `outputs/figures/03_hospitalisation_risk_profile.png` — Unvaccinated vs. vaccinated hospitalisation rates and age x chronic condition savings heatmap.
* `outputs/figures/04_model_cost_comparison.png` — OOF town cost (`0.50` vs. Bayes policy + Treat None/All/Oracle lines) and AUC vs. savings scatter.
* `outputs/figures/05_calibration_and_thresholds.png` — Probability calibration curves and global threshold sweep vs. per-person Bayes policy.
* `outputs/figures/06_explainability_shap_importance.png` — Top 18 features by LightGBM `mean |SHAP|`.
* `outputs/figures/07_cost_savings_by_segment.png` — Per-person town cost and vaccine allocation rate across age groups.
* `outputs/figures/08_explainability_shap_beeswarm.png` — Directional SHAP impact plot showing whether higher/lower values push acceptance up or down.
* `outputs/figures/09_meta_weights_and_sensitivity.png` — Calibrated stack weights and cross-scenario sensitivity heatmap.
* `submissions/challenge_submission_group_<X>_order_<1..5>.csv` — Top 5 ranked prediction files ready for Canvas upload.
* `submissions/submission_manifest.csv` — Summary metadata for the 5 exported submission files (model, policy, OOF cost, test positive counts).

