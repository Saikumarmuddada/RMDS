# RMDS Hackathon — Dataset C (COVID-19) Implementation Report

## 1. Dataset Overview
- **Training rows:** 4,756 (`covid_vaccine = 1`: 1,554 / 32.67%)
- **Testing rows:** 4,749
- **Features:** 29 raw features -> 54 cleaned + engineered features

## 2. Town Cost Baselines (Training OOF Population, n=4,756)
- **Treat None (0% sent):** $19,326,349 ($4,063.57/person)
- **Treat All (100% sent):** $18,905,005 ($3,974.98/person)
- **Oracle Lower Bound (perfect foresight):** $14,102,005 ($2,965.10/person)

## 3. 5-Fold Stratified CV Model Comparison

| Model | ROC-AUC | LogLoss | Cost @ 0.50 ($) | Cost @ Bayes-Exact ($) | Per-Person ($) | Oracle Savings Captured |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated_Stack | 0.8670 | 0.4173 | $16,232,548 | $15,748,199 | $3,311.23 | 65.7% |
| XGBoost | 0.8654 | 0.4199 | $16,306,148 | $15,766,822 | $3,315.14 | 65.3% |
| LightGBM | 0.8635 | 0.4219 | $16,310,480 | $15,784,433 | $3,318.85 | 65.0% |
| CatBoost | 0.8656 | 0.4188 | $16,271,939 | $15,789,697 | $3,319.95 | 64.9% |
| RandomForest | 0.8543 | 0.4357 | $16,469,697 | $15,810,110 | $3,324.25 | 64.4% |
| LogisticRegression | 0.8652 | 0.4227 | $16,264,090 | $15,836,969 | $3,329.89 | 63.9% |
| HistGradientBoosting | 0.8542 | 0.4440 | $16,408,570 | $15,844,339 | $3,331.44 | 63.7% |

## 4. Calibrated Stacking Meta-Weights (Log-Odds Coefficients)
- `CatBoost`: 0.3556
- `LogisticRegression`: 0.3099
- `XGBoost`: 0.1955
- `LightGBM`: 0.1789
- `HistGradientBoosting`: -0.0061
- `RandomForest`: -0.0095

## 5. Top 10 Explainable AI Drivers (LightGBM Mean |SHAP|)
- `health_insurance`: 0.5161
- `opinion_eff_x_risk`: 0.4070
- `doctor_x_effective`: 0.3832
- `doctor_x_risk`: 0.2625
- `employment_sector`: 0.1597
- `health_worker`: 0.1295
- `age_ordinal`: 0.1219
- `opinion_covid_vacc_effective`: 0.1061
- `health_insurance_is_missing`: 0.1045
- `opinion_pro_vaccine_index`: 0.0952

## 6. Top 5 Ranked Submission Files

| Order | Filename | Model | Policy | OOF Total Cost ($) | Test Predicted Vaccinations |
| ---: | :--- | :--- | :--- | ---: | ---: |
| 1 | `challenge_submission_group_0_order_1.csv` | Calibrated_Stack | bayes_exact | $15,748,199 | 1,679 (35.4%) |
| 2 | `challenge_submission_group_0_order_2.csv` | XGBoost | bayes_exact | $15,766,822 | 1,662 (35.0%) |
| 3 | `challenge_submission_group_0_order_3.csv` | LightGBM | bayes_exact | $15,784,433 | 1,663 (35.0%) |
| 4 | `challenge_submission_group_0_order_4.csv` | CatBoost | bayes_exact | $15,789,697 | 1,660 (35.0%) |
| 5 | `challenge_submission_group_0_order_5.csv` | RandomForest | bayes_exact | $15,810,110 | 1,698 (35.8%) |
