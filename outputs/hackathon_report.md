# RMDS Hackathon — Dataset C (COVID-19) Implementation Report

## 1. Dataset Overview
- **Training rows:** 4,756 (`covid_vaccine = 1`: 1,554 / 32.67%)
- **Testing rows:** 4,749
- **Features:** 29 raw features -> 49 cleaned + validated features

## 2. Town Cost Baselines (Training OOF Population, n=4,756)
- **Treat None (0% sent):** $17,152,332 ($3,606.46/person)
- **Treat All (100% sent):** $17,290,453 ($3,635.50/person)
- **Oracle Lower Bound (perfect foresight):** $12,487,453 ($2,625.62/person)

## 3. Nested 5-Fold Stratified CV Model & Calibration Comparison

| Model | ROC-AUC | LogLoss | Brier | ECE | Cost @ 0.50 ($) | Cost @ Bayes-Exact ($) | Allocated | Wasted | Law Comp | Exp. Hosp |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated_Stack | 0.8670 | 0.4173 | 0.1313 | 0.0088 | $14,415,518 | $13,970,808 | 1,476 | 481 | 559 | 266.3 |
| Calibrated_Stack | 0.8670 | 0.4173 | 0.1313 | 0.0088 | $14,415,518 | $13,970,808 | 1,476 | 481 | 559 | 266.3 |
| RandomForest | 0.8586 | 0.4351 | 0.1378 | 0.0342 | $14,688,404 | $14,004,588 | 1,432 | 510 | 632 | 266.1 |
| XGBoost | 0.8618 | 0.4249 | 0.1342 | 0.0137 | $14,487,463 | $14,030,042 | 1,442 | 475 | 587 | 268.0 |
| LogisticRegression | 0.8654 | 0.4218 | 0.1321 | 0.0147 | $14,431,588 | $14,033,902 | 1,531 | 534 | 557 | 265.9 |
| LightGBM | 0.8622 | 0.4238 | 0.1341 | 0.0128 | $14,515,615 | $14,037,696 | 1,471 | 489 | 572 | 267.6 |
| CatBoost | 0.8640 | 0.4204 | 0.1326 | 0.0136 | $14,502,418 | $14,072,497 | 1,437 | 477 | 594 | 268.9 |
| HistGradientBoosting | 0.8526 | 0.4485 | 0.1400 | 0.0408 | $14,610,676 | $14,073,918 | 1,464 | 467 | 557 | 269.3 |

## 4. Meta-Learner Logistic Regression Coefficients (Log-Odds)
- `LogisticRegression`: 0.3685
- `CatBoost`: 0.2150
- `XGBoost`: 0.1793
- `LightGBM`: 0.1325
- `RandomForest`: 0.0944
- `HistGradientBoosting`: 0.0512

## 5. Top 10 Explainable AI Drivers (5-Fold OOF LightGBM Mean |SHAP| ± SD)
- `health_insurance` (numeric): 0.5331 ± 0.0595
- `opinion_covid_risk` (numeric): 0.4552 ± 0.0685
- `doctor_x_effective` (numeric): 0.3811 ± 0.0398
- `opinion_covid_vacc_effective` (numeric): 0.2923 ± 0.0128
- `doctor_x_risk` (numeric): 0.2565 ± 0.0747
- `employment_sector` (categorical): 0.1644 ± 0.0345
- `age_ordinal` (numeric): 0.1326 ± 0.0223
- `health_worker` (numeric): 0.1308 ± 0.0097
- `covid_knowledge` (numeric): 0.1168 ± 0.0280
- `health_insurance_is_missing` (numeric): 0.1037 ± 0.0132

## 6. Top 5 Ranked Submission Files

| Order | Filename | Model | Policy | OOF Total Cost ($) | Test Predicted Vaccinations |
| ---: | :--- | :--- | :--- | ---: | ---: |
| 1 | `challenge_submission_group_0_order_1.csv` | Calibrated_Stack | bayes_exact | $13,970,808 | 1,546 (32.6%) |
| 2 | `challenge_submission_group_0_order_2.csv` | CatBoost | bayes_scaled | $13,992,147 | 1,682 (35.4%) |
| 3 | `challenge_submission_group_0_order_3.csv` | RandomForest | bayes_exact | $14,004,588 | 1,495 (31.5%) |
| 4 | `challenge_submission_group_0_order_4.csv` | LightGBM | bayes_scaled | $14,021,900 | 1,296 (27.3%) |
| 5 | `challenge_submission_group_0_order_5.csv` | XGBoost | bayes_exact | $14,030,042 | 1,503 (31.6%) |
