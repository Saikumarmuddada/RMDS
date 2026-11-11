# RMDS Hackathon — Dataset C (COVID-19) Implementation Report

## 1. Dataset Overview
- **Training rows:** 4,756 (`covid_vaccine = 1`: 1,554 / 32.67%)
- **Testing rows:** 4,749
- **Features:** 29 raw features -> 49 cleaned + validated features

## 2. Town Cost Baselines (Training OOF Population, n=4,756)
- **Treat None (0% sent):** $17,600,550 ($3,700.70/person)
- **Treat All (100% sent):** $17,581,366 ($3,696.67/person)
- **Oracle Lower Bound (perfect foresight):** $12,778,366 ($2,686.79/person)

## 3. Nested 5-Fold Stratified CV Model & Calibration Comparison

| Model | ROC-AUC | LogLoss | Brier | ECE | Cost @ 0.50 ($) | Cost @ Bayes-Exact ($) | Allocated | Wasted | Law Comp | Exp. Hosp |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uncalibrated_Stack | 0.8670 | 0.4173 | 0.1313 | 0.0088 | $14,740,402 | $14,281,676 | 1,487 | 491 | 558 | 273.5 |
| Calibrated_Stack | 0.8670 | 0.4173 | 0.1313 | 0.0088 | $14,740,402 | $14,281,676 | 1,487 | 491 | 558 | 273.5 |
| RandomForest | 0.8586 | 0.4351 | 0.1378 | 0.0342 | $15,031,856 | $14,308,264 | 1,448 | 524 | 630 | 273.0 |
| LogisticRegression | 0.8654 | 0.4218 | 0.1321 | 0.0147 | $14,759,858 | $14,336,991 | 1,536 | 538 | 556 | 273.1 |
| LightGBM | 0.8622 | 0.4238 | 0.1341 | 0.0128 | $14,845,302 | $14,343,198 | 1,480 | 497 | 571 | 274.8 |
| XGBoost | 0.8618 | 0.4249 | 0.1342 | 0.0137 | $14,816,648 | $14,347,335 | 1,456 | 488 | 586 | 275.2 |
| CatBoost | 0.8640 | 0.4204 | 0.1326 | 0.0136 | $14,831,290 | $14,364,624 | 1,444 | 482 | 592 | 275.9 |
| HistGradientBoosting | 0.8526 | 0.4485 | 0.1400 | 0.0408 | $14,947,879 | $14,390,153 | 1,471 | 474 | 557 | 276.8 |

## 4. Cost Policy & Component Breakdown (Uncalibrated_Stack, 5-Fold OOF)

| Policy | Total Cost ($) | Vaccine Send ($) | Wasted ($) | Law Comp ($) | Exp. Hosp ($) | Allocated | Accepted | Wasted | Unmet | Exp. Hosp | Savings vs 0.50 ($) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed Threshold (0.50) | $14,740,402 | $1,944,000 | $450,000 | $837,000 | $11,959,402 | 1,296 | 996 | 300 | 558 | 291.7 | $0 |
| CV-Tuned Global Threshold (Out-of-Fold) | $14,713,139 | $2,721,000 | $933,000 | $543,000 | $11,449,139 | 1,814 | 1,192 | 622 | 362 | 279.2 | $27,264 |
| Person-Specific Bayes Threshold (Analytical) | $14,281,676 | $2,230,500 | $736,500 | $837,000 | $11,214,176 | 1,487 | 996 | 491 | 558 | 273.5 | $458,727 |
| CV-Scaled Bayes Threshold (Out-of-Fold) | $14,283,408 | $2,158,500 | $708,000 | $880,500 | $11,244,408 | 1,439 | 967 | 472 | 587 | 274.3 | $456,995 |

## 5. Meta-Learner Logistic Regression Coefficients (Log-Odds)
- `LogisticRegression`: 0.3685
- `CatBoost`: 0.2150
- `XGBoost`: 0.1793
- `LightGBM`: 0.1325
- `RandomForest`: 0.0944
- `HistGradientBoosting`: 0.0512

## 6. Top 10 Explainable AI Drivers (5-Fold OOF LightGBM Mean |SHAP| ± SD)
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

## 7. Top 5 Ranked Submission Files

| Order | Filename | Model | Policy | OOF Total Cost ($) | Test Predicted Vaccinations |
| ---: | :--- | :--- | :--- | ---: | ---: |
| 1 | `challenge_submission_group_0_order_1.csv` | Calibrated_Stack | bayes_exact | $14,281,676 | 1,555 (32.7%) |
| 2 | `challenge_submission_group_0_order_2.csv` | CatBoost | bayes_scaled | $14,300,969 | 1,663 (35.0%) |
| 3 | `challenge_submission_group_0_order_3.csv` | RandomForest | bayes_exact | $14,308,264 | 1,505 (31.7%) |
| 4 | `challenge_submission_group_0_order_4.csv` | LightGBM | bayes_scaled | $14,321,612 | 1,315 (27.7%) |
| 5 | `challenge_submission_group_0_order_5.csv` | LogisticRegression | bayes_exact | $14,336,991 | 1,589 (33.5%) |
