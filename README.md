# RMDS Data Science Challenge Hackathon — Dataset C (COVID-19)

End-to-end solution for the RMDS Hackathon (**Dataset C: COVID-19 Vaccine Acceptance & Town Cost Minimization**), covering EDA, missingness-aware preprocessing, feature engineering, hospitalisation probability estimation, 5-fold Stratified CV multi-model training, cost-optimal Bayes decision thresholding, SHAP explainability, and Canvas-ready submission generation.

## Project Structure

```text
RMDS/
├── DOCs/                                # Original datasets & Dataset B PDF
├── Images/                              # Hackathon rules, cost metric, Dataset C sheets
├── src/
│   ├── config.py                        # Paths, feature lists, Dataset C costs ($1,500 / $41,000)
│   ├── data_prep.py                     # Cleaning (whitespace, 0->NaN flags) & feature engineering
│   ├── cost_model.py                    # Hospitalisation IHR model, cost matrix, Bayes threshold
│   ├── eda.py                           # EDA stats & visualizations (Figures 01–03)
│   ├── modeling.py                      # Nested 5x4 leakage-free CV (6 models + cross-fitted stack)
│   └── explainability.py                # Multi-fold OOF SHAP, calibration/ECE & sensitivity plots
├── notebooks/
│   └── RMDS_Hackathon_Dataset_C.ipynb   # Google Colab & Jupyter ready walkthrough notebook
├── tests/
│   └── test_pipeline.py                 # 11 pytest checks (preprocessing, leakage, cost, SHAP, schema)
├── outputs/
│   ├── figures/                         # 10 publication-grade PNG charts (01–09 + 08b)
│   ├── model_comparison.csv             # Full nested CV metrics across all 8 models & cost breakdowns
│   ├── sensitivity_matrix.csv           # Cross-scenario cost & oracle savings matrix
│   ├── ve_and_risk_sensitivity_grid.csv # IHR scenario x VE regime x Sex sensitivity matrix
│   ├── stacking_calibration_audit.csv   # Inner OOF LogLoss calibration selection audit (none/platt/isotonic)
│   ├── preprocessing_validation.json    # Automated dataset integrity & zero-sentinel audit
│   ├── feature_ablation_report.csv      # Leakage-free CV feature-group ablation benchmark
│   ├── shap_feature_importance.csv      # 5-fold OOF SHAP rankings with fold SD stability
│   └── hackathon_report.md              # Generated summary report
├── TECHNICAL_AUDIT_REPORT.md            # Complete technical audit, literature grounding & Before/After comparison
├── PROJECT_GUIDE.md                     # Compliance audit + task-by-task + file-by-file guide
├── PRESENTATION.md                      # 6-slide presentation deck outline with figures & metrics
├── submissions/
│   ├── challenge_submission_group_0_order_1.csv  # #1 Uncalibrated_Stack (Bayes-Exact, = Calibrated_Stack)
│   ├── challenge_submission_group_0_order_2.csv  # #2 CatBoost (Bayes-Scaled 0.88)
│   ├── challenge_submission_group_0_order_3.csv  # #3 RandomForest (Bayes-Exact)
│   ├── challenge_submission_group_0_order_4.csv  # #4 LightGBM (Bayes-Scaled 1.17)
│   ├── challenge_submission_group_0_order_5.csv  # #5 XGBoost (Bayes-Exact)
│   └── submission_manifest.csv
└── main.py                              # Single CLI entrypoint
```

## Quick Start

### 1. Run Locally
```powershell
# Using the configured virtual environment
.venv\Scripts\python.exe main.py --group 0

# Replace 0 with your assigned hackathon group number X:
.venv\Scripts\python.exe main.py --group 3

# Run unit tests
.venv\Scripts\python.exe -m pytest -v
```

### 2. Run in Google Colab
Open `notebooks/RMDS_Hackathon_Dataset_C.ipynb` in Google Colab, upload or clone this repository, and run all cells sequentially.

## Cost Optimization Formula (`Target Metric Information`)

Under the town rules:
- **Vaccine cost:** `$1,500` (paid if sent $\hat{y}_i = 1$, or paid as local-law compensation if $\hat{y}_i = 0$ and $y_i = 1$)
- **Hospitalisation cost:** `$41,000` (100% infection assumption)
- **Spoilage:** sent vaccines not taken ($\hat{y}_i = 1, y_i = 0$) are ruined.

Sending a vaccine reduces expected town cost whenever the calibrated acceptance probability $\hat{p}_i = P(y_i = 1 \mid x_i)$ exceeds the individual's break-even threshold:

$$\tau^*(x_i) = \frac{C_v}{C_v + C_h \cdot \big(p_{\text{hosp, unvacc}}(x_i) - p_{\text{hosp, vacc}}(x_i)\big)}$$

