# RMDS Data Science Challenge Hackathon — Presentation Deck (Dataset C)

Slide-by-slide presentation outline ready for your team's pitch deck, backed directly by verified CV metrics and figures in `outputs/figures/`.

---

## Slide 1: Problem Statement & Town Cost Objective
* **Challenge:** Predict individual COVID-19 vaccine acceptance (`covid_vaccine` in `{0, 1}`) on `4,749` test respondents (`4,756` labeled training rows).
* **Real Goal (`Target Metric Information`):** Minimize **Total Town Cost** under 100% winter-party infection:
  * **Vaccine Cost:** `\$1,500` per shipped dose (spoils if unused) OR paid to willing citizens who don't receive one (local law).
  * **Hospitalisation Cost:** `\$41,000` per hospitalised resident.
* **Core Insight:** Because missed high-risk acceptors cost up to `\$14,600` in expected hospitalisations while a wasted dose costs `\$1,500`, **a fixed `0.50` classification cutoff leaves money on the table**.

---

## Slide 2: Dataset Quirks & Exploratory Insights
* **Target Balance:** `32.67%` accepted (`1,554` / `4,756`), `67.33%` declined (`3,202`).
* **3 Hidden Data Signals We Discovered:**
  1. **Informative Missingness:** Respondents missing `health_insurance` (~40–42% of rows) accept at only **17.3%** (`n=1,908`), vs. **45.8%** for insured respondents (`n=2,544`).
  2. **Doctor Recommendation Power:** `doctor_recc_covid = 1` triples acceptance from **21.7%** to **69.9%**.
  3. **Data Cleaning Fixes:** Fixed double spacing in `"MSA, Not Principle  City"` and flagged undocumented `0`s (`92` train / `107` test in effectiveness; `96` train / `104` test in risk) as `*_is_zero` + `NaN`.

![Target Distribution and Missingness](outputs/figures/01_target_and_missingness.png)
![Key Drivers of Vaccine Acceptance](outputs/figures/02_key_drivers_acceptance.png)

---

## Slide 3: Hospitalisation Risk Matrix & Bayes-Optimal Decision Rule
* **Hospitalisation Risk Factors:** Estimated $p_{\text{hosp, unvacc}}(x_i)$ and $p_{\text{hosp, vacc}}(x_i)$ from `age_group`, `number_chronic_med_conditions`, `sex`, `health_insurance`, and `income_poverty` (`85%` vaccine protection, `80%` for `65+` / `>=2` chronic conditions).
* **Exact Break-Even Threshold:** Send vaccine ($\hat{y}_i = 1$) whenever predicted acceptance probability satisfies:
  $$\hat{p}_i \ge \tau^*(x_i) = \frac{C_v}{C_v + C_h \cdot \big(p_{\text{hosp, unvacc}}(x_i) - p_{\text{hosp, vacc}}(x_i)\big)}$$
* **Why It Works (Real Test Examples):**
  * **High-Risk Senior (`ID=4759`, 65+ Male):** `p_unvacc=18.6%`, break-even $\tau^* = 19.7\%$, model $\hat{p} = 29.3\%$ → **Send (`1`)** (naive `0.50` misses `\$6,101` expected hospitalisation savings).
  * **Low-Risk Young Adult (`ID=4763`, 18–34 Male):** `p_unvacc=1.2%`, break-even $\tau^* = 78.2\%$, model $\hat{p} = 56.1\%$ → **Don't Send (`0`)** (naive `0.50` risks `\$1,500` spoilage for only `\$418` savings).

![Hospitalisation Risk Profile](outputs/figures/03_hospitalisation_risk_profile.png)

---

## Slide 4: Multi-Model Benchmark (5-Fold Stratified CV)
* Trained **6 distinct model families** (`LogisticRegression`, `RandomForest`, `HistGradientBoosting`, `LightGBM`, `XGBoost`, `CatBoost`) + out-of-fold logit **`Calibrated_Stack`** over `54` cleaned/engineered features.
* **`Calibrated_Stack` achieved #1 across every metric:**
  * **ROC-AUC:** `0.8670` | **LogLoss:** `0.4173` | **Brier Score:** `0.1313`
  * **Town Cost (OOF):** **`\$15,748,199`** (`\$3,311.23/person`)
  * **Savings:** **`\$3,578,150`** vs. Treat None (`\$19.33M`), **`\$3,156,806`** vs. Treat All (`\$18.91M`), and **`\$484,349`** extra saved over a naive `0.50` threshold (`\$16.23M`)—capturing **65.7%** of theoretical oracle savings.

![Model Cost Comparison](outputs/figures/04_model_cost_comparison.png)
![Calibration and Threshold Sweep](outputs/figures/05_calibration_and_thresholds.png)

---

## Slide 5: Explainable AI (SHAP Drivers & Segment Impact)
* **Top Global Drivers (`mean |SHAP|`):** `health_insurance` (`0.516`), `opinion_eff_x_risk` (`0.407`), `doctor_x_effective` (`0.383`), `doctor_x_risk` (`0.263`), `employment_sector` (`0.160`), `health_worker` (`0.130`), `age_ordinal` (`0.122`).
* **Allocation Shift Across Ages:** Bayes policy targets **60.8%** of `65+ Years` vs. only **8.6%** of `18–34 Years` on OOF validation.

![SHAP Importance](outputs/figures/06_explainability_shap_importance.png)
![SHAP Beeswarm](outputs/figures/08_explainability_shap_beeswarm.png)
![Cost Savings by Segment](outputs/figures/07_cost_savings_by_segment.png)

---

## Slide 6: Robustness Check & Top 5 Canvas Submissions
* **Cross-Scenario Sensitivity:** Tested `conservative` (`0.5x` risk odds), `base` (`1.0x`), and `severe` (`1.5x`) hospitalisation profiles—`Calibrated_Stack` ranked **#1 in all 3 scenarios** (`59.2%`, `65.7%`, `59.0%` oracle savings captured).
* **Top 5 Exported Submissions (`submissions/`):**
  1. `Calibrated_Stack` (`1,679` test vaccines / `35.4%`)
  2. `XGBoost` (`1,662` / `35.0%`)
  3. `LightGBM` (`1,663` / `35.0%`)
  4. `CatBoost` (`1,660` / `35.0%`)
  5. `RandomForest` (`1,698` / `35.8%`)

![Meta Weights and Sensitivity Matrix](outputs/figures/09_meta_weights_and_sensitivity.png)

