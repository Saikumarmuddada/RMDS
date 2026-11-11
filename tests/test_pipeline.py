"""Comprehensive automated verification suite across preprocessing, leakage isolation, cost optimality, SHAP stability, and submission schema."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DATASET_C_CONFIG, SUBMISSIONS_DIR, DatasetConfig
from src.cost_model import (
    bayes_optimal_thresholds,
    compute_ece,
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    expected_decision_cost,
    predict_with_cost_policy,
)
from src.data_prep import (
    audit_dataset_preprocessing,
    engineer_features,
    get_feature_columns,
    load_raw_datasets,
)
from src.explainability import _compute_and_plot_shap
from src.modeling import (
    fit_outer_fold_models,
    train_and_evaluate_all_models,
)


# ---------------------------------------------------------------------------
# Category A & B: Preprocessing and Feature Engineering Integrity
# ---------------------------------------------------------------------------


def test_data_cleaning_fixes_whitespace_and_zero_opinions():
    raw_train, raw_test, _ = load_raw_datasets(DATASET_C_CONFIG)
    feat_train = engineer_features(raw_train, DATASET_C_CONFIG)
    feat_test = engineer_features(raw_test, DATASET_C_CONFIG)

    # Whitespace collapse on census_msa
    assert not any("  " in str(v) for v in feat_train["census_msa"].dropna().unique())

    # Undocumented 0 values mapped to NaN with explicit indicators
    assert (feat_train["opinion_covid_vacc_effective"] == 0).sum() == 0
    assert (feat_train["opinion_covid_risk"] == 0).sum() == 0
    assert feat_train["opinion_covid_vacc_effective_is_zero"].sum() == 92
    assert feat_train["opinion_covid_risk_is_zero"].sum() == 96

    # Feature columns strictly match between train and test, excluding ID and target
    tr_num, tr_cat = get_feature_columns(feat_train, DATASET_C_CONFIG)
    te_num, te_cat = get_feature_columns(feat_test, DATASET_C_CONFIG)
    assert tr_num == te_num
    assert tr_cat == te_cat
    assert DATASET_C_CONFIG.id_col not in tr_num + tr_cat
    assert DATASET_C_CONFIG.target_col not in tr_num + tr_cat

    # Collinear opinion composites were pruned after CV ablation
    for pruned_col in (
        "opinion_net_benefit",
        "opinion_risk_minus_worry",
        "opinion_pro_vaccine_index",
        "opinion_eff_x_risk",
        "concern_x_knowledge",
    ):
        assert pruned_col not in feat_train.columns


def test_preprocessing_determinism_no_target_leakage_and_unseen_safety():
    raw_train, raw_test, _ = load_raw_datasets(DATASET_C_CONFIG)
    feat_tr_1 = engineer_features(raw_train, DATASET_C_CONFIG)
    feat_tr_2 = engineer_features(raw_train, DATASET_C_CONFIG)
    feat_test = engineer_features(raw_test, DATASET_C_CONFIG)

    audit = audit_dataset_preprocessing(
        raw_train, raw_test, feat_tr_1, feat_test, DATASET_C_CONFIG
    )
    assert audit["id_overlap_count"] == 0
    assert not audit["id_in_features"]
    assert not audit["target_in_features"]
    assert audit["train_exact_duplicates"] == 0
    assert audit["train_test_feature_column_match"]

    # Deterministic repeatability
    num_cols, cat_cols = get_feature_columns(feat_tr_1, DATASET_C_CONFIG)
    pd.testing.assert_frame_equal(feat_tr_1[num_cols + cat_cols], feat_tr_2[num_cols + cat_cols])

    # Changing target labels must NEVER alter any engineered feature column
    corrupted = raw_train.copy()
    corrupted[DATASET_C_CONFIG.target_col] = 1 - corrupted[DATASET_C_CONFIG.target_col]
    feat_corrupted = engineer_features(corrupted, DATASET_C_CONFIG)
    pd.testing.assert_frame_equal(
        feat_tr_1[num_cols + cat_cols], feat_corrupted[num_cols + cat_cols]
    )

    # Unseen categories in synthetic test rows do not crash feature engineering
    unseen_row = raw_test.iloc[:2].copy()
    unseen_row.loc[unseen_row.index[0], "age_group"] = "99+ Centuries"
    unseen_row.loc[unseen_row.index[1], "employment_sector"] = "unseen_brand_new_sector"
    fe_unseen = engineer_features(unseen_row, DATASET_C_CONFIG)
    assert len(fe_unseen) == 2
    assert pd.isna(fe_unseen.loc[fe_unseen.index[0], "age_ordinal"])


# ---------------------------------------------------------------------------
# Category C & D: Validation Leakage, Stacking Isolation & Calibration Tests
# ---------------------------------------------------------------------------


def test_outer_validation_isolation_and_no_leakage():
    """Verify structurally and empirically that outer validation rows/labels never leak."""
    # Structural verification: fit_outer_fold_models cannot receive y_va
    sig = inspect.signature(fit_outer_fold_models)
    assert "y_va" not in sig.parameters

    raw_train, raw_test, _ = load_raw_datasets(DATASET_C_CONFIG)
    # Stratified 220-row subset for rapid multi-model leakage test
    pos = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 1].iloc[:75]
    neg = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 0].iloc[:145]
    sub_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)
    feat_sub = engineer_features(sub_df, DATASET_C_CONFIG)
    feat_te_sub = engineer_features(raw_test.iloc[:40].copy(), DATASET_C_CONFIG)
    num_cols, cat_cols = get_feature_columns(feat_sub, DATASET_C_CONFIG)
    feat_cols = num_cols + cat_cols

    X_tr = feat_sub.iloc[:170][feat_cols]
    y_tr = feat_sub.iloc[:170][DATASET_C_CONFIG.target_col].to_numpy(dtype=int)
    X_va_clean = feat_sub.iloc[170:][feat_cols].copy()
    X_te = feat_te_sub[feat_cols].copy()

    res_a = fit_outer_fold_models(
        X_tr, y_tr, X_va_clean, X_te, num_cols, cat_cols, outer_fold_idx=0, inner_folds=2, fast_mode=True
    )

    # Corrupt row 0 of X_va with extreme values + unseen category to prove zero fit leakage
    X_va_poisoned = X_va_clean.copy()
    X_va_poisoned.iloc[0, X_va_poisoned.columns.get_loc(num_cols[0])] = 999999.0
    X_va_poisoned.iloc[0, X_va_poisoned.columns.get_loc(cat_cols[0])] = "POISONED_CATEGORY"
    res_b = fit_outer_fold_models(
        X_tr, y_tr, X_va_poisoned, X_te, num_cols, cat_cols, outer_fold_idx=0, inner_folds=2, fast_mode=True
    )

    # All untouched validation rows [1:] and ALL test predictions must be bit-for-bit identical
    for m_name in res_a["val_preds"]:
        np.testing.assert_allclose(res_a["val_preds"][m_name][1:], res_b["val_preds"][m_name][1:], atol=1e-12)
        np.testing.assert_allclose(res_a["test_preds"][m_name], res_b["test_preds"][m_name], atol=1e-12)

    np.testing.assert_allclose(res_a["raw_stack_val"][1:], res_b["raw_stack_val"][1:], atol=1e-12)
    np.testing.assert_allclose(res_a["cal_stack_val"][1:], res_b["cal_stack_val"][1:], atol=1e-12)
    np.testing.assert_allclose(res_a["cal_stack_test"], res_b["cal_stack_test"], atol=1e-12)
    np.testing.assert_allclose(res_a["meta_coefs"], res_b["meta_coefs"], atol=1e-12)

    # Early stopping split stays strictly inside outer training size (170), never using va (50)
    es_sizes = res_a["outer_base"]["es_split_sizes"]
    assert es_sizes["fit"] + es_sizes["es"] == len(X_tr)
    assert res_a["inner_oof_size"] == len(X_tr)
    assert res_a["selected_calibrator"] in {"none", "platt", "isotonic"}


def test_ece_metric_and_calibration_bounds():
    # Perfectly calibrated predictions
    y_perf = np.array([0, 0, 1, 1] * 25)
    p_perf = np.array([0.0, 0.0, 1.0, 1.0] * 25)
    assert compute_ece(y_perf, p_perf, n_bins=10) == pytest.approx(0.0, abs=1e-12)

    # Constant 0.90 probability on 50% positive outcomes => exact ECE = 0.40
    y_half = np.array([0, 1] * 50)
    p_90 = np.full(100, 0.90)
    assert compute_ece(y_half, p_90, n_bins=10) == pytest.approx(0.40, abs=1e-12)


# ---------------------------------------------------------------------------
# Category E: Scientific Hospitalisation Probability Estimation Tests
# ---------------------------------------------------------------------------


def test_hospitalisation_risk_monotonicity():
    df = pd.DataFrame(
        {
            "age_group": ["18 - 34 Years", "35 - 44 Years", "55 - 64 Years", "65+ Years"],
            "number_chronic_med_conditions": [0, 1, 2, 3],
            "sex": ["Female", "Female", "Male", "Male"],
            "health_insurance": [1.0, 1.0, 0.0, 0.0],
            "income_poverty": ["Above Poverty", "Above Poverty", "Below Poverty", "Below Poverty"],
        }
    )
    pu_base, pv_base = estimate_hospitalisation_probabilities(df, scenario="base")
    pu_cons, _ = estimate_hospitalisation_probabilities(df, scenario="conservative")
    pu_sev, _ = estimate_hospitalisation_probabilities(df, scenario="severe")

    assert np.all(np.diff(pu_base) > 0)
    assert np.all((pu_base >= 0.0) & (pu_base <= 1.0))
    assert np.all((pv_base >= 0.0) & (pv_base < pu_base))
    assert np.all(pu_cons < pu_base) and np.all(pu_base < pu_sev)


def test_hospitalisation_model_literature_constraints_and_missing_inputs():
    # Chronic cap >=3 matches Ko et al. (2021) category without unbounded exponential growth
    df_ch = pd.DataFrame(
        {
            "age_group": ["45 - 54 Years"] * 4,
            "number_chronic_med_conditions": [2, 3, 5, np.nan],
            "sex": ["Male", "Male", "Male", np.nan],
            "health_insurance": [1.0, 0.0, 0.0, np.nan],
            "income_poverty": ["Above Poverty", "Below Poverty", "Below Poverty", np.nan],
        }
    )
    pu, pv = estimate_hospitalisation_probabilities(df_ch, scenario="base")
    assert np.all(np.isfinite(pu)) and np.all(np.isfinite(pv))
    assert pu[1] == pytest.approx(pu[2], abs=1e-12)  # 3 and 5 both map to >=3 Ko et al. bucket
    assert pu[1] > pu[0]

    # Unsupported socioeconomic multipliers (health_insurance, income_poverty) must NOT shift risk
    df_se = pd.DataFrame(
        {
            "age_group": ["45 - 54 Years", "45 - 54 Years"],
            "number_chronic_med_conditions": [1, 1],
            "sex": ["Female", "Female"],
            "health_insurance": [1.0, 0.0],
            "income_poverty": ["Above Poverty", "Below Poverty"],
        }
    )
    pu_se, pv_se = estimate_hospitalisation_probabilities(df_se, scenario="base")
    assert pu_se[0] == pytest.approx(pu_se[1], abs=1e-12)
    assert pv_se[0] == pytest.approx(pv_se[1], abs=1e-12)

    # VE regimes ordering: boosted < base < waned_omicron vaccinated hospitalisation probability
    _, pv_boost = estimate_hospitalisation_probabilities(df_se, ve_regime="boosted")
    _, pv_base = estimate_hospitalisation_probabilities(df_se, ve_regime="base")
    _, pv_wane = estimate_hospitalisation_probabilities(df_se, ve_regime="waned_omicron")
    assert pv_boost[0] < pv_base[0] < pv_wane[0]


# ---------------------------------------------------------------------------
# Category F: Cost Optimisation Derivation & Edge Cases
# ---------------------------------------------------------------------------


def test_town_cost_rules_exact():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    p_u = np.array([0.10, 0.10, 0.20, 0.10])
    p_v = np.array([0.02, 0.02, 0.04, 0.02])

    bd = compute_town_cost(y_true, y_pred, p_u, p_v, DATASET_C_CONFIG)
    expected_hosp = 0.02 + 0.10 + 0.20 + 0.10
    expected_total = 4500.0 + expected_hosp * 41000.0

    assert bd.vaccine_send_cost == pytest.approx(3000.0)
    assert bd.accepted_vaccine_cost == pytest.approx(1500.0)
    assert bd.wasted_vaccine_cost == pytest.approx(1500.0)
    assert bd.law_penalty_cost == pytest.approx(1500.0)
    assert bd.vaccines_allocated == 2
    assert bd.wasted_vaccines == 1
    assert bd.law_compensation_count == 1
    assert bd.expected_hospitalisations == pytest.approx(expected_hosp)
    assert bd.total_cost == pytest.approx(expected_total)
    # Full accounting conservation identity
    assert (
        bd.accepted_vaccine_cost + bd.wasted_vaccine_cost + bd.law_penalty_cost + bd.hospitalisation_cost
    ) == pytest.approx(bd.total_cost)


def test_bayes_threshold_consistency():
    p_u = np.array([0.10, 0.02])
    p_v = np.array([0.02, 0.004])
    tau = bayes_optimal_thresholds(p_u, p_v, DATASET_C_CONFIG)

    # At p = tau*, E[Cost(send=1)] equals E[Cost(send=0)] identically
    c_send = expected_decision_cost(tau, np.ones_like(tau, dtype=int), p_u, p_v, DATASET_C_CONFIG)
    c_skip = expected_decision_cost(tau, np.zeros_like(tau, dtype=int), p_u, p_v, DATASET_C_CONFIG)
    np.testing.assert_allclose(c_send, c_skip, atol=1e-7)


def test_cost_edge_cases_zero_or_adverse_ve_and_invalid_probs():
    # Zero VE (q1 == q0) and adverse VE (q1 > q0) must NEVER receive a vaccine even at p=1.0
    p_u = np.array([0.05, 0.05, 0.10])
    p_v = np.array([0.05, 0.08, 0.01])
    tau = bayes_optimal_thresholds(p_u, p_v, DATASET_C_CONFIG)
    assert tau[0] == pytest.approx(1.0)
    assert tau[1] == pytest.approx(1.0)
    assert 0.0 < tau[2] < 1.0

    preds = predict_with_cost_policy(
        np.array([1.0, 1.0, 0.90]), p_u, p_v, policy="bayes", threshold_param=0.7, cfg=DATASET_C_CONFIG
    )
    assert preds.tolist() == [0, 0, 1]

    # Invalid / NaN probabilities handled safely under strict=False, rejected under strict=True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tau_safe = bayes_optimal_thresholds(
            np.array([np.nan, -0.2]), np.array([0.01, 0.01]), DATASET_C_CONFIG, strict=False
        )
        preds_safe = predict_with_cost_policy(
            np.array([np.nan, 0.9]), np.array([0.1, 0.1]), np.array([0.01, 0.01]), cfg=DATASET_C_CONFIG
        )
    assert tau_safe[0] == pytest.approx(1.0)
    assert preds_safe[0] == 0

    with pytest.raises(ValueError):
        bayes_optimal_thresholds(np.array([np.nan]), np.array([0.01]), DATASET_C_CONFIG, strict=True)
    with pytest.raises(ValueError):
        bayes_optimal_thresholds(np.array([1.5]), np.array([0.01]), DATASET_C_CONFIG, strict=True)
    with pytest.raises(ValueError):
        compute_town_cost(
            np.array([1]), np.array([1]), np.array([np.nan]), np.array([0.01]), DATASET_C_CONFIG, strict=True
        )

    # Free vaccine edge case (vaccine_cost = 0): vaccinate when q0 > q1, skip when q0 <= q1
    free_cfg = DatasetConfig(
        name="free",
        train_path=DATASET_C_CONFIG.train_path,
        test_path=DATASET_C_CONFIG.test_path,
        example_submission_path=DATASET_C_CONFIG.example_submission_path,
        id_col=DATASET_C_CONFIG.id_col,
        target_col=DATASET_C_CONFIG.target_col,
        disease_prefix=DATASET_C_CONFIG.disease_prefix,
        vaccine_cost=0.0,
        hospitalisation_cost=41000.0,
    )
    tau_free = bayes_optimal_thresholds(np.array([0.05, 0.05]), np.array([0.01, 0.05]), free_cfg)
    assert tau_free[0] == pytest.approx(0.0)
    assert tau_free[1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Category G: Multi-Fold SHAP Explainability & Stability Tests
# ---------------------------------------------------------------------------


def test_multifold_shap_stability_and_separation(tmp_path: Path):
    raw_train, raw_test, _ = load_raw_datasets(DATASET_C_CONFIG)
    pos = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 1].iloc[:80]
    neg = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 0].iloc[:160]
    sub_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)

    feat_tr = engineer_features(sub_df, DATASET_C_CONFIG)
    feat_te = engineer_features(raw_test.iloc[:30].copy(), DATASET_C_CONFIG)

    out_dir = tmp_path / "outputs"
    fig_dir = tmp_path / "figures"
    results, comp_df, folds, artifacts = train_and_evaluate_all_models(
        feat_tr, feat_te, DATASET_C_CONFIG, out_dir, n_folds=2, inner_folds=2, fast_mode=True
    )

    assert "Uncalibrated_Stack" in comp_df["model"].values
    assert "Calibrated_Stack" in comp_df["model"].values
    assert "ece" in comp_df.columns
    assert (out_dir / "stacking_calibration_audit.csv").exists()

    shap_df = _compute_and_plot_shap(feat_tr, feat_te, artifacts, fig_dir, out_dir)
    assert "fold_std_abs_shap" in shap_df.columns
    assert "fold_0_mean_abs_shap" in shap_df.columns
    assert "fold_1_mean_abs_shap" in shap_df.columns
    assert set(shap_df["feature_type"]).issubset({"numeric", "categorical"})
    assert (fig_dir / "06_explainability_shap_importance.png").exists()
    assert (fig_dir / "08_explainability_shap_beeswarm.png").exists()
    assert (fig_dir / "08b_explainability_categorical_shap.png").exists()


# ---------------------------------------------------------------------------
# Category H: Submission Schema Validation Tests
# ---------------------------------------------------------------------------


def test_generated_submissions_match_example_schema():
    example_sub = pd.read_csv(DATASET_C_CONFIG.example_submission_path)
    manifest_path = SUBMISSIONS_DIR / "submission_manifest.csv"
    assert manifest_path.exists(), "Run main.py before verifying generated submissions."
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 5

    for idx in range(1, 6):
        sub_path = SUBMISSIONS_DIR / f"challenge_submission_group_0_order_{idx}.csv"
        assert sub_path.exists(), f"Missing submission: {sub_path}"
        sub = pd.read_csv(sub_path)
        assert list(sub.columns) == list(example_sub.columns)
        assert len(sub) == len(example_sub) == 4749
        assert np.array_equal(sub["respondent_id"].values, example_sub["respondent_id"].values)
        assert not sub["covid_vaccine"].isna().any()
        assert set(sub["covid_vaccine"].unique()).issubset({0, 1})
