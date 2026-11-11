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
    MAX_MODELLED_IHR,
    bayes_optimal_thresholds,
    compute_ece,
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    expected_decision_cost,
    predict_with_cost_policy,
    summarize_cost_policies_table,
    tune_threshold_on_oof,
    zhang_yu_rr_to_or,
)
from src.data_prep import (
    audit_dataset_preprocessing,
    engineer_features,
    get_feature_columns,
    load_raw_datasets,
    run_feature_ablation,
)
from src.explainability import _compute_and_plot_shap
from src.modeling import (
    _coerce_sklearn_frame,
    _fit_base_models_on_train_only,
    _make_sklearn_onehot_preprocessor,
    evaluate_calibration_nested,
    fit_lgbm_with_early_stopping,
    fit_outer_fold_models,
    select_and_fit_posthoc_calibration,
    train_and_evaluate_all_models,
)


@pytest.fixture(scope="module")
def raw_data():
    """Load Dataset C raw splits once per test module without mutation."""
    return load_raw_datasets(DATASET_C_CONFIG)


# ---------------------------------------------------------------------------
# Category A & B: Preprocessing and Feature Engineering Integrity
# ---------------------------------------------------------------------------


def test_data_cleaning_fixes_whitespace_and_zero_opinions(raw_data):
    raw_train, raw_test, _ = raw_data
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


def test_preprocessing_determinism_no_target_leakage_and_unseen_safety(raw_data):
    raw_train, raw_test, _ = raw_data
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


def test_preprocessing_duplicates_types_and_missing_value_imputation(raw_data):
    raw_train, raw_test, _ = raw_data
    sample_tr = raw_train.iloc[:40].copy()
    sample_te = raw_test.iloc[:15].copy()

    # Exact duplicate row should be detected cleanly by audit_dataset_preprocessing
    dup_tr = pd.concat([sample_tr, sample_tr.iloc[[0]]], ignore_index=True)
    feat_dup_tr = engineer_features(dup_tr, DATASET_C_CONFIG)
    feat_te = engineer_features(sample_te, DATASET_C_CONFIG)
    audit_dup = audit_dataset_preprocessing(
        dup_tr, sample_te, feat_dup_tr, feat_te, DATASET_C_CONFIG
    )
    assert audit_dup["train_exact_duplicates"] == 1
    assert audit_dup["undocumented_zero_audit"]["policy_basis"].startswith("Empirical")

    # Malformed numeric string is coerced to NaN; valid numeric string is parsed; valid 0s stay 0
    dirty = sample_tr.iloc[:3].copy()
    dirty["household_adults"] = ["2", "invalid_num", 0]
    dirty["covid_concern"] = [0, 1, 3]
    fe_dirty = engineer_features(dirty, DATASET_C_CONFIG)
    assert fe_dirty["household_adults"].iloc[0] == pytest.approx(2.0)
    assert pd.isna(fe_dirty["household_adults"].iloc[1])
    assert fe_dirty["covid_concern"].iloc[0] == pytest.approx(0.0)

    # Sklearn preprocessor coerces all numerics cleanly and eliminates NaNs without Infs
    num_cols, cat_cols = get_feature_columns(fe_dirty, DATASET_C_CONFIG)
    coerced = _coerce_sklearn_frame(fe_dirty[num_cols + cat_cols], num_cols, cat_cols)
    for col in num_cols:
        assert coerced[col].dtype == np.float64
    pre = _make_sklearn_onehot_preprocessor(num_cols, cat_cols)
    transformed = pre.fit_transform(coerced)
    assert np.all(np.isfinite(transformed))


def test_engineered_feature_calculations_and_boundaries(raw_data):
    raw_train, _, _ = raw_data
    rows = raw_train.iloc[:3].copy().reset_index(drop=True)

    # Row 0: age 55-64 (ordinal 3.0) + 1 chronic -> high_risk_elderly=1; doctor_recc=1 + missing eff -> 3.0
    rows.loc[0, "age_group"] = "55 - 64 Years"
    rows.loc[0, "number_chronic_med_conditions"] = 1
    rows.loc[0, "doctor_recc_covid"] = 1
    rows.loc[0, "opinion_covid_vacc_effective"] = 0  # mapped to NaN -> imputed to midpoint 3.0 in interaction
    rows.loc[0, "opinion_covid_risk"] = 5
    rows.loc[0, "opinion_covid_sick_from_vacc"] = 2
    rows.loc[0, "household_adults"] = 2.0
    rows.loc[0, "household_children"] = 1.0
    rows.loc[0, "employment_sector"] = "healthcare"
    rows.loc[0, "health_insurance"] = 0.0
    for col in (
        "behavioral_antiviral_meds",
        "behavioral_avoidance",
        "behavioral_face_mask",
        "behavioral_wash_hands",
        "behavioral_large_gatherings",
        "behavioral_outside_home",
        "behavioral_touch_face",
    ):
        rows.loc[0, col] = 1.0

    # Row 1: age 45-54 (ordinal 2.0) + 3 chronic -> high_risk_elderly=0 (age boundary)
    rows.loc[1, "age_group"] = "45 - 54 Years"
    rows.loc[1, "number_chronic_med_conditions"] = 3

    # Row 2: age 65+ (ordinal 4.0) + 0 chronic -> high_risk_elderly=0 (chronic boundary)
    rows.loc[2, "age_group"] = "65+ Years"
    rows.loc[2, "number_chronic_med_conditions"] = 0

    fe = engineer_features(rows, DATASET_C_CONFIG)
    assert fe.loc[0, "age_ordinal"] == pytest.approx(3.0)
    assert fe.loc[0, "behavioral_sum"] == pytest.approx(7.0)
    assert fe.loc[0, "behavioral_mean"] == pytest.approx(1.0)
    assert fe.loc[0, "doctor_x_effective"] == pytest.approx(3.0)
    assert fe.loc[0, "doctor_x_risk"] == pytest.approx(5.0)
    assert fe.loc[0, "doctor_x_not_worried"] == pytest.approx(4.0)
    assert fe.loc[0, "household_total"] == pytest.approx(4.0)
    assert fe.loc[0, "has_chronic_condition"] == pytest.approx(1.0)
    assert fe.loc[0, "high_risk_elderly"] == pytest.approx(1.0)
    assert fe.loc[0, "is_healthcare_sector"] == pytest.approx(1.0)
    assert fe.loc[0, "uninsured_flag"] == pytest.approx(1.0)
    assert fe.loc[1, "high_risk_elderly"] == pytest.approx(0.0)
    assert fe.loc[2, "high_risk_elderly"] == pytest.approx(0.0)


def test_feature_ablation_metrics_and_structure(raw_data):
    raw_train, _, _ = raw_data
    pos = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 1].iloc[:65]
    neg = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 0].iloc[:135]
    sub_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)

    ablation = run_feature_ablation(sub_df, DATASET_C_CONFIG, n_folds=2, random_state=42)
    required_cols = {
        "variant",
        "feature_group",
        "evaluation_type",
        "reference_variant",
        "description",
        "n_features",
        "lr_roc_auc",
        "lr_log_loss",
        "lr_brier",
        "lr_ece",
        "lgb_roc_auc",
        "lgb_log_loss",
        "lgb_brier",
        "lgb_ece",
        "lr_auc_delta_vs_ref",
        "lr_logloss_delta_vs_ref",
        "lgb_auc_delta_vs_ref",
        "lgb_logloss_delta_vs_ref",
        "decision",
        "reason",
    }
    assert required_cols.issubset(set(ablation.columns))
    assert "add_missing_flags" in ablation["variant"].values
    assert "ablate_without_ordinal" in ablation["variant"].values
    assert "plus_pruned_opinion_composites" in ablation["variant"].values
    for col in ("lr_roc_auc", "lr_log_loss", "lr_brier", "lr_ece", "lgb_roc_auc", "lgb_log_loss", "lgb_brier", "lgb_ece"):
        assert np.all(np.isfinite(ablation[col].to_numpy()))


# ---------------------------------------------------------------------------
# Category C & D: Validation Leakage, Stacking Isolation & Calibration Tests
# ---------------------------------------------------------------------------


def test_outer_validation_isolation_and_no_leakage(raw_data):
    """Verify structurally and empirically that outer validation rows/labels never leak."""
    # Structural verification: fit_outer_fold_models cannot receive y_va
    sig = inspect.signature(fit_outer_fold_models)
    assert "y_va" not in sig.parameters

    raw_train, raw_test, _ = raw_data
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
    assert len(res_a["nested_c_per_fold"]) == 2

    # Outer permutation invariance: shuffling X_va rows yields identically shuffled predictions
    perm = np.arange(len(X_va_clean))[::-1]
    res_perm = fit_outer_fold_models(
        X_tr, y_tr, X_va_clean.iloc[perm], X_te, num_cols, cat_cols, outer_fold_idx=0, inner_folds=2, fast_mode=True
    )
    np.testing.assert_allclose(res_a["cal_stack_val"][perm], res_perm["cal_stack_val"], atol=1e-12)


def test_boosting_models_independent_execution_api_compat_and_reproducibility(raw_data):
    """Verify LightGBM, XGBoost, and CatBoost individually, eval_set fallback, Stratification, and determinism."""
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    raw_train, raw_test, _ = raw_data
    pos = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 1].iloc[:75]
    neg = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 0].iloc[:150]
    sub_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)
    feat_tr = engineer_features(sub_df, DATASET_C_CONFIG)
    feat_te = engineer_features(raw_test.iloc[:30].copy(), DATASET_C_CONFIG)
    num_cols, cat_cols = get_feature_columns(feat_tr, DATASET_C_CONFIG)
    feat_cols = num_cols + cat_cols

    X_tr = feat_tr.iloc[:175][feat_cols]
    y_tr = feat_tr.iloc[:175][DATASET_C_CONFIG.target_col].to_numpy(dtype=int)
    X_va = feat_tr.iloc[175:][feat_cols]
    X_te = feat_te[feat_cols]

    run1 = _fit_base_models_on_train_only(X_tr, y_tr, X_va, X_te, num_cols, cat_cols, seed=42, fast_mode=True)
    run2 = _fit_base_models_on_train_only(X_tr, y_tr, X_va, X_te, num_cols, cat_cols, seed=42, fast_mode=True)

    for model_name in ("LightGBM", "XGBoost", "CatBoost", "LogisticRegression", "RandomForest", "HistGradientBoosting"):
        p_va_1 = run1["val_preds"][model_name]
        p_te_1 = run1["test_preds"][model_name]
        assert p_va_1.shape == (len(X_va),)
        assert p_te_1.shape == (len(X_te),)
        assert np.all(np.isfinite(p_va_1)) and np.all((p_va_1 >= 0.0) & (p_va_1 <= 1.0))
        assert np.all(np.isfinite(p_te_1)) and np.all((p_te_1 >= 0.0) & (p_te_1 <= 1.0))
        np.testing.assert_allclose(p_va_1, run2["val_preds"][model_name], atol=1e-12)
        np.testing.assert_allclose(p_te_1, run2["test_preds"][model_name], atol=1e-12)

    # Verify version-safe LightGBM eval_set fallback for LightGBM 4.5.x compatibility
    class _LegacyLGBMWrapper:
        def __init__(self):
            self._m = LGBMClassifier(n_estimators=50, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)
            self.used_eval_set = False

        def fit(self, X, y, eval_set=None, callbacks=None):
            self.used_eval_set = eval_set is not None and len(eval_set) == 1
            return self._m.fit(X, y, eval_set=eval_set, callbacks=callbacks)

    X_num_only = _coerce_sklearn_frame(X_tr, num_cols, [])
    wrapper = _LegacyLGBMWrapper()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_lgbm_with_early_stopping(wrapper, X_num_only.iloc[:130], y_tr[:130], X_num_only.iloc[130:], y_tr[130:], 15)
    assert wrapper.used_eval_set

    # Stratified fold class ratio check
    y_full = raw_train[DATASET_C_CONFIG.target_col].to_numpy(dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_rate = float(np.mean(y_full))
    for _, va_idx in skf.split(raw_train, y_full):
        assert abs(float(np.mean(y_full[va_idx])) - base_rate) < 0.002


def test_stacking_respondent_level_isolation_and_active_calibration(raw_data):
    """Verify respondent-level OOF & nested calibration isolation plus active Platt selection."""
    from sklearn.model_selection import StratifiedKFold

    raw_train, _, _ = raw_data
    pos = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 1].iloc[:60]
    neg = raw_train[raw_train[DATASET_C_CONFIG.target_col] == 0].iloc[:120]
    sub_df = pd.concat([pos, neg], axis=0).reset_index(drop=True)
    feat_tr = engineer_features(sub_df, DATASET_C_CONFIG)
    num_cols, cat_cols = get_feature_columns(feat_tr, DATASET_C_CONFIG)
    X_tr = feat_tr[num_cols + cat_cols]
    y_tr = feat_tr[DATASET_C_CONFIG.target_col].to_numpy(dtype=int)

    # Respondent-level isolation check across inner folds:
    # For inner fold k=0, flipping labels y_tr[va_0] MUST NOT alter inner_oof_probs[va_0]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    splits = list(skf.split(X_tr, y_tr))
    tr_0, va_0 = splits[0]
    y_flipped = y_tr.copy()
    y_flipped[va_0] = 1 - y_flipped[va_0]

    res_clean = _fit_base_models_on_train_only(
        X_tr.iloc[tr_0], y_tr[tr_0], X_tr.iloc[va_0], None, num_cols, cat_cols, seed=42, fast_mode=True
    )
    res_label_flipped = _fit_base_models_on_train_only(
        X_tr.iloc[tr_0], y_flipped[tr_0], X_tr.iloc[va_0], None, num_cols, cat_cols, seed=42, fast_mode=True
    )
    for m_name in res_clean["val_preds"]:
        np.testing.assert_allclose(
            res_clean["val_preds"][m_name], res_label_flipped["val_preds"][m_name], atol=1e-12
        )

    # Verify active calibration triggers when probabilities are genuinely overconfident
    rng = np.random.default_rng(7)
    z_true = rng.normal(0.0, 1.2, size=600)
    p_true = 1.0 / (1.0 + np.exp(-z_true))
    y_syn = rng.binomial(1, p_true)
    p_overconfident = 1.0 / (1.0 + np.exp(-3.0 * z_true - 0.6))
    fold_ids = np.arange(600) % 4
    best_mode, calibrator, cal_eval = select_and_fit_posthoc_calibration(
        p_overconfident, y_syn, fold_ids, seed=42
    )
    assert best_mode == "platt"
    assert calibrator is not None
    assert cal_eval["platt"] < cal_eval["none"] - 0.05

    # Nested calibration audit: flipping validation labels strictly inside cal_va (fold 0)
    # MUST NOT alter fold 0's chosen regularization C or nested_raw_oof[fold == 0] predictions.
    Z_syn = np.column_stack([z_true, 0.8 * z_true + rng.normal(0.0, 0.4, size=600)])
    inner_ids = np.repeat(np.arange(3), 200)
    y_flip_f0 = y_syn.copy()
    cal_va_0 = inner_ids == 0
    y_flip_f0[cal_va_0] = 1 - y_flip_f0[cal_va_0]

    nest_clean = evaluate_calibration_nested(Z_syn, y_syn, inner_ids, seed=42)
    nest_flipped = evaluate_calibration_nested(Z_syn, y_flip_f0, inner_ids, seed=42)
    assert nest_clean["nested_c_per_fold"][0] == nest_flipped["nested_c_per_fold"][0]
    np.testing.assert_allclose(
        nest_clean["nested_raw_oof"][cal_va_0],
        nest_flipped["nested_raw_oof"][cal_va_0],
        atol=1e-12,
    )


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


def test_hospitalisation_effect_measure_consistency_and_bounds():
    # Low-risk stratum: cloglog RR model agrees closely with p0 * RR and Zhang-Yu conversion
    df_young = pd.DataFrame(
        {
            "age_group": ["18 - 34 Years", "18 - 34 Years"],
            "number_chronic_med_conditions": [0, 3],
            "sex": ["Female", "Female"],
        }
    )
    pu_young, _ = estimate_hospitalisation_probabilities(
        df_young, scenario="base", include_sex_adjustment=False
    )
    rr_ratio = pu_young[1] / pu_young[0]
    assert 4.5 <= rr_ratio <= 5.05  # true aRR = 5.0 behavior at low baseline risk

    # Extreme severe 65+ >=3 stratum remains smoothly bounded below MAX_MODELLED_IHR without overflow
    df_extreme = pd.DataFrame(
        {
            "age_group": ["65+ Years"],
            "number_chronic_med_conditions": [3],
            "sex": ["Male"],
        }
    )
    pu_ext, pv_ext = estimate_hospitalisation_probabilities(df_extreme, scenario="severe")
    assert 0.50 < pu_ext[0] <= MAX_MODELLED_IHR
    assert 0.0 < pv_ext[0] < pu_ext[0]

    # Zhang-Yu utility agrees with exact OR conversion formula at low risk and stays bounded at high risk
    expected_zy_or = 5.0 * (1.0 - 0.006) / (1.0 - 5.0 * 0.006)
    assert zhang_yu_rr_to_or(5.0, 0.006) == pytest.approx(expected_zy_or, abs=1e-6)
    assert np.isfinite(zhang_yu_rr_to_or(5.0, 0.25))


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


def test_cost_policy_fallback_cv_and_summary_table():
    rng = np.random.default_rng(123)
    y_t = rng.binomial(1, 0.3, size=100)
    probs = np.clip(0.15 + 0.5 * y_t + rng.normal(0.0, 0.1, size=100), 0.01, 0.99)
    pu = np.full(100, 0.06)
    pv = np.full(100, 0.01)

    # Even when folds=None, tune_threshold_on_oof MUST enforce internal out-of-fold CV
    pol = tune_threshold_on_oof(y_t, probs, pu, pv, cfg=DATASET_C_CONFIG, folds=None)
    assert pol["global_tuned"]["evaluation"] == "strictly_out_of_fold_cv"
    assert pol["bayes_scaled"]["evaluation"] == "strictly_out_of_fold_cv"
    assert pol["bayes_exact"]["evaluation"] == "analytical_zero_tuning"

    tbl = summarize_cost_policies_table(pol)
    assert len(tbl) == 4
    assert {"policy_name", "vaccines_accepted", "savings_vs_fixed_0.5"}.issubset(set(tbl.columns))
    for _, row in tbl.iterrows():
        assert row["vaccine_send_cost"] + row["law_penalty_cost"] + row["hospitalisation_cost"] == pytest.approx(
            row["total_cost"], abs=1e-6
        )


# ---------------------------------------------------------------------------
# Category G: Multi-Fold SHAP Explainability & Stability Tests
# ---------------------------------------------------------------------------


def test_multifold_shap_stability_and_separation(raw_data, tmp_path: Path):
    raw_train, raw_test, _ = raw_data
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shap_df = _compute_and_plot_shap(feat_tr, feat_te, artifacts, fig_dir, out_dir)
    shap_list_warnings = [w for w in caught if "output has changed to a list of ndarray" in str(w.message)]
    assert len(shap_list_warnings) == 0

    assert "fold_std_abs_shap" in shap_df.columns
    assert "fold_0_mean_abs_shap" in shap_df.columns
    assert "fold_1_mean_abs_shap" in shap_df.columns
    assert set(shap_df["feature_type"]).issubset({"numeric", "categorical"})
    assert (fig_dir / "06_explainability_shap_importance.png").exists()
    assert (fig_dir / "08_explainability_shap_beeswarm.png").exists()
    assert (fig_dir / "08b_explainability_categorical_shap.png").exists()

    # Exact weighted aggregation consistency across outer folds
    fold_sizes = np.array([(folds == 0).sum(), (folds == 1).sum()], dtype=np.float64)
    expected_overall = (
        shap_df["fold_0_mean_abs_shap"].to_numpy() * fold_sizes[0]
        + shap_df["fold_1_mean_abs_shap"].to_numpy() * fold_sizes[1]
    ) / fold_sizes.sum()
    np.testing.assert_allclose(shap_df["mean_abs_shap"].to_numpy(), expected_overall, atol=1e-12)


# ---------------------------------------------------------------------------
# Category H: Submission Schema Validation Tests
# ---------------------------------------------------------------------------


def test_generated_submissions_match_example_schema():
    example_sub = pd.read_csv(DATASET_C_CONFIG.example_submission_path)
    manifest_path = SUBMISSIONS_DIR / "submission_manifest.csv"
    assert manifest_path.exists(), "Run main.py before verifying generated submissions."
    manifest = pd.read_csv(manifest_path)
    assert len(manifest) == 5
    assert list(manifest["oof_total_cost"]) == sorted(manifest["oof_total_cost"])

    pred_signatures = set()
    for idx in range(1, 6):
        sub_path = SUBMISSIONS_DIR / f"challenge_submission_group_0_order_{idx}.csv"
        assert sub_path.exists(), f"Missing submission: {sub_path}"
        sub = pd.read_csv(sub_path)
        assert list(sub.columns) == list(example_sub.columns)
        assert len(sub) == len(example_sub) == 4749
        assert sub["respondent_id"].is_unique
        assert np.array_equal(sub["respondent_id"].values, example_sub["respondent_id"].values)
        assert not sub["covid_vaccine"].isna().any()
        assert set(sub["covid_vaccine"].unique()).issubset({0, 1})
        positives = int(sub["covid_vaccine"].sum())
        assert 0 < positives < len(sub)
        pred_signatures.add(tuple(sub["covid_vaccine"].tolist()))

    assert len(pred_signatures) >= 4
