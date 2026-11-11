"""Multi-model Stratified CV training, cost evaluation, and submission export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    DATASET_C_CONFIG,
    DatasetConfig,
    N_FOLDS,
    OUTPUT_DIR,
    RANDOM_SEED,
    SUBMISSIONS_DIR,
)
from src.cost_model import (
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    predict_with_cost_policy,
    tune_threshold_on_oof,
)
from src.data_prep import get_feature_columns


@dataclass
class ModelRunResult:
    """Out-of-fold and test predictions plus evaluation metrics for one model."""

    name: str
    oof_probs: np.ndarray
    test_probs: np.ndarray
    roc_auc: float
    log_loss_val: float
    brier: float
    accuracy_05: float
    f1_05: float
    policy_results: Dict[str, dict]
    scenario_costs: Dict[str, float]


def _make_sklearn_onehot_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def _make_sklearn_ordinal_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "ord",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-2,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def _prepare_category_frames(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: pd.DataFrame,
    cat_cols: List[str],
    for_catboost: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr, va, te = X_tr.copy(), X_va.copy(), X_te.copy()
    for col in cat_cols:
        tr_s = tr[col].fillna("Missing").astype(str)
        va_s = va[col].fillna("Missing").astype(str)
        te_s = te[col].fillna("Missing").astype(str)
        if for_catboost:
            tr[col], va[col], te[col] = tr_s, va_s, te_s
        else:
            cats = sorted(tr_s.unique().tolist())
            if "Unknown" not in cats:
                cats.append("Unknown")
            va_s = va_s.where(va_s.isin(cats), "Unknown")
            te_s = te_s.where(te_s.isin(cats), "Unknown")
            dtype = pd.CategoricalDtype(categories=cats)
            tr[col] = tr_s.astype(dtype)
            va[col] = va_s.astype(dtype)
            te[col] = te_s.astype(dtype)
    return tr, va, te


def _evaluate_model_predictions(
    name: str,
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    test_probs: np.ndarray,
    folds: np.ndarray,
    feat_train: pd.DataFrame,
    cfg: DatasetConfig,
) -> ModelRunResult:
    p_unvacc_base, p_vacc_base = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    policy_results = tune_threshold_on_oof(
        y_true, oof_probs, p_unvacc_base, p_vacc_base, folds=folds, cfg=cfg
    )

    # Evaluate Bayes-exact policy across Conservative / Base / Severe risk profiles
    scenario_costs = {}
    for scen in ("conservative", "base", "severe"):
        pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario=scen)
        preds = predict_with_cost_policy(
            oof_probs, pu, pv, policy="bayes", threshold_param=1.0, cfg=cfg
        )
        scenario_costs[scen] = compute_town_cost(y_true, preds, pu, pv, cfg=cfg).total_cost

    pred_05 = (oof_probs >= 0.5).astype(np.int64)
    return ModelRunResult(
        name=name,
        oof_probs=oof_probs,
        test_probs=test_probs,
        roc_auc=float(roc_auc_score(y_true, oof_probs)),
        log_loss_val=float(log_loss(y_true, oof_probs)),
        brier=float(brier_score_loss(y_true, oof_probs)),
        accuracy_05=float(accuracy_score(y_true, pred_05)),
        f1_05=float(f1_score(y_true, pred_05)),
        policy_results=policy_results,
        scenario_costs=scenario_costs,
    )


def train_and_evaluate_all_models(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    output_dir: Path = OUTPUT_DIR,
) -> Tuple[List[ModelRunResult], pd.DataFrame, np.ndarray, dict]:
    """Train 6 diverse model families + 1 calibrated stacking ensemble with 5-fold Stratified CV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols, cat_cols = get_feature_columns(feat_train, cfg=cfg)
    feature_cols = num_cols + cat_cols

    X = feat_train[feature_cols].copy()
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    X_test = feat_test[feature_cols].copy()

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = np.zeros(len(y), dtype=np.int64)
    for fold_idx, (_, val_idx) in enumerate(skf.split(X, y)):
        folds[val_idx] = fold_idx

    model_names = [
        "LogisticRegression",
        "RandomForest",
        "HistGradientBoosting",
        "LightGBM",
        "XGBoost",
        "CatBoost",
    ]
    oof_dict = {m: np.zeros(len(y), dtype=np.float64) for m in model_names}
    test_dict = {m: np.zeros(len(X_test), dtype=np.float64) for m in model_names}

    best_cb_models: List[CatBoostClassifier] = []
    best_lgbm_models: List[LGBMClassifier] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X.iloc[va_idx], y[va_idx]

        # 1. Logistic Regression (L2 linear baseline)
        lr_pipe = Pipeline(
            [
                ("prep", _make_sklearn_onehot_preprocessor(num_cols, cat_cols)),
                ("clf", LogisticRegression(C=0.5, max_iter=2000, random_state=RANDOM_SEED)),
            ]
        )
        lr_pipe.fit(X_tr, y_tr)
        oof_dict["LogisticRegression"][va_idx] = lr_pipe.predict_proba(X_va)[:, 1]
        test_dict["LogisticRegression"] += lr_pipe.predict_proba(X_test)[:, 1] / N_FOLDS

        # 2. Random Forest (bagging ensemble)
        rf_pipe = Pipeline(
            [
                ("prep", _make_sklearn_ordinal_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=14,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        n_jobs=-1,
                        random_state=RANDOM_SEED + fold_idx,
                    ),
                ),
            ]
        )
        rf_pipe.fit(X_tr, y_tr)
        oof_dict["RandomForest"][va_idx] = rf_pipe.predict_proba(X_va)[:, 1]
        test_dict["RandomForest"] += rf_pipe.predict_proba(X_test)[:, 1] / N_FOLDS

        # 3. HistGradientBoosting (sklearn native GBDT)
        cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
        hgb_pipe = Pipeline(
            [
                ("prep", _make_sklearn_ordinal_preprocessor(num_cols, cat_cols)),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=2.0,
                        categorical_features=cat_indices,
                        random_state=RANDOM_SEED + fold_idx,
                    ),
                ),
            ]
        )
        hgb_pipe.fit(X_tr, y_tr)
        oof_dict["HistGradientBoosting"][va_idx] = hgb_pipe.predict_proba(X_va)[:, 1]
        test_dict["HistGradientBoosting"] += hgb_pipe.predict_proba(X_test)[:, 1] / N_FOLDS

        # Prepare category-typed frames for LightGBM & XGBoost
        X_tr_cat, X_va_cat, X_te_cat = _prepare_category_frames(
            X_tr, X_va, X_test, cat_cols, for_catboost=False
        )

        # 4. LightGBM
        from lightgbm import early_stopping

        lgb_model = LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.025,
            num_leaves=23,
            min_child_samples=35,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=RANDOM_SEED + fold_idx,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_cat,
            y_tr,
            eval_set=[(X_va_cat, y_va)],
            callbacks=[early_stopping(stopping_rounds=80, verbose=False)],
        )
        oof_dict["LightGBM"][va_idx] = lgb_model.predict_proba(X_va_cat)[:, 1]
        test_dict["LightGBM"] += lgb_model.predict_proba(X_te_cat)[:, 1] / N_FOLDS
        best_lgbm_models.append(lgb_model)

        # 5. XGBoost
        xgb_model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=3.0,
            enable_categorical=True,
            eval_metric="logloss",
            early_stopping_rounds=80,
            random_state=RANDOM_SEED + fold_idx,
            n_jobs=-1,
        )
        xgb_model.fit(X_tr_cat, y_tr, eval_set=[(X_va_cat, y_va)], verbose=False)
        oof_dict["XGBoost"][va_idx] = xgb_model.predict_proba(X_va_cat)[:, 1]
        test_dict["XGBoost"] += xgb_model.predict_proba(X_te_cat)[:, 1] / N_FOLDS

        # 6. CatBoost
        X_tr_cb, X_va_cb, X_te_cb = _prepare_category_frames(
            X_tr, X_va, X_test, cat_cols, for_catboost=True
        )
        cb_model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=6.0,
            eval_metric="Logloss",
            random_seed=RANDOM_SEED + fold_idx,
            verbose=False,
        )
        cb_model.fit(
            X_tr_cb,
            y_tr,
            cat_features=cat_cols,
            eval_set=(X_va_cb, y_va),
            early_stopping_rounds=100,
            verbose=False,
        )
        oof_dict["CatBoost"][va_idx] = cb_model.predict_proba(X_va_cb)[:, 1]
        test_dict["CatBoost"] += cb_model.predict_proba(X_te_cb)[:, 1] / N_FOLDS
        best_cb_models.append(cb_model)

    # 7. Calibrated Stacking Meta-Ensemble (strict out-of-fold logit stacking)
    oof_stack_matrix = np.column_stack(
        [logit(np.clip(oof_dict[m], 1e-5, 1 - 1e-5)) for m in model_names]
    )
    test_stack_matrix = np.column_stack(
        [logit(np.clip(test_dict[m], 1e-5, 1 - 1e-5)) for m in model_names]
    )

    stack_oof = np.zeros(len(y), dtype=np.float64)
    stack_test = np.zeros(len(X_test), dtype=np.float64)
    meta_coefs = []
    for fold_idx in range(N_FOLDS):
        tr_mask = folds != fold_idx
        va_mask = folds == fold_idx
        meta = LogisticRegression(C=0.3, max_iter=1000, random_state=RANDOM_SEED)
        meta.fit(oof_stack_matrix[tr_mask], y[tr_mask])
        stack_oof[va_mask] = meta.predict_proba(oof_stack_matrix[va_mask])[:, 1]
        stack_test += meta.predict_proba(test_stack_matrix)[:, 1] / N_FOLDS
        meta_coefs.append(meta.coef_[0])

    oof_dict["Calibrated_Stack"] = stack_oof
    test_dict["Calibrated_Stack"] = stack_test

    # Evaluate all 7 models
    results: List[ModelRunResult] = []
    for m_name in model_names + ["Calibrated_Stack"]:
        res = _evaluate_model_predictions(
            m_name, y, oof_dict[m_name], test_dict[m_name], folds, feat_train, cfg
        )
        results.append(res)

    # Build comparison table sorted by Bayes-Exact Base Scenario Town Cost
    rows = []
    for r in results:
        b_exact = r.policy_results["bayes_exact"]["breakdown"]
        b_scaled = r.policy_results["bayes_scaled"]["breakdown"]
        g_05 = r.policy_results["global_0.5"]["breakdown"]
        g_tuned = r.policy_results["global_tuned"]["breakdown"]
        rows.append(
            {
                "model": r.name,
                "roc_auc": r.roc_auc,
                "log_loss": r.log_loss_val,
                "brier": r.brier,
                "acc_at_0.5": r.accuracy_05,
                "f1_at_0.5": r.f1_05,
                "cost_global_0.5": g_05.total_cost,
                "cost_global_tuned": g_tuned.total_cost,
                "global_tuned_thr": r.policy_results["global_tuned"]["param"],
                "cost_bayes_exact": b_exact.total_cost,
                "cost_bayes_scaled": b_scaled.total_cost,
                "bayes_scale_param": r.policy_results["bayes_scaled"]["param"],
                "bayes_exact_per_person": b_exact.cost_per_person,
                "savings_vs_treat_none": b_exact.savings_vs_treat_none,
                "savings_vs_treat_all": b_exact.savings_vs_treat_all,
                "captured_oracle_pct": b_exact.captured_oracle_savings_pct,
                "pred_vacc_rate_bayes": b_exact.predicted_positive_rate,
                "cost_conservative": r.scenario_costs["conservative"],
                "cost_severe": r.scenario_costs["severe"],
            }
        )

    comp_df = pd.DataFrame(rows).sort_values("cost_bayes_exact", ascending=True).reset_index(drop=True)
    comp_df.to_csv(output_dir / "model_comparison.csv", index=False)

    oof_df = pd.DataFrame({cfg.id_col: feat_train[cfg.id_col], cfg.target_col: y, **oof_dict})
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)

    test_df = pd.DataFrame({cfg.id_col: feat_test[cfg.id_col], **test_dict})
    test_df.to_csv(output_dir / "test_probabilities.csv", index=False)

    artifacts = {
        "folds": folds,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "lgbm_models": best_lgbm_models,
        "catboost_models": best_cb_models,
        "meta_weights": dict(zip(model_names, np.mean(meta_coefs, axis=0).tolist())),
    }
    return results, comp_df, folds, artifacts


def export_top5_submissions(
    results: List[ModelRunResult],
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    example_sub: pd.DataFrame,
    group_id: int = 0,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    submissions_dir: Path = SUBMISSIONS_DIR,
) -> pd.DataFrame:
    """Rank candidate model+policy pipelines by CV town cost and export top 5 submissions."""
    submissions_dir.mkdir(parents=True, exist_ok=True)
    p_unvacc_test, p_vacc_test = estimate_hospitalisation_probabilities(feat_test, scenario="base")
    p_unvacc_train, p_vacc_train = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    y_true = feat_train[cfg.target_col].to_numpy(dtype=np.int64)

    # Consider each model's best Bayes-exact and Bayes-scaled strategies, keeping top 5 distinct models
    # so the 5 submissions offer both top performance and model diversity.
    candidates = []
    for r in results:
        for policy_key in ("bayes_exact", "bayes_scaled"):
            entry = r.policy_results[policy_key]
            bd = entry["breakdown"]
            candidates.append(
                {
                    "model": r.name,
                    "policy_key": policy_key,
                    "policy": entry["policy"],
                    "param": entry["param"],
                    "oof_cost": bd.total_cost,
                    "oof_per_person": bd.cost_per_person,
                    "captured_oracle_pct": bd.captured_oracle_savings_pct,
                    "roc_auc": r.roc_auc,
                    "test_probs": r.test_probs,
                    "oof_probs": r.oof_probs,
                }
            )

    # Pick the best policy per model first, then rank models from 1 to 5
    best_per_model = {}
    for c in sorted(candidates, key=lambda x: x["oof_cost"]):
        if c["model"] not in best_per_model:
            best_per_model[c["model"]] = c

    ranked = sorted(best_per_model.values(), key=lambda x: x["oof_cost"])[:5]

    manifest_rows = []
    for order_idx, item in enumerate(ranked, start=1):
        test_preds = predict_with_cost_policy(
            item["test_probs"],
            p_unvacc_test,
            p_vacc_test,
            policy=item["policy"],
            threshold_param=item["param"],
            cfg=cfg,
        )
        oof_preds = predict_with_cost_policy(
            item["oof_probs"],
            p_unvacc_train,
            p_vacc_train,
            policy=item["policy"],
            threshold_param=item["param"],
            cfg=cfg,
        )
        oof_bd = compute_town_cost(y_true, oof_preds, p_unvacc_train, p_vacc_train, cfg=cfg)

        filename = f"challenge_submission_group_{group_id}_order_{order_idx}.csv"
        sub_df = pd.DataFrame(
            {
                cfg.id_col: feat_test[cfg.id_col].to_numpy(dtype=np.int64),
                cfg.target_col: test_preds.astype(np.int64),
            }
        )
        # Strict validation against example submission IDs and columns
        assert list(sub_df.columns) == list(example_sub.columns), "Submission column mismatch!"
        assert np.array_equal(
            sub_df[cfg.id_col].values, example_sub[cfg.id_col].values
        ), "Respondent IDs do not match example submission!"
        assert set(sub_df[cfg.target_col].unique()).issubset({0, 1}), "Non-binary predictions!"

        sub_df.to_csv(submissions_dir / filename, index=False)
        manifest_rows.append(
            {
                "order": order_idx,
                "filename": filename,
                "model": item["model"],
                "policy": item["policy_key"],
                "threshold_param": item["param"],
                "roc_auc": item["roc_auc"],
                "oof_total_cost": oof_bd.total_cost,
                "oof_cost_per_person": oof_bd.cost_per_person,
                "captured_oracle_pct": oof_bd.captured_oracle_savings_pct,
                "test_pred_positive_count": int(test_preds.sum()),
                "test_pred_positive_pct": float(test_preds.mean() * 100.0),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(submissions_dir / "submission_manifest.csv", index=False)
    return manifest_df
