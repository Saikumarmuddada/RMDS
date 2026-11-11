"""Leakage-free nested cross-fitted model training, probability calibration, stacking, and submission export.

Validation & Stacking Architecture:
1. Outer Stratified K-Fold CV (`folds`, K=5) partitions respondents into outer train
   (`tr_idx`) and untouched outer validation (`va_idx`) sets. Outer validation labels
   `y[va_idx]` are NEVER passed to any model fit, preprocessor, early-stopping
   callback, meta-learner, hyperparameter search, or calibration procedure.
2. For boosting models (LightGBM, XGBoost, CatBoost), early stopping uses an internal
   stratified validation split (`stratified 85%/15%`) carved strictly from the active
   training partition only. Preprocessing and categorical vocabularies are also fitted
   exclusively on the active fitting partition.
3. For stacking, each outer training fold `tr_idx` runs an independent inner Stratified
   K-Fold cross-fitting loop (`K_inner=4`). Base models + inner early stopping fit
   only on inner-train partitions, producing strictly uncontaminated inner OOF base
   logits `Z_inner` on `tr_idx`. Meta-learner regularization `C` and isolated post-hoc
   probability calibration (comparing uncalibrated sigmoid meta vs Platt vs Isotonic)
   are selected and fitted using held-out inner folds of `(Z_inner, y_tr)` before
   producing predictions for outer `va_idx` and `X_test`.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping
from scipy.special import logit
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import DATASET_C_CONFIG, DatasetConfig, N_FOLDS, RANDOM_SEED, SUBMISSIONS_DIR
from src.cost_model import (
    bayes_optimal_thresholds,
    compute_ece,
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    predict_with_cost_policy,
    tune_threshold_on_oof,
)
from src.data_prep import get_feature_columns

BASE_MODEL_NAMES = [
    "LogisticRegression",
    "RandomForest",
    "HistGradientBoosting",
    "LightGBM",
    "XGBoost",
    "CatBoost",
]


@dataclass
class ModelRunResult:
    """Container storing OOF/test probabilities and classification + cost metrics."""

    name: str
    oof_probs: np.ndarray
    test_probs: np.ndarray
    roc_auc: float
    log_loss_val: float
    brier: float
    ece: float
    accuracy_05: float
    f1_05: float
    policy_results: Dict[str, dict]
    scenario_costs: Dict[str, float]


def _coerce_sklearn_frame(
    df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
) -> pd.DataFrame:
    """Ensure clean numeric float64 and object string/np.nan categorical columns for sklearn."""
    out = pd.DataFrame(index=df.index)
    for col in num_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    for col in cat_cols:
        s = df[col].astype("string").str.strip()
        out[col] = s.astype(object).where(s.notna(), np.nan)
    return out


def _make_sklearn_onehot_preprocessor(
    num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def _make_sklearn_ordinal_preprocessor(
    num_cols: List[str], cat_cols: List[str]
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        (
                            "ordinal",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value", unknown_value=-1
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def apply_category_vocab(
    df: pd.DataFrame,
    cat_vocabs: Dict[str, pd.Index],
    *,
    for_catboost: bool = False,
) -> pd.DataFrame:
    """Apply frozen training-fold categorical vocabularies to any DataFrame cleanly."""
    out = df.copy()
    for col, cats in cat_vocabs.items():
        if col not in out.columns:
            continue
        s = out[col].astype("string").str.strip().fillna("Missing")
        s = s.where(s.isin(cats), "Missing")
        if for_catboost:
            out[col] = s.astype(str)
        else:
            out[col] = pd.Categorical(s, categories=cats)
    return out


def _prepare_category_frames(
    X_tr: pd.DataFrame,
    X_va: pd.DataFrame,
    X_te: Optional[pd.DataFrame],
    cat_cols: List[str],
    for_catboost: bool = False,
    *,
    X_es: Optional[pd.DataFrame] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Dict[str, pd.Index],
]:
    """Fit category vocabularies strictly on X_tr and apply safely without leakage."""
    cat_vocabs: Dict[str, pd.Index] = {}
    for col in cat_cols:
        tr_vals = X_tr[col].astype("string").str.strip().dropna().unique()
        cat_vocabs[col] = pd.Index(tr_vals).union(pd.Index(["Missing"])).sort_values()

    tr_out = apply_category_vocab(X_tr, cat_vocabs, for_catboost=for_catboost)
    va_out = apply_category_vocab(X_va, cat_vocabs, for_catboost=for_catboost)
    te_out = (
        apply_category_vocab(X_te, cat_vocabs, for_catboost=for_catboost)
        if X_te is not None
        else None
    )
    es_out = (
        apply_category_vocab(X_es, cat_vocabs, for_catboost=for_catboost)
        if X_es is not None
        else None
    )
    return tr_out, va_out, te_out, es_out, cat_vocabs


def fit_lgbm_with_early_stopping(
    model: LGBMClassifier,
    X_fit: pd.DataFrame,
    y_fit: np.ndarray,
    X_es: pd.DataFrame,
    y_es: np.ndarray,
    stopping_rounds: int,
) -> LGBMClassifier:
    """Fit LGBMClassifier with version-safe early stopping across LightGBM >=4.5 and 4.7+."""
    fit_params = inspect.signature(model.fit).parameters
    callbacks = [early_stopping(stopping_rounds=stopping_rounds, verbose=False)]
    if "eval_X" in fit_params and "eval_y" in fit_params:
        return model.fit(
            X_fit,
            y_fit,
            eval_X=X_es,
            eval_y=y_es,
            callbacks=callbacks,
        )
    return model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_es, y_es)],
        callbacks=callbacks,
    )


def _fit_base_models_on_train_only(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: Optional[pd.DataFrame],
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    *,
    fast_mode: bool = False,
) -> dict:
    """Fit all 6 base model families using ONLY (X_tr, y_tr).

    Notice y_va is NOT an argument to this function, structurally preventing any
    validation-fold label leakage into preprocessing, early stopping, or model training.
    """
    n_iter_lgb = 80 if fast_mode else 1800
    n_iter_xgb = 80 if fast_mode else 1800
    n_iter_cb = 60 if fast_mode else 1200
    es_rounds_lgb = 15 if fast_mode else 80
    es_rounds_cb = 15 if fast_mode else 60
    rf_trees = 40 if fast_mode else 500
    hgb_iter = 60 if fast_mode else 350
    job_count = 1 if fast_mode else -1

    # Prepare sklearn-safe frame
    X_tr_sk = _coerce_sklearn_frame(X_tr, num_cols, cat_cols)
    X_va_sk = _coerce_sklearn_frame(X_va, num_cols, cat_cols)
    X_te_sk = _coerce_sklearn_frame(X_te, num_cols, cat_cols) if X_te is not None else None

    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, Optional[np.ndarray]] = {}

    # 1. Logistic Regression
    lr_pipe = Pipeline(
        [
            ("prep", _make_sklearn_onehot_preprocessor(num_cols, cat_cols)),
            ("clf", LogisticRegression(C=0.25, max_iter=1500, random_state=seed)),
        ]
    )
    lr_pipe.fit(X_tr_sk, y_tr)
    val_preds["LogisticRegression"] = lr_pipe.predict_proba(X_va_sk)[:, 1]
    test_preds["LogisticRegression"] = (
        lr_pipe.predict_proba(X_te_sk)[:, 1] if X_te_sk is not None else None
    )

    # 2. Random Forest
    rf_pipe = Pipeline(
        [
            ("prep", _make_sklearn_ordinal_preprocessor(num_cols, cat_cols)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=rf_trees,
                    max_depth=14,
                    min_samples_leaf=8,
                    max_features="sqrt",
                    n_jobs=job_count,
                    random_state=seed,
                ),
            ),
        ]
    )
    rf_pipe.fit(X_tr_sk, y_tr)
    val_preds["RandomForest"] = rf_pipe.predict_proba(X_va_sk)[:, 1]
    test_preds["RandomForest"] = (
        rf_pipe.predict_proba(X_te_sk)[:, 1] if X_te_sk is not None else None
    )

    # 3. HistGradientBoosting
    cat_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    hgb_pipe = Pipeline(
        [
            ("prep", _make_sklearn_ordinal_preprocessor(num_cols, cat_cols)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=hgb_iter,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.5,
                    categorical_features=cat_indices if cat_indices else None,
                    random_state=seed,
                ),
            ),
        ]
    )
    hgb_pipe.fit(X_tr_sk, y_tr)
    val_preds["HistGradientBoosting"] = hgb_pipe.predict_proba(X_va_sk)[:, 1]
    test_preds["HistGradientBoosting"] = (
        hgb_pipe.predict_proba(X_te_sk)[:, 1] if X_te_sk is not None else None
    )

    # Create STRICT internal early-stopping validation split inside (X_tr, y_tr) ONLY
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    boost_fit_sub, boost_es_sub = next(sss.split(X_tr, y_tr))
    X_bfit, y_bfit = X_tr.iloc[boost_fit_sub], y_tr[boost_fit_sub]
    X_bes, y_bes = X_tr.iloc[boost_es_sub], y_tr[boost_es_sub]

    # 4. LightGBM & 5. XGBoost (vocabularies fitted strictly on X_bfit)
    X_fit_cat, X_va_cat, X_te_cat, X_es_cat, boost_vocabs = _prepare_category_frames(
        X_bfit, X_va, X_te, cat_cols, for_catboost=False, X_es=X_bes
    )

    lgb_model = LGBMClassifier(
        n_estimators=n_iter_lgb,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=job_count,
        verbose=-1,
    )
    fit_lgbm_with_early_stopping(
        lgb_model,
        X_fit_cat,
        y_bfit,
        X_es_cat,
        y_bes,
        stopping_rounds=es_rounds_lgb,
    )
    val_preds["LightGBM"] = lgb_model.predict_proba(X_va_cat)[:, 1]
    test_preds["LightGBM"] = (
        lgb_model.predict_proba(X_te_cat)[:, 1] if X_te_cat is not None else None
    )

    xgb_model = XGBClassifier(
        n_estimators=n_iter_xgb,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=es_rounds_lgb,
        random_state=seed,
        n_jobs=job_count,
    )
    xgb_model.fit(X_fit_cat, y_bfit, eval_set=[(X_es_cat, y_bes)], verbose=False)
    val_preds["XGBoost"] = xgb_model.predict_proba(X_va_cat)[:, 1]
    test_preds["XGBoost"] = (
        xgb_model.predict_proba(X_te_cat)[:, 1] if X_te_cat is not None else None
    )

    # 6. CatBoost (native categorical text features, vocab fitted on X_bfit)
    X_fit_cb, X_va_cb, X_te_cb, X_es_cb, _ = _prepare_category_frames(
        X_bfit, X_va, X_te, cat_cols, for_catboost=True, X_es=X_bes
    )
    cb_model = CatBoostClassifier(
        iterations=n_iter_cb,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=5.0,
        eval_metric="Logloss",
        random_seed=seed,
        thread_count=job_count,
        allow_writing_files=False,
        verbose=False,
    )
    cb_model.fit(
        X_fit_cb,
        y_bfit,
        cat_features=cat_cols,
        eval_set=(X_es_cb, y_bes),
        early_stopping_rounds=es_rounds_cb,
        verbose=False,
    )
    val_preds["CatBoost"] = cb_model.predict_proba(X_va_cb)[:, 1]
    test_preds["CatBoost"] = (
        cb_model.predict_proba(X_te_cb)[:, 1] if X_te_cb is not None else None
    )

    return {
        "val_preds": val_preds,
        "test_preds": test_preds,
        "lgbm_model": lgb_model,
        "catboost_model": cb_model,
        "boost_vocabs": boost_vocabs,
        "es_split_sizes": {"fit": len(X_bfit), "es": len(X_bes)},
    }


def _apply_calibrator(calibrator, mode: str, raw_probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if mode == "none" or calibrator is None:
        return p
    if mode == "platt":
        z = logit(p).reshape(-1, 1)
        return np.clip(calibrator.predict_proba(z)[:, 1], 1e-6, 1.0 - 1e-6)
    if mode == "isotonic":
        return np.clip(calibrator.predict(p), 1e-6, 1.0 - 1e-6)
    raise ValueError(f"Unsupported calibration mode: {mode}")


def _fit_calibrator(mode: str, raw_probs: np.ndarray, y: np.ndarray, seed: int):
    p = np.clip(np.asarray(raw_probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if mode == "none":
        return None
    if mode == "platt":
        lr = LogisticRegression(C=10.0, max_iter=500, random_state=seed)
        lr.fit(logit(p).reshape(-1, 1), y)
        return lr
    if mode == "isotonic":
        iso = IsotonicRegression(y_min=1e-5, y_max=1.0 - 1e-5, out_of_bounds="clip")
        iso.fit(p, y)
        return iso
    raise ValueError(f"Unsupported calibration mode: {mode}")


def _select_best_meta_c(
    Z: np.ndarray,
    y: np.ndarray,
    cv_ids: np.ndarray,
    seed: int,
    c_grid: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
) -> Tuple[float, np.ndarray]:
    """Select LogisticRegression meta regularization C strictly out-of-fold over (Z, y)."""
    unique_folds = np.unique(cv_ids)
    best_c, best_c_ll = float(c_grid[0]), float("inf")
    oof_by_c: Dict[float, np.ndarray] = {}
    for c_val in c_grid:
        c_oof = np.zeros(len(y), dtype=np.float64)
        for fid in unique_folds:
            tr_m = cv_ids != fid
            va_m = cv_ids == fid
            m_clf = LogisticRegression(C=float(c_val), max_iter=1000, random_state=seed)
            m_clf.fit(Z[tr_m], y[tr_m])
            c_oof[va_m] = m_clf.predict_proba(Z[va_m])[:, 1]
        oof_by_c[float(c_val)] = c_oof
        ll = float(log_loss(y, c_oof))
        if ll < best_c_ll:
            best_c, best_c_ll = float(c_val), ll
    return best_c, oof_by_c[best_c]


def evaluate_calibration_nested(
    Z_inner: np.ndarray,
    y_tr: np.ndarray,
    inner_fold_ids: np.ndarray,
    seed: int,
    c_grid: Tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0),
    *,
    min_improvement: float = 1e-4,
) -> dict:
    """Nested calibration evaluation where meta-C and calibrator fit NEVER see cal_va labels.

    For each inner fold j:
      1. Select c_j and out-of-fold sub-meta scores strictly inside cal_tr (inner_fold_ids != j).
      2. Fit LogisticRegression(C=c_j) on cal_tr and predict untouched cal_va -> nested_raw_oof[cal_va].
      3. Fit each candidate calibrator (none, platt, isotonic) on (sub_meta_tr, y_tr[cal_tr])
         and score strictly out-of-sample on (nested_raw_oof[cal_va], y_tr[cal_va]).
    """
    unique_folds = np.unique(inner_fold_ids)
    cal_modes = ("none", "platt", "isotonic")
    nested_raw_oof = np.zeros(len(y_tr), dtype=np.float64)
    cal_oof_by_mode = {m: np.zeros(len(y_tr), dtype=np.float64) for m in cal_modes}
    nested_c_per_fold: List[float] = []

    for j in unique_folds:
        cal_tr = inner_fold_ids != j
        cal_va = inner_fold_ids == j
        Z_tr_j, y_tr_j = Z_inner[cal_tr], y_tr[cal_tr]
        tr_sub_ids = inner_fold_ids[cal_tr]
        if len(np.unique(tr_sub_ids)) < 2:
            skf_sub = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed + int(j))
            tr_sub_ids = np.zeros(len(y_tr_j), dtype=int)
            for s_idx, (_, s_va) in enumerate(skf_sub.split(Z_tr_j, y_tr_j)):
                tr_sub_ids[s_va] = s_idx

        c_j, sub_meta_oof_tr = _select_best_meta_c(
            Z_tr_j, y_tr_j, tr_sub_ids, seed + int(j), c_grid
        )
        nested_c_per_fold.append(c_j)

        meta_j = LogisticRegression(C=c_j, max_iter=1000, random_state=seed + int(j))
        meta_j.fit(Z_tr_j, y_tr_j)
        raw_va_j = meta_j.predict_proba(Z_inner[cal_va])[:, 1]
        nested_raw_oof[cal_va] = raw_va_j

        for mode in cal_modes:
            cal_j = _fit_calibrator(mode, sub_meta_oof_tr, y_tr_j, seed + int(j))
            cal_oof_by_mode[mode][cal_va] = _apply_calibrator(cal_j, mode, raw_va_j)

    cal_eval = {m: float(log_loss(y_tr, cal_oof_by_mode[m])) for m in cal_modes}
    best_cal_mode = "none"
    best_cal_ll = cal_eval["none"]
    for mode in ("platt", "isotonic"):
        if cal_eval[mode] < best_cal_ll - min_improvement:
            best_cal_ll = cal_eval[mode]
            best_cal_mode = mode

    final_calibrator = _fit_calibrator(best_cal_mode, nested_raw_oof, y_tr, seed)
    reason = (
        f"Identity calibration retained: nested OOF LogLoss none={cal_eval['none']:.5f} <= "
        f"platt={cal_eval['platt']:.5f}, isotonic={cal_eval['isotonic']:.5f}"
        if best_cal_mode == "none"
        else f"Selected {best_cal_mode}: improved nested OOF LogLoss from {cal_eval['none']:.5f} to {best_cal_ll:.5f}"
    )
    return {
        "best_cal_mode": best_cal_mode,
        "final_calibrator": final_calibrator,
        "cal_eval": cal_eval,
        "nested_raw_oof": nested_raw_oof,
        "nested_c_per_fold": nested_c_per_fold,
        "reason": reason,
    }


def select_and_fit_posthoc_calibration(
    meta_inner_oof: np.ndarray,
    y_tr: np.ndarray,
    inner_fold_ids: np.ndarray,
    seed: int,
    *,
    min_improvement: float = 1e-4,
) -> Tuple[str, object, Dict[str, float]]:
    """Select calibration mode out-of-sample on fixed inner OOF scores and fit on y_tr."""
    cal_modes = ("none", "platt", "isotonic")
    cal_eval: Dict[str, float] = {}
    unique_folds = np.unique(inner_fold_ids)

    for mode in cal_modes:
        cal_oof = np.zeros(len(y_tr), dtype=np.float64)
        for j in unique_folds:
            tr_m = inner_fold_ids != j
            va_m = inner_fold_ids == j
            cal_j = _fit_calibrator(mode, meta_inner_oof[tr_m], y_tr[tr_m], seed + int(j))
            cal_oof[va_m] = _apply_calibrator(cal_j, mode, meta_inner_oof[va_m])
        cal_eval[mode] = float(log_loss(y_tr, cal_oof))

    best_cal_mode = "none"
    best_cal_ll = cal_eval["none"]
    for mode in ("platt", "isotonic"):
        if cal_eval[mode] < best_cal_ll - min_improvement:
            best_cal_ll = cal_eval[mode]
            best_cal_mode = mode

    final_calibrator = _fit_calibrator(best_cal_mode, meta_inner_oof, y_tr, seed)
    return best_cal_mode, final_calibrator, cal_eval


def fit_outer_fold_models(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    X_te: Optional[pd.DataFrame],
    num_cols: List[str],
    cat_cols: List[str],
    outer_fold_idx: int,
    *,
    inner_folds: int = 4,
    seed: int = RANDOM_SEED,
    fast_mode: bool = False,
) -> dict:
    """Fit all base models + nested cross-fitted stack + isolated calibration on ONE outer fold.

    Because y_va is NOT passed into this function, outer validation observations remain
    100% isolated from preprocessing, early stopping, inner stacking, C tuning, and calibration.
    """
    fold_seed = seed + outer_fold_idx

    # 1. Fit outer base models strictly on (X_tr, y_tr) -> predict untouched X_va & X_te
    outer_base = _fit_base_models_on_train_only(
        X_tr, y_tr, X_va, X_te, num_cols, cat_cols, fold_seed, fast_mode=fast_mode
    )

    # 2. Inner K-fold cross-fitting strictly inside (X_tr, y_tr) to construct uncontaminated Z_inner
    skf_inner = StratifiedKFold(
        n_splits=inner_folds, shuffle=True, random_state=seed + 100 + outer_fold_idx
    )
    inner_fold_ids = np.empty(len(y_tr), dtype=int)
    inner_oof_probs = np.zeros((len(y_tr), len(BASE_MODEL_NAMES)), dtype=np.float64)

    for j, (in_tr_idx, in_va_idx) in enumerate(skf_inner.split(X_tr, y_tr)):
        inner_fold_ids[in_va_idx] = j
        in_res = _fit_base_models_on_train_only(
            X_tr.iloc[in_tr_idx],
            y_tr[in_tr_idx],
            X_tr.iloc[in_va_idx],
            None,
            num_cols,
            cat_cols,
            fold_seed * 10 + j,
            fast_mode=fast_mode,
        )
        for m_idx, m_name in enumerate(BASE_MODEL_NAMES):
            inner_oof_probs[in_va_idx, m_idx] = in_res["val_preds"][m_name]

    Z_inner = logit(np.clip(inner_oof_probs, 1e-5, 1.0 - 1e-5))
    Z_va_outer = np.column_stack(
        [
            logit(np.clip(outer_base["val_preds"][m], 1e-5, 1.0 - 1e-5))
            for m in BASE_MODEL_NAMES
        ]
    )
    Z_te_outer = (
        np.column_stack(
            [
                logit(np.clip(outer_base["test_preds"][m], 1e-5, 1.0 - 1e-5))
                for m in BASE_MODEL_NAMES
            ]
        )
        if X_te is not None
        else None
    )

    # 3. Nested calibration evaluation (c_j tuned strictly inside cal_tr, zero cal_va leakage)
    c_grid = (0.05, 0.1, 0.25, 0.5, 1.0)
    cal_res = evaluate_calibration_nested(Z_inner, y_tr, inner_fold_ids, fold_seed, c_grid=c_grid)
    best_cal_mode = cal_res["best_cal_mode"]
    final_calibrator = cal_res["final_calibrator"]
    cal_eval = cal_res["cal_eval"]

    # 4. Select final outer-fold meta-C strictly inside complete clean Z_inner -> fit meta_full
    best_c, _ = _select_best_meta_c(Z_inner, y_tr, inner_fold_ids, fold_seed, c_grid=c_grid)
    meta_full = LogisticRegression(C=best_c, max_iter=1000, random_state=fold_seed)
    meta_full.fit(Z_inner, y_tr)
    raw_stack_va = meta_full.predict_proba(Z_va_outer)[:, 1]
    raw_stack_te = (
        meta_full.predict_proba(Z_te_outer)[:, 1] if Z_te_outer is not None else None
    )

    cal_stack_va = _apply_calibrator(final_calibrator, best_cal_mode, raw_stack_va)
    cal_stack_te = (
        _apply_calibrator(final_calibrator, best_cal_mode, raw_stack_te)
        if raw_stack_te is not None
        else None
    )

    return {
        "outer_base": outer_base,
        "val_preds": outer_base["val_preds"],
        "test_preds": outer_base["test_preds"],
        "raw_stack_val": raw_stack_va,
        "raw_stack_test": raw_stack_te,
        "cal_stack_val": cal_stack_va,
        "cal_stack_test": cal_stack_te,
        "meta_coefs": meta_full.coef_[0].copy(),
        "best_meta_c": best_c,
        "nested_c_per_fold": cal_res["nested_c_per_fold"],
        "selected_calibrator": best_cal_mode,
        "cal_selection_reason": cal_res["reason"],
        "inner_cal_logloss": cal_eval,
        "inner_oof_size": len(y_tr),
    }


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

    scenario_costs = {}
    for scen in ("conservative", "base", "severe"):
        pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario=scen)  # type: ignore[arg-type]
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
        ece=float(compute_ece(y_true, oof_probs, n_bins=10)),
        accuracy_05=float(accuracy_score(y_true, pred_05)),
        f1_05=float(f1_score(y_true, pred_05)),
        policy_results=policy_results,
        scenario_costs=scenario_costs,
    )


def train_and_evaluate_all_models(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    output_dir: Path = Path("outputs"),
    *,
    n_folds: int = N_FOLDS,
    inner_folds: int = 4,
    fast_mode: bool = False,
) -> Tuple[List[ModelRunResult], pd.DataFrame, np.ndarray, dict]:
    """Run leakage-free nested Stratified CV across 6 base models + nested cross-fitted stacking."""
    output_dir.mkdir(parents=True, exist_ok=True)
    num_cols, cat_cols = get_feature_columns(feat_train, cfg=cfg)
    feature_cols = num_cols + cat_cols

    X = feat_train[feature_cols].copy()
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    X_test = feat_test[feature_cols].copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    folds = np.zeros(len(y), dtype=np.int64)

    oof_dict: Dict[str, np.ndarray] = {m: np.zeros(len(y), dtype=np.float64) for m in BASE_MODEL_NAMES}
    test_dict: Dict[str, np.ndarray] = {m: np.zeros(len(X_test), dtype=np.float64) for m in BASE_MODEL_NAMES}
    oof_dict["Uncalibrated_Stack"] = np.zeros(len(y), dtype=np.float64)
    test_dict["Uncalibrated_Stack"] = np.zeros(len(X_test), dtype=np.float64)
    oof_dict["Calibrated_Stack"] = np.zeros(len(y), dtype=np.float64)
    test_dict["Calibrated_Stack"] = np.zeros(len(X_test), dtype=np.float64)

    best_lgbm_models = []
    best_cb_models = []
    lgbm_fold_vocabs = []
    meta_coefs = []
    fold_calibration_audit = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        folds[va_idx] = fold_idx
        fold_res = fit_outer_fold_models(
            X.iloc[tr_idx],
            y[tr_idx],
            X.iloc[va_idx],
            X_test,
            num_cols,
            cat_cols,
            outer_fold_idx=fold_idx,
            inner_folds=inner_folds,
            seed=RANDOM_SEED,
            fast_mode=fast_mode,
        )

        for m_name in BASE_MODEL_NAMES:
            oof_dict[m_name][va_idx] = fold_res["val_preds"][m_name]
            test_dict[m_name] += fold_res["test_preds"][m_name] / n_folds

        oof_dict["Uncalibrated_Stack"][va_idx] = fold_res["raw_stack_val"]
        test_dict["Uncalibrated_Stack"] += fold_res["raw_stack_test"] / n_folds
        oof_dict["Calibrated_Stack"][va_idx] = fold_res["cal_stack_val"]
        test_dict["Calibrated_Stack"] += fold_res["cal_stack_test"] / n_folds

        best_lgbm_models.append(fold_res["outer_base"]["lgbm_model"])
        best_cb_models.append(fold_res["outer_base"]["catboost_model"])
        lgbm_fold_vocabs.append(fold_res["outer_base"]["boost_vocabs"])
        meta_coefs.append(fold_res["meta_coefs"])
        fold_calibration_audit.append(
            {
                "outer_fold": fold_idx,
                "best_meta_c": fold_res["best_meta_c"],
                "nested_cal_c_per_fold": ",".join(str(c) for c in fold_res["nested_c_per_fold"]),
                "selected_calibrator": fold_res["selected_calibrator"],
                "inner_oof_logloss_none": fold_res["inner_cal_logloss"]["none"],
                "inner_oof_logloss_platt": fold_res["inner_cal_logloss"]["platt"],
                "inner_oof_logloss_isotonic": fold_res["inner_cal_logloss"]["isotonic"],
                "es_fit_rows": fold_res["outer_base"]["es_split_sizes"]["fit"],
                "es_val_rows": fold_res["outer_base"]["es_split_sizes"]["es"],
                "cal_selection_reason": fold_res["cal_selection_reason"],
            }
        )

    all_model_names = BASE_MODEL_NAMES + ["Uncalibrated_Stack", "Calibrated_Stack"]
    results = [
        _evaluate_model_predictions(
            m, y, oof_dict[m], test_dict[m], folds, feat_train, cfg
        )
        for m in all_model_names
    ]

    rows = []
    for r in results:
        g_05 = r.policy_results["global_0.5"]["breakdown"]
        g_tuned = r.policy_results["global_tuned"]["breakdown"]
        b_exact = r.policy_results["bayes_exact"]["breakdown"]
        b_scaled = r.policy_results["bayes_scaled"]["breakdown"]
        rows.append(
            {
                "model": r.name,
                "roc_auc": r.roc_auc,
                "log_loss": r.log_loss_val,
                "brier": r.brier,
                "ece": r.ece,
                "acc_at_0.5": r.accuracy_05,
                "f1_at_0.5": r.f1_05,
                "cost_global_0.5": g_05.total_cost,
                "cost_global_tuned": g_tuned.total_cost,
                "global_tuned_thr": r.policy_results["global_tuned"]["param"],
                "cost_bayes_exact": b_exact.total_cost,
                "cost_bayes_scaled": b_scaled.total_cost,
                "bayes_scale_param": r.policy_results["bayes_scaled"]["param"],
                "bayes_exact_per_person": b_exact.cost_per_person,
                "vaccines_allocated": b_exact.vaccines_allocated,
                "vaccines_accepted": b_exact.vaccines_accepted,
                "wasted_vaccines": b_exact.wasted_vaccines,
                "law_compensation_count": b_exact.law_compensation_count,
                "vaccine_send_cost": b_exact.vaccine_send_cost,
                "wasted_vaccine_cost": b_exact.wasted_vaccine_cost,
                "law_penalty_cost": b_exact.law_penalty_cost,
                "expected_hospitalisations": b_exact.expected_hospitalisations,
                "hospitalisation_cost": b_exact.hospitalisation_cost,
                "savings_vs_treat_none": b_exact.savings_vs_treat_none,
                "savings_vs_treat_all": b_exact.savings_vs_treat_all,
                "captured_oracle_pct": b_exact.captured_oracle_savings_pct,
                "pred_vacc_rate_bayes": b_exact.predicted_positive_rate,
                "fixed05_vaccines_allocated": g_05.vaccines_allocated,
                "fixed05_expected_hospitalisations": g_05.expected_hospitalisations,
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

    cal_audit_df = pd.DataFrame(fold_calibration_audit)
    cal_audit_df.to_csv(output_dir / "stacking_calibration_audit.csv", index=False)

    artifacts = {
        "folds": folds,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "lgbm_models": best_lgbm_models,
        "lgbm_fold_vocabs": lgbm_fold_vocabs,
        "catboost_models": best_cb_models,
        "meta_weights": dict(zip(BASE_MODEL_NAMES, np.mean(meta_coefs, axis=0).tolist())),
        "meta_coefs_per_fold": np.vstack(meta_coefs).tolist(),
        "calibration_audit": fold_calibration_audit,
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
    """Rank candidate model+policy pipelines by CV town cost and export top 5 diverse submissions."""
    submissions_dir.mkdir(parents=True, exist_ok=True)
    p_unvacc_test, p_vacc_test = estimate_hospitalisation_probabilities(feat_test, scenario="base")
    p_unvacc_train, p_vacc_train = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    y_true = feat_train[cfg.target_col].to_numpy(dtype=np.int64)

    candidates = []
    for r in results:
        for policy_key in ("bayes_exact", "bayes_scaled"):
            entry = r.policy_results[policy_key]
            bd = entry["breakdown"]
            candidates.append(
                {
                    "model": r.name,
                    "diversity_family": "StackEnsemble" if r.name.endswith("_Stack") else r.name,
                    "policy_key": policy_key,
                    "policy": entry["policy"],
                    "param": entry["param"],
                    "oof_cost": bd.total_cost,
                    "oof_per_person": bd.cost_per_person,
                    "captured_oracle_pct": bd.captured_oracle_savings_pct,
                    "roc_auc": r.roc_auc,
                    "log_loss": r.log_loss_val,
                    "ece": r.ece,
                    "test_probs": r.test_probs,
                    "oof_probs": r.oof_probs,
                }
            )

    # Keep the best-cost policy per model family (preferring Calibrated_Stack on exact Stack ties)
    best_per_family = {}
    sorted_candidates = sorted(
        candidates,
        key=lambda x: (x["oof_cost"], -int(x["model"] == "Calibrated_Stack"), x["log_loss"]),
    )
    for c in sorted_candidates:
        fam = c["diversity_family"]
        if fam not in best_per_family:
            best_per_family[fam] = c

    ranked = sorted(
        best_per_family.values(),
        key=lambda x: (x["oof_cost"], -int(x["model"] == "Calibrated_Stack"), x["log_loss"]),
    )[:5]

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
        _ = compute_town_cost(y_true, oof_preds, p_unvacc_train, p_vacc_train, cfg=cfg)

        filename = f"challenge_submission_group_{group_id}_order_{order_idx}.csv"
        sub_df = pd.DataFrame(
            {
                cfg.id_col: feat_test[cfg.id_col].to_numpy(dtype=np.int64),
                cfg.target_col: test_preds.astype(np.int64),
            }
        )
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
                "log_loss": item["log_loss"],
                "ece": item["ece"],
                "oof_total_cost": item["oof_cost"],
                "oof_cost_per_person": item["oof_per_person"],
                "captured_oracle_pct": item["captured_oracle_pct"],
                "test_pred_positive_count": int(test_preds.sum()),
                "test_pred_positive_pct": float(test_preds.mean() * 100.0),
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(submissions_dir / "submission_manifest.csv", index=False)
    return manifest_df
