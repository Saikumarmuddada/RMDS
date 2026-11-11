"""Data loading, cleaning, and feature engineering for Dataset C."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.config import (
    AGE_ORDINAL_MAP,
    BEHAVIORAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATASET_C_CONFIG,
    DatasetConfig,
    EDUCATION_ORDINAL_MAP,
    INCOME_ORDINAL_MAP,
    OTHER_BINARY_FEATURES,
)


def load_raw_datasets(
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load training, testing, and example submission DataFrames."""
    train_df = pd.read_csv(cfg.train_path)
    test_df = pd.read_csv(cfg.test_path)
    example_sub = pd.read_csv(cfg.example_submission_path)
    return train_df, test_df, example_sub


def _normalize_whitespace(series: pd.Series) -> pd.Series:
    """Collapse repeated whitespace in non-null categorical strings."""
    return series.map(
        lambda val: re.sub(r"\s+", " ", val.strip()) if isinstance(val, str) else val
    )


def engineer_features(
    df: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> pd.DataFrame:
    """Clean raw quirks and add domain-driven features without target leakage."""
    out = df.copy()
    p = cfg.disease_prefix

    # Normalize string whitespace (e.g., "MSA, Not Principle  City")
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = _normalize_whitespace(out[col])

    # Defensive numeric coercion across numeric/ordinal survey inputs
    numeric_raw_candidates = [
        f"{p}_concern",
        f"{p}_knowledge",
        *BEHAVIORAL_FEATURES,
        f"doctor_recc_{p}",
        *OTHER_BINARY_FEATURES,
        f"opinion_{p}_vacc_effective",
        f"opinion_{p}_risk",
        f"opinion_{p}_sick_from_vacc",
        "household_adults",
        "household_children",
        "number_chronic_med_conditions",
    ]
    for col in numeric_raw_candidates:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Explicit missingness flags before imputation
    for col in ("health_insurance", "employment_sector", "income_poverty", "rent_or_own"):
        if col in out.columns:
            out[f"{col}_is_missing"] = out[col].isna().astype(np.float32)

    raw_feature_cols = [
        c for c in out.columns if c not in {cfg.id_col, cfg.target_col}
    ]
    out["row_missing_count"] = out[raw_feature_cols].isna().sum(axis=1).astype(np.float32)

    # Undocumented 0 values in 1-5 opinion scales represent unknown/unanswered:
    # Empirically in training data, 0 on opinion_covid_vacc_effective has 25.0% acceptance
    # (between rank 3 [16.2%] and rank 4 [30.5%], vs rank 1 [3.2%]), and 0 on
    # opinion_covid_risk has 27.1% acceptance (vs rank 1 [12.5%]), matching missing NaNs
    # on opinion_covid_sick_from_vacc (27.8%). Map 0 -> NaN and track explicit binary flags.
    eff_col = f"opinion_{p}_vacc_effective"
    risk_col = f"opinion_{p}_risk"
    worry_col = f"opinion_{p}_sick_from_vacc"
    recc_col = f"doctor_recc_{p}"

    for col in (eff_col, risk_col):
        if col in out.columns:
            out[f"{col}_is_zero"] = (out[col] == 0).astype(np.float32)
            out[col] = out[col].replace(0, np.nan)

    # Ordered demographic encodings (unseen labels safely map to NaN)
    out["age_ordinal"] = out["age_group"].map(AGE_ORDINAL_MAP).astype(np.float32)
    out["education_ordinal"] = out["education"].map(EDUCATION_ORDINAL_MAP).astype(np.float32)
    out["income_ordinal"] = out["income_poverty"].map(INCOME_ORDINAL_MAP).astype(np.float32)

    # Behavioral aggregates
    beh_cols = [c for c in BEHAVIORAL_FEATURES if c in out.columns]
    out["behavioral_sum"] = out[beh_cols].sum(axis=1, min_count=1).astype(np.float32)
    out["behavioral_mean"] = out[beh_cols].mean(axis=1).astype(np.float32)

    # Note: Collinear attitude composites (opinion_net_benefit, opinion_risk_minus_worry,
    # opinion_pro_vaccine_index, opinion_eff_x_risk, concern_x_knowledge) were pruned after
    # CV ablation showed they degraded both LogisticRegression LogLoss (0.4226 -> 0.4218 when
    # dropped) and LightGBM LogLoss/ROC-AUC (0.4248/0.8616 -> 0.4221/0.8636 when dropped).

    # Doctor recommendation interactions (strongest clinical cue).
    # LIKERT_NEUTRAL_MIDPOINT (3.0 on deterministic 1-5 scale, zero data leakage)
    # preserves distinction between doctor_recc==1 with missing opinion vs doctor_recc==0.
    likert_neutral_midpoint = 3.0
    out["doctor_x_effective"] = (
        out[recc_col] * out[eff_col].fillna(likert_neutral_midpoint)
    ).astype(np.float32)
    out["doctor_x_risk"] = (
        out[recc_col] * out[risk_col].fillna(likert_neutral_midpoint)
    ).astype(np.float32)
    out["doctor_x_not_worried"] = (
        out[recc_col] * (6.0 - out[worry_col].fillna(likert_neutral_midpoint))
    ).astype(np.float32)

    # Household & vulnerability indicators
    out["household_total"] = (
        1.0 + out["household_adults"].fillna(0.0) + out["household_children"].fillna(0.0)
    ).astype(np.float32)
    out["has_chronic_condition"] = (
        out["number_chronic_med_conditions"].fillna(0.0) > 0
    ).astype(np.float32)
    out["high_risk_elderly"] = (
        (out["age_ordinal"] >= 3.0) & (out["number_chronic_med_conditions"].fillna(0.0) >= 1)
    ).astype(np.float32)
    out["is_healthcare_sector"] = (
        out["employment_sector"] == "healthcare"
    ).astype(np.float32)
    out["uninsured_flag"] = (out["health_insurance"] == 0.0).astype(np.float32)

    return out


def get_feature_columns(
    df: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> Tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) excluding id and target."""
    exclude = {cfg.id_col, cfg.target_col}
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_cols = [
        c for c in df.columns if c not in exclude and c not in set(cat_cols)
    ]
    return num_cols, cat_cols


def save_cleaned_summary(train_df: pd.DataFrame, test_df: pd.DataFrame, out_path: Path) -> None:
    """Write a concise dataset verification summary JSON."""
    import json

    summary = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "target_positive_rate": float(train_df[DATASET_C_CONFIG.target_col].mean()),
        "binary_features": BEHAVIORAL_FEATURES + OTHER_BINARY_FEATURES,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def audit_dataset_preprocessing(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> dict:
    """Verify data integrity, unseen categories, zero-opinion sentinels, and zero leakage."""
    raw_feat_cols = [c for c in raw_test.columns if c != cfg.id_col]
    num_cols, cat_cols = get_feature_columns(feat_train, cfg)

    train_ids = set(raw_train[cfg.id_col].astype(int))
    test_ids = set(raw_test[cfg.id_col].astype(int))

    unseen_test_cats: dict[str, list[str]] = {}
    for col in cat_cols:
        tr_vals = set(feat_train[col].dropna().astype(str).unique())
        te_vals = set(feat_test[col].dropna().astype(str).unique())
        unseen = sorted(te_vals - tr_vals)
        if unseen:
            unseen_test_cats[col] = unseen

    zero_audit = {}
    p = cfg.disease_prefix
    for oc in (f"opinion_{p}_vacc_effective", f"opinion_{p}_risk"):
        if oc in raw_train.columns:
            zero_mask_tr = raw_train[oc] == 0
            one_mask_tr = raw_train[oc] == 1
            zero_audit[oc] = {
                "train_zero_count": int(zero_mask_tr.sum()),
                "test_zero_count": int((raw_test[oc] == 0).sum()) if oc in raw_test.columns else 0,
                "train_zero_target_rate": float(raw_train.loc[zero_mask_tr, cfg.target_col].mean())
                if zero_mask_tr.any()
                else None,
                "train_rank1_target_rate": float(raw_train.loc[one_mask_tr, cfg.target_col].mean())
                if one_mask_tr.any()
                else None,
            }

    return {
        "train_rows": int(len(raw_train)),
        "test_rows": int(len(raw_test)),
        "train_exact_duplicates": int(raw_train.duplicated().sum()),
        "train_feature_duplicates": int(raw_train.duplicated(subset=raw_feat_cols).sum()),
        "test_feature_duplicates": int(raw_test.duplicated(subset=raw_feat_cols).sum()),
        "id_overlap_count": int(len(train_ids & test_ids)),
        "id_in_features": bool(cfg.id_col in num_cols or cfg.id_col in cat_cols),
        "target_in_features": bool(cfg.target_col in num_cols or cfg.target_col in cat_cols),
        "numeric_feature_count": int(len(num_cols)),
        "categorical_feature_count": int(len(cat_cols)),
        "unseen_test_categories": unseen_test_cats,
        "undocumented_zero_audit": {
            "policy_basis": "Empirical distribution decision (not stated in official PDF/image docs): 0 on opinion_covid_vacc_effective (25.0% acceptance) and opinion_covid_risk (27.1%) aligns with rank-3/missingness rather than rank-1 (3.2%/12.5%); mapped 0 -> NaN with explicit _is_zero indicators.",
            "features": zero_audit,
        },
        "train_test_feature_column_match": bool(
            get_feature_columns(feat_train, cfg) == get_feature_columns(feat_test, cfg)
        ),
    }


def run_feature_ablation(
    raw_train: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate feature groups via leakage-free CV (LogisticRegression & inner-ES LightGBM)."""
    from lightgbm import LGBMClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from src.modeling import fit_lgbm_with_early_stopping

    y = raw_train[cfg.target_col].to_numpy(dtype=int)

    raw_base = raw_train.drop(columns=[cfg.id_col, cfg.target_col]).copy()
    for col in CATEGORICAL_FEATURES:
        if col in raw_base.columns:
            raw_base[col] = _normalize_whitespace(raw_base[col])

    p = cfg.disease_prefix
    raw_zero_clean = raw_base.copy()
    for oc in (f"opinion_{p}_vacc_effective", f"opinion_{p}_risk"):
        if oc in raw_zero_clean.columns:
            raw_zero_clean[f"{oc}_is_zero"] = (raw_zero_clean[oc] == 0).astype(np.float32)
            raw_zero_clean[oc] = raw_zero_clean[oc].replace(0, np.nan)

    fe_full = engineer_features(raw_train, cfg=cfg)
    num_all, cat_all = get_feature_columns(fe_full, cfg=cfg)
    all_cols = num_all + cat_all
    cat_set = set(cat_all)

    groups = {
        "missing_flags": [
            "health_insurance_is_missing",
            "employment_sector_is_missing",
            "income_poverty_is_missing",
            "rent_or_own_is_missing",
            "row_missing_count",
        ],
        "zero_flags": [
            f"opinion_{p}_vacc_effective_is_zero",
            f"opinion_{p}_risk_is_zero",
        ],
        "ordinal": ["age_ordinal", "education_ordinal", "income_ordinal"],
        "behavioral": ["behavioral_sum", "behavioral_mean"],
        "doctor_interactions": ["doctor_x_effective", "doctor_x_risk", "doctor_x_not_worried"],
        "vulnerability_flags": [
            "household_total",
            "has_chronic_condition",
            "high_risk_elderly",
            "is_healthcare_sector",
            "uninsured_flag",
        ],
    }

    def _eval(df_sub: pd.DataFrame, label: str) -> dict:
        df_sub = df_sub.copy()
        cat_c = [c for c in df_sub.columns if c in cat_set]
        num_c = [c for c in df_sub.columns if c not in cat_set]
        for c in cat_c:
            df_sub[c] = df_sub[c].astype(object).where(df_sub[c].notna(), np.nan)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        oof_lr = np.zeros(len(y), dtype=np.float64)
        oof_lgb = np.zeros(len(y), dtype=np.float64)

        for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(df_sub, y)):
            X_tr, y_tr = df_sub.iloc[tr_idx], y[tr_idx]
            X_va = df_sub.iloc[va_idx]

            pre = ColumnTransformer(
                [
                    (
                        "num",
                        Pipeline(
                            [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
                        ),
                        num_c,
                    ),
                    (
                        "cat",
                        Pipeline(
                            [
                                ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                            ]
                        ),
                        cat_c,
                    ),
                ]
            )
            lr = Pipeline(
                [
                    ("pre", pre),
                    ("clf", LogisticRegression(C=0.25, max_iter=1500, random_state=random_state)),
                ]
            )
            lr.fit(X_tr, y_tr)
            oof_lr[va_idx] = lr.predict_proba(X_va)[:, 1]

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=0.15, random_state=random_state + fold_idx
            )
            fit_sub, es_sub = next(sss.split(X_tr, y_tr))
            X_fit, y_fit = X_tr.iloc[fit_sub].copy(), y_tr[fit_sub]
            X_es, y_es = X_tr.iloc[es_sub].copy(), y_tr[es_sub]
            X_va_c = X_va.copy()
            for col in cat_c:
                cats = (
                    pd.Index(X_fit[col].dropna().astype(str).unique())
                    .union(pd.Index(["Missing"]))
                    .sort_values()
                )
                for part in (X_fit, X_es, X_va_c):
                    s = part[col].astype("string").fillna("Missing")
                    s = s.where(s.isin(cats), "Missing")
                    part[col] = pd.Categorical(s, categories=cats)

            lgb = LGBMClassifier(
                n_estimators=1500,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=25,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=random_state,
                n_jobs=1,
                verbose=-1,
            )
            fit_lgbm_with_early_stopping(
                lgb,
                X_fit,
                y_fit,
                X_es,
                y_es,
                stopping_rounds=80,
            )
            oof_lgb[va_idx] = lgb.predict_proba(X_va_c)[:, 1]

        return {
            "variant": label,
            "n_features": int(df_sub.shape[1]),
            "lr_roc_auc": float(roc_auc_score(y, oof_lr)),
            "lr_log_loss": float(log_loss(y, oof_lr)),
            "lgb_roc_auc": float(roc_auc_score(y, oof_lgb)),
            "lgb_log_loss": float(log_loss(y, oof_lgb)),
        }

    fe_with_pruned = fe_full[all_cols].copy()
    eff_c, risk_c, worry_c = (
        f"opinion_{p}_vacc_effective",
        f"opinion_{p}_risk",
        f"opinion_{p}_sick_from_vacc",
    )
    fe_with_pruned["opinion_net_benefit"] = (
        fe_full[eff_c] + fe_full[risk_c] - fe_full[worry_c]
    ).astype(np.float32)
    fe_with_pruned["opinion_risk_minus_worry"] = (
        fe_full[risk_c] - fe_full[worry_c]
    ).astype(np.float32)
    fe_with_pruned["opinion_pro_vaccine_index"] = (
        fe_full[eff_c].fillna(3.0) + fe_full[risk_c].fillna(3.0) - fe_full[worry_c].fillna(3.0)
    ).astype(np.float32)
    fe_with_pruned["opinion_eff_x_risk"] = (
        fe_full[eff_c] * fe_full[risk_c]
    ).astype(np.float32)
    fe_with_pruned["concern_x_knowledge"] = (
        fe_full[f"{p}_concern"] * fe_full[f"{p}_knowledge"]
    ).astype(np.float32)

    records = [
        _eval(raw_base, "raw_baseline_0_untouched"),
        _eval(raw_zero_clean, "raw_plus_zero_as_nan"),
        _eval(fe_full[all_cols], "validated_engineered_full"),
        _eval(fe_with_pruned, "plus_pruned_opinion_composites"),
    ]
    for gname, gcols in groups.items():
        keep = [c for c in all_cols if c not in set(gcols)]
        records.append(_eval(fe_full[keep], f"ablate_without_{gname}"))

    return pd.DataFrame(records)


