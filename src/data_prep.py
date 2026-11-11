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

    # Explicit missingness flags before imputation
    for col in ("health_insurance", "employment_sector", "income_poverty", "rent_or_own"):
        if col in out.columns:
            out[f"{col}_is_missing"] = out[col].isna().astype(np.float32)

    raw_feature_cols = [
        c for c in out.columns if c not in {cfg.id_col, cfg.target_col}
    ]
    out["row_missing_count"] = out[raw_feature_cols].isna().sum(axis=1).astype(np.float32)

    # Undocumented 0 values in 1-5 opinion scales represent unknown/unanswered
    eff_col = f"opinion_{p}_vacc_effective"
    risk_col = f"opinion_{p}_risk"
    worry_col = f"opinion_{p}_sick_from_vacc"
    recc_col = f"doctor_recc_{p}"
    concern_col = f"{p}_concern"
    knowledge_col = f"{p}_knowledge"

    for col in (eff_col, risk_col):
        if col in out.columns:
            out[f"{col}_is_zero"] = (out[col] == 0).astype(np.float32)
            out[col] = out[col].replace(0, np.nan)

    # Ordered demographic encodings
    out["age_ordinal"] = out["age_group"].map(AGE_ORDINAL_MAP)
    out["education_ordinal"] = out["education"].map(EDUCATION_ORDINAL_MAP)
    out["income_ordinal"] = out["income_poverty"].map(INCOME_ORDINAL_MAP)

    # Behavioral aggregates
    beh_cols = [c for c in BEHAVIORAL_FEATURES if c in out.columns]
    out["behavioral_sum"] = out[beh_cols].sum(axis=1, min_count=1)
    out["behavioral_mean"] = out[beh_cols].mean(axis=1)

    # Attitude & risk-perception balances
    out["opinion_net_benefit"] = out[eff_col] - out[worry_col]
    out["opinion_risk_minus_worry"] = out[risk_col] - out[worry_col]
    out["opinion_pro_vaccine_index"] = out[eff_col] + out[risk_col] - out[worry_col]
    out["opinion_eff_x_risk"] = out[eff_col] * out[risk_col]
    out["concern_x_knowledge"] = out[concern_col] * out[knowledge_col]

    # Doctor recommendation interactions (strongest clinical cue)
    out["doctor_x_effective"] = out[recc_col] * out[eff_col].fillna(3.0)
    out["doctor_x_risk"] = out[recc_col] * out[risk_col].fillna(3.0)
    out["doctor_x_not_worried"] = out[recc_col] * (6.0 - out[worry_col].fillna(3.0))

    # Household & vulnerability indicators
    out["household_total"] = (
        1.0 + out["household_adults"].fillna(0.0) + out["household_children"].fillna(0.0)
    )
    out["has_chronic_condition"] = (
        out["number_chronic_med_conditions"] > 0
    ).astype(np.float32)
    out["high_risk_elderly"] = (
        (out["age_ordinal"] >= 3.0) & (out["number_chronic_med_conditions"] >= 1)
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

