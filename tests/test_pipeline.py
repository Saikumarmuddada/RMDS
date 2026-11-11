"""Unit tests for data cleaning, hospitalisation risk model, cost matrix, and submissions."""

import numpy as np
import pandas as pd

from src.config import DATASET_C_CONFIG, SUBMISSIONS_DIR
from src.cost_model import (
    bayes_optimal_thresholds,
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    predict_with_cost_policy,
)
from src.data_prep import engineer_features, load_raw_datasets


def test_data_cleaning_fixes_whitespace_and_zero_opinions():
    raw_train, raw_test, _ = load_raw_datasets(DATASET_C_CONFIG)
    feat_train = engineer_features(raw_train, DATASET_C_CONFIG)
    feat_test = engineer_features(raw_test, DATASET_C_CONFIG)

    assert "MSA, Not Principle  City" not in feat_train["census_msa"].dropna().unique()
    assert "MSA, Not Principle City" in feat_train["census_msa"].dropna().unique()
    assert not (feat_train["opinion_covid_vacc_effective"] == 0).any()
    assert not (feat_test["opinion_covid_risk"] == 0).any()
    assert feat_train["opinion_covid_vacc_effective_is_zero"].sum() == 92
    assert feat_test["opinion_covid_risk_is_zero"].sum() == 104


def test_hospitalisation_risk_monotonicity():
    sample = pd.DataFrame(
        {
            "age_group": ["18 - 34 Years", "65+ Years", "65+ Years"],
            "number_chronic_med_conditions": [0, 0, 3],
            "sex": ["Female", "Female", "Male"],
            "health_insurance": [1.0, 1.0, 0.0],
            "income_poverty": ["> $75,000", "> $75,000", "Below Poverty"],
        }
    )
    pu, pv = estimate_hospitalisation_probabilities(sample, scenario="base")
    assert np.all(pv < pu)
    assert pu[0] < pu[1] < pu[2]
    assert np.all((pu > 0) & (pu < 1))


def test_town_cost_rules_exact():
    # Single-person edge cases under Dataset C ($1,500 vaccine, $41,000 hospitalisation)
    pu = np.array([0.10])
    pv = np.array([0.02])

    # Case 1: refuses (0), predict 0 -> $0 vaccine + 0.10 * 41,000 = $4,100
    bd_00 = compute_town_cost(np.array([0]), np.array([0]), pu, pv, DATASET_C_CONFIG)
    assert np.isclose(bd_00.total_cost, 4100.0)

    # Case 2: refuses (0), predict 1 -> $1,500 wasted + 0.10 * 41,000 = $5,600
    bd_01 = compute_town_cost(np.array([0]), np.array([1]), pu, pv, DATASET_C_CONFIG)
    assert np.isclose(bd_01.total_cost, 5600.0)

    # Case 3: accepts (1), predict 0 -> $1,500 local law penalty + 0.10 * 41,000 = $5,600
    bd_10 = compute_town_cost(np.array([1]), np.array([0]), pu, pv, DATASET_C_CONFIG)
    assert np.isclose(bd_10.total_cost, 5600.0)

    # Case 4: accepts (1), predict 1 -> $1,500 vaccine + 0.02 * 41,000 = $2,320
    bd_11 = compute_town_cost(np.array([1]), np.array([1]), pu, pv, DATASET_C_CONFIG)
    assert np.isclose(bd_11.total_cost, 2320.0)


def test_bayes_threshold_consistency():
    # Threshold tau* = 1500 / (1500 + 41000*(0.10 - 0.02)) = 1500 / 4780 = 0.3138075
    pu = np.array([0.10, 0.10])
    pv = np.array([0.02, 0.02])
    tau = bayes_optimal_thresholds(pu, pv, DATASET_C_CONFIG)
    assert np.allclose(tau, 1500.0 / 4780.0)

    preds = predict_with_cost_policy(
        np.array([0.30, 0.33]), pu, pv, policy="bayes", threshold_param=1.0, cfg=DATASET_C_CONFIG
    )
    assert np.array_equal(preds, np.array([0, 1]))


def test_generated_submissions_match_example_schema():
    _, _, example_sub = load_raw_datasets(DATASET_C_CONFIG)
    for order in range(1, 6):
        path = SUBMISSIONS_DIR / f"challenge_submission_group_0_order_{order}.csv"
        assert path.exists(), f"Missing submission file: {path}"
        sub = pd.read_csv(path)
        assert list(sub.columns) == list(example_sub.columns)
        assert len(sub) == len(example_sub)
        assert np.array_equal(sub["respondent_id"].values, example_sub["respondent_id"].values)
        assert set(sub["covid_vaccine"].unique()).issubset({0, 1})

