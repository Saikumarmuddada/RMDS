"""Hospitalisation probability estimation and town-cost optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd

from src.config import DATASET_C_CONFIG, DatasetConfig


RiskScenario = Literal["conservative", "base", "severe"]

# Age-stratified Infection Hospitalisation Rates (IHR) for SARS-CoV-2
# under a 100% infection assumption (base profile anchored to public health estimates).
BASE_AGE_IHR = {
    "18 - 34 Years": 0.010,
    "35 - 44 Years": 0.020,
    "45 - 54 Years": 0.040,
    "55 - 64 Years": 0.075,
    "65+ Years": 0.160,
}

SCENARIO_ODDS_MULTIPLIER = {
    "conservative": 0.50,
    "base": 1.00,
    "severe": 1.50,
}


@dataclass(frozen=True)
class CostBreakdown:
    """Detailed financial breakdown for a set of binary vaccine predictions."""

    total_cost: float
    cost_per_person: float
    vaccine_send_cost: float
    law_penalty_cost: float
    total_vaccine_related_cost: float
    expected_hospitalisations: float
    hospitalisation_cost: float
    treat_none_cost: float
    treat_all_cost: float
    oracle_min_cost: float
    savings_vs_treat_none: float
    savings_vs_treat_all: float
    captured_oracle_savings_pct: float
    predicted_positive_rate: float


def _logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clipped / (1.0 - p_clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def estimate_hospitalisation_probabilities(
    df: pd.DataFrame,
    scenario: RiskScenario = "base",
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate P(hospitalisation | features, unvaccinated) and P(hospitalisation | features, vaccinated).

    Factors incorporated from Dataset C:
    1. age_group: strongest driver of COVID-19 infection-hospitalisation risk.
    2. number_chronic_med_conditions: each chronic condition raises hospitalisation odds (~1.55x).
    3. sex: male sex carries modestly higher severe outcome odds (~1.20x vs female ~0.90x).
    4. health_insurance / income_poverty: uninsured or below-poverty status modestly raises risk.
    5. vaccination status: ~85% relative risk reduction (80% in older/multi-comorbidity patients).
    """
    if scenario not in SCENARIO_ODDS_MULTIPLIER:
        raise ValueError(f"Unknown risk scenario: {scenario}")

    base_age = df["age_group"].map(BASE_AGE_IHR).fillna(0.045).to_numpy(dtype=np.float64)
    log_odds = _logit(base_age)

    # Scenario calibration shift
    log_odds += np.log(SCENARIO_ODDS_MULTIPLIER[scenario])

    # Chronic medical conditions (capped at 5 effective conditions to avoid extreme saturation)
    chronic = df["number_chronic_med_conditions"].fillna(0).clip(0, 5).to_numpy(dtype=np.float64)
    log_odds += chronic * np.log(1.55)

    # Sex adjustment (centered around 1.0)
    sex_mult = np.where(
        df["sex"] == "Male",
        1.20,
        np.where(df["sex"] == "Female", 0.90, 1.00),
    )
    log_odds += np.log(sex_mult)

    # Socioeconomic / access adjustment
    uninsured = (df["health_insurance"] == 0).to_numpy(dtype=bool)
    below_poverty = (df["income_poverty"] == "Below Poverty").to_numpy(dtype=bool)
    log_odds += np.where(uninsured, np.log(1.15), 0.0)
    log_odds += np.where(below_poverty, np.log(1.12), 0.0)

    p_unvacc = np.clip(_sigmoid(log_odds), 0.002, 0.85)

    # Vaccine effectiveness against hospitalisation given infection
    high_frailty = (df["age_group"] == "65+ Years").to_numpy(dtype=bool) | (chronic >= 2)
    ve_hosp = np.where(high_frailty, 0.80, 0.85)
    p_vacc = p_unvacc * (1.0 - ve_hosp)

    return p_unvacc, p_vacc


def compute_town_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> CostBreakdown:
    """Compute exact expected town cost under the Target Metric Information rules."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)

    cv = cfg.vaccine_cost
    ch = cfg.hospitalisation_cost

    def _raw_cost(pred: np.ndarray) -> Tuple[float, float, float, float]:
        send_cost = float(np.sum(pred == 1) * cv)
        law_cost = float(np.sum((pred == 0) & (y_t == 1)) * cv)
        vaccinated = (pred == 1) & (y_t == 1)
        p_hosp = np.where(vaccinated, p_hosp_vacc, p_hosp_unvacc)
        exp_hosp = float(np.sum(p_hosp))
        hosp_cost = exp_hosp * ch
        return send_cost, law_cost, exp_hosp, hosp_cost

    send_cost, law_cost, exp_hosp, hosp_cost = _raw_cost(y_p)
    total_cost = send_cost + law_cost + hosp_cost

    _, none_law, _, none_hosp = _raw_cost(np.zeros_like(y_t))
    treat_none_cost = none_law + none_hosp

    all_send, _, _, all_hosp = _raw_cost(np.ones_like(y_t))
    treat_all_cost = all_send + all_hosp

    oracle_send, oracle_law, _, oracle_hosp = _raw_cost(y_t)
    oracle_min_cost = oracle_send + oracle_law + oracle_hosp

    baseline_best = min(treat_none_cost, treat_all_cost)
    max_possible_savings = max(baseline_best - oracle_min_cost, 1e-9)
    captured_pct = ((baseline_best - total_cost) / max_possible_savings) * 100.0

    return CostBreakdown(
        total_cost=total_cost,
        cost_per_person=total_cost / len(y_t),
        vaccine_send_cost=send_cost,
        law_penalty_cost=law_cost,
        total_vaccine_related_cost=send_cost + law_cost,
        expected_hospitalisations=exp_hosp,
        hospitalisation_cost=hosp_cost,
        treat_none_cost=treat_none_cost,
        treat_all_cost=treat_all_cost,
        oracle_min_cost=oracle_min_cost,
        savings_vs_treat_none=treat_none_cost - total_cost,
        savings_vs_treat_all=treat_all_cost - total_cost,
        captured_oracle_savings_pct=captured_pct,
        predicted_positive_rate=float(np.mean(y_p)),
    )


def bayes_optimal_thresholds(
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> np.ndarray:
    """Compute exact per-person acceptance probability threshold tau*(x_i).

    Predicting 1 beats predicting 0 in expected cost iff:
        P(y=1 | x) > C_v / (C_v + C_h * (p_hosp_unvacc - p_hosp_vacc))
    """
    delta_hosp_savings = np.maximum(p_hosp_unvacc - p_hosp_vacc, 0.0) * cfg.hospitalisation_cost
    return cfg.vaccine_cost / (cfg.vaccine_cost + delta_hosp_savings)


def predict_with_cost_policy(
    y_prob: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    policy: str = "bayes",
    threshold_param: float = 1.0,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> np.ndarray:
    """Convert predicted probabilities into 0/1 vaccine send decisions.

    policy='bayes': threshold_param scales tau*(x_i), i.e., y_prob >= clip(tau* * scale, 0, 1).
    policy='global': threshold_param is a constant probability cutoff across all individuals.
    """
    probs = np.asarray(y_prob, dtype=np.float64)
    if policy == "bayes":
        tau = bayes_optimal_thresholds(p_hosp_unvacc, p_hosp_vacc, cfg=cfg)
        cutoff = np.clip(tau * threshold_param, 0.0, 1.0)
        return (probs >= cutoff).astype(np.int64)
    if policy == "global":
        return (probs >= threshold_param).astype(np.int64)
    raise ValueError(f"Unsupported policy: {policy}")


def tune_threshold_on_oof(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    folds: np.ndarray | None = None,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> dict:
    """Compare 0.5 cutoff, global tuned, Bayes-optimal (zero tuning), and nested-CV Bayes-scaled."""
    y_t = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(oof_probs, dtype=np.float64)
    results = {}

    # 1. Naive 0.5 cutoff
    pred_05 = (probs >= 0.5).astype(np.int64)
    results["global_0.5"] = {
        "policy": "global",
        "param": 0.5,
        "breakdown": compute_town_cost(y_t, pred_05, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    # 2. Global threshold (nested CV if folds provided, else direct)
    def _best_global(yt: np.ndarray, pr: np.ndarray, pu: np.ndarray, pv: np.ndarray) -> float:
        best_t, best_c = 0.5, float("inf")
        for t in np.linspace(0.05, 0.95, 181):
            c = compute_town_cost(yt, (pr >= t).astype(np.int64), pu, pv, cfg).total_cost
            if c < best_c:
                best_c, best_t = c, float(t)
        return best_t

    def _best_scale(yt: np.ndarray, pr: np.ndarray, pu: np.ndarray, pv: np.ndarray) -> float:
        best_s, best_c = 1.0, float("inf")
        for s in np.linspace(0.70, 1.30, 121):
            preds = predict_with_cost_policy(
                pr, pu, pv, policy="bayes", threshold_param=float(s), cfg=cfg
            )
            c = compute_town_cost(yt, preds, pu, pv, cfg).total_cost
            if c < best_c:
                best_c, best_s = c, float(s)
        return best_s

    if folds is not None:
        pred_gt = np.zeros_like(y_t)
        pred_bs = np.zeros_like(y_t)
        g_params, s_params = [], []
        for f in np.unique(folds):
            tr_mask = folds != f
            va_mask = folds == f
            t_fold = _best_global(
                y_t[tr_mask], probs[tr_mask], p_hosp_unvacc[tr_mask], p_hosp_vacc[tr_mask]
            )
            s_fold = _best_scale(
                y_t[tr_mask], probs[tr_mask], p_hosp_unvacc[tr_mask], p_hosp_vacc[tr_mask]
            )
            g_params.append(t_fold)
            s_params.append(s_fold)
            pred_gt[va_mask] = (probs[va_mask] >= t_fold).astype(np.int64)
            pred_bs[va_mask] = predict_with_cost_policy(
                probs[va_mask],
                p_hosp_unvacc[va_mask],
                p_hosp_vacc[va_mask],
                policy="bayes",
                threshold_param=s_fold,
                cfg=cfg,
            )
        best_g_param = float(np.median(g_params))
        best_b_scale = float(np.median(s_params))
    else:
        best_g_param = _best_global(y_t, probs, p_hosp_unvacc, p_hosp_vacc)
        pred_gt = (probs >= best_g_param).astype(np.int64)
        best_b_scale = _best_scale(y_t, probs, p_hosp_unvacc, p_hosp_vacc)
        pred_bs = predict_with_cost_policy(
            probs, p_hosp_unvacc, p_hosp_vacc, policy="bayes", threshold_param=best_b_scale, cfg=cfg
        )

    results["global_tuned"] = {
        "policy": "global",
        "param": best_g_param,
        "breakdown": compute_town_cost(y_t, pred_gt, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    # 3. Exact theoretical Bayes threshold (scale = 1.0, zero tuning)
    pred_bayes = predict_with_cost_policy(
        probs, p_hosp_unvacc, p_hosp_vacc, policy="bayes", threshold_param=1.0, cfg=cfg
    )
    results["bayes_exact"] = {
        "policy": "bayes",
        "param": 1.0,
        "breakdown": compute_town_cost(y_t, pred_bayes, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    # 4. Scaled Bayes threshold
    results["bayes_scaled"] = {
        "policy": "bayes",
        "param": best_b_scale,
        "breakdown": compute_town_cost(y_t, pred_bs, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    return results
