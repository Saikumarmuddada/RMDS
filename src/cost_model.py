"""Cost model, mathematically derived Bayesian decision policy, and modelled hospitalisation risk.

Important methodological notice:
Dataset C does NOT contain an empirical COVID-19 hospitalisation outcome column.
Individual hospitalisation probabilities under unvaccinated (q0) and vaccinated (q1)
states are therefore explicit modelled epidemiological estimates anchored to published
peer-reviewed studies and CDC surveillance analyses, evaluated under sensitivity ranges.
They are NOT clinical risk scores trained on Dataset C outcomes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
import pandas as pd

from src.config import DATASET_C_CONFIG, DatasetConfig

# ---------------------------------------------------------------------------
# Scientifically Grounded Modelled Parameters & Explicit Literature Citations
# ---------------------------------------------------------------------------
# Baseline unvaccinated infection-to-hospitalisation risk (IHR) mapped onto
# Dataset C adult age brackets.
# Primary anchors:
# 1. Herrera-Esposito & de los Campos (2022), "Age-specific rate of severe and
#    critical SARS-CoV-2 infections estimated with multi-country seroprevalence
#    studies", BMC Infectious Diseases 22:311, doi:10.1186/s12879-022-07262-0.
#    Reported pre-vaccination ISR: 20-29: 0.47% (0.28-0.74%), 30-39: 0.99%
#    (0.61-1.53%), 40-49: 2.11% (1.31-3.22%), 50-59: 4.49% (2.80-6.84%),
#    60-69: 9.54% (5.99-14.49%).
# 2. Verity et al. (2020), "Estimates of the severity of coronavirus disease 2019:
#    a model-based analysis", Lancet Infect Dis 20(6):669-677,
#    doi:10.1016/S1473-3099(20)30243-7. Reported Mainland China early-wave IHR:
#    20-29: 1.0%, 30-39: 3.4%, 40-49: 4.3%, 50-59: 8.2%, 60-69: 11.8%,
#    70-79: 16.6%, 80+: 18.4%.
# 3. Salje et al. (2020), "Estimating the burden of SARS-CoV-2 in France",
#    Science 369:208-211, doi:10.1126/science.abc3517 (overall ~2.6-2.9%, 80+ >=8.3%).
#
# Limitation: Survey age brackets (18-34, 65+) aggregate finer 10-year age bands
# in published studies; midpoint values below sit between Herrera-Esposito (2022)
# seroprevalence estimates and Verity et al. (2020) upper bounds.
BASE_AGE_IHR: Dict[str, float] = {
    "18 - 34 Years": 0.006,
    "35 - 44 Years": 0.015,
    "45 - 54 Years": 0.031,
    "55 - 64 Years": 0.066,
    "65+ Years": 0.145,
}
DEFAULT_UNKNOWN_AGE_IHR: float = 0.035  # Adult mid-distribution prior if age_group is missing

# Explicit bracket crosswalk documenting how finer epidemiological age strata map onto Dataset C:
# Herrera-Esposito & de los Campos (2022) report ancestral multi-country seroprevalence ISR
# (20-29: 0.47%, 30-39: 0.99%, 40-49: 2.11%, 50-59: 4.49%, 60-69: 9.54%); Verity et al. (2020)
# report Mainland China early-wave IHR (20-29: 1.0%, 30-39: 3.4%, 40-49: 4.3%, 50-59: 8.2%,
# 60-69: 11.8%, 70-79: 16.6%, 80+: 18.4%). Because Dataset C pools all elders into one open
# `65+ Years` bracket (spanning 65-69, 70-79, and 80+ community-dwelling respondents), `0.145`
# sits centered between Verity 60-69 (11.8%) and 70-79/80+ (16.6-18.4%) and Herrera-Esposito
# 60-69/70+ gradient, while conservative (0.5x odds) and severe (1.5x odds) scenarios bracket uncertainty.
AGE_BRACKET_MAPPING_DOC: Dict[str, str] = {
    "18 - 34 Years": "Spans 20-29 (Herrera-Esposito 0.47%, Verity 1.0%) & lower 30-39 (0.99%); midpoint 0.006.",
    "35 - 44 Years": "Spans upper 30-39 (0.99%) & lower 40-49 (2.11%, Verity 4.3%); midpoint 0.015.",
    "45 - 54 Years": "Spans upper 40-49 (2.11%) & lower 50-59 (4.49%, Verity 8.2%); midpoint 0.031.",
    "55 - 64 Years": "Spans upper 50-59 (4.49%) & lower 60-69 (9.54%, Verity 11.8%); midpoint 0.066.",
    "65+ Years": "Community-dwelling open bracket pooling 65-69 (~10-12%), 70-79 (Verity 16.6%), and 80+ (Verity 18.4%); composite prior 0.145 bounded via conservative/severe sensitivity.",
}

# Chronic medical condition adjusted Rate Ratios (aRR, NOT Odds Ratios):
# Ko et al. (2021), "Risk Factors for Coronavirus Disease 2019 (COVID-19)-Associated
# Hospitalization: COVID-19-Associated Hospitalization Surveillance Network and Behavioral Risk
# Factor Surveillance System", Clin Infect Dis 72(11):e695-e703, doi:10.1093/cid/ciaa1419.
# US adult COVID-NET/BRFSS >=3 conditions vs 0: aRR = 5.0 (95% CI: 3.9-6.3), adjusted for age,
# sex, race/ethnicity. Levels 1 and 2 use explicit concave geometric interpolation (5^(1/3), 5^(2/3))
# as modelled parameters (Ko et al. reported category >=3 rather than separate integer counts 1 & 2).
CHRONIC_CONDITION_RR_MAP: Dict[int, float] = {
    0: 1.00,
    1: 1.70,
    2: 2.90,
    3: 5.00,  # >= 3 conditions capped to Ko et al. (2021) >=3 category
}

# Sex adjustment on the ODDS RATIO scale (consistent with logistic multivariable aOR):
# - Petrilli et al. (2020), "Factors associated with hospital admission and critical illness among
#   5279 people with coronavirus disease 2019 in New York City", BMJ 369:m1966,
#   doi:10.1136/bmj.m1966: Male hospital admission multivariable aOR = 2.8 (95% CI: 2.4-3.2),
#   critical illness aOR = 1.5 (95% CI: 1.3-1.8).
# - Peckham et al. (2020), Nat Commun 11:6317, doi:10.1038/s41467-020-19741-6: male ITU OR = 2.84, death OR = 1.39.
# - Williamson et al. (2020, OpenSAFELY), Nature 584:430-436, doi:10.1038/s41586-020-2521-4: death HR = 1.59.
# Conservative synthesis centers OR = 1.45 near geometric mean 1.0 (toggleable via include_sex_adjustment).
SEX_ODDS_MULTIPLIER: Dict[str, float] = {
    "Male": 1.20,
    "Female": 1.20 / 1.45,  # ~0.8276 (OR Male/Female = 1.45)
}

# Scenario multipliers capturing variant severity, healthcare-threshold, and age-mix uncertainty.
SCENARIO_ODDS_MULTIPLIER: Dict[str, float] = {
    "conservative": 0.50,
    "base": 1.00,
    "severe": 1.50,
}

# Vaccine effectiveness against COVID-19 hospitalisation given infection:
# Tuple: (standard_adult_ve, high_frailty_ve_65plus_or_3plus_chronic)
# Primary anchors:
# - Tenforde et al. (2021), "Association Between mRNA Vaccination and COVID-19 Hospitalization and
#   Disease Severity", JAMA 326(20):2043-2054, doi:10.1001/jama.2021.19499 (PMID 34734975):
#   Delta hospitalization VE = 86% overall (>120d Moderna 85%, Pfizer 64%); lower in
#   immunocompromised subgroups (motivating illustrative high-frailty sensitivity = 80%).
# - Ferdinands et al. (2022), "Waning 2-Dose and 3-Dose Effectiveness of mRNA Vaccines Against
#   COVID-19-Associated Emergency Department and Urgent Care Encounters and Hospitalizations Among
#   Adults During Periods of Delta and Omicron Variant Predominance - VISION Network",
#   MMWR 71(7):255-263, doi:10.15585/mmwr.mm7107e2: Omicron hospitalization VE was 91% (<2m 3-dose),
#   78% (>=4m 3-dose), 71% (<2m 2-dose), and 54% (>=5m 2-dose); `waned_omicron` uses 65%/55% midpoint.
VE_REGIMES: Dict[str, Tuple[float, float]] = {
    "waned_omicron": (0.65, 0.55),
    "base": (0.86, 0.80),
    "boosted": (0.91, 0.86),
}

MIN_MODELLED_IHR = 0.002
MAX_MODELLED_IHR = 0.850

HOSPITALISATION_SCIENTIFIC_SOURCES = {
    "age_ihr": {
        "citations": [
            "Herrera-Esposito & de los Campos (2022), BMC Infect Dis 22:311, doi:10.1186/s12879-022-07262-0",
            "Verity et al. (2020), Lancet Infect Dis 20(6):669-677, doi:10.1016/S1473-3099(20)30243-7",
            "Salje et al. (2020), Science 369:208-211, doi:10.1126/science.abc3517",
        ],
        "statistic_type": "Infection-Hospitalisation / Infection-Severe Probability (IHR / ISR)",
        "bracket_mapping": AGE_BRACKET_MAPPING_DOC,
        "limitation": "Pre-vaccination ancestral SARS-CoV-2 seroprevalence/Wave-1 cohorts across non-US & multi-country settings; 5-yr offset requires bracket midpoint interpolation and 65+ weighting.",
    },
    "chronic_conditions": {
        "citations": [
            "Ko et al. (2021), Clin Infect Dis 72(11):e695-e703, doi:10.1093/cid/ciaa1419",
            "Zhang & Yu (1998), JAMA 280(19):1690-1691, doi:10.1001/jama.280.19.1690",
        ],
        "statistic_type": "Adjusted Rate Ratio (aRR = 5.0, 95% CI 3.9-6.3 for >=3 vs 0 conditions)",
        "transformation": "Applied on the complementary log-log (cloglog) proportional-rate scale p_u = 1 - (1 - p_base)^aRR rather than silently treating aRR as an odds ratio; preserves RR agreement at low p_base while staying strictly bounded in (0, 1) without >1.0 overflow in elderly strata. Levels 1 & 2 (1.70, 2.90) are explicit concave geometric assumptions (5^(1/3), 5^(2/3)).",
        "limitation": "COVID-NET/BRFSS US adult population rate ratio mixes exposure and severity; 1-2 count interpolation is uncalibrated on Dataset C.",
    },
    "sex": {
        "citations": [
            "Petrilli et al. (2020), BMJ 369:m1966, doi:10.1136/bmj.m1966",
            "Peckham et al. (2020), Nat Commun 11:6317, doi:10.1038/s41467-020-19741-6",
            "Williamson et al. (2020, OpenSAFELY), Nature 584:430-436, doi:10.1038/s41586-020-2521-4",
        ],
        "statistic_type": "Multivariable Odds Ratio (OR = 1.45 conservative Male/Female synthesis on log-odds scale)",
        "limitation": "Petrilli NYC Wave-1 admission aOR (2.8) exceeded global critical/mortality ratios; conservative centered 1.45 OR tested via include_sex_adjustment=False sensitivity.",
    },
    "vaccine_effectiveness": {
        "citations": [
            "Tenforde et al. (2021), JAMA 326(20):2043-2054, doi:10.1001/jama.2021.19499 (PMID 34734975)",
            "Ferdinands et al. (2022), MMWR 71(7):255-263, doi:10.15585/mmwr.mm7107e2",
        ],
        "statistic_type": "Test-negative Hospitalisation VE (1 - aOR_hospitalisation)",
        "estimate": "Tenforde Delta VE = 86%; Ferdinands Omicron 3-dose VE = 91% (<2m) to 78% (>=4m) and 2-dose 71% (<2m) to 54% (>=5m). High-frailty (65+ or >=3 chronic) downward shift (80%/55%/86%) is an explicit sensitivity assumption grounded in weaker immunocompromised protection.",
    },
    "population_compatibility": (
        "CRITICAL CAVEAT: Parameters synthesize US COVID-NET/VISION/NYC cohorts (2020-2022, ancestral/Delta/Omicron) "
        "and multi-country seroprevalence meta-analyses (2020-2021). Dataset C itself contains NO empirical COVID-19 "
        "hospitalisation column; (q0, q1) are explicit modelled epidemiological estimates, NOT clinically validated risk scores."
    ),
}

RiskScenario = Literal["conservative", "base", "severe"]
VERegime = Literal["waned_omicron", "base", "boosted"]


@dataclass
class CostBreakdown:
    """Full accounting breakdown of expected town financial expenditure."""

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
    # Extended granular accounting fields (defaults preserve backward construction)
    accepted_vaccine_cost: float = 0.0
    wasted_vaccine_cost: float = 0.0
    vaccines_allocated: int = 0
    vaccines_accepted: int = 0
    wasted_vaccines: int = 0
    law_compensation_count: int = 0


def _logit(p: np.ndarray) -> np.ndarray:
    p_clip = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p_clip / (1.0 - p_clip))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def _prepare_probabilities(
    arr: np.ndarray | pd.Series,
    name: str,
    *,
    strict: bool = False,
    nan_fill: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and safely coerce probability arrays into [0, 1].

    Returns (clean_array, valid_mask_before_fill).
    """
    vals = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(vals)
    out_of_range = finite_mask & ((vals < -1e-9) | (vals > 1.0 + 1e-9))

    if strict:
        if not np.all(finite_mask):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf).")
        if np.any(out_of_range):
            raise ValueError(f"{name} contains out-of-range probabilities outside [0, 1].")

    if not np.all(finite_mask):
        warnings.warn(
            f"{name} contains non-finite values; applying safe conservative fallback.",
            RuntimeWarning,
            stacklevel=3,
        )
    if np.any(out_of_range):
        warnings.warn(
            f"{name} contains values outside [0, 1]; clipping to [0, 1].",
            RuntimeWarning,
            stacklevel=3,
        )

    cleaned = np.where(finite_mask, vals, nan_fill)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    return cleaned, finite_mask


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) over uniform [0, 1] confidence bins."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp, _ = _prepare_probabilities(y_prob, "y_prob", strict=False, nan_fill=0.0)
    if len(yt) == 0:
        return 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = float(len(yt))
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (yp >= lo) & (yp <= hi)
        else:
            mask = (yp >= lo) & (yp < hi)
        if np.any(mask):
            conf = float(np.mean(yp[mask]))
            acc = float(np.mean(yt[mask]))
            ece += (float(np.sum(mask)) / n) * abs(acc - conf)
    return float(ece)


def zhang_yu_rr_to_or(
    rr: np.ndarray | float,
    p0: np.ndarray | float,
) -> np.ndarray:
    """Convert Risk/Rate Ratio RR to Odds Ratio OR at baseline risk p0 (Zhang & Yu, JAMA 1998)."""
    r = np.asarray(rr, dtype=np.float64)
    b = np.clip(np.asarray(p0, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    # Exact Zhang & Yu relation OR = RR * (1 - p0) / (1 - p0 * RR), bounded safely near unity
    adjusted_denom = np.maximum(1.0 - np.minimum(b * r, 0.99), 1e-6)
    return (r * (1.0 - b)) / adjusted_denom


def estimate_hospitalisation_probabilities(
    df: pd.DataFrame,
    scenario: RiskScenario = "base",
    *,
    ve_regime: VERegime = "base",
    include_sex_adjustment: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate individual hospitalisation probabilities under 100% infection risk.

    Effect-measure pipeline:
      1. Odds-ratio stage (log-odds scale): age-stratum IHR prior + scenario severity odds +
         optional multivariable sex OR -> p_base in (0, 1).
      2. Rate-ratio stage (complementary log-log proportional-rate link): applies Ko et al. (2021)
         chronic aRR via p_u = 1 - (1 - p_base)^aRR, keeping effect measures consistent without
         equating aRR to OR or overflowing 1.0 in elderly severe strata.
      3. Vaccine effectiveness: Tenforde et al. (2021) / Ferdinands et al. (2022) relative reduction.
    """
    if scenario not in SCENARIO_ODDS_MULTIPLIER:
        raise ValueError(f"Unsupported risk scenario: {scenario}")
    if ve_regime not in VE_REGIMES:
        raise ValueError(f"Unsupported VE regime: {ve_regime}")

    n = len(df)
    if n == 0:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty.copy()

    if "age_group" in df.columns:
        age_clean = df["age_group"].astype("string").str.strip()
        base_ihr = (
            age_clean.map(BASE_AGE_IHR).fillna(DEFAULT_UNKNOWN_AGE_IHR).to_numpy(dtype=np.float64)
        )
    else:
        age_clean = pd.Series(["Missing"] * n, dtype="string")
        base_ihr = np.full(n, DEFAULT_UNKNOWN_AGE_IHR, dtype=np.float64)

    # Stage 1: OR-scale adjustments (age prior + scenario odds + centered sex OR)
    log_odds = _logit(base_ihr) + np.log(SCENARIO_ODDS_MULTIPLIER[scenario])
    if include_sex_adjustment and "sex" in df.columns:
        sex_clean = df["sex"].astype("string").str.strip()
        sex_mult = sex_clean.map(SEX_ODDS_MULTIPLIER).fillna(1.00).to_numpy(dtype=np.float64)
        log_odds += np.log(sex_mult)
    p_age_sex = np.clip(_sigmoid(log_odds), MIN_MODELLED_IHR, MAX_MODELLED_IHR)

    # Stage 2: Chronic condition aRR transformed via bounded cloglog survival link: 1 - (1 - p0)^aRR
    if "number_chronic_med_conditions" in df.columns:
        chronic_raw = pd.to_numeric(df["number_chronic_med_conditions"], errors="coerce")
        chronic_bucket = chronic_raw.fillna(0).clip(lower=0, upper=3).round().astype(int)
    else:
        chronic_bucket = pd.Series(0, index=df.index, dtype=int)
    chronic_rr = (
        chronic_bucket.map(CHRONIC_CONDITION_RR_MAP).fillna(1.0).to_numpy(dtype=np.float64)
    )
    p_unvacc_raw = 1.0 - np.power(np.clip(1.0 - p_age_sex, 1e-9, 1.0), chronic_rr)
    p_unvacc = np.clip(p_unvacc_raw, MIN_MODELLED_IHR, MAX_MODELLED_IHR)

    # Stage 3: Vaccine effectiveness against hospitalisation given infection
    std_ve, frail_ve = VE_REGIMES[ve_regime]
    is_high_frailty = (age_clean == "65+ Years").to_numpy(dtype=bool) | (
        chronic_bucket.to_numpy() >= 3
    )
    ve_hosp = np.where(is_high_frailty, frail_ve, std_ve)
    p_vacc = np.clip(p_unvacc * (1.0 - ve_hosp), 0.0, MAX_MODELLED_IHR)

    return p_unvacc, p_vacc


def bayes_optimal_thresholds(
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Compute exact individual Bayes-optimal acceptance probability threshold tau*(x_i).

    Derivation under 100% infection risk:
        E[Cost(send=1)] = C_v + C_h * q0 - p * C_h * (q0 - q1)
        E[Cost(send=0)] = p * C_v + C_h * q0
        Send iff E[Cost(send=1)] <= E[Cost(send=0)]
            <=> (1 - p) * C_v <= p * C_h * (q0 - q1)
            <=> p >= C_v / (C_v + C_h * (q0 - q1))  when (q0 - q1) > 0.
    When q1 >= q0 (vaccine yields zero/negative benefit) or inputs are non-finite,
    returns threshold = 1.0 (and policy callers block allocation even at p=1.0).
    """
    q0, valid_q0 = _prepare_probabilities(
        p_hosp_unvacc, "p_hosp_unvacc", strict=strict, nan_fill=0.0
    )
    q1, valid_q1 = _prepare_probabilities(
        p_hosp_vacc, "p_hosp_vacc", strict=strict, nan_fill=0.0
    )
    if q0.shape != q1.shape:
        raise ValueError("p_hosp_unvacc and p_hosp_vacc must have identical shapes.")

    cv = float(cfg.vaccine_cost)
    ch = float(cfg.hospitalisation_cost)
    if cv < 0 or ch < 0:
        raise ValueError("Costs must be non-negative.")

    delta = q0 - q1
    valid_both = valid_q0 & valid_q1
    beneficial = valid_both & (delta > 0.0)

    denom = cv + ch * np.maximum(delta, 0.0)
    raw_tau = np.where(beneficial & (denom > 0.0), cv / np.maximum(denom, 1e-12), 1.0)
    return np.clip(raw_tau, 0.0, 1.0)


def expected_decision_cost(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Compute per-person closed-form expected financial expenditure given y_prob and action y_pred."""
    p, _ = _prepare_probabilities(y_prob, "y_prob", strict=strict, nan_fill=0.0)
    q0, _ = _prepare_probabilities(p_hosp_unvacc, "p_hosp_unvacc", strict=strict, nan_fill=0.0)
    q1, _ = _prepare_probabilities(p_hosp_vacc, "p_hosp_vacc", strict=strict, nan_fill=0.0)
    pred = np.asarray(y_pred, dtype=np.int64)
    if not (p.shape == q0.shape == q1.shape == pred.shape):
        raise ValueError("Input arrays must share the same shape.")

    cv = float(cfg.vaccine_cost)
    ch = float(cfg.hospitalisation_cost)
    cost_send = cv + ch * (q0 - p * (q0 - q1))
    cost_skip = p * cv + ch * q0
    return np.where(pred == 1, cost_send, cost_skip)


def predict_with_cost_policy(
    y_prob: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    policy: str = "bayes",
    threshold_param: float = 1.0,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    *,
    strict: bool = False,
) -> np.ndarray:
    """Convert predicted probabilities into 0/1 vaccine send decisions.

    policy='bayes': y_prob >= clip(tau*(x) * threshold_param, 0, 1), gated by (q0 > q1)
                    and valid probabilities so zero/harmful VE never receives a dose.
    policy='global': constant probability cutoff across all individuals (fixed baseline).
    """
    probs, valid_p = _prepare_probabilities(y_prob, "y_prob", strict=strict, nan_fill=0.0)
    if policy == "bayes":
        q0, valid_q0 = _prepare_probabilities(
            p_hosp_unvacc, "p_hosp_unvacc", strict=strict, nan_fill=0.0
        )
        q1, valid_q1 = _prepare_probabilities(
            p_hosp_vacc, "p_hosp_vacc", strict=strict, nan_fill=0.0
        )
        tau = bayes_optimal_thresholds(q0, q1, cfg=cfg, strict=False)
        cutoff = np.clip(tau * float(threshold_param), 0.0, 1.0)
        beneficial = valid_p & valid_q0 & valid_q1 & ((q0 - q1) > 0.0)
        return (beneficial & (probs >= cutoff)).astype(np.int64)
    if policy == "global":
        return (valid_p & (probs >= float(threshold_param))).astype(np.int64)
    raise ValueError(f"Unsupported policy: {policy}")


def compute_town_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    *,
    strict: bool = True,
) -> CostBreakdown:
    """Compute exact expected town cost under the Target Metric Information rules."""
    y_t = np.asarray(y_true, dtype=np.int64)
    y_p = np.asarray(y_pred, dtype=np.int64)
    if y_t.shape != y_p.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    if len(y_t) == 0:
        raise ValueError("Cannot compute town cost on empty arrays.")
    if not np.all(np.isin(y_t, (0, 1))):
        raise ValueError("y_true must contain only binary values {0, 1}.")
    if not np.all(np.isin(y_p, (0, 1))):
        raise ValueError("y_pred must contain only binary values {0, 1}.")

    q0, _ = _prepare_probabilities(
        p_hosp_unvacc, "p_hosp_unvacc", strict=strict, nan_fill=DEFAULT_UNKNOWN_AGE_IHR
    )
    q1, _ = _prepare_probabilities(
        p_hosp_vacc, "p_hosp_vacc", strict=strict, nan_fill=DEFAULT_UNKNOWN_AGE_IHR
    )
    if q0.shape != y_t.shape or q1.shape != y_t.shape:
        raise ValueError("Hospitalisation probability arrays must match y_true shape.")

    cv = float(cfg.vaccine_cost)
    ch = float(cfg.hospitalisation_cost)

    def _raw_cost(pred: np.ndarray) -> Tuple[float, float, float, float, float, float, int, int, int, int]:
        allocated_mask = pred == 1
        accepted_mask = allocated_mask & (y_t == 1)
        wasted_mask = allocated_mask & (y_t == 0)
        law_mask = (pred == 0) & (y_t == 1)

        vaccines_alloc = int(np.sum(allocated_mask))
        vaccines_acc = int(np.sum(accepted_mask))
        wasted_cnt = int(np.sum(wasted_mask))
        law_cnt = int(np.sum(law_mask))

        send_cost = float(vaccines_alloc * cv)
        accepted_cost = float(vaccines_acc * cv)
        wasted_cost = float(wasted_cnt * cv)
        law_cost = float(law_cnt * cv)

        p_hosp = np.where(accepted_mask, q1, q0)
        exp_hosp = float(np.sum(p_hosp))
        hosp_cost = exp_hosp * ch
        return (
            send_cost,
            law_cost,
            exp_hosp,
            hosp_cost,
            accepted_cost,
            wasted_cost,
            vaccines_alloc,
            vaccines_acc,
            wasted_cnt,
            law_cnt,
        )

    (
        send_cost,
        law_cost,
        exp_hosp,
        hosp_cost,
        accepted_cost,
        wasted_cost,
        vaccines_alloc,
        vaccines_acc,
        wasted_cnt,
        law_cnt,
    ) = _raw_cost(y_p)
    total_cost = send_cost + law_cost + hosp_cost

    _, none_law, _, none_hosp, *_ = _raw_cost(np.zeros_like(y_t))
    treat_none_cost = none_law + none_hosp

    all_send, _, _, all_hosp, *_ = _raw_cost(np.ones_like(y_t))
    treat_all_cost = all_send + all_hosp

    # Exact Oracle allocates to y_t == 1 iff vaccination actually lowers hospitalisation risk
    oracle_pred = ((y_t == 1) & (q0 > q1)).astype(np.int64)
    oracle_send, oracle_law, _, oracle_hosp, *_ = _raw_cost(oracle_pred)
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
        accepted_vaccine_cost=accepted_cost,
        wasted_vaccine_cost=wasted_cost,
        vaccines_allocated=vaccines_alloc,
        vaccines_accepted=vaccines_acc,
        wasted_vaccines=wasted_cnt,
        law_compensation_count=law_cnt,
    )


def tune_threshold_on_oof(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    p_hosp_unvacc: np.ndarray,
    p_hosp_vacc: np.ndarray,
    folds: np.ndarray | None = None,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> dict:
    """Compare 0.50 cutoff, global tuned, Bayes-optimal (zero tuning), and CV Bayes-scaled."""
    y_t = np.asarray(y_true, dtype=np.int64)
    probs, _ = _prepare_probabilities(oof_probs, "oof_probs", strict=False, nan_fill=0.0)
    results = {}

    # 1. Naive 0.50 cutoff
    pred_05 = (probs >= 0.5).astype(np.int64)
    results["global_0.5"] = {
        "policy": "global",
        "param": 0.5,
        "breakdown": compute_town_cost(y_t, pred_05, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

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

    if folds is None:
        from sklearn.model_selection import StratifiedKFold

        skf_fallback = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        folds_arr = np.zeros(len(y_t), dtype=np.int64)
        for f_idx, (_, va_idx) in enumerate(skf_fallback.split(probs.reshape(-1, 1), y_t)):
            folds_arr[va_idx] = f_idx
    else:
        folds_arr = np.asarray(folds, dtype=np.int64)

    pred_gt = np.zeros_like(y_t)
    pred_bs = np.zeros_like(y_t)
    g_params, s_params = [], []
    for f in np.unique(folds_arr):
        tr_mask = folds_arr != f
        va_mask = folds_arr == f
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

    results["global_tuned"] = {
        "policy": "global",
        "param": best_g_param,
        "evaluation": "strictly_out_of_fold_cv",
        "breakdown": compute_town_cost(y_t, pred_gt, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    # 3. Exact theoretical Bayes threshold (scale = 1.0, zero tuning)
    pred_bayes = predict_with_cost_policy(
        probs, p_hosp_unvacc, p_hosp_vacc, policy="bayes", threshold_param=1.0, cfg=cfg
    )
    results["bayes_exact"] = {
        "policy": "bayes",
        "param": 1.0,
        "evaluation": "analytical_zero_tuning",
        "breakdown": compute_town_cost(y_t, pred_bayes, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    # 4. Scaled Bayes threshold
    results["bayes_scaled"] = {
        "policy": "bayes",
        "param": best_b_scale,
        "evaluation": "strictly_out_of_fold_cv",
        "breakdown": compute_town_cost(y_t, pred_bs, p_hosp_unvacc, p_hosp_vacc, cfg),
    }

    return results


def summarize_cost_policies_table(policy_results: dict) -> pd.DataFrame:
    """Build complete cost-component breakdown across decision policies."""
    fixed_cost = policy_results["global_0.5"]["breakdown"].total_cost
    order = [
        ("global_0.5", "Fixed Threshold (0.50)"),
        ("global_tuned", "CV-Tuned Global Threshold (Out-of-Fold)"),
        ("bayes_exact", "Person-Specific Bayes Threshold (Analytical)"),
        ("bayes_scaled", "CV-Scaled Bayes Threshold (Out-of-Fold)"),
    ]
    rows = []
    for key, label in order:
        entry = policy_results[key]
        bd: CostBreakdown = entry["breakdown"]
        rows.append(
            {
                "policy_key": key,
                "policy_name": label,
                "param": float(entry["param"]),
                "evaluation_mode": entry.get("evaluation", "fixed_cutoff"),
                "vaccines_allocated": bd.vaccines_allocated,
                "vaccines_accepted": bd.vaccines_accepted,
                "wasted_vaccines": bd.wasted_vaccines,
                "law_compensation_count": bd.law_compensation_count,
                "expected_hospitalisations": bd.expected_hospitalisations,
                "accepted_vaccine_cost": bd.accepted_vaccine_cost,
                "wasted_vaccine_cost": bd.wasted_vaccine_cost,
                "vaccine_send_cost": bd.vaccine_send_cost,
                "law_penalty_cost": bd.law_penalty_cost,
                "hospitalisation_cost": bd.hospitalisation_cost,
                "total_cost": bd.total_cost,
                "savings_vs_treat_none": bd.savings_vs_treat_none,
                "savings_vs_fixed_0.5": fixed_cost - bd.total_cost,
            }
        )
    return pd.DataFrame(rows)


def evaluate_cost_sensitivity_grid(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cfg: DatasetConfig = DATASET_C_CONFIG,
) -> pd.DataFrame:
    """Evaluate fixed-0.50 vs person-specific Bayes decision policies across IHR x VE x Sex sensitivity."""
    rows = []
    specs = [
        ("conservative", "waned_omicron", True),
        ("conservative", "base", True),
        ("base", "waned_omicron", True),
        ("base", "base", True),
        ("base", "base", False),  # Sex-neutral sensitivity check
        ("base", "boosted", True),
        ("severe", "waned_omicron", True),
        ("severe", "base", True),
        ("severe", "boosted", True),
    ]
    y_t = np.asarray(y_true, dtype=np.int64)
    probs, _ = _prepare_probabilities(y_prob, "y_prob", strict=False, nan_fill=0.0)

    for scen, ve_reg, use_sex in specs:
        pu, pv = estimate_hospitalisation_probabilities(
            df,
            scenario=scen,  # type: ignore[arg-type]
            ve_regime=ve_reg,  # type: ignore[arg-type]
            include_sex_adjustment=use_sex,
        )
        pred_bayes = predict_with_cost_policy(probs, pu, pv, policy="bayes", threshold_param=1.0, cfg=cfg)
        pred_fixed = predict_with_cost_policy(probs, pu, pv, policy="global", threshold_param=0.5, cfg=cfg)
        cb_b = compute_town_cost(y_t, pred_bayes, pu, pv, cfg=cfg)
        cb_f = compute_town_cost(y_t, pred_fixed, pu, pv, cfg=cfg)
        rows.append(
            {
                "ihr_scenario": scen,
                "ve_regime": ve_reg,
                "include_sex_adj": use_sex,
                "mean_p_unvacc": float(np.mean(pu)),
                "mean_p_vacc": float(np.mean(pv)),
                "fixed_05_total_cost": cb_f.total_cost,
                "bayes_total_cost": cb_b.total_cost,
                "bayes_savings_vs_fixed_05": cb_f.total_cost - cb_b.total_cost,
                "bayes_vaccines_allocated": cb_b.vaccines_allocated,
                "bayes_wasted_vaccines": cb_b.wasted_vaccines,
                "bayes_law_comp_count": cb_b.law_compensation_count,
                "bayes_expected_hosp": cb_b.expected_hospitalisations,
                "bayes_captured_oracle_pct": cb_b.captured_oracle_savings_pct,
            }
        )
    return pd.DataFrame(rows)
