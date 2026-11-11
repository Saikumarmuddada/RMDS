"""Exploratory Data Analysis (EDA) and publication-ready visualization generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import DATASET_C_CONFIG, DatasetConfig, FIGURES_DIR, OUTPUT_DIR
from src.cost_model import estimate_hospitalisation_probabilities


sns.set_theme(style="whitegrid", context="notebook", palette="deep")


def run_eda(
    raw_train: pd.DataFrame,
    raw_test: pd.DataFrame,
    feat_train: pd.DataFrame,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    output_dir: Path = OUTPUT_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> Dict:
    """Compute EDA statistics and save embellished EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    target = cfg.target_col
    y = raw_train[target]

    # Missingness rates comparison between train and test
    feature_cols = [c for c in raw_test.columns if c != cfg.id_col]
    missing_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "train_missing_pct": [raw_train[c].isna().mean() * 100.0 for c in feature_cols],
            "test_missing_pct": [raw_test[c].isna().mean() * 100.0 for c in feature_cols],
        }
    ).sort_values("train_missing_pct", ascending=False)
    missing_df.to_csv(output_dir / "eda_missingness.csv", index=False)

    # Vaccine acceptance rate by key subgroups
    subgroup_cols = [
        f"doctor_recc_{cfg.disease_prefix}",
        f"opinion_{cfg.disease_prefix}_vacc_effective",
        f"opinion_{cfg.disease_prefix}_risk",
        f"opinion_{cfg.disease_prefix}_sick_from_vacc",
        "age_group",
        "health_worker",
        "number_chronic_med_conditions",
        "health_insurance",
    ]
    subgroup_rates = {}
    for col in subgroup_cols:
        grp = (
            raw_train.assign(_val=raw_train[col].fillna("Missing").astype(str))
            .groupby("_val")[target]
            .agg(["count", "mean"])
            .rename(columns={"mean": "vaccine_rate"})
        )
        subgroup_rates[col] = grp.to_dict(orient="index")

    # Numeric feature correlations with target
    num_cols = feat_train.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in {cfg.id_col, target}]
    corrs = (
        feat_train[num_cols]
        .corrwith(feat_train[target])
        .dropna()
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )
    corrs.to_frame("pearson_corr_with_target").to_csv(output_dir / "eda_correlations.csv")

    p_unvacc, p_vacc = estimate_hospitalisation_probabilities(feat_train, scenario="base")

    eda_summary = {
        "train_rows": int(len(raw_train)),
        "test_rows": int(len(raw_test)),
        "raw_feature_count": len(feature_cols),
        "engineered_feature_count": int(feat_train.shape[1] - 2),
        "target_positive_count": int(y.sum()),
        "target_negative_count": int((1 - y).sum()),
        "target_positive_rate": float(y.mean()),
        "top_missing_features": missing_df.head(8).to_dict(orient="records"),
        "top_target_correlations": [
            {"feature": k, "corr": float(v)} for k, v in corrs.head(12).items()
        ],
        "mean_estimated_unvacc_hosp_prob": float(np.mean(p_unvacc)),
        "mean_estimated_vacc_hosp_prob": float(np.mean(p_vacc)),
        "subgroup_rates": subgroup_rates,
    }
    (output_dir / "eda_summary.json").write_text(
        json.dumps(eda_summary, indent=2), encoding="utf-8"
    )

    _plot_target_and_missingness(raw_train, missing_df, target, figures_dir)
    _plot_key_drivers(raw_train, cfg, figures_dir)
    _plot_hospitalisation_risk_profile(feat_train, p_unvacc, p_vacc, target, figures_dir)

    return eda_summary


def _plot_target_and_missingness(
    raw_train: pd.DataFrame,
    missing_df: pd.DataFrame,
    target: str,
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    counts = raw_train[target].value_counts().sort_index()
    labels = ["No Vaccine (0)", "Accepted Vaccine (1)"]
    colors = ["#5C7AEA", "#2EC4B6"]
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.8)
    total = len(raw_train)
    for bar, val in zip(bars, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            val + 45,
            f"{val:,}\n({val / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    axes[0].set_ylim(0, counts.max() * 1.18)
    axes[0].set_title("Training Target Distribution (covid_vaccine)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Number of Respondents")

    top_miss = missing_df[missing_df["train_missing_pct"] > 0.3].head(12).iloc[::-1]
    y_pos = np.arange(len(top_miss))
    h = 0.38
    axes[1].barh(y_pos - h / 2, top_miss["train_missing_pct"], height=h, label="Train", color="#3D5A80")
    axes[1].barh(y_pos + h / 2, top_miss["test_missing_pct"], height=h, label="Test", color="#EE6C4D")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_miss["feature"])
    axes[1].set_xlabel("Missing Values (%)")
    axes[1].set_title("Missingness Pattern: Train vs. Test Consistency", fontsize=13, fontweight="bold")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figures_dir / "01_target_and_missingness.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_key_drivers(
    raw_train: pd.DataFrame,
    cfg: DatasetConfig,
    figures_dir: Path,
) -> None:
    p = cfg.disease_prefix
    target = cfg.target_col
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    panels = [
        (f"doctor_recc_{p}", "Doctor Recommended COVID Vaccine", axes[0, 0]),
        (f"opinion_{p}_vacc_effective", "Opinion: Vaccine Effectiveness (0=Unknown, 1-5)", axes[0, 1]),
        (f"opinion_{p}_risk", "Opinion: Risk Without Vaccine (0=Unknown, 1-5)", axes[1, 0]),
        ("age_group", "Vaccine Acceptance by Age Group", axes[1, 1]),
    ]

    for col, title, ax in panels:
        order = sorted(raw_train[col].dropna().unique(), key=lambda x: str(x))
        if col == "age_group":
            order = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
        sub = (
            raw_train.assign(_k=raw_train[col].fillna("Missing"))
            .groupby("_k")[target]
            .agg(["mean", "count"])
        )
        keys = [k for k in order if k in sub.index]
        if "Missing" in sub.index:
            keys.append("Missing")
        sub = sub.loc[keys]

        x = np.arange(len(sub))
        bars = ax.bar(x, sub["mean"] * 100.0, color="#2A9D8F", edgecolor="black", linewidth=0.7)
        ax.axhline(raw_train[target].mean() * 100.0, color="#E76F51", linestyle="--", linewidth=1.4, label="Overall Avg (32.7%)")
        for bar, (_, row) in zip(bars, sub.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{row['mean'] * 100:.1f}%\n(n={int(row['count'])})",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in keys], rotation=15 if col == "age_group" else 0)
        ax.set_ylim(0, min(105, max(sub["mean"] * 100.0) + 16))
        ax.set_ylabel("Vaccine Acceptance Rate (%)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(figures_dir / "02_key_drivers_acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_hospitalisation_risk_profile(
    feat_train: pd.DataFrame,
    p_unvacc: np.ndarray,
    p_vacc: np.ndarray,
    target: str,
    figures_dir: Path,
) -> None:
    df_plot = feat_train[["age_group", "number_chronic_med_conditions", target]].copy()
    df_plot["p_unvacc"] = p_unvacc * 100.0
    df_plot["p_vacc"] = p_vacc * 100.0
    df_plot["expected_hosp_saving"] = (p_unvacc - p_vacc) * 41000.0

    age_order = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    age_grp = df_plot.groupby("age_group")[["p_unvacc", "p_vacc"]].mean().reindex(age_order)
    x = np.arange(len(age_order))
    w = 0.36
    axes[0].bar(x - w / 2, age_grp["p_unvacc"], width=w, label="Unvaccinated Risk (%)", color="#E63946")
    axes[0].bar(x + w / 2, age_grp["p_vacc"], width=w, label="Vaccinated Risk (%)", color="#457B9D")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(age_order, rotation=15)
    axes[0].set_ylabel("Estimated Hospitalisation Probability (%)")
    axes[0].set_title("Base Hospitalisation Risk Model by Age Group", fontsize=12.5, fontweight="bold")
    axes[0].legend()

    df_plot["chronic_bucket"] = df_plot["number_chronic_med_conditions"].fillna(0).clip(0, 3).astype(int)
    pivot = df_plot.pivot_table(
        index="age_group",
        columns="chronic_bucket",
        values="expected_hosp_saving",
        aggfunc="mean",
    ).reindex(age_order)
    pivot.columns = ["0 Cond.", "1 Cond.", "2 Cond.", "3+ Cond."]
    sns.heatmap(
        pivot,
        annot=True,
        fmt=",.0f",
        cmap="YlOrRd",
        ax=axes[1],
        cbar_kws={"label": "Expected Hospitalisation Cost Saved ($)"},
    )
    axes[1].set_title("Expected Hospitalisation Savings ($) if Vaccinated", fontsize=12.5, fontweight="bold")
    axes[1].set_xlabel("Chronic Medical Conditions")
    axes[1].set_ylabel("Age Group")

    plt.tight_layout()
    plt.savefig(figures_dir / "03_hospitalisation_risk_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

