"""Explainable AI (SHAP + Permutation Importance) and embellished presentation figures."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.calibration import calibration_curve

from src.config import DATASET_C_CONFIG, DatasetConfig, FIGURES_DIR, OUTPUT_DIR
from src.cost_model import compute_town_cost, estimate_hospitalisation_probabilities, predict_with_cost_policy
from src.modeling import ModelRunResult, _prepare_category_frames


sns.set_theme(style="whitegrid", context="notebook")


def run_explainability_and_plots(
    results: List[ModelRunResult],
    comp_df: pd.DataFrame,
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    artifacts: dict,
    cfg: DatasetConfig = DATASET_C_CONFIG,
    output_dir: Path = OUTPUT_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> Dict:
    """Compute SHAP global feature importances and save model/cost comparison visuals."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_model_cost_comparison(comp_df, feat_train, cfg, figures_dir)
    _plot_calibration_and_thresholds(results, feat_train, cfg, figures_dir)

    shap_df = _compute_and_plot_shap(feat_train, feat_test, artifacts, figures_dir, output_dir)
    _plot_cost_savings_by_segment(results, feat_train, cfg, figures_dir)
    _plot_meta_weights_and_sensitivity(results, feat_train, artifacts, cfg, figures_dir, output_dir)

    return {
        "top_shap_features": shap_df.head(15).to_dict(orient="records"),
    }


def _plot_model_cost_comparison(
    comp_df: pd.DataFrame,
    feat_train: pd.DataFrame,
    cfg: DatasetConfig,
    figures_dir: Path,
) -> None:
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    bd_none = compute_town_cost(y, np.zeros_like(y), pu, pv, cfg)
    bd_all = compute_town_cost(y, np.ones_like(y), pu, pv, cfg)
    bd_oracle = compute_town_cost(y, y, pu, pv, cfg)

    df_s = comp_df.sort_values("cost_bayes_exact", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    y_pos = np.arange(len(df_s))
    h = 0.35
    axes[0].barh(
        y_pos - h / 2,
        df_s["cost_global_0.5"] / 1e6,
        height=h,
        label="Naive Threshold (0.50)",
        color="#8D99AE",
    )
    axes[0].barh(
        y_pos + h / 2,
        df_s["cost_bayes_exact"] / 1e6,
        height=h,
        label="Cost-Optimal Bayes Policy",
        color="#2EC4B6",
    )
    axes[0].axvline(
        bd_none.total_cost / 1e6,
        color="#E63946",
        linestyle="--",
        linewidth=1.5,
        label=f"Treat None (${bd_none.total_cost / 1e6:.2f}M)",
    )
    axes[0].axvline(
        bd_all.total_cost / 1e6,
        color="#F4A261",
        linestyle=":",
        linewidth=1.8,
        label=f"Treat All (${bd_all.total_cost / 1e6:.2f}M)",
    )
    axes[0].axvline(
        bd_oracle.total_cost / 1e6,
        color="#1D3557",
        linestyle="-.",
        linewidth=1.5,
        label=f"Oracle Lower Bound (${bd_oracle.total_cost / 1e6:.2f}M)",
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_s["model"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Expected Total Town Cost ($ Millions, OOF)")
    axes[0].set_title("Model Town Cost: Naive 0.5 vs. Bayes Cost-Optimal", fontsize=12.5, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="lower right")

    # Right panel: ROC-AUC vs Cost Savings Captured (%)
    axes[1].scatter(
        df_s["roc_auc"],
        df_s["captured_oracle_pct"],
        s=150,
        c=np.arange(len(df_s)),
        cmap="viridis",
        edgecolor="black",
    )
    for _, row in df_s.iterrows():
        axes[1].annotate(
            row["model"],
            (row["roc_auc"], row["captured_oracle_pct"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    axes[1].set_xlabel("5-Fold CV ROC-AUC")
    axes[1].set_ylabel("Oracle Achievable Cost Savings Captured (%)")
    axes[1].set_title("Discrimination (ROC-AUC) vs. Cost Optimization Performance", fontsize=12.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "04_model_cost_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_calibration_and_thresholds(
    results: List[ModelRunResult],
    feat_train: pd.DataFrame,
    cfg: DatasetConfig,
    figures_dir: Path,
) -> None:
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    best_res = min(results, key=lambda r: r.policy_results["bayes_exact"]["breakdown"].total_cost)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for r in results:
        if r.name in {"LogisticRegression", "RandomForest", "LightGBM", "CatBoost", "Calibrated_Stack"}:
            frac_pos, mean_pred = calibration_curve(y, r.oof_probs, n_bins=10, strategy="quantile")
            axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=1.8, label=f"{r.name} (Brier={r.brier:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].set_xlabel("Predicted Probability P(covid_vaccine = 1)")
    axes[0].set_ylabel("Observed Acceptance Fraction")
    axes[0].set_title("Probability Calibration Curves (5-Fold OOF)", fontsize=12.5, fontweight="bold")
    axes[0].legend(fontsize=8.5)

    # Sweep global threshold vs cost and compare to Bayes-exact line
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = [
        compute_town_cost(
            y, (best_res.oof_probs >= t).astype(np.int64), pu, pv, cfg
        ).total_cost / 1e6
        for t in thresholds
    ]
    bayes_cost = best_res.policy_results["bayes_exact"]["breakdown"].total_cost / 1e6
    axes[1].plot(thresholds, costs, color="#457B9D", linewidth=2.2, label="Global Constant Threshold")
    axes[1].axhline(bayes_cost, color="#2EC4B6", linestyle="--", linewidth=2.2, label=f"Per-Person Bayes Policy (${bayes_cost:.3f}M)")
    axes[1].axvline(0.50, color="#E63946", linestyle=":", linewidth=1.6, label="Naive 0.50 Cutoff")
    axes[1].set_xlabel("Global Probability Threshold")
    axes[1].set_ylabel("Total Town Cost ($ Millions)")
    axes[1].set_title(f"Why Per-Person Risk Thresholding Beats Fixed Cutoffs ({best_res.name})", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / "05_calibration_and_thresholds.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _compute_and_plot_shap(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    artifacts: dict,
    figures_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    feature_cols = artifacts["feature_cols"]
    cat_cols = artifacts["cat_cols"]
    lgbm_model = artifacts["lgbm_models"][0]
    folds = artifacts["folds"]

    val_mask = folds == 0
    X_tr = feat_train.loc[~val_mask, feature_cols]
    X_va = feat_train.loc[val_mask, feature_cols]
    X_te = feat_test[feature_cols]
    _, X_va_cat, _ = _prepare_category_frames(X_tr, X_va, X_te, cat_cols, for_catboost=False)

    explainer = shap.TreeExplainer(lgbm_model)
    shap_values = explainer.shap_values(X_va_cat)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame(
        {"feature": feature_cols, "mean_abs_shap": mean_abs_shap}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    top_18 = shap_df.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top_18["feature"], top_18["mean_abs_shap"], color="#3D5A80", edgecolor="black", linewidth=0.6)
    ax.set_xlabel("Mean |SHAP Value| (Impact on Log-Odds of Vaccine Acceptance)")
    ax.set_title("Explainable AI: Top 18 Drivers of COVID-19 Vaccine Acceptance (LightGBM SHAP)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "06_explainability_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Directional SHAP beeswarm plot (convert categoricals to numeric codes only for dot coloring)
    X_va_disp = X_va_cat.copy()
    for col in cat_cols:
        X_va_disp[col] = X_va_disp[col].cat.codes.replace(-1, np.nan).astype(float)

    plt.figure(figsize=(11, 7.5))
    shap.summary_plot(shap_vals, X_va_disp, max_display=16, show=False)
    plt.title("Explainable AI: SHAP Directional Impact (Red = Higher Feature Value)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "08_explainability_shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    return shap_df


def _plot_cost_savings_by_segment(
    results: List[ModelRunResult],
    feat_train: pd.DataFrame,
    cfg: DatasetConfig,
    figures_dir: Path,
) -> None:
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    best_res = min(results, key=lambda r: r.policy_results["bayes_exact"]["breakdown"].total_cost)

    pred_bayes = predict_with_cost_policy(
        best_res.oof_probs, pu, pv, policy="bayes", threshold_param=1.0, cfg=cfg
    )
    pred_05 = (best_res.oof_probs >= 0.5).astype(np.int64)

    df_seg = feat_train[["age_group"]].copy()
    df_seg["y_true"] = y
    df_seg["pred_bayes"] = pred_bayes
    df_seg["pred_05"] = pred_05

    age_order = ["18 - 34 Years", "35 - 44 Years", "45 - 54 Years", "55 - 64 Years", "65+ Years"]
    rows = []
    for age in age_order:
        mask = (df_seg["age_group"] == age).to_numpy()
        bd_b = compute_town_cost(y[mask], pred_bayes[mask], pu[mask], pv[mask], cfg)
        bd_05 = compute_town_cost(y[mask], pred_05[mask], pu[mask], pv[mask], cfg)
        bd_none = compute_town_cost(y[mask], np.zeros(mask.sum(), dtype=int), pu[mask], pv[mask], cfg)
        rows.append(
            {
                "age_group": age,
                "Treat None": bd_none.cost_per_person,
                "Naive 0.5": bd_05.cost_per_person,
                "Bayes Optimal": bd_b.cost_per_person,
                "bayes_send_rate": pred_bayes[mask].mean() * 100.0,
            }
        )
    seg_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(age_order))
    w = 0.25
    axes[0].bar(x - w, seg_df["Treat None"], width=w, label="Treat None", color="#E63946")
    axes[0].bar(x, seg_df["Naive 0.5"], width=w, label="Naive 0.5 Cutoff", color="#8D99AE")
    axes[0].bar(x + w, seg_df["Bayes Optimal"], width=w, label="Bayes Cost-Optimal", color="#2EC4B6")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(age_order, rotation=15)
    axes[0].set_ylabel("Expected Cost per Person ($)")
    axes[0].set_title("Per-Person Town Cost by Age Segment", fontsize=12.5, fontweight="bold")
    axes[0].legend()

    axes[1].bar(x, seg_df["bayes_send_rate"], color="#457B9D", edgecolor="black", linewidth=0.7)
    for idx, val in enumerate(seg_df["bayes_send_rate"]):
        axes[1].text(idx, val + 1.5, f"{val:.1f}%", ha="center", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(age_order, rotation=15)
    axes[1].set_ylim(0, 108)
    axes[1].set_ylabel("Share Sent a Vaccine (%)")
    axes[1].set_title("Vaccine Allocation Rate Under Cost-Optimal Policy", fontsize=12.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(figures_dir / "07_cost_savings_by_segment.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_meta_weights_and_sensitivity(
    results: List[ModelRunResult],
    feat_train: pd.DataFrame,
    artifacts: dict,
    cfg: DatasetConfig,
    figures_dir: Path,
    output_dir: Path,
) -> None:
    """Save multi-scenario robustness matrix (sensitivity across risk assumptions) and meta weights plot."""
    y = feat_train[cfg.target_col].to_numpy(dtype=np.int64)
    sens_rows = []
    for r in results:
        for scen in ("conservative", "base", "severe"):
            pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario=scen)
            pred_b = predict_with_cost_policy(
                r.oof_probs, pu, pv, policy="bayes", threshold_param=1.0, cfg=cfg
            )
            bd = compute_town_cost(y, pred_b, pu, pv, cfg)
            sens_rows.append(
                {
                    "model": r.name,
                    "scenario": scen,
                    "total_cost": bd.total_cost,
                    "cost_per_person": bd.cost_per_person,
                    "captured_oracle_pct": bd.captured_oracle_savings_pct,
                }
            )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(output_dir / "sensitivity_matrix.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    meta_weights = artifacts.get("meta_weights", {})
    if meta_weights:
        mw = pd.Series(meta_weights).sort_values()
        colors = ["#2EC4B6" if v >= 0 else "#E63946" for v in mw.values]
        axes[0].barh(mw.index, mw.values, color=colors, edgecolor="black", linewidth=0.6)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Log-Odds Meta-Learner Coefficient")
        axes[0].set_title("Calibrated Stack Blend Weights", fontsize=12.5, fontweight="bold")

    piv = sens_df.pivot(index="model", columns="scenario", values="captured_oracle_pct")[
        ["conservative", "base", "severe"]
    ].sort_values("base", ascending=False)
    sns.heatmap(
        piv,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        ax=axes[1],
        cbar_kws={"label": "Oracle Savings Captured (%)"},
    )
    axes[1].set_title("Robustness Across Hospitalisation Risk Scenarios (%)", fontsize=12.5, fontweight="bold")
    axes[1].set_xlabel("Hospitalisation Risk Scenario")
    axes[1].set_ylabel("Model")

    plt.tight_layout()
    plt.savefig(figures_dir / "09_meta_weights_and_sensitivity.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

