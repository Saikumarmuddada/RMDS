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
from src.cost_model import (
    compute_town_cost,
    estimate_hospitalisation_probabilities,
    evaluate_cost_sensitivity_grid,
    predict_with_cost_policy,
)
from src.modeling import ModelRunResult, apply_category_vocab


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
        label_txt = row["model"]
        offset = (6, 4)
        ha = "left"
        if label_txt == "Uncalibrated_Stack":
            offset = (-10, 4)
            ha = "right"
        elif label_txt == "Calibrated_Stack":
            offset = (-10, -14)
            ha = "right"
        elif label_txt == "LightGBM":
            offset = (6, -10)
        axes[1].annotate(
            label_txt,
            (row["roc_auc"], row["captured_oracle_pct"]),
            textcoords="offset points",
            xytext=offset,
            ha=ha,
            fontsize=8.8,
        )
    auc_span = max(df_s["roc_auc"].max() - df_s["roc_auc"].min(), 1e-3)
    axes[1].set_xlim(df_s["roc_auc"].min() - 0.08 * auc_span, df_s["roc_auc"].max() + 0.18 * auc_span)
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

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))

    display_models = {
        "LogisticRegression",
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "Uncalibrated_Stack",
        "Calibrated_Stack",
    }
    for r in results:
        if r.name in display_models:
            frac_pos, mean_pred = calibration_curve(y, r.oof_probs, n_bins=10, strategy="quantile")
            axes[0].plot(
                mean_pred,
                frac_pos,
                marker="o",
                linewidth=1.8,
                label=f"{r.name} (Brier={r.brier:.4f}, ECE={r.ece:.4f})",
            )
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    axes[0].set_xlabel("Predicted Probability P(covid_vaccine = 1)")
    axes[0].set_ylabel("Observed Acceptance Fraction")
    axes[0].set_title("Out-of-Fold Probability Calibration Curves (ECE & Brier)", fontsize=12.5, fontweight="bold")
    axes[0].legend(fontsize=8.0)

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


def _extract_positive_class_shap(raw_shap) -> np.ndarray:
    """Normalize SHAP output across versions into a 2D array (n_samples, n_features)."""
    if isinstance(raw_shap, list):
        return np.asarray(raw_shap[1], dtype=np.float64)
    arr = np.asarray(raw_shap, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return arr[:, :, 1]
    return arr


def _compute_and_plot_shap(
    feat_train: pd.DataFrame,
    feat_test: pd.DataFrame,
    artifacts: dict,
    figures_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_cols: List[str] = artifacts["feature_cols"]
    num_cols: List[str] = artifacts["num_cols"]
    cat_cols: List[str] = artifacts["cat_cols"]
    cat_set = set(cat_cols)
    lgbm_models = artifacts["lgbm_models"]
    lgbm_fold_vocabs = artifacts.get("lgbm_fold_vocabs")
    folds = artifacts["folds"]
    unique_folds = sorted(np.unique(folds).tolist())

    # Out-of-fold SHAP matrix across ALL outer folds (never fitted and explained on same fold)
    oof_shap = np.zeros((len(feat_train), len(feature_cols)), dtype=np.float64)
    fold_mean_abs = np.zeros((len(unique_folds), len(feature_cols)), dtype=np.float64)

    for idx_pos, f_id in enumerate(unique_folds):
        val_mask = folds == f_id
        X_va = feat_train.loc[val_mask, feature_cols]
        if lgbm_fold_vocabs and idx_pos < len(lgbm_fold_vocabs):
            X_va_cat = apply_category_vocab(X_va, lgbm_fold_vocabs[idx_pos], for_catboost=False)
        else:
            X_tr = feat_train.loc[~val_mask, feature_cols]
            vocabs = {}
            for col in cat_cols:
                vals = X_tr[col].astype("string").str.strip().dropna().unique()
                vocabs[col] = pd.Index(vals).union(pd.Index(["Missing"])).sort_values()
            X_va_cat = apply_category_vocab(X_va, vocabs, for_catboost=False)

        explainer = shap.TreeExplainer(lgbm_models[idx_pos])
        sv = _extract_positive_class_shap(explainer.shap_values(X_va_cat))
        oof_shap[val_mask, :] = sv
        fold_mean_abs[idx_pos, :] = np.abs(sv).mean(axis=0)

    overall_mean_abs = np.abs(oof_shap).mean(axis=0)
    fold_mean = fold_mean_abs.mean(axis=0)
    fold_std = fold_mean_abs.std(axis=0, ddof=1) if len(unique_folds) > 1 else np.zeros_like(fold_mean)

    # Fold rank stability
    fold_ranks = np.argsort(np.argsort(-fold_mean_abs, axis=1), axis=1) + 1
    mean_rank = fold_ranks.mean(axis=0)
    std_rank = fold_ranks.std(axis=0, ddof=1) if len(unique_folds) > 1 else np.zeros_like(mean_rank)

    shap_data = {
        "feature": feature_cols,
        "feature_type": ["categorical" if c in cat_set else "numeric" for c in feature_cols],
        "mean_abs_shap": overall_mean_abs,
        "fold_mean_abs_shap": fold_mean,
        "fold_std_abs_shap": fold_std,
        "mean_rank": mean_rank,
        "std_rank": std_rank,
    }
    for idx_pos, f_id in enumerate(unique_folds):
        shap_data[f"fold_{f_id}_mean_abs_shap"] = fold_mean_abs[idx_pos]

    shap_df = pd.DataFrame(shap_data).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    # 1. Global SHAP Importance with across-fold SD error bars & Numeric/Categorical color key
    top_18 = shap_df.head(18).iloc[::-1]
    type_colors = [
        "#E07A5F" if ft == "categorical" else "#3D5A80"
        for ft in top_18["feature_type"]
    ]
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.barh(
        top_18["feature"],
        top_18["mean_abs_shap"],
        xerr=top_18["fold_std_abs_shap"],
        capsize=3,
        color=type_colors,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.92,
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#3D5A80", edgecolor="black", label="Numeric / Ordinal / Binary"),
            Patch(facecolor="#E07A5F", edgecolor="black", label="Nominal Categorical"),
        ],
        loc="lower right",
    )
    ax.set_xlabel("Out-of-Fold Mean |SHAP Value| ± 1 Fold SD (Impact on Acceptance Log-Odds)")
    ax.set_title(
        "Explainable AI: Multi-Fold OOF LightGBM SHAP Importance & Stability",
        fontsize=12.5,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(figures_dir / "06_explainability_shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # 2. Numeric-ONLY Directional SHAP Beeswarm (never map categorical strings to integer codes)
    num_indices = [feature_cols.index(c) for c in num_cols]
    if num_indices:
        X_num_oof = feat_train[num_cols].apply(pd.to_numeric, errors="coerce")
        oof_shap_num = oof_shap[:, num_indices]
        plt.figure(figsize=(11, 7.5))
        shap.summary_plot(oof_shap_num, X_num_oof, max_display=16, show=False)
        plt.title(
            "Explainable AI: Numeric & Ordinal Feature SHAP Beeswarm (All Outer OOF Folds)",
            fontsize=12.5,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(figures_dir / "08_explainability_shap_beeswarm.png", dpi=180, bbox_inches="tight")
        plt.close()

    # 3. Separate Categorical Feature Level SHAP Visualization (true category labels, no .cat.codes!)
    cat_top = (
        shap_df[shap_df["feature_type"] == "categorical"]
        .head(4)["feature"]
        .tolist()
    )
    if cat_top:
        fig, axes = plt.subplots(len(cat_top), 1, figsize=(11, 3.2 * len(cat_top)))
        if len(cat_top) == 1:
            axes = [axes]
        for ax_c, col in zip(axes, cat_top):
            c_idx = feature_cols.index(col)
            tmp = pd.DataFrame(
                {
                    "category": feat_train[col].astype("string").str.strip().fillna("Missing"),
                    "shap_value": oof_shap[:, c_idx],
                }
            )
            cat_stats = (
                tmp.groupby("category", as_index=False)["shap_value"]
                .agg(["mean", "count"])
                .sort_values("mean")
            )
            # Keep top/bottom categories if high cardinality
            if len(cat_stats) > 10:
                cat_stats = cat_stats.sort_values("count", ascending=False).head(10).sort_values("mean")
            bar_cols = ["#2EC4B6" if v >= 0 else "#E63946" for v in cat_stats["mean"]]
            labels = [f"{cat} (n={int(cnt)})" for cat, cnt in zip(cat_stats["category"], cat_stats["count"])]
            ax_c.barh(labels, cat_stats["mean"], color=bar_cols, edgecolor="black", linewidth=0.5)
            ax_c.axvline(0.0, color="black", linewidth=0.8)
            ax_c.set_title(f"Categorical Driver `{col}`: Mean OOF SHAP by Category", fontsize=11, fontweight="bold")
            ax_c.set_xlabel("Mean Signed SHAP Value (Log-Odds Contribution)")
        plt.tight_layout()
        plt.savefig(figures_dir / "08b_explainability_categorical_shap.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

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

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    meta_weights = artifacts.get("meta_weights", {})
    if meta_weights:
        mw = pd.Series(meta_weights)
        meta_per_fold = artifacts.get("meta_coefs_per_fold")
        if meta_per_fold is not None:
            coef_arr = np.asarray(meta_per_fold, dtype=np.float64)
            std_s = pd.Series(coef_arr.std(axis=0, ddof=1), index=list(meta_weights.keys())) if coef_arr.shape[0] > 1 else pd.Series(0.0, index=mw.index)
        else:
            std_s = pd.Series(0.0, index=mw.index)
        order_idx = mw.sort_values().index
        mw = mw.loc[order_idx]
        std_s = std_s.loc[order_idx]
        colors = ["#2EC4B6" if v >= 0 else "#E63946" for v in mw.values]
        axes[0].barh(mw.index, mw.values, xerr=std_s.values, capsize=3, color=colors, edgecolor="black", linewidth=0.6)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Meta-Learner Logistic Regression Log-Odds Coefficient (± 1 Fold SD)")
        axes[0].set_title("Meta-Learner Logistic Regression Coefficients (Log-Odds)", fontsize=12.2, fontweight="bold")

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

    best_res = min(results, key=lambda r: r.policy_results["bayes_exact"]["breakdown"].total_cost)
    ve_sens_df = evaluate_cost_sensitivity_grid(feat_train, y, best_res.oof_probs, cfg=cfg)
    ve_sens_df.to_csv(output_dir / "ve_and_risk_sensitivity_grid.csv", index=False)

