"""Main entry point for the RMDS Hackathon Dataset C end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import DATASET_C_CONFIG, FIGURES_DIR, OUTPUT_DIR, SUBMISSIONS_DIR
from src.cost_model import compute_town_cost, estimate_hospitalisation_probabilities
from src.data_prep import engineer_features, load_raw_datasets
from src.eda import run_eda
from src.explainability import run_explainability_and_plots
from src.modeling import export_top5_submissions, train_and_evaluate_all_models


def write_markdown_report(
    eda_summary: dict,
    comp_df,
    manifest_df,
    explain_summary: dict,
    meta_weights: dict,
    feat_train,
    out_path: Path,
) -> None:
    """Write a complete hackathon summary report with observed CV and cost results."""
    y = feat_train[DATASET_C_CONFIG.target_col].to_numpy(dtype=int)
    pu, pv = estimate_hospitalisation_probabilities(feat_train, scenario="base")
    bd_none = compute_town_cost(y, y * 0, pu, pv, DATASET_C_CONFIG)
    bd_all = compute_town_cost(y, y * 0 + 1, pu, pv, DATASET_C_CONFIG)
    bd_oracle = compute_town_cost(y, y, pu, pv, DATASET_C_CONFIG)

    lines = [
        "# RMDS Hackathon — Dataset C (COVID-19) Implementation Report",
        "",
        "## 1. Dataset Overview",
        f"- **Training rows:** {eda_summary['train_rows']:,} (`covid_vaccine = 1`: {eda_summary['target_positive_count']:,} / {eda_summary['target_positive_rate']*100:.2f}%)",
        f"- **Testing rows:** {eda_summary['test_rows']:,}",
        f"- **Features:** {eda_summary['raw_feature_count']} raw features -> {eda_summary['engineered_feature_count']} cleaned + engineered features",
        "",
        "## 2. Town Cost Baselines (Training OOF Population, n=4,756)",
        f"- **Treat None (0% sent):** ${bd_none.total_cost:,.0f} (${bd_none.cost_per_person:,.2f}/person)",
        f"- **Treat All (100% sent):** ${bd_all.total_cost:,.0f} (${bd_all.cost_per_person:,.2f}/person)",
        f"- **Oracle Lower Bound (perfect foresight):** ${bd_oracle.total_cost:,.0f} (${bd_oracle.cost_per_person:,.2f}/person)",
        "",
        "## 3. 5-Fold Stratified CV Model Comparison",
        "",
        "| Model | ROC-AUC | LogLoss | Cost @ 0.50 ($) | Cost @ Bayes-Exact ($) | Per-Person ($) | Oracle Savings Captured |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in comp_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['roc_auc']:.4f} | {row['log_loss']:.4f} | "
            f"${row['cost_global_0.5']:,.0f} | ${row['cost_bayes_exact']:,.0f} | "
            f"${row['bayes_exact_per_person']:,.2f} | {row['captured_oracle_pct']:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## 4. Calibrated Stacking Meta-Weights (Log-Odds Coefficients)",
        ]
    )
    for m, w in sorted(meta_weights.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{m}`: {w:.4f}")

    lines.extend(
        [
            "",
            "## 5. Top 10 Explainable AI Drivers (LightGBM Mean |SHAP|)",
        ]
    )
    for item in explain_summary["top_shap_features"][:10]:
        lines.append(f"- `{item['feature']}`: {item['mean_abs_shap']:.4f}")

    lines.extend(
        [
            "",
            "## 6. Top 5 Ranked Submission Files",
            "",
            "| Order | Filename | Model | Policy | OOF Total Cost ($) | Test Predicted Vaccinations |",
            "| ---: | :--- | :--- | :--- | ---: | ---: |",
        ]
    )
    for _, row in manifest_df.iterrows():
        lines.append(
            f"| {int(row['order'])} | `{row['filename']}` | {row['model']} | {row['policy']} | "
            f"${row['oof_total_cost']:,.0f} | {int(row['test_pred_positive_count']):,} ({row['test_pred_positive_pct']:.1f}%) |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RMDS Dataset C Hackathon pipeline.")
    parser.add_argument("--group", type=int, default=0, help="Hackathon group number X for submission filenames.")
    args = parser.parse_args()

    print("[1/5] Loading and engineering Dataset C features...")
    raw_train, raw_test, example_sub = load_raw_datasets(DATASET_C_CONFIG)
    feat_train = engineer_features(raw_train, DATASET_C_CONFIG)
    feat_test = engineer_features(raw_test, DATASET_C_CONFIG)

    print("[2/5] Running EDA and generating figures...")
    eda_summary = run_eda(raw_train, raw_test, feat_train, DATASET_C_CONFIG, OUTPUT_DIR, FIGURES_DIR)

    print("[3/5] Training 6 diverse model families + calibrated stacking ensemble (5-fold Stratified CV)...")
    results, comp_df, _, artifacts = train_and_evaluate_all_models(
        feat_train, feat_test, DATASET_C_CONFIG, OUTPUT_DIR
    )

    print("[4/5] Running Explainable AI (SHAP) and cost-analysis visualizations...")
    explain_summary = run_explainability_and_plots(
        results, comp_df, feat_train, feat_test, artifacts, DATASET_C_CONFIG, OUTPUT_DIR, FIGURES_DIR
    )

    print(f"[5/5] Exporting top 5 submission CSVs for Group {args.group}...")
    manifest_df = export_top5_submissions(
        results,
        feat_train,
        feat_test,
        example_sub,
        group_id=args.group,
        cfg=DATASET_C_CONFIG,
        submissions_dir=SUBMISSIONS_DIR,
    )

    write_markdown_report(
        eda_summary,
        comp_df,
        manifest_df,
        explain_summary,
        artifacts["meta_weights"],
        feat_train,
        OUTPUT_DIR / "hackathon_report.md",
    )

    (OUTPUT_DIR / "meta_summary.json").write_text(
        json.dumps({"meta_weights": artifacts["meta_weights"]}, indent=2), encoding="utf-8"
    )
    print("\n=== MODEL COMPARISON (sorted by Bayes-Exact OOF Town Cost) ===")
    print(comp_df[["model", "roc_auc", "log_loss", "cost_global_0.5", "cost_bayes_exact", "captured_oracle_pct"]].to_string(index=False))
    print("\n=== TOP 5 SUBMISSIONS ===")
    print(manifest_df[["order", "filename", "model", "policy", "oof_total_cost", "test_pred_positive_count"]].to_string(index=False))


if __name__ == "__main__":
    main()

