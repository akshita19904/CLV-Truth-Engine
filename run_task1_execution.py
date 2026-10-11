"""Script to execute Task 1 ML pipeline, generate comparison tables, SHAP values, and final scores."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PATHS
from src.features import build_feature_matrix, get_modeling_population
from src.task1_models import (
    evaluate_raw_and_calibrated_models,
    evaluate_expansion_prediction,
    generate_treeshap_explanations,
    generate_final_scores,
    ALL_FEATURE_COLS
)


def main():
    print("Loading cleaned data...")
    cleaned_path = PATHS["processed_data_dir"] / "cleaned_data.csv"
    df_clean = pd.read_csv(cleaned_path)
    
    print(f"Total rows: {len(df_clean)}, Labelled rows: {(df_clean['is_label_known'] == 1).sum()}")
    
    # Build candidate feature matrix
    df_features = build_feature_matrix(df_clean)
    X_full = df_features[ALL_FEATURE_COLS].copy()
    
    # Filter to labelled population for training and cross-validation using canonical helper
    X_lab, y_lab, groups_lab, meta_lab = get_modeling_population(df_features)
    df_lab = df_clean[df_clean["is_label_known"] == 1].copy().reset_index(drop=True)
    
    print("\n1. Running GroupKFold Cross-Validation for Raw and Calibrated Churn Models...")
    comparison_df, all_oofs = evaluate_raw_and_calibrated_models(
        X=X_lab,
        y=y_lab,
        groups=groups_lab,
        n_outer_splits=5,
        n_inner_splits=3,
        random_state=42
    )
    
    comp_path = PATHS["reports_dir"] / "task1_churn_model_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"Saved model comparison to: {comp_path}")
    print("\n--- MODEL COMPARISON RESULTS ---")
    print(comparison_df.to_string(index=False))
    
    print("\n2. Evaluating Capacity Expansion Prediction (High vs. Non-High, No Leakage)...")
    exp_metrics, exp_oof_probs = evaluate_expansion_prediction(
        X_raw=X_lab,
        df_clean=df_lab,
        groups=groups_lab,
        n_splits=5,
        random_state=42
    )
    print("Expansion Model Metrics (without expansion target features):")
    for k, v in exp_metrics.items():
        print(f"  {k}: {v}")
        
    print("\n3. Generating TreeSHAP Explanations for XGBoost...")
    meta_lab = df_lab[["array_id", "account_id"]]
    shap_long_df, shap_summary_df = generate_treeshap_explanations(
        X=X_lab,
        y=y_lab,
        metadata=meta_lab,
        random_state=42,
        model_version="v1.0-xgb"
    )
    shap_path = PATHS["reports_dir"] / "task1_shap_values.csv"
    shap_long_df.to_csv(shap_path, index=False)
    print(f"Saved SHAP values ({len(shap_long_df)} rows) to: {shap_path}")
    print("\nTop 5 Features by Mean Absolute TreeSHAP:")
    print(shap_summary_df.head(5).to_string(index=False))
    
    print("\n4. Generating Final Scoring Dataset for All 2,000 Arrays...")
    final_scores_df = generate_final_scores(
        df_clean=df_clean,
        X_full=X_full,
        labelled_oofs=all_oofs,
        exp_oof_probs=exp_oof_probs,
        random_state=42,
        model_version="v1.0-task1",
        scoring_run_id="run_task1_final"
    )
    scores_path = PATHS["reports_dir"] / "task1_final_scores.csv"
    final_scores_df.to_csv(scores_path, index=False)
    print(f"Saved final scores ({len(final_scores_df)} rows) to: {scores_path}")
    print("\nFinal Scores Summary Preview:")
    print(final_scores_df.head(5).to_string(index=False))
    print("\nSummary Statistics of Predicted Probabilities:")
    print(final_scores_df[[
        "churn_probability_xgb",
        "churn_probability_rf",
        "churn_probability_logistic",
        "selected_churn_probability",
        "renewal_survival_score",
        "expansion_score",
        "expansion_probability"
    ]].describe().to_string())
    
    print("\nTask 1 execution completed successfully!")


if __name__ == "__main__":
    main()
