"""Automated Unit Tests for Task 1: ML Models, Grouped Validation, SHAP, and Final Scoring."""

import pandas as pd
import numpy as np
import pytest

from src.config import PATHS
from src.features import build_feature_matrix, get_modeling_population, ALL_FEATURE_COLS
from src.task1_models import EXPANSION_FEATURE_COLS, get_preprocessor


@pytest.fixture(scope="module")
def feature_data():
    """Provides full feature matrix and labelled subset."""
    df_feat = build_feature_matrix()
    X, y, groups, meta = get_modeling_population(df_feat)
    return df_feat, X, y, groups, meta


@pytest.fixture(scope="module")
def final_scores():
    """Loads reports/task1_final_scores.csv."""
    path = PATHS["reports_dir"] / "task1_final_scores.csv"
    assert path.exists(), f"Final scores file not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def shap_values():
    """Loads reports/task1_shap_values.csv."""
    path = PATHS["reports_dir"] / "task1_shap_values.csv"
    assert path.exists(), f"SHAP values file not found at {path}"
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def model_comparison():
    """Loads reports/task1_churn_model_comparison.csv."""
    path = PATHS["reports_dir"] / "task1_churn_model_comparison.csv"
    assert path.exists(), f"Model comparison file not found at {path}"
    return pd.read_csv(path)


def test_xgb_and_rf_predictions_bounded_zero_one(final_scores):
    """TEST: XGBoost and Random Forest probabilities must lie strictly in [0, 1]."""
    for col in ["churn_probability_xgb", "churn_probability_rf", "churn_probability_logistic", "selected_churn_probability"]:
        assert col in final_scores.columns, f"Missing column {col} in final scores"
        probs = final_scores[col]
        assert probs.notnull().all(), f"Found NaN probabilities in {col}"
        assert (probs >= 0.0).all(), f"Found probabilities < 0 in {col}"
        assert (probs <= 1.0).all(), f"Found probabilities > 1 in {col}"


def test_no_target_leakage_in_features():
    """TEST: Churn target must never be present in the candidate feature columns."""
    assert "churn_event_next_12m" not in ALL_FEATURE_COLS, "churn_event_next_12m found in ALL_FEATURE_COLS!"
    assert "target" not in ALL_FEATURE_COLS


def test_no_sibling_target_leakage_in_features():
    """TEST: Sibling churn label must never be present in the feature columns."""
    forbidden = ["sibling_churn", "sibling_churn_event_next_12m", "account_churn_rate"]
    for col in forbidden:
        assert col not in ALL_FEATURE_COLS, f"Forbidden sibling target leakage column {col} in ALL_FEATURE_COLS!"


def test_expansion_target_exclusion_in_expansion_features():
    """TEST: Expansion model features must strictly exclude expansion_potential_signal and sibling_expansion."""
    assert "expansion_potential_signal" not in EXPANSION_FEATURE_COLS, (
        "Target leakage: expansion_potential_signal found in EXPANSION_FEATURE_COLS!"
    )
    assert "sibling_expansion" not in EXPANSION_FEATURE_COLS, (
        "Target leakage: sibling_expansion found in EXPANSION_FEATURE_COLS!"
    )


def test_unique_array_id_in_final_scores(final_scores):
    """TEST: Final scores dataset must contain exactly 2,000 rows with unique array_id."""
    assert len(final_scores) == 2000, f"Expected 2,000 rows in final scores, got {len(final_scores)}"
    assert final_scores["array_id"].nunique() == 2000, "Duplicate array_id found in final scores!"
    assert final_scores["account_id"].nunique() == 1000, "Expected exactly 1,000 unique accounts!"


def test_shap_row_integrity_and_columns(shap_values, feature_data):
    """TEST: TreeSHAP dataset must have required columns, no NaNs, and cover all labelled rows."""
    _, X, _, _, _ = feature_data
    required_cols = ["array_id", "account_id", "feature", "feature_value", "shap_value", "model_version"]
    for col in required_cols:
        assert col in shap_values.columns, f"Missing required SHAP column: {col}"
        
    assert shap_values["shap_value"].notnull().all(), "Found NaN in shap_value!"
    assert shap_values["array_id"].nunique() == len(X), (
        f"Expected {len(X)} unique arrays in SHAP dataset, got {shap_values['array_id'].nunique()}"
    )


def test_survival_scenario_score_bounded(final_scores):
    """TEST: Long-horizon renewal survival score must be strictly in [0, 1]."""
    scores = final_scores["renewal_survival_score"]
    assert scores.notnull().all(), "Found NaNs in renewal_survival_score"
    assert (scores >= 0.0).all(), "Found negative renewal_survival_score"
    assert (scores <= 1.0).all(), "Found renewal_survival_score > 1.0"


def test_model_comparison_file_integrity(model_comparison):
    """TEST: Model comparison report must contain all required candidate models and metrics."""
    required_models = ["Static 5% Baseline", "XGBoost", "Random Forest"]
    for m in required_models:
        matches = model_comparison[model_comparison["model"].str.startswith(m)]
        assert len(matches) > 0, f"Model {m} not found in model comparison table"
        
    cols = ["model", "log_loss", "brier_score", "roc_auc", "pr_auc", "calibration_status"]
    for c in cols:
        assert c in model_comparison.columns, f"Missing column {c} in comparison table"
