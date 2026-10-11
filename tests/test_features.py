"""Automated test suite for Phase 2: Feature Engineering & Target Integrity."""

import pandas as pd
import numpy as np
import pytest

from src.config import PATHS
from src.features import (
    build_feature_matrix,
    get_modeling_population,
    add_sibling_features,
    ALL_FEATURE_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture(scope="module")
def feature_df():
    """Fixture providing full feature matrix."""
    return build_feature_matrix()


def test_exactly_one_sibling_per_array(feature_df):
    """TEST: Verifies exactly 2 arrays per account and exactly 1 sibling per array."""
    counts = feature_df["account_id"].value_counts()
    assert (counts == 2).all(), "Every account must contain exactly 2 arrays"
    assert "sibling_array_id" in feature_df.columns
    assert feature_df["sibling_array_id"].notnull().all(), "No array can have a missing sibling"


def test_no_self_sibling_relationships(feature_df):
    """TEST: An array must never be assigned as its own sibling."""
    assert (feature_df["array_id"] != feature_df["sibling_array_id"]).all(), (
        "Self-sibling relationship detected!"
    )
    # Verify account IDs match
    assert (feature_df["account_id"] == feature_df["sibling_account_id"]).all(), (
        "Sibling account ID must match parent array account ID"
    )


def test_no_target_derived_features(feature_df):
    """TEST: CRITICAL LEAKAGE TEST. Sibling churn target and account-level target features must be absent."""
    forbidden_terms = [
        "churn",
        "target",
        "label",
        "y_true",
        "outcome",
    ]
    
    for col in ALL_FEATURE_COLS:
        for term in forbidden_terms:
            assert term not in col.lower(), (
                f"Target leakage detected in feature matrix! Column '{col}' contains forbidden term '{term}'."
            )
            
    # Explicitly check sibling target does not exist
    assert "sibling_churn_event_next_12m" not in feature_df.columns
    assert "account_churn_rate" not in feature_df.columns
    assert "account_average_churn" not in feature_df.columns


def test_feature_matrix_dimensions_and_nulls(feature_df):
    """TEST: Feature matrix must have 2,000 rows and zero nulls across all candidate features."""
    assert len(feature_df) == 2000, f"Expected 2,000 rows, got {len(feature_df)}"
    for col in ALL_FEATURE_COLS:
        assert feature_df[col].isnull().sum() == 0, f"Feature '{col}' contains null values!"


def test_derived_hardware_metrics_correctness(feature_df):
    """TEST: Verifies capacity_pressure and profit_per_tb formulas."""
    expected_pressure = feature_df["current_capacity_tb"] * feature_df["utilization_capped"]
    np.testing.assert_allclose(feature_df["capacity_pressure"], expected_pressure, rtol=1e-5)
    
    expected_profit_per_tb = feature_df["historical_15yr_net_profit_usd"] / np.maximum(feature_df["current_capacity_tb"], 1.0)
    np.testing.assert_allclose(feature_df["profit_per_tb"], expected_profit_per_tb, rtol=1e-5)
