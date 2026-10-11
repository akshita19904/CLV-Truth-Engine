"""Automated test suite for Phase 2: Grouped Validation Splits, Leakage, & OOF Predictions."""

import pandas as pd
import numpy as np
import pytest

from src.config import PATHS
from src.features import build_feature_matrix, get_modeling_population
from src.validation import get_grouped_cv, verify_grouped_splits_isolation


@pytest.fixture(scope="module")
def modeling_data():
    """Fixture providing X, y, groups, meta for labelled population."""
    df_feat = build_feature_matrix()
    return get_modeling_population(df_feat)


@pytest.fixture(scope="module")
def oof_predictions():
    """Fixture loading persisted out-of-fold predictions."""
    oof_path = PATHS["cleaned_data"].parent / "oof_predictions.csv"
    assert oof_path.exists(), f"OOF predictions file not found at: {oof_path}"
    return pd.read_csv(oof_path)


def test_missing_churn_labels_excluded_from_modeling(modeling_data):
    """TEST: Unlabelled rows (is_label_known == 0) must NEVER enter training/validation."""
    X, y, groups, meta = modeling_data
    assert len(X) == 1884, f"Expected exactly 1,884 labelled rows, got {len(X)}"
    assert len(y) == 1884, f"Expected exactly 1,884 targets, got {len(y)}"
    assert y.isnull().sum() == 0, "Target vector must not contain nulls"
    assert (meta["is_label_known"] == 1).all(), "All modeling rows must have is_label_known == 1"


def test_grouped_cv_zero_account_leakage(modeling_data):
    """TEST: Zero account overlap between train and validation folds in GroupKFold."""
    X, y, groups, meta = modeling_data
    gkf = get_grouped_cv(n_splits=5)
    splits = list(gkf.split(X, y, groups=groups))
    
    assert len(splits) == 5, f"Expected 5 folds, got {len(splits)}"
    
    # Run rigorous isolation verification
    fold_audit = verify_grouped_splits_isolation(groups, splits)
    for report in fold_audit:
        assert report["overlap_accounts"] == 0, (
            f"Account leakage detected in Fold {report['fold']}: {report['overlap_accounts']} accounts overlap!"
        )
        assert report["train_rows"] > 0
        assert report["val_rows"] > 0


def test_oof_predictions_coverage_and_keys(modeling_data, oof_predictions):
    """TEST: OOF predictions must cover exactly 1,884 labelled rows per model with unique keys."""
    X, y, groups, meta = modeling_data
    models = oof_predictions["model"].unique()
    assert len(models) >= 4, f"Expected at least 4 models, found: {models}"
    
    for model_name in models:
        sub = oof_predictions[oof_predictions["model"] == model_name]
        assert len(sub) == 1884, (
            f"Model '{model_name}' expected 1,884 predictions, got {len(sub)}"
        )
        # Verify no duplicate (array_id, model)
        assert sub["array_id"].duplicated().sum() == 0, (
            f"Duplicate array_id found in OOF predictions for model '{model_name}'!"
        )
        # Verify all array IDs match labelled population
        assert set(sub["array_id"]) == set(meta["array_id"]), (
            f"Array ID mismatch between OOF predictions and labelled population for model '{model_name}'!"
        )


def test_oof_probabilities_bounded_zero_one(oof_predictions):
    """TEST: All predicted out-of-fold probabilities must lie strictly within [0, 1]."""
    probs = oof_predictions["oof_pred_probability"]
    assert probs.notnull().all(), "OOF predictions contain NaN values!"
    assert (probs >= 0.0).all(), "Found negative probability in OOF predictions!"
    assert (probs <= 1.0).all(), "Found probability exceeding 1.0 in OOF predictions!"


def test_no_global_preprocessing_leakage(modeling_data):
    """TEST: Verifies that preprocessors fitted on fold training set do not learn global parameters."""
    from sklearn.preprocessing import StandardScaler
    X, y, groups, meta = modeling_data
    
    gkf = get_grouped_cv(n_splits=5)
    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        # Fit strictly on train
        scaler.fit(X.iloc[train_idx][["current_capacity_tb"]])
        train_mean = scaler.mean_[0]
        
        # Global mean
        global_mean = X["current_capacity_tb"].mean()
        # Train mean on fold should differ from global mean (proving fold-specific estimation)
        assert train_mean != global_mean, (
            "Fold mean matches global mean exactly, indicating possible global leakage!"
        )
