"""Automated test suite for Phase 1: Data Audit and Cleaning."""

import hashlib
import pandas as pd
import numpy as np
import pytest

from src.config import PATHS
from src.cleaning import load_raw_data, clean_dataset, audit_account_array_structure

EXPECTED_RAW_SHA256 = "be100dc8625aa5ce05e937540186a150817f60a5d6a58b50d85804f8f0aa3c31"


@pytest.fixture(scope="module")
def raw_df():
    """Fixture providing loaded raw dataframe."""
    return load_raw_data(PATHS["raw_data"])


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    """Fixture providing cleaned dataframe."""
    return clean_dataset(raw_df)


def test_raw_dataset_hash_unmodified():
    """TEST 1: Verifies raw dataset has not been modified or corrupted."""
    with open(PATHS["raw_data"], "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest().lower()
    assert file_hash == EXPECTED_RAW_SHA256, (
        f"Raw dataset hash mismatch! Expected {EXPECTED_RAW_SHA256}, got {file_hash}"
    )


def test_processed_row_count(raw_df, cleaned_df):
    """TEST 2: Processed dataset must have exactly 2000 rows without silent drops."""
    assert len(cleaned_df) == 2000, f"Expected 2000 rows, got {len(cleaned_df)}"
    assert len(cleaned_df) == len(raw_df), "Cleaned dataframe length does not match raw length"


def test_missing_churn_labels_preserved(cleaned_df):
    """TEST 3: Missing churn labels must remain NaN and not imputed to 0."""
    missing_count = cleaned_df["churn_event_next_12m"].isnull().sum()
    assert missing_count == 116, f"Expected 116 missing churn labels, got {missing_count}"
    
    # Check is_label_known flag
    assert "is_label_known" in cleaned_df.columns
    assert (cleaned_df["is_label_known"] == 0).sum() == 116
    assert (cleaned_df["is_label_known"] == 1).sum() == 1884
    
    # Verify that where label is unknown, churn_event_next_12m is indeed null
    unknown_mask = cleaned_df["is_label_known"] == 0
    assert cleaned_df.loc[unknown_mask, "churn_event_next_12m"].isnull().all()


def test_negative_profits_preserved(cleaned_df):
    """TEST 4: Negative profits must remain negative and flagged."""
    neg_profits = cleaned_df[cleaned_df["historical_15yr_net_profit_usd"] < 0]
    assert len(neg_profits) == 58, f"Expected 58 negative-profit rows, got {len(neg_profits)}"
    assert (cleaned_df["negative_profit_flag"] == 1).sum() == 58
    assert (cleaned_df["negative_profit_flag"] == 0).sum() == 1942


def test_over_capacity_rows_preserved(cleaned_df):
    """TEST 5: Over-capacity rows (> 1.0) must not be deleted and correctly flagged."""
    over_cap = cleaned_df[cleaned_df["utilization_rate_pct"] > 1.0]
    assert len(over_cap) == 862, f"Expected 862 over-capacity rows, got {len(over_cap)}"
    assert (cleaned_df["over_capacity"] == 1).sum() == 862
    assert (cleaned_df["over_capacity"] == 0).sum() == 1138
    
    # Verify utilization_capped is at most 1.0
    assert (cleaned_df["utilization_capped"] <= 1.0).all()
    assert (cleaned_df["utilization_excess"] >= 0.0).all()
    # Where over_capacity == 0, excess must be 0
    assert (cleaned_df.loc[cleaned_df["over_capacity"] == 0, "utilization_excess"] == 0.0).all()


def test_unknown_expansion_preserved(cleaned_df):
    """TEST 6: UNKNOWN expansion categories must remain represented as distinct category."""
    counts = cleaned_df["expansion_potential_signal"].value_counts()
    assert "UNKNOWN" in counts.index
    assert counts["UNKNOWN"] == 111, f"Expected 111 UNKNOWN expansion records, got {counts['UNKNOWN']}"


def test_array_uniqueness_and_account_structure(cleaned_df):
    """TEST 7: Array IDs must remain strictly unique, with exactly 2 arrays per account."""
    assert cleaned_df["array_id"].nunique() == 2000
    assert cleaned_df["account_id"].nunique() == 1000
    
    audit = audit_account_array_structure(cleaned_df)
    assert audit["exactly_two_arrays_per_account"] is True
    assert audit["full_duplicate_rows"] == 0
    assert audit["duplicate_array_ids"] == 0
    assert audit["duplicate_account_array_pairs"] == 0


def test_profit_signed_log_transformation(cleaned_df):
    """TEST 8: Signed log transformation correctly handles positive and negative profits."""
    # Negative values should have negative signed log
    neg_mask = cleaned_df["historical_15yr_net_profit_usd"] < 0
    pos_mask = cleaned_df["historical_15yr_net_profit_usd"] > 0
    
    assert (cleaned_df.loc[neg_mask, "profit_signed_log"] < 0).all()
    assert (cleaned_df.loc[pos_mask, "profit_signed_log"] > 0).all()
