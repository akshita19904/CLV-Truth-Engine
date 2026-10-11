"""CLV Truth Engine - Feature Engineering Module

Constructs leakage-safe candidate feature matrix including array-level,
contract-temporal, and sibling-level operational metrics.
Guarantees that target labels never enter the feature matrix.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PATHS, config


# Reference date for contract maturity calculation
DEFAULT_REFERENCE_DATE = config.get("environment", {}).get("reference_date", "2026-10-01")

# Explicit feature definitions
NUMERICAL_FEATURES = [
    "historical_15yr_net_profit_usd",
    "current_capacity_tb",
    "utilization_rate_pct",
    "utilization_capped",
    "utilization_excess",
    "over_capacity",
    "negative_profit_flag",
    "profit_signed_log",
    "months_to_contract_end",
    "capacity_pressure",
    "profit_per_tb",
    "sibling_utilization",
    "sibling_capacity",
]

CATEGORICAL_FEATURES = [
    "expansion_potential_signal",
    "sibling_expansion",
]

ALL_FEATURE_COLS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def add_contract_timing_feature(df: pd.DataFrame, reference_date: str = DEFAULT_REFERENCE_DATE) -> pd.DataFrame:
    """Calculates elapsed months from reference date to contract expiration date."""
    out = df.copy()
    ref_dt = pd.to_datetime(reference_date)
    end_dt = pd.to_datetime(out["contract_end_date"])
    out["months_to_contract_end"] = ((end_dt - ref_dt).dt.days / 30.4375).round(2)
    return out


def add_derived_hardware_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Computes capacity pressure and hardware profit density."""
    out = df.copy()
    # Safe division guarding non-positive capacity
    safe_cap = np.maximum(out["current_capacity_tb"], 1.0)
    out["capacity_pressure"] = out["current_capacity_tb"] * out["utilization_capped"]
    out["profit_per_tb"] = out["historical_15yr_net_profit_usd"] / safe_cap
    return out


def add_sibling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constructs sibling features across paired arrays within the same customer account.
    
    CRITICAL LEAKAGE CONSTRAINT:
    Sibling churn outcomes (churn_event_next_12m) are strictly prohibited from entering
    the feature matrix. Only operational and pre-contract attributes are shared.
    """
    out = df.copy()
    
    # Sort deterministically by account_id and array_id
    out = out.sort_values(["account_id", "array_id"]).reset_index(drop=True)
    
    # Verify exactly 2 arrays per account
    counts = out["account_id"].value_counts()
    if not (counts == 2).all():
        raise ValueError("Each account must have exactly 2 arrays to establish 1-to-1 sibling relationship.")
    
    # Position index within account: 0 for first array, 1 for second array
    pos = out.groupby("account_id").cumcount()
    
    # Sibling index: if pos == 0, sibling is index + 1; if pos == 1, sibling is index - 1
    sibling_idx = np.where(pos == 0, out.index + 1, out.index - 1)
    
    # Extract sibling features
    out["sibling_array_id"] = out.loc[sibling_idx, "array_id"].values
    out["sibling_account_id"] = out.loc[sibling_idx, "account_id"].values
    out["sibling_utilization"] = out.loc[sibling_idx, "utilization_rate_pct"].values
    out["sibling_capacity"] = out.loc[sibling_idx, "current_capacity_tb"].values
    out["sibling_expansion"] = out.loc[sibling_idx, "expansion_potential_signal"].values
    
    # Rigorous validation checks
    assert (out["account_id"] == out["sibling_account_id"]).all(), "Sibling account ID mismatch!"
    assert (out["array_id"] != out["sibling_array_id"]).all(), "Self-sibling relationship detected!"
    assert out["sibling_utilization"].notnull().all(), "Missing sibling utilization values!"
    assert out["sibling_capacity"].notnull().all(), "Missing sibling capacity values!"
    assert out["sibling_expansion"].notnull().all(), "Missing sibling expansion values!"
    
    return out


def build_feature_matrix(cleaned_path_or_df=None, output_path: Path = None) -> pd.DataFrame:
    """End-to-end feature engineering pipeline producing candidate feature matrix."""
    if isinstance(cleaned_path_or_df, pd.DataFrame):
        df = cleaned_path_or_df.copy()
    else:
        cleaned_path = cleaned_path_or_df if cleaned_path_or_df is not None else PATHS["cleaned_data"]
        if not Path(cleaned_path).exists():
            raise FileNotFoundError(f"Cleaned dataset not found at: {cleaned_path}")
        df = pd.read_csv(cleaned_path)
    
    # 1. Add contract timing
    df = add_contract_timing_feature(df)
    
    # 2. Add derived hardware metrics
    df = add_derived_hardware_metrics(df)
    
    # 3. Add sibling features (leakage-safe)
    df = add_sibling_features(df)
    
    # Persist feature matrix
    if output_path is None:
        output_path = PATHS["model_features"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    return df


def get_modeling_population(df_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Isolates the labelled population (is_label_known == 1).
    
    Returns:
        X (pd.DataFrame): Candidate feature matrix (15 features).
        y (pd.Series): Target 12-month churn binary labels (0.0 or 1.0).
        groups (pd.Series): Account IDs used for GroupKFold splitting.
        metadata (pd.DataFrame): Identifiers (account_id, array_id, is_label_known).
    """
    labelled = df_features[df_features["is_label_known"] == 1].copy().reset_index(drop=True)
    
    X = labelled[ALL_FEATURE_COLS].copy()
    y = labelled["churn_event_next_12m"].astype(int)
    groups = labelled["account_id"].copy()
    metadata = labelled[["account_id", "array_id", "is_label_known"]].copy()
    
    return X, y, groups, metadata


if __name__ == "__main__":
    df_feat = build_feature_matrix()
    print(f"Feature matrix built successfully: {df_feat.shape}")
    X, y, groups, meta = get_modeling_population(df_feat)
    print(f"Labelled modeling population: {X.shape[0]} rows, {X.shape[1]} features")
    print(f"Positive churn count: {y.sum()} ({y.mean()*100:.2f}%)")
