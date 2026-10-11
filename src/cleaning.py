"""CLV Truth Engine - Data Cleaning and Audit Module

Performs comprehensive data auditing, anomaly detection, and leakage-safe cleaning
on the raw storage array CLV dataset. Preserves raw files and produces the cleaned
analytical dataset along with data quality reports.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import PATHS


def load_raw_data(filepath: Path = None) -> pd.DataFrame:
    """Loads raw storage array dataset without modifications."""
    if filepath is None:
        filepath = PATHS["raw_data"]
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def generate_data_quality_report(df: pd.DataFrame, output_path: Path = None) -> pd.DataFrame:
    """Generates an audit report summarizing schema, missingness, cardinality, and statistics."""
    records = []
    total_rows = len(df)
    
    for col in df.columns:
        s = df[col]
        dtype_str = str(s.dtype)
        n_missing = int(s.isnull().sum())
        pct_missing = float(n_missing / total_rows * 100.0)
        n_unique = int(s.nunique(dropna=True))
        
        is_num = pd.api.types.is_numeric_dtype(s)
        val_min = float(s.min()) if is_num and s.dropna().size > 0 else None
        val_max = float(s.max()) if is_num and s.dropna().size > 0 else None
        val_mean = float(s.mean()) if is_num and s.dropna().size > 0 else None
        val_median = float(s.median()) if is_num and s.dropna().size > 0 else None
        
        records.append({
            "column": col,
            "dtype": dtype_str,
            "missing_count": n_missing,
            "missing_pct": round(pct_missing, 2),
            "unique_count": n_unique,
            "min": round(val_min, 4) if val_min is not None else "",
            "max": round(val_max, 4) if val_max is not None else "",
            "mean": round(val_mean, 4) if val_mean is not None else "",
            "median": round(val_median, 4) if val_median is not None else "",
        })
    
    report_df = pd.DataFrame(records)
    
    if output_path is None:
        output_path = PATHS["figures_dir"].parent / "data_quality_report.csv"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)
    return report_df


def audit_account_array_structure(df: pd.DataFrame) -> dict:
    """Verifies account and storage array relationships and checks duplicates."""
    n_rows = len(df)
    n_accounts = df["account_id"].nunique()
    n_arrays = df["array_id"].nunique()
    arrays_per_account = df["account_id"].value_counts()
    
    full_duplicates = int(df.duplicated().sum())
    array_duplicates = int(df["array_id"].duplicated().sum())
    pair_duplicates = int(df.duplicated(subset=["account_id", "array_id"]).sum())
    
    summary = {
        "total_rows": n_rows,
        "unique_accounts": n_accounts,
        "unique_arrays": n_arrays,
        "min_arrays_per_account": int(arrays_per_account.min()),
        "max_arrays_per_account": int(arrays_per_account.max()),
        "exactly_two_arrays_per_account": bool((arrays_per_account == 2).all()),
        "full_duplicate_rows": full_duplicates,
        "duplicate_array_ids": array_duplicates,
        "duplicate_account_array_pairs": pair_duplicates,
    }
    return summary


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Applies non-destructive cleaning and anomaly transformation rules.
    
    Rules applied:
    1. is_label_known: 1 if churn label is present, 0 if missing (keeps missing as NaN).
    2. utilization_capped: min(utilization, 1.0) to bound physical loading.
    3. over_capacity: 1 if utilization > 1.0, 0 otherwise.
    4. utilization_excess: max(utilization - 1.0, 0.0) quantifying excess load.
    5. negative_profit_flag: 1 if historical profit < 0, 0 otherwise.
    6. profit_signed_log: sign(profit) * ln(1 + |profit|) for numeric stability.
    7. UNKNOWN expansion categories preserved as distinct category.
    """
    cleaned = df.copy()
    
    # 1. Target label isolation
    cleaned["is_label_known"] = cleaned["churn_event_next_12m"].notnull().astype(int)
    # Ensure missing labels remain NaN in the target column
    
    # 2. Utilization transformations (utilization is proportion 0.35 - 1.45)
    cleaned["utilization_capped"] = np.minimum(cleaned["utilization_rate_pct"], 1.0)
    cleaned["over_capacity"] = (cleaned["utilization_rate_pct"] > 1.0).astype(int)
    cleaned["utilization_excess"] = np.maximum(cleaned["utilization_rate_pct"] - 1.0, 0.0)
    
    # 3. Profit transformations (handling negative profits without deleting)
    cleaned["negative_profit_flag"] = (cleaned["historical_15yr_net_profit_usd"] < 0).astype(int)
    cleaned["profit_signed_log"] = (
        np.sign(cleaned["historical_15yr_net_profit_usd"]) * 
        np.log1p(np.abs(cleaned["historical_15yr_net_profit_usd"]))
    )
    
    return cleaned


def run_audit_and_cleaning(raw_path: Path = None, output_path: Path = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executes the complete audit and cleaning process, saving output artifacts."""
    df_raw = load_raw_data(raw_path)
    
    # Generate data quality report
    report_df = generate_data_quality_report(df_raw)
    
    # Clean dataset
    df_cleaned = clean_dataset(df_raw)
    
    # Save cleaned data
    if output_path is None:
        output_path = PATHS["cleaned_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    
    return df_cleaned, report_df


if __name__ == "__main__":
    df_clean, report = run_audit_and_cleaning()
    print("Data audit and cleaning completed successfully.")
    print(f"Cleaned shape: {df_clean.shape}")
    print(f"Report rows: {len(report)}")
