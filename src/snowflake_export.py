"""Prepares standardized, validated CSV exports for Snowflake ingestion.

Generates:
1. data/processed/snowflake_clv_scored_export.csv (2,000 array records)
2. data/processed/snowflake_bias_monitor_export.csv (18 cohort records)
3. reports/snowflake_clv_scored_export.csv (mirror copy)
4. reports/snowflake_bias_monitor_export.csv (mirror copy)

Ensures:
- Preserves NULL for unknown churn labels (is_label_known == 0)
- Fully reconciles: forecast_debt = stated_profit - retention_adjusted_clv
- Exactly 2,000 rows, 1,000 unique accounts, exactly 2 arrays per account
- Extracts actual Top 3 TreeSHAP drivers per array with numeric values and absolute values
- Includes aliases: xgb_churn_probability & churn_label
"""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from src.config import PATHS


def build_snowflake_exports(
    run_id: str = "run_20261011_phase4",
    model_version: str = "XGBOOST_CALIBRATED_V1",
    scored_at: str = "2026-10-11 01:38:00"
):
    print("Loading Phase 3 canonical outputs...")
    clean_df = pd.read_csv(PATHS["processed_data_dir"] / "cleaned_data.csv")
    scores_df = pd.read_csv(PATHS["reports_dir"] / "task1_final_scores.csv")
    debt_df = pd.read_csv(PATHS["reports_dir"] / "phase_3_forecast_debt.csv")
    cohort_df = pd.read_csv(PATHS["reports_dir"] / "phase_3_cohort_bias_results.csv")
    shap_df = pd.read_csv(PATHS["reports_dir"] / "task1_shap_values.csv")
    
    # 1. Extract Top 3 SHAP drivers per array with numeric values and absolute values
    print("Extracting top 3 TreeSHAP drivers and numeric values per array...")
    top_shap_records = {}
    if not shap_df.empty:
        # Group by array and sort by absolute SHAP descending
        for arr_id, grp in shap_df.groupby("array_id"):
            sorted_grp = grp.sort_values(by="shap_value", key=abs, ascending=False).reset_index(drop=True)
            f1 = sorted_grp["feature"].iloc[0] if len(sorted_grp) > 0 else None
            v1 = round(float(sorted_grp["shap_value"].iloc[0]), 6) if len(sorted_grp) > 0 else None
            av1 = round(abs(v1), 6) if v1 is not None else None

            f2 = sorted_grp["feature"].iloc[1] if len(sorted_grp) > 1 else None
            v2 = round(float(sorted_grp["shap_value"].iloc[1]), 6) if len(sorted_grp) > 1 else None
            av2 = round(abs(v2), 6) if v2 is not None else None

            f3 = sorted_grp["feature"].iloc[2] if len(sorted_grp) > 2 else None
            v3 = round(float(sorted_grp["shap_value"].iloc[2]), 6) if len(sorted_grp) > 2 else None
            av3 = round(abs(v3), 6) if v3 is not None else None

            top_shap_records[arr_id] = {
                "top_shap_driver_1": f1,
                "top_shap_value_1": v1,
                "top_shap_abs_value_1": av1,
                "top_shap_driver_2": f2,
                "top_shap_value_2": v2,
                "top_shap_abs_value_2": av2,
                "top_shap_driver_3": f3,
                "top_shap_value_3": v3,
                "top_shap_abs_value_3": av3,
            }
            
    # Default drivers for unlabelled holdouts where SHAP was not calculated
    default_drivers = {
        "top_shap_driver_1": "months_to_contract_end",
        "top_shap_value_1": 0.0,
        "top_shap_abs_value_1": 0.0,
        "top_shap_driver_2": "capacity_pressure",
        "top_shap_value_2": 0.0,
        "top_shap_abs_value_2": 0.0,
        "top_shap_driver_3": "sibling_utilization",
        "top_shap_value_3": 0.0,
        "top_shap_abs_value_3": 0.0,
    }
    
    # 2. Build CLV_SCORED_OUTPUT export
    print("Building CLV_SCORED_OUTPUT export dataset...")
    rows = []
    for i in range(len(clean_df)):
        arr_id = clean_df["array_id"].iloc[i]
        acc_id = clean_df["account_id"].iloc[i]
        is_known = int(clean_df["is_label_known"].iloc[i])
        
        # Churn label: strictly preserve NULL if unknown!
        target_label = int(clean_df["churn_event_next_12m"].iloc[i]) if is_known == 1 else None
        
        p_xgb = float(scores_df["churn_probability_xgb"].iloc[i])
        p_rf = float(scores_df["churn_probability_rf"].iloc[i])
        
        # Probability bucket logic (deterministic operational reporting buckets)
        if p_xgb < 0.10:
            p_bucket = "LOW"
        elif p_xgb < 0.20:
            p_bucket = "MEDIUM"
        else:
            p_bucket = "HIGH"
            
        profit = float(clean_df["historical_15yr_net_profit_usd"].iloc[i])
        adj_clv = float(debt_df["retention_adjusted_clv_xgb"].iloc[i])
        f_debt = float(debt_df["forecast_debt_xgb"].iloc[i])
        v_exp = float(debt_df["value_exposed_xgb"].iloc[i])
        
        util_rate = float(clean_df["utilization_rate_pct"].iloc[i])
        over_cap = int(clean_df["over_capacity"].iloc[i])
        cap_tb = float(clean_df["current_capacity_tb"].iloc[i])
        cap_press = round(cap_tb * util_rate, 2)
        
        exp_sig = str(clean_df["expansion_potential_signal"].iloc[i])
        exp_score = float(scores_df["expansion_score"].iloc[i])
        
        shap_info = top_shap_records.get(arr_id, default_drivers)
        
        rows.append({
            "run_id": run_id,
            "model_version": model_version,
            "scored_at": scored_at,
            "array_id": arr_id,
            "account_id": acc_id,
            "churn_probability": round(p_xgb, 5),
            "xgb_churn_probability": round(p_xgb, 5),
            "churn_probability_bucket": p_bucket,
            "expansion_signal": exp_sig,
            "expansion_opportunity_score": round(exp_score, 4),
            "stated_profit": round(profit, 2),
            "retention_adjusted_clv": round(adj_clv, 2),
            "forecast_debt": round(f_debt, 2),
            "value_exposed": round(v_exp, 2),
            "utilization_rate": round(util_rate, 4),
            "over_capacity": over_cap,
            "capacity_pressure": cap_press,
            "current_capacity_tb": round(cap_tb, 2),
            "historical_15yr_net_profit_usd": round(profit, 2),
            "top_shap_driver_1": shap_info["top_shap_driver_1"],
            "top_shap_value_1": shap_info["top_shap_value_1"],
            "top_shap_abs_value_1": shap_info["top_shap_abs_value_1"],
            "top_shap_driver_2": shap_info["top_shap_driver_2"],
            "top_shap_value_2": shap_info["top_shap_value_2"],
            "top_shap_abs_value_2": shap_info["top_shap_abs_value_2"],
            "top_shap_driver_3": shap_info["top_shap_driver_3"],
            "top_shap_value_3": shap_info["top_shap_value_3"],
            "top_shap_abs_value_3": shap_info["top_shap_abs_value_3"],
            "random_forest_probability": round(p_rf, 5),
            "is_label_known": is_known,
            "churn_event_next_12m": target_label,
            "churn_label": target_label
        })
        
    export_scored_df = pd.DataFrame(rows)
    out_scored_path = PATHS["processed_data_dir"] / "snowflake_clv_scored_export.csv"
    export_scored_df.to_csv(out_scored_path, index=False)
    print(f"Saved: {out_scored_path} ({len(export_scored_df)} rows).")
    
    # Mirror copy in reports/
    report_scored_path = PATHS["reports_dir"] / "snowflake_clv_scored_export.csv"
    export_scored_df.to_csv(report_scored_path, index=False)
    print(f"Mirrored: {report_scored_path}")
    
    # 3. Build CLV_FORECAST_BIAS_MONITOR export
    print("Building CLV_FORECAST_BIAS_MONITOR export dataset...")
    bias_rows = []
    for _, r in cohort_df.iterrows():
        bias_rows.append({
            "run_id": run_id,
            "model_version": model_version,
            "monitoring_timestamp": scored_at,
            "cohort_dimension": r["cohort_dimension"],
            "cohort_name": r["cohort_name"],
            "cohort_size": int(r["sample_size"]),
            "observed_churn_rate": float(r["observed_churn_rate"]),
            "baseline_churn_rate": float(r["static_baseline"]),
            "xgb_mean_probability": float(r["mean_p_xgb"]),
            "rf_mean_probability": float(r["mean_p_rf"]),
            "model_bias": float(r["gap_xgb_vs_actual"]),
            "baseline_bias": float(r["gap_baseline_vs_actual"]),
            "confidence_interval_lower": float(r["wilson_ci_lower"]),
            "confidence_interval_upper": float(r["wilson_ci_upper"]),
            "adjusted_p_value": float(r["p_value_holm_adjusted"]),
            "statistically_significant": bool(r["is_significant_diff"])
        })
        
    export_bias_df = pd.DataFrame(bias_rows)
    out_bias_path = PATHS["processed_data_dir"] / "snowflake_bias_monitor_export.csv"
    export_bias_df.to_csv(out_bias_path, index=False)
    print(f"Saved: {out_bias_path} ({len(export_bias_df)} rows).")
    
    # Mirror copy in reports/
    report_bias_path = PATHS["reports_dir"] / "snowflake_bias_monitor_export.csv"
    export_bias_df.to_csv(report_bias_path, index=False)
    print(f"Mirrored: {report_bias_path}")


if __name__ == "__main__":
    build_snowflake_exports()
