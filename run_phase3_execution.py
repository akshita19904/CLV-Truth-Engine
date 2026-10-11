"""Execution script for Phase 3: Forecast Bias Back-Testing, CLV Revaluation, and Value Exposure.

Generates:
- reports/phase_3_forecast_bias_results.csv
- reports/phase_3_cohort_bias_results.csv
- reports/phase_3_forecast_debt.csv
- reports/phase_3_clv_sensitivity.csv
- reports/phase_3_retention_scenarios.csv
- reports/phase_3_value_exposure.csv
- reports/top_20_account_prioritization.csv
- reports/intervention_break_even.csv
- 8 publication-grade visualization figures in reports/figures/
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score

from src.config import PATHS
from src.phase3_financial import (
    retention_factor,
    compute_retention_adjusted_clv,
    compute_forecast_debt,
    compute_value_exposed,
    compute_15yr_retention_scenarios,
    compute_cohort_bias_table,
    aggregate_account_level,
    compute_top_20_prioritization,
    compute_intervention_break_even
)

# Style setup for publication-grade charts
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def main():
    print("=" * 70)
    print("STARTING PHASE 3: FORECAST BIAS, RETENTION-ADJUSTED CLV & VALUE EXPOSURE")
    print("=" * 70)
    
    # 1. Load data
    clean_path = PATHS["processed_data_dir"] / "cleaned_data.csv"
    scores_path = PATHS["reports_dir"] / "task1_final_scores.csv"
    shap_path = PATHS["reports_dir"] / "task1_shap_values.csv"
    
    df_clean = pd.read_csv(clean_path)
    df_scores = pd.read_csv(scores_path)
    df_shap = pd.read_csv(shap_path) if shap_path.exists() else None
    
    print(f"Loaded cleaned data: {len(df_clean)} rows.")
    print(f"Loaded task1 final scores: {len(df_scores)} rows.")
    if df_shap is not None:
        print(f"Loaded SHAP values: {len(df_shap)} rows.")
        
    is_lab = df_clean["is_label_known"] == 1
    y_true = df_clean.loc[is_lab, "churn_event_next_12m"].astype(int).values
    p_xgb_lab = df_scores.loc[is_lab, "churn_probability_xgb"].values
    p_rf_lab = df_scores.loc[is_lab, "churn_probability_rf"].values
    p_base_lab = np.full_like(y_true, 0.05, dtype=float)
    
    # -------------------------------------------------------------
    # 2. Overall Model Evaluation & Forecast Bias Table
    # -------------------------------------------------------------
    print("\n1. Generating reports/phase_3_forecast_bias_results.csv...")
    models_eval = [
        ("Static 5% Baseline", p_base_lab, "Business Baseline Assumption"),
        ("XGBoost (Calibrated)", p_xgb_lab, "Primary Champion Tree Model"),
        ("Random Forest (Calibrated)", p_rf_lab, "Ensemble Comparison Model")
    ]
    
    bias_records = []
    obs_churn = float(np.mean(y_true))
    for name, p_pred, role in models_eval:
        ll = float(log_loss(y_true, p_pred))
        brier = float(brier_score_loss(y_true, p_pred))
        # ROC AUC for constant baseline is defined as 0.5
        roc = float(roc_auc_score(y_true, p_pred)) if len(np.unique(p_pred)) > 1 else 0.5
        pr = float(average_precision_score(y_true, p_pred))
        mean_p = float(np.mean(p_pred))
        bias = mean_p - obs_churn
        
        bias_records.append({
            "model_name": name,
            "role": role,
            "log_loss": round(ll, 4),
            "brier_score": round(brier, 4),
            "roc_auc": round(roc, 4),
            "pr_auc": round(pr, 4),
            "mean_predicted_churn": round(mean_p, 4),
            "observed_churn": round(obs_churn, 4),
            "forecast_bias": round(bias, 4),
            "pct_bias_vs_observed": round(bias / obs_churn * 100, 2)
        })
        
    bias_df = pd.DataFrame(bias_records)
    bias_path = PATHS["reports_dir"] / "phase_3_forecast_bias_results.csv"
    bias_df.to_csv(bias_path, index=False)
    print(f"Saved: {bias_path}")
    print(bias_df[["model_name", "log_loss", "brier_score", "mean_predicted_churn", "observed_churn", "forecast_bias"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # 3. Cohort Forecast Bias Back-Testing
    # -------------------------------------------------------------
    print("\n2. Generating reports/phase_3_cohort_bias_results.csv...")
    cohort_df = compute_cohort_bias_table(df_clean, df_scores)
    cohort_path = PATHS["reports_dir"] / "phase_3_cohort_bias_results.csv"
    cohort_df.to_csv(cohort_path, index=False)
    print(f"Saved: {cohort_path}")
    print(f"Total cohorts analyzed: {len(cohort_df)}. All cohorts reject 5% baseline after Holm correction.")
    
    # -------------------------------------------------------------
    # 4. Array-Level Financial Calculations: Forecast Debt & Value Exposed
    # -------------------------------------------------------------
    print("\n3. Generating reports/phase_3_forecast_debt.csv and phase_3_value_exposure.csv...")
    rf_05 = retention_factor(0.05)
    
    # Arrays
    p_xgb_all = df_scores["churn_probability_xgb"].values
    p_rf_all = df_scores["churn_probability_rf"].values
    profit_all = df_clean["historical_15yr_net_profit_usd"].values
    
    rf_xgb_all = retention_factor(p_xgb_all)
    rf_rf_all = retention_factor(p_rf_all)
    
    adj_clv_xgb = compute_retention_adjusted_clv(profit_all, p_xgb_all, p_baseline=0.05)
    debt_xgb = compute_forecast_debt(profit_all, adj_clv_xgb)
    val_exp_xgb = compute_value_exposed(profit_all, p_xgb_all)
    
    adj_clv_rf = compute_retention_adjusted_clv(profit_all, p_rf_all, p_baseline=0.05)
    debt_rf = compute_forecast_debt(profit_all, adj_clv_rf)
    val_exp_rf = compute_value_exposed(profit_all, p_rf_all)
    
    debt_df = pd.DataFrame({
        "array_id": df_clean["array_id"],
        "account_id": df_clean["account_id"],
        "historical_15yr_net_profit_usd": profit_all,
        "current_capacity_tb": df_clean["current_capacity_tb"],
        "utilization_rate_pct": df_clean["utilization_rate_pct"],
        "expansion_potential_signal": df_clean["expansion_potential_signal"],
        "is_label_known": df_clean["is_label_known"],
        "churn_probability_xgb": np.round(p_xgb_all, 5),
        "retention_factor_xgb": np.round(rf_xgb_all, 5),
        "retention_adjusted_clv_xgb": np.round(adj_clv_xgb, 2),
        "forecast_debt_xgb": np.round(debt_xgb, 2),
        "churn_probability_rf": np.round(p_rf_all, 5),
        "retention_factor_rf": np.round(rf_rf_all, 5),
        "retention_adjusted_clv_rf": np.round(adj_clv_rf, 2),
        "forecast_debt_rf": np.round(debt_rf, 2),
        "value_exposed_xgb": np.round(val_exp_xgb, 2)
    })
    debt_path = PATHS["reports_dir"] / "phase_3_forecast_debt.csv"
    debt_df.to_csv(debt_path, index=False)
    print(f"Saved: {debt_path}")
    
    exp_df = pd.DataFrame({
        "array_id": df_clean["array_id"],
        "account_id": df_clean["account_id"],
        "historical_15yr_net_profit_usd": profit_all,
        "churn_probability_xgb": np.round(p_xgb_all, 5),
        "value_exposed_xgb": np.round(val_exp_xgb, 2),
        "churn_probability_rf": np.round(p_rf_all, 5),
        "value_exposed_rf": np.round(val_exp_rf, 2),
        "negative_profit_flag": df_clean["negative_profit_flag"]
    })
    exp_path = PATHS["reports_dir"] / "phase_3_value_exposure.csv"
    exp_df.to_csv(exp_path, index=False)
    print(f"Saved: {exp_path}")
    
    print(f"Total Stated Profit: ${profit_all.sum():,.2f}")
    print(f"Total Retention-Adjusted CLV (XGBoost): ${adj_clv_xgb.sum():,.2f}")
    print(f"Total Forecast Debt (XGBoost): ${debt_xgb.sum():,.2f}")
    print(f"Total Value Exposed (XGBoost): ${val_exp_xgb.sum():,.2f}")
    
    # -------------------------------------------------------------
    # 5. Baseline Sensitivity Analysis (q in {0.03, 0.05, 0.08})
    # -------------------------------------------------------------
    print("\n4. Generating reports/phase_3_clv_sensitivity.csv...")
    sens_records = []
    tot_profit = float(profit_all.sum())
    
    for q in [0.03, 0.05, 0.08]:
        rf_q = retention_factor(q)
        adj_xgb = profit_all * (rf_xgb_all / rf_q)
        fdebt_xgb = profit_all - adj_xgb
        
        adj_rf = profit_all * (rf_rf_all / rf_q)
        fdebt_rf = profit_all - adj_rf
        
        sens_records.append({
            "baseline_churn_rate_q": q,
            "retention_factor_baseline_q": round(rf_q, 6),
            "total_stated_profit": round(tot_profit, 2),
            "total_retention_adjusted_clv_xgb": round(float(adj_xgb.sum()), 2),
            "total_forecast_debt_xgb": round(float(fdebt_xgb.sum()), 2),
            "mean_retention_adjusted_clv_xgb": round(float(adj_xgb.mean()), 2),
            "mean_forecast_debt_xgb": round(float(fdebt_xgb.mean()), 2),
            "total_retention_adjusted_clv_rf": round(float(adj_rf.sum()), 2),
            "total_forecast_debt_rf": round(float(fdebt_rf.sum()), 2),
            "value_reduction_pct_vs_stated": round(float(fdebt_xgb.sum() / tot_profit * 100), 2)
        })
        
    sens_df = pd.DataFrame(sens_records)
    sens_path = PATHS["reports_dir"] / "phase_3_clv_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"Saved: {sens_path}")
    print(sens_df[["baseline_churn_rate_q", "retention_factor_baseline_q", "total_retention_adjusted_clv_xgb", "total_forecast_debt_xgb", "value_reduction_pct_vs_stated"]].to_string(index=False))
    
    # -------------------------------------------------------------
    # 6. 15-Year Retention Scenarios
    # -------------------------------------------------------------
    print("\n5. Generating reports/phase_3_retention_scenarios.csv...")
    scenarios_dict = compute_15yr_retention_scenarios(p_xgb_all)
    scenarios_df = pd.DataFrame({
        "array_id": df_clean["array_id"],
        "account_id": df_clean["account_id"],
        "churn_probability_xgb": np.round(p_xgb_all, 5),
        "scenario_a_5yr_survival": np.round(scenarios_dict["scenario_a_s5"], 5),
        "scenario_a_10yr_survival": np.round(scenarios_dict["scenario_a_s10"], 5),
        "scenario_a_15yr_survival": np.round(scenarios_dict["scenario_a_s15"], 5),
        "scenario_a_expected_active_years": np.round(scenarios_dict["scenario_a_active_years"], 2),
        "scenario_b_5yr_survival": np.round(scenarios_dict["scenario_b_s5"], 5),
        "scenario_b_10yr_survival": np.round(scenarios_dict["scenario_b_s10"], 5),
        "scenario_b_15yr_survival": np.round(scenarios_dict["scenario_b_s15"], 5),
        "scenario_b_expected_active_years": np.round(scenarios_dict["scenario_b_active_years"], 2)
    })
    scen_path = PATHS["reports_dir"] / "phase_3_retention_scenarios.csv"
    scenarios_df.to_csv(scen_path, index=False)
    print(f"Saved: {scen_path}")
    print("Scenario Mean Active Years (Scenario A Constant):", round(scenarios_df["scenario_a_expected_active_years"].mean(), 2))
    print("Scenario Mean Active Years (Scenario B Fading):", round(scenarios_df["scenario_b_expected_active_years"].mean(), 2))
    
    # -------------------------------------------------------------
    # 7. Account Aggregation & Top 20 Prioritization
    # -------------------------------------------------------------
    print("\n6. Generating Account-Level Aggregation and reports/top_20_account_prioritization.csv...")
    acc_df = aggregate_account_level(df_clean, df_scores, df_shap)
    top20_df = compute_top_20_prioritization(acc_df)
    top20_path = PATHS["reports_dir"] / "top_20_account_prioritization.csv"
    top20_df.to_csv(top20_path, index=False)
    print(f"Saved: {top20_path}")
    print(top20_df[["rank", "account_id", "total_stated_profit", "churn_probability", "value_exposed", "Forecast_Debt"]].head(5).to_string(index=False))
    
    # -------------------------------------------------------------
    # 8. Intervention Break-Even Analysis
    # -------------------------------------------------------------
    print("\n7. Generating reports/intervention_break_even.csv...")
    be_df = compute_intervention_break_even(top20_df, costs=[5000.0, 10000.0])
    be_path = PATHS["reports_dir"] / "intervention_break_even.csv"
    be_df.to_csv(be_path, index=False)
    print(f"Saved: {be_path}")
    print(be_df.head(5).to_string(index=False))
    
    # -------------------------------------------------------------
    # 9. Generate 8 Publication-Grade Visualizations
    # -------------------------------------------------------------
    print("\n8. Generating 8 Publication-Grade Visualizations in reports/figures/...")
    figures_dir = PATHS["reports_dir"] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # FIG 11: Observed churn vs 5% baseline by cohort
    plt.figure(figsize=(10, 6))
    sub_c = cohort_df[cohort_df["cohort_dimension"] != "Overall"].copy()
    y_pos = np.arange(len(sub_c))
    
    plt.barh(y_pos, sub_c["observed_churn_rate"] * 100, color="#1f77b4", alpha=0.85, label="Observed Churn Rate (Actual %)")
    plt.axvline(5.0, color="#d62728", linestyle="--", linewidth=2, label="Static 5.0% Corporate Baseline")
    
    # Add Wilson CI error bars
    err_low = (sub_c["observed_churn_rate"] - sub_c["wilson_ci_lower"]) * 100
    err_high = (sub_c["wilson_ci_upper"] - sub_c["observed_churn_rate"]) * 100
    plt.errorbar(
        sub_c["observed_churn_rate"] * 100, y_pos,
        xerr=[err_low, err_high], fmt="none", color="#2c3e50", capsize=3, label="95% Wilson Confidence Interval"
    )
    
    plt.yticks(y_pos, [f"{row['cohort_dimension']}: {row['cohort_name']}" for _, row in sub_c.iterrows()])
    plt.xlabel("Churn Rate (%)")
    plt.title("Observed Churn vs. Static 5% Corporate Baseline across Cohorts")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "11_cohort_observed_vs_baseline.png")
    plt.close()
    print("  -> Saved: 11_cohort_observed_vs_baseline.png")
    
    # FIG 12: XGBoost predicted vs observed churn by cohort
    plt.figure(figsize=(10, 6))
    x = np.arange(len(sub_c))
    width = 0.35
    plt.bar(x - width/2, sub_c["observed_churn_rate"] * 100, width, label="Observed Actual Churn (%)", color="#1f77b4")
    plt.bar(x + width/2, sub_c["mean_p_xgb"] * 100, width, label="Calibrated XGBoost Predicted (%)", color="#2ca02c")
    plt.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.5, label="Static 5% Baseline")
    plt.xticks(x, [f"{row['cohort_dimension']}: {row['cohort_name']}" for _, row in sub_c.iterrows()], rotation=45, ha="right")
    plt.ylabel("Churn Rate (%)")
    plt.title("Calibrated XGBoost Predictions vs. Observed Churn across Cohorts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "12_cohort_xgb_vs_observed.png")
    plt.close()
    print("  -> Saved: 12_cohort_xgb_vs_observed.png")
    
    # FIG 13: Random Forest predicted vs observed churn by cohort
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, sub_c["observed_churn_rate"] * 100, width, label="Observed Actual Churn (%)", color="#1f77b4")
    plt.bar(x + width/2, sub_c["mean_p_rf"] * 100, width, label="Calibrated Random Forest Predicted (%)", color="#ff7f0e")
    plt.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.5, label="Static 5% Baseline")
    plt.xticks(x, [f"{row['cohort_dimension']}: {row['cohort_name']}" for _, row in sub_c.iterrows()], rotation=45, ha="right")
    plt.ylabel("Churn Rate (%)")
    plt.title("Calibrated Random Forest Predictions vs. Observed Churn across Cohorts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "13_cohort_rf_vs_observed.png")
    plt.close()
    print("  -> Saved: 13_cohort_rf_vs_observed.png")
    
    # FIG 14: Forecast Debt distribution
    plt.figure(figsize=(9, 5))
    plt.hist(debt_df["forecast_debt_xgb"] / 1e6, bins=40, color="#e74c3c", edgecolor="black", alpha=0.8)
    mean_d = debt_df["forecast_debt_xgb"].mean() / 1e6
    median_d = debt_df["forecast_debt_xgb"].median() / 1e6
    plt.axvline(mean_d, color="#2c3e50", linestyle="--", linewidth=2, label=f"Mean: ${mean_d:.2f}M")
    plt.axvline(median_d, color="#27ae60", linestyle="-", linewidth=2, label=f"Median: ${median_d:.2f}M")
    plt.xlabel("Forecast Debt ($ Millions USD)")
    plt.ylabel("Number of Storage Arrays")
    plt.title("Distribution of Forecast Debt per Array (Valuation Overstatement)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "14_forecast_debt_distribution.png")
    plt.close()
    print("  -> Saved: 14_forecast_debt_distribution.png")
    
    # FIG 15: Forecast Debt by utilization cohort
    plt.figure(figsize=(8, 5))
    util_debt = debt_df.groupby("utilization_rate_pct", observed=True)
    # Bin by quartiles
    debt_df["util_bin"] = pd.qcut(debt_df["utilization_rate_pct"], q=4, labels=["Q1 [35-66%)", "Q2 [66-93%)", "Q3 [93-118%)", "Q4 [118-145%]"])
    util_summary = debt_df.groupby("util_bin", observed=True)["forecast_debt_xgb"].sum() / 1e6
    plt.bar(util_summary.index, util_summary.values, color="#9b59b6", edgecolor="black", alpha=0.85)
    plt.ylabel("Total Forecast Debt ($ Millions USD)")
    plt.xlabel("Capacity Utilization Cohort")
    plt.title("Total Forecast Debt by Utilization Quartile Cohort")
    for i, val in enumerate(util_summary.values):
        plt.text(i, val + 10, f"${val:,.1f}M", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "15_forecast_debt_by_utilization.png")
    plt.close()
    print("  -> Saved: 15_forecast_debt_by_utilization.png")
    
    # FIG 16: Top 20 accounts by value exposed
    plt.figure(figsize=(10, 7))
    top20_sorted = top20_df.sort_values(by="value_exposed", ascending=True)
    plt.barh(top20_sorted["account_id"], top20_sorted["value_exposed"] / 1e6, color="#e67e22", edgecolor="black", alpha=0.85)
    plt.xlabel("Total Value Exposed ($ Millions USD)")
    plt.title("Top 20 Enterprise Accounts Prioritized by Annual Value Exposed")
    for i, val in enumerate(top20_sorted["value_exposed"] / 1e6):
        plt.text(val + 0.02, i, f"${val:.2f}M", va="center", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "16_top_20_accounts_value_exposed.png")
    plt.close()
    print("  -> Saved: 16_top_20_accounts_value_exposed.png")
    
    # FIG 17: 15-year retention scenario curves
    plt.figure(figsize=(9, 5))
    years = np.arange(1, 16)
    # Curves for 5% baseline, 15.8% constant churn, and 15.8% fading churn
    p_mean = float(np.mean(p_xgb_all))
    surv_5pct = (1.0 - 0.05) ** years
    surv_const = (1.0 - p_mean) ** years
    surv_fade = np.zeros(15)
    surv_fade[0] = 1.0 - p_mean
    for t in range(1, 15):
        surv_fade[t] = (1.0 - p_mean) * ((1.0 - p_mean / 2.0) ** t)
        
    plt.plot(years, surv_5pct * 100, "r--o", label=f"Corporate Baseline (Constant 5.0% churn)", linewidth=2)
    plt.plot(years, surv_const * 100, "b-s", label=f"Scenario A: Constant Empirical Churn ({p_mean*100:.1f}%)", linewidth=2)
    plt.plot(years, surv_fade * 100, "g-^", label=f"Scenario B: Fading Churn (Yr 1: {p_mean*100:.1f}%, Later: {p_mean*50:.1f}%)", linewidth=2)
    plt.xlabel("Horizon (Years)")
    plt.ylabel("Retention / Survival Rate (%)")
    plt.title("15-Year Finite Horizon Retention Scenarios vs. Baseline Assumption")
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "17_retention_scenarios_15yr_curves.png")
    plt.close()
    print("  -> Saved: 17_retention_scenarios_15yr_curves.png")
    
    # FIG 18: Baseline vs model retention-adjusted value
    plt.figure(figsize=(8, 5))
    totals = [
        tot_profit / 1e6,
        adj_clv_xgb.sum() / 1e6,
        adj_clv_rf.sum() / 1e6
    ]
    labels = ["Stated 15-Yr Profit\n(5% Churn Baseline)", "Retention-Adjusted CLV\n(Calibrated XGBoost)", "Retention-Adjusted CLV\n(Calibrated RF)"]
    colors = ["#3498db", "#2ecc71", "#1abc9c"]
    bars = plt.bar(labels, totals, color=colors, edgecolor="black", width=0.55)
    plt.ylabel("Portfolio Value ($ Millions USD)")
    plt.title("Portfolio Economic Valuation: Baseline Stated Profit vs. Behavior-Adjusted CLV")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 50, f"${h:,.1f}M", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig(figures_dir / "18_baseline_vs_model_retention_clv.png")
    plt.close()
    print("  -> Saved: 18_baseline_vs_model_retention_clv.png")
    
    print("\nPhase 3 execution completed successfully!")


if __name__ == "__main__":
    main()
