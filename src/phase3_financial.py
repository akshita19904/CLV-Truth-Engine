"""Phase 3 Financial & Back-Testing Engine: Forecast Bias, Retention-Adjusted CLV, and Value Exposure.

Implements Task 2 and the financial revaluation layer:
- Analytical Retention Factor: RF(p) = [1 - (1-p)^15] / (15*p)
- Retention-Adjusted CLV: stated_profit * RF(p_model) / RF(0.05)
- Forecast Debt: stated_profit - Retention_Adjusted_CLV
- Value Exposed: p_model * max(stated_profit, 0)
- Cohort-level Forecast Bias back-testing on labelled rows with Wilson CIs and Holm-adjusted tests
- Account-level risk-and-value aggregation (1,000 accounts, 2 arrays per account)
- 15-year retention scenarios (Constant Churn and Fading Churn)
- Baseline sensitivity analysis (q in {0.03, 0.05, 0.08})
- Non-causal intervention break-even modeling ($5k, $10k)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint


def retention_factor(p: Union[float, int, np.ndarray, pd.Series]) -> Union[float, np.ndarray]:
    """Computes the 15-year finite-horizon retention factor RF(p).
    
    Formula:
        RF(p) = (1/15) * sum_{t=0}^{14} (1-p)^t = [1 - (1-p)^15] / (15 * p)  for p > 0
        RF(0) = 1.0
        
    Args:
        p: Annual churn probability (scalar or array-like in [0, 1]).
        
    Returns:
        Retention factor value(s) in (0, 1].
    """
    p_arr = np.asarray(p, dtype=float)
    scalar_input = p_arr.ndim == 0
    p_arr = np.atleast_1d(p_arr)
    
    if np.any((p_arr < 0.0) | (p_arr > 1.0)):
        raise ValueError("Churn probability p must lie within [0, 1].")
        
    rf = np.ones_like(p_arr)
    mask = p_arr > 0.0
    rf[mask] = (1.0 - (1.0 - p_arr[mask]) ** 15) / (15.0 * p_arr[mask])
    
    # Handle numerical edge cases
    rf[p_arr == 0.0] = 1.0
    
    if scalar_input:
        return float(rf[0])
    return rf


def compute_retention_adjusted_clv(
    stated_profit: Union[float, np.ndarray, pd.Series],
    p_model: Union[float, np.ndarray, pd.Series],
    p_baseline: float = 0.05
) -> Union[float, np.ndarray]:
    """Calculates Retention-Adjusted CLV revaluation.
    
    Formula:
        Retention_Adjusted_CLV = stated_profit * (RF(p_model) / RF(p_baseline))
    """
    rf_baseline = retention_factor(p_baseline)
    rf_m = retention_factor(p_model)
    return stated_profit * (rf_m / rf_baseline)


def compute_forecast_debt(
    stated_profit: Union[float, np.ndarray, pd.Series],
    retention_adjusted_clv: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray]:
    """Calculates Forecast Debt (model-based valuation gap).
    
    Formula:
        Forecast_Debt = stated_profit - Retention_Adjusted_CLV
    """
    return stated_profit - retention_adjusted_clv


def compute_value_exposed(
    stated_profit: Union[float, np.ndarray, pd.Series],
    p_model: Union[float, np.ndarray, pd.Series]
) -> Union[float, np.ndarray]:
    """Calculates single-year value exposed to churn risk.
    
    Formula:
        value_exposed = p_model * max(stated_profit, 0)
        
    Negative profit records carry zero economic value exposed to retention loss.
    """
    profit_arr = np.asarray(stated_profit, dtype=float)
    p_arr = np.asarray(p_model, dtype=float)
    
    pos_profit = np.maximum(profit_arr, 0.0)
    return p_arr * pos_profit


def compute_15yr_retention_scenarios(
    p_model: Union[float, np.ndarray, pd.Series]
) -> Dict[str, Union[float, np.ndarray]]:
    """Calculates 15-year survival scenarios under Constant and Fading churn assumptions.
    
    NOTE: These are illustrative geometric survival scenarios, NOT supervised
    15-year renewal predictions.
    
    Scenario A (Constant annual churn p):
        S_A(t) = (1 - p)^t
        Expected active years = sum_{t=0}^{14} (1-p)^t = 15 * RF(p)
        
    Scenario B (Fading annual churn: Year 1 = p, later years = p/2):
        S_B(1) = 1 - p
        S_B(t) = (1 - p) * (1 - p/2)^(t-1) for t >= 2
        Expected active years = 1 + (1 - p) * [1 - (1 - p/2)^14] / (p/2)
    """
    p_arr = np.asarray(p_model, dtype=float)
    scalar_input = p_arr.ndim == 0
    p_arr = np.atleast_1d(p_arr)
    
    # Scenario A: Constant churn
    scen_a_s5 = (1.0 - p_arr) ** 5
    scen_a_s10 = (1.0 - p_arr) ** 10
    scen_a_s15 = (1.0 - p_arr) ** 15
    scen_a_active_years = 15.0 * retention_factor(p_arr)
    
    # Scenario B: Fading churn
    half_p = p_arr / 2.0
    scen_b_s1 = 1.0 - p_arr
    scen_b_s5 = (1.0 - p_arr) * ((1.0 - half_p) ** 4)
    scen_b_s10 = (1.0 - p_arr) * ((1.0 - half_p) ** 9)
    scen_b_s15 = (1.0 - p_arr) * ((1.0 - half_p) ** 14)
    
    # Expected active years for fading churn
    scen_b_active_years = np.full_like(p_arr, 15.0)
    mask = half_p > 0.0
    scen_b_active_years[mask] = 1.0 + (1.0 - p_arr[mask]) * (
        (1.0 - (1.0 - half_p[mask]) ** 14) / half_p[mask]
    )
    
    if scalar_input:
        return {
            "scenario_a_s5": float(scen_a_s5[0]),
            "scenario_a_s10": float(scen_a_s10[0]),
            "scenario_a_s15": float(scen_a_s15[0]),
            "scenario_a_active_years": float(scen_a_active_years[0]),
            "scenario_b_s5": float(scen_b_s5[0]),
            "scenario_b_s10": float(scen_b_s10[0]),
            "scenario_b_s15": float(scen_b_s15[0]),
            "scenario_b_active_years": float(scen_b_active_years[0]),
        }
        
    return {
        "scenario_a_s5": scen_a_s5,
        "scenario_a_s10": scen_a_s10,
        "scenario_a_s15": scen_a_s15,
        "scenario_a_active_years": scen_a_active_years,
        "scenario_b_s5": scen_b_s5,
        "scenario_b_s10": scen_b_s10,
        "scenario_b_s15": scen_b_s15,
        "scenario_b_active_years": scen_b_active_years,
    }


def compute_cohort_bias_table(
    df_clean: pd.DataFrame,
    scores_df: pd.DataFrame
) -> pd.DataFrame:
    """Builds the comprehensive Task 2 cohort forecast bias back-testing table.
    
    Filters strictly to the 1,884 labelled rows (is_label_known == 1).
    Evaluates:
    - Overall
    - Utilization cohorts (quartiles)
    - Expansion signal cohorts (Low, Medium, High, UNKNOWN)
    - Capacity cohorts (quartiles)
    - Historical Profit cohorts (quartiles + Negative Profit)
    
    Calculates sample size, observed churn, static 5% baseline, model mean probabilities,
    gaps, 95% Wilson confidence intervals, exact binomial test p-values, and
    Holm-Bonferroni adjusted p-values.
    """
    merged = pd.merge(
        df_clean,
        scores_df[["array_id", "churn_probability_xgb", "churn_probability_rf"]],
        on="array_id"
    )
    
    # Exclude holdout / unlabelled rows strictly from empirical back-testing
    lab = merged[merged["is_label_known"] == 1].copy()
    
    def _eval_group(sub: pd.DataFrame, dim: str, name: str) -> dict:
        n = len(sub)
        k = int(sub["churn_event_next_12m"].sum())
        obs = k / n if n > 0 else 0.0
        ci_low, ci_high = proportion_confint(k, n, alpha=0.05, method="wilson") if n > 0 else (0.0, 0.0)
        p_val = stats.binomtest(k, n, p=0.05).pvalue if n > 0 else 1.0
        mean_xgb = float(sub["churn_probability_xgb"].mean())
        mean_rf = float(sub["churn_probability_rf"].mean())
        
        return {
            "cohort_dimension": dim,
            "cohort_name": name,
            "sample_size": n,
            "churn_count": k,
            "observed_churn_rate": round(obs, 4),
            "static_baseline": 0.05,
            "mean_p_xgb": round(mean_xgb, 4),
            "mean_p_rf": round(mean_rf, 4),
            "gap_xgb_vs_actual": round(mean_xgb - obs, 4),
            "gap_rf_vs_actual": round(mean_rf - obs, 4),
            "gap_baseline_vs_actual": round(0.05 - obs, 4),
            "wilson_ci_lower": round(ci_low, 4),
            "wilson_ci_upper": round(ci_high, 4),
            "p_value_vs_baseline": p_val
        }

    cohort_records = []
    
    # 1. Overall
    cohort_records.append(_eval_group(lab, "Overall", "All Labelled Arrays"))
    
    # 2. Utilization quartiles
    lab["util_q"] = pd.qcut(
        lab["utilization_rate_pct"],
        q=4,
        labels=["Q1 [0.35, 0.66)", "Q2 [0.66, 0.93)", "Q3 [0.93, 1.18)", "Q4 [1.18, 1.45]"]
    )
    for q_name, group in lab.groupby("util_q", observed=True):
        cohort_records.append(_eval_group(group, "Utilization", str(q_name)))
        
    # 3. Expansion signal
    for sig in ["Low", "Medium", "High", "UNKNOWN"]:
        group = lab[lab["expansion_potential_signal"] == sig]
        cohort_records.append(_eval_group(group, "Expansion Signal", sig))
        
    # 4. Capacity quartiles
    lab["cap_q"] = pd.qcut(
        lab["current_capacity_tb"],
        q=4,
        labels=["Q1 [53, 674TB)", "Q2 [674, 1306TB)", "Q3 [1306, 1880TB)", "Q4 [1880, 2498TB]"]
    )
    for q_name, group in lab.groupby("cap_q", observed=True):
        cohort_records.append(_eval_group(group, "Capacity", str(q_name)))
        
    # 5. Profit quartiles
    lab["profit_q"] = pd.qcut(
        lab["historical_15yr_net_profit_usd"],
        q=4,
        labels=["Q1 [-93.8k, 997k)", "Q2 [997k, 2.13M)", "Q3 [2.13M, 3.28M)", "Q4 [3.28M, 4.50M]"]
    )
    for q_name, group in lab.groupby("profit_q", observed=True):
        cohort_records.append(_eval_group(group, "Historical Profit", str(q_name)))
        
    # 6. Negative profit subset
    neg_profit_sub = lab[lab["historical_15yr_net_profit_usd"] < 0]
    cohort_records.append(_eval_group(neg_profit_sub, "Historical Profit", "Negative Profit (<$0)"))
    
    cohort_df = pd.DataFrame(cohort_records)
    
    # Holm-Bonferroni correction
    reject, p_corrected, _, _ = multipletests(cohort_df["p_value_vs_baseline"], alpha=0.05, method="holm")
    cohort_df["p_value_holm_adjusted"] = p_corrected
    cohort_df["is_significant_diff"] = reject
    
    return cohort_df


def aggregate_account_level(
    df_clean: pd.DataFrame,
    scores_df: pd.DataFrame,
    shap_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Aggregates array-level records to exactly 1,000 enterprise accounts.
    
    Preserves total economic metrics:
    - total_stated_profit
    - total_retention_adjusted_clv
    - total_forecast_debt
    - total_value_exposed
    - avg_churn_probability and max_churn_probability
    - avg_utilization
    - capacity_pressure
    - expansion_signal (highest tier present across account arrays)
    - over_capacity_count (number of arrays > 100% capacity)
    - top_shap_drivers (extracted from TreeSHAP values)
    """
    merged = pd.merge(
        df_clean,
        scores_df[[
            "array_id", "churn_probability_xgb", "churn_probability_rf", "expansion_score"
        ]],
        on="array_id"
    )
    
    if "capacity_pressure" not in merged.columns:
        merged["capacity_pressure"] = np.round(
            merged["current_capacity_tb"] * merged["utilization_rate_pct"], 2
        )
    
    # Financial computations at array level using primary champion XGBoost
    rf_05 = retention_factor(0.05)
    merged["rf_xgb"] = retention_factor(merged["churn_probability_xgb"].values)
    merged["retention_adjusted_clv_xgb"] = compute_retention_adjusted_clv(
        merged["historical_15yr_net_profit_usd"].values,
        merged["churn_probability_xgb"].values,
        p_baseline=0.05
    )
    merged["forecast_debt_xgb"] = compute_forecast_debt(
        merged["historical_15yr_net_profit_usd"].values,
        merged["retention_adjusted_clv_xgb"].values
    )
    merged["value_exposed_xgb"] = compute_value_exposed(
        merged["historical_15yr_net_profit_usd"].values,
        merged["churn_probability_xgb"].values
    )
    
    # Precompute SHAP top features per account if available
    acc_top_shap = {}
    if shap_df is not None and not shap_df.empty:
        # Group SHAP by account_id and feature
        shap_grouped = shap_df.groupby(["account_id", "feature"])["shap_value"].apply(
            lambda x: x.abs().mean()
        ).reset_index()
        for acc_id, group in shap_grouped.groupby("account_id"):
            top_feats = group.sort_values(by="shap_value", ascending=False).head(3)["feature"].tolist()
            acc_top_shap[acc_id] = ", ".join(top_feats)
            
    # Expansion signal hierarchy
    sig_order = {"High": 3, "Medium": 2, "Low": 1, "UNKNOWN": 0}
    inv_order = {3: "High", 2: "Medium", 1: "Low", 0: "UNKNOWN"}
    
    rows = []
    for acc_id, grp in merged.groupby("account_id"):
        assert len(grp) == 2, f"Account {acc_id} does not have exactly 2 arrays (found {len(grp)})."
        
        tot_profit = float(grp["historical_15yr_net_profit_usd"].sum())
        tot_adj_clv = float(grp["retention_adjusted_clv_xgb"].sum())
        tot_debt = float(grp["forecast_debt_xgb"].sum())
        tot_exp = float(grp["value_exposed_xgb"].sum())
        
        avg_p = float(grp["churn_probability_xgb"].mean())
        max_p = float(grp["churn_probability_xgb"].max())
        
        avg_util = float(grp["utilization_rate_pct"].mean())
        cap_press = float(grp["capacity_pressure"].max())
        over_cap_cnt = int(grp["over_capacity"].sum())
        
        # Max expansion signal tier across the 2 arrays
        max_sig_val = max(grp["expansion_potential_signal"].map(sig_order))
        acc_sig = inv_order.get(max_sig_val, "UNKNOWN")
        
        shap_str = acc_top_shap.get(acc_id, "contract_end, capacity, utilization")
        
        rows.append({
            "account_id": acc_id,
            "total_stated_profit": round(tot_profit, 2),
            "total_retention_adjusted_clv": round(tot_adj_clv, 2),
            "total_forecast_debt": round(tot_debt, 2),
            "total_value_exposed": round(tot_exp, 2),
            "avg_churn_probability": round(avg_p, 5),
            "max_churn_probability": round(max_p, 5),
            "avg_utilization": round(avg_util, 4),
            "capacity_pressure": round(cap_press, 4),
            "expansion_signal": acc_sig,
            "over_capacity_count": over_cap_cnt,
            "top_shap_drivers": shap_str
        })
        
    acc_df = pd.DataFrame(rows)
    return acc_df


def compute_top_20_prioritization(account_df: pd.DataFrame) -> pd.DataFrame:
    """Prioritizes Top 20 enterprise accounts by Total Value Exposed.
    
    Framed strictly as 'Risk-and-value prioritization' rather than
    'top churners', reflecting limited individual-level discrimination.
    """
    top20 = account_df.sort_values(by="total_value_exposed", ascending=False).head(20).copy().reset_index(drop=True)
    top20["rank"] = range(1, 21)
    
    # Format and select final columns
    cols_order = [
        "rank",
        "account_id",
        "total_stated_profit",
        "avg_churn_probability",
        "total_value_exposed",
        "total_forecast_debt",
        "avg_utilization",
        "capacity_pressure",
        "expansion_signal",
        "over_capacity_count",
        "top_shap_drivers"
    ]
    return top20[cols_order].rename(columns={
        "avg_churn_probability": "churn_probability",
        "total_value_exposed": "value_exposed",
        "total_forecast_debt": "Forecast_Debt",
        "avg_utilization": "utilization",
        "over_capacity_count": "over_capacity_status"
    })


def compute_intervention_break_even(
    top20_df: pd.DataFrame,
    costs: List[float] = [5000.0, 10000.0]
) -> pd.DataFrame:
    """Calculates non-causal break-even required effectiveness for customer interventions.
    
    Formulas:
        required_effectiveness = C / (p * value_exposed)
        break_even_relative_churn_reduction = C / value_exposed
    """
    records = []
    for _, row in top20_df.iterrows():
        acc = row["account_id"]
        profit = row["total_stated_profit"]
        p = row["churn_probability"]
        val_exp = row["value_exposed"]
        
        rec = {
            "account_id": acc,
            "total_stated_profit": profit,
            "churn_probability": p,
            "value_exposed": val_exp
        }
        for c in costs:
            c_tag = f"{int(c//1000)}k"
            req_eff = c / (p * val_exp) if (p * val_exp) > 0 else np.nan
            rel_churn_red = c / val_exp if val_exp > 0 else np.nan
            rec[f"cost_{c_tag}"] = c
            rec[f"required_effectiveness_{c_tag}"] = round(req_eff, 4) if np.isfinite(req_eff) else np.nan
            rec[f"break_even_churn_reduction_{c_tag}"] = round(rel_churn_red, 4) if np.isfinite(rel_churn_red) else np.nan
            
        records.append(rec)
        
    return pd.DataFrame(records)
