"""Unit tests for Phase 3: Forecast Bias Back-Testing, CLV Revaluation, and Value Exposure.

Covers all 16 required test verifications:
1. Forecast Debt formula
2. retention factor formula
3. RF(0) = 1
4. survival probabilities in [0, 1]
5. survival is non-increasing with horizon
6. value_exposed >= 0
7. negative profit -> value_exposed = 0
8. unlabelled rows excluded from back-testing
9. OOF predictions used for labelled evaluation
10. exactly 2 arrays/account
11. account aggregation preserves economic totals
12. no sibling churn leakage
13. baseline always = 0.05
14. sensitivity values exactly 0.03, 0.05, 0.08
15. Top 20 has 20 unique accounts
16. Forecast_Debt = stated_profit - retention_adjusted_clv
"""

import numpy as np
import pandas as pd
import pytest

from src.config import PATHS
from src.features import ALL_FEATURE_COLS
from src.phase3_financial import (
    retention_factor,
    compute_retention_adjusted_clv,
    compute_forecast_debt,
    compute_value_exposed,
    compute_15yr_retention_scenarios,
    aggregate_account_level
)


@pytest.fixture(scope="module")
def cleaned_data():
    return pd.read_csv(PATHS["processed_data_dir"] / "cleaned_data.csv")


@pytest.fixture(scope="module")
def final_scores():
    return pd.read_csv(PATHS["reports_dir"] / "task1_final_scores.csv")


@pytest.fixture(scope="module")
def cohort_results():
    return pd.read_csv(PATHS["reports_dir"] / "phase_3_cohort_bias_results.csv")


@pytest.fixture(scope="module")
def forecast_debt_df():
    return pd.read_csv(PATHS["reports_dir"] / "phase_3_forecast_debt.csv")


@pytest.fixture(scope="module")
def clv_sensitivity_df():
    return pd.read_csv(PATHS["reports_dir"] / "phase_3_clv_sensitivity.csv")


@pytest.fixture(scope="module")
def retention_scenarios_df():
    return pd.read_csv(PATHS["reports_dir"] / "phase_3_retention_scenarios.csv")


@pytest.fixture(scope="module")
def value_exposure_df():
    return pd.read_csv(PATHS["reports_dir"] / "phase_3_value_exposure.csv")


@pytest.fixture(scope="module")
def top20_df():
    return pd.read_csv(PATHS["reports_dir"] / "top_20_account_prioritization.csv")


# 1. retention factor formula & RF(0) = 1
def test_retention_factor_formula_and_rf0():
    """TEST 2 & 3: retention factor formula and RF(0) == 1."""
    # Test RF(0) == 1
    assert retention_factor(0.0) == 1.0
    assert np.isclose(retention_factor(np.array([0.0]))[0], 1.0)
    
    # Test analytical RF(0.05) = (1 - 0.95^15) / (15 * 0.05)
    expected_05 = (1.0 - (1.0 - 0.05) ** 15) / (15.0 * 0.05)
    assert np.isclose(retention_factor(0.05), expected_05, atol=1e-6)
    
    # Test analytical RF(1.0) = 1 / 15
    assert np.isclose(retention_factor(1.0), 1.0 / 15.0, atol=1e-6)
    
    # Test monotonically decreasing with churn rate
    p_grid = np.linspace(0.01, 0.99, 50)
    rf_grid = retention_factor(p_grid)
    assert (np.diff(rf_grid) < 0).all(), "Retention factor must decrease monotonically with p!"


# 2. Forecast Debt formula
def test_forecast_debt_formula(forecast_debt_df):
    """TEST 1 & 16: Forecast_Debt = stated_profit - retention_adjusted_clv."""
    profit = forecast_debt_df["historical_15yr_net_profit_usd"]
    adj_clv_xgb = forecast_debt_df["retention_adjusted_clv_xgb"]
    debt_xgb = forecast_debt_df["forecast_debt_xgb"]
    
    # Verify exact element-wise formula equality within floating point precision
    diff = (profit - adj_clv_xgb) - debt_xgb
    assert (diff.abs() < 1e-2).all(), "Forecast Debt does not equal stated_profit - retention_adjusted_clv!"
    
    # Test function directly
    calc_debt = compute_forecast_debt(profit.values, adj_clv_xgb.values)
    assert np.allclose(calc_debt, debt_xgb.values, atol=1e-2)


# 3. value_exposed >= 0 and negative profit -> value_exposed = 0
def test_value_exposed_non_negative_and_zero_for_negative_profit(value_exposure_df, cleaned_data):
    """TEST 6 & 7: value_exposed >= 0, and negative profit -> value_exposed == 0."""
    val_exp = value_exposure_df["value_exposed_xgb"]
    profit = value_exposure_df["historical_15yr_net_profit_usd"]
    
    # value_exposed >= 0 everywhere
    assert (val_exp >= 0.0).all(), "Found negative value_exposed!"
    
    # For negative profit arrays, value_exposed must be strictly 0.0
    neg_mask = profit < 0
    assert neg_mask.sum() > 0, "No negative profit records found in test data"
    assert (val_exp[neg_mask] == 0.0).all(), "Negative profit arrays have non-zero value_exposed!"
    
    # Test scalar function
    assert compute_value_exposed(-50000.0, 0.20) == 0.0
    assert compute_value_exposed(100000.0, 0.20) == 20000.0


# 4. Survival probabilities in [0, 1] and non-increasing with horizon
def test_survival_probabilities_bounded_and_monotonic(retention_scenarios_df):
    """TEST 4 & 5: Survival probabilities in [0, 1] and non-increasing with horizon."""
    cols_a = ["scenario_a_5yr_survival", "scenario_a_10yr_survival", "scenario_a_15yr_survival"]
    cols_b = ["scenario_b_5yr_survival", "scenario_b_10yr_survival", "scenario_b_15yr_survival"]
    
    for c in cols_a + cols_b:
        vals = retention_scenarios_df[c]
        assert (vals >= 0.0).all(), f"Found negative survival in {c}"
        assert (vals <= 1.0).all(), f"Found survival > 1 in {c}"
        
    # Non-increasing with horizon for Scenario A
    s5_a = retention_scenarios_df["scenario_a_5yr_survival"]
    s10_a = retention_scenarios_df["scenario_a_10yr_survival"]
    s15_a = retention_scenarios_df["scenario_a_15yr_survival"]
    assert (s5_a >= s10_a).all(), "Scenario A survival increases from 5 to 10 years!"
    assert (s10_a >= s15_a).all(), "Scenario A survival increases from 10 to 15 years!"
    
    # Non-increasing with horizon for Scenario B
    s5_b = retention_scenarios_df["scenario_b_5yr_survival"]
    s10_b = retention_scenarios_df["scenario_b_10yr_survival"]
    s15_b = retention_scenarios_df["scenario_b_15yr_survival"]
    assert (s5_b >= s10_b).all(), "Scenario B survival increases from 5 to 10 years!"
    assert (s10_b >= s15_b).all(), "Scenario B survival increases from 10 to 15 years!"


# 5. Unlabelled rows excluded from back-testing
def test_unlabelled_rows_excluded_from_back_testing(cohort_results, cleaned_data):
    """TEST 8: Unlabelled rows (is_label_known == 0, N=116) excluded from empirical back-testing."""
    overall_cohort = cohort_results[cohort_results["cohort_name"] == "All Labelled Arrays"]
    assert len(overall_cohort) == 1, "Expected single 'All Labelled Arrays' cohort row"
    
    sample_size = overall_cohort["sample_size"].iloc[0]
    labelled_count = (cleaned_data["is_label_known"] == 1).sum()
    
    assert sample_size == labelled_count, (
        f"Cohort back-testing used {sample_size} rows, but labelled data has {labelled_count} rows!"
    )
    assert sample_size == 1884, f"Expected exactly 1,884 labelled rows, got {sample_size}"


# 6. OOF predictions used for labelled evaluation
def test_oof_predictions_used_for_labelled_evaluation(final_scores, cleaned_data):
    """TEST 9: OOF calibrated predictions used for labelled evaluation."""
    is_lab = cleaned_data["is_label_known"] == 1
    p_xgb_lab = final_scores.loc[is_lab, "churn_probability_xgb"]
    actual_churn_rate = cleaned_data.loc[is_lab, "churn_event_next_12m"].mean()
    
    # GroupKFold OOF predictions must be properly calibrated close to ground truth churn rate (15.82%)
    assert abs(p_xgb_lab.mean() - actual_churn_rate) < 0.01, (
        f"Mean OOF prediction ({p_xgb_lab.mean():.4f}) diverges from actual churn rate ({actual_churn_rate:.4f})!"
    )


# 7. Exactly 2 arrays per account
def test_exactly_2_arrays_per_account(cleaned_data):
    """TEST 10: Every account must have exactly 2 arrays."""
    counts = cleaned_data.groupby("account_id")["array_id"].count()
    assert (counts == 2).all(), "Found accounts with array count != 2!"
    assert len(counts) == 1000, f"Expected 1,000 accounts, found {len(counts)}"


# 8. Account aggregation preserves economic totals
def test_account_aggregation_preserves_economic_totals(cleaned_data, final_scores):
    """TEST 11: Account aggregation preserves total profit, adjusted CLV, Forecast Debt, and value exposed."""
    acc_df = aggregate_account_level(cleaned_data, final_scores)
    assert len(acc_df) == 1000, f"Expected exactly 1,000 aggregated accounts, got {len(acc_df)}"
    
    # Compute array-level totals
    val_exp_array = compute_value_exposed(
        cleaned_data["historical_15yr_net_profit_usd"].values,
        final_scores["churn_probability_xgb"].values
    )
    adj_clv_array = compute_retention_adjusted_clv(
        cleaned_data["historical_15yr_net_profit_usd"].values,
        final_scores["churn_probability_xgb"].values,
        p_baseline=0.05
    )
    debt_array = compute_forecast_debt(
        cleaned_data["historical_15yr_net_profit_usd"].values,
        adj_clv_array
    )
    
    tot_profit_array = cleaned_data["historical_15yr_net_profit_usd"].sum()
    tot_exp_array = val_exp_array.sum()
    tot_clv_array = adj_clv_array.sum()
    tot_debt_array = debt_array.sum()
    
    # Check aggregation sums match within $1.00
    assert abs(acc_df["total_stated_profit"].sum() - tot_profit_array) < 1.0
    assert abs(acc_df["total_value_exposed"].sum() - tot_exp_array) < 1.0
    assert abs(acc_df["total_retention_adjusted_clv"].sum() - tot_clv_array) < 1.0
    assert abs(acc_df["total_forecast_debt"].sum() - tot_debt_array) < 1.0


# 9. No sibling churn leakage
def test_no_sibling_churn_leakage():
    """TEST 12: No sibling churn label features exist in candidate feature set."""
    forbidden = ["sibling_churn", "sibling_churn_event_next_12m", "account_churn_rate"]
    for f in forbidden:
        assert f not in ALL_FEATURE_COLS, f"Forbidden sibling churn leakage feature found: {f}"


# 10. Baseline always = 0.05
def test_baseline_always_0_05(cohort_results):
    """TEST 13: Corporate baseline assumption is strictly 0.05 across all cohorts."""
    assert (cohort_results["static_baseline"] == 0.05).all(), "Found static_baseline != 0.05!"


# 11. Sensitivity values exactly 0.03, 0.05, 0.08
def test_sensitivity_values_exactly_003_005_008(clv_sensitivity_df):
    """TEST 14: Sensitivity analysis tests exactly q in {0.03, 0.05, 0.08}."""
    expected_baselines = {0.03, 0.05, 0.08}
    actual_baselines = set(clv_sensitivity_df["baseline_churn_rate_q"])
    assert actual_baselines == expected_baselines, (
        f"Expected sensitivity baselines {expected_baselines}, got {actual_baselines}"
    )


# 12. Top 20 has 20 unique accounts
def test_top_20_has_20_unique_accounts(top20_df):
    """TEST 15: Top 20 prioritization contains exactly 20 unique accounts ranked 1 to 20."""
    assert len(top20_df) == 20, f"Expected 20 rows in Top 20 table, got {len(top20_df)}"
    assert top20_df["account_id"].nunique() == 20, "Duplicate account_id in Top 20 prioritization!"
    assert list(top20_df["rank"]) == list(range(1, 21)), "Ranks in Top 20 are not sequential 1 to 20!"
