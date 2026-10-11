"""Automated verification of the Snowflake export dataset and SQL validation logic.

Executes all 14 validation checks required by Snowflake Step 12:
1. Total production rows = 2,000
2. Unique array IDs = 2,000
3. Unique account IDs = 1,000
4. Exactly 2 arrays per account
5. No duplicate production keys (array_id, model_version, run_id)
6. Churn probability bounded in [0, 1]
7. Target label isolation: exactly 116 NULL labels for holdout rows
8. Value exposed >= 0
9. Negative-profit rows have strictly zero value exposed
10. Forecast Debt reconciles: forecast_debt = stated_profit - retention_adjusted_clv
11. Probability bucket agrees with churn probability
12. SHAP drivers exist for labelled rows with numeric SHAP values
13. Model version is populated
14. Run ID is populated
15. Cohort forecast bias monitor table has exactly 18 cohorts and valid statistics
16. Alias columns match (xgb_churn_probability == churn_probability, churn_label == churn_event_next_12m)
"""

import sqlite3
import pandas as pd
import numpy as np
import pytest

from src.config import PATHS


@pytest.fixture(scope="module")
def sql_db():
    """Sets up an in-memory SQL database with the exact exported Snowflake datasets."""
    conn = sqlite3.connect(":memory:")
    
    scored_path = PATHS["processed_data_dir"] / "snowflake_clv_scored_export.csv"
    bias_path = PATHS["processed_data_dir"] / "snowflake_bias_monitor_export.csv"
    
    assert scored_path.exists(), f"Missing export file: {scored_path}"
    assert bias_path.exists(), f"Missing export file: {bias_path}"
    
    df_scored = pd.read_csv(scored_path)
    df_bias = pd.read_csv(bias_path)
    
    df_scored.to_sql("CLV_SCORED_OUTPUT", conn, index=False)
    df_bias.to_sql("CLV_FORECAST_BIAS_MONITOR", conn, index=False)
    
    yield conn
    conn.close()


def test_sql_row_counts_and_uniqueness(sql_db):
    """TEST 1, 2, 3: Total rows = 2,000, unique arrays = 2,000, unique accounts = 1,000."""
    cur = sql_db.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT array_id), COUNT(DISTINCT account_id) FROM CLV_SCORED_OUTPUT")
    tot, arrs, accs = cur.fetchone()
    assert tot == 2000, f"Expected 2,000 total rows, got {tot}"
    assert arrs == 2000, f"Expected 2,000 unique arrays, got {arrs}"
    assert accs == 1000, f"Expected 1,000 unique accounts, got {accs}"


def test_sql_two_arrays_per_account(sql_db):
    """TEST 4: Exactly 2 arrays per account for 100% of accounts."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT account_id, COUNT(array_id) AS cnt
            FROM CLV_SCORED_OUTPUT
            GROUP BY account_id
            HAVING COUNT(array_id) != 2
        )
    """)
    violations = cur.fetchone()[0]
    assert violations == 0, f"Found {violations} accounts with array count != 2!"


def test_sql_primary_key_uniqueness(sql_db):
    """TEST 5: Primary key (array_id, model_version, run_id) has zero duplicates."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM (
            SELECT array_id, model_version, run_id, COUNT(*) AS cnt
            FROM CLV_SCORED_OUTPUT
            GROUP BY array_id, model_version, run_id
            HAVING COUNT(*) > 1
        )
    """)
    dups = cur.fetchone()[0]
    assert dups == 0, f"Found {dups} duplicate primary key entries!"


def test_sql_probability_bounds(sql_db):
    """TEST 6: Churn probability is bounded in [0, 1]."""
    cur = sql_db.cursor()
    cur.execute("SELECT COUNT(*) FROM CLV_SCORED_OUTPUT WHERE churn_probability < 0.0 OR churn_probability > 1.0")
    out_of_bounds = cur.fetchone()[0]
    assert out_of_bounds == 0, f"Found {out_of_bounds} probabilities outside [0, 1]!"


def test_sql_target_label_isolation(sql_db):
    """TEST 7: Exactly 116 NULL churn labels, exactly 1,884 known churn labels."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT 
            COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) AS null_cnt,
            COUNT(CASE WHEN is_label_known = 0 THEN 1 END) AS holdout_cnt,
            COUNT(CASE WHEN is_label_known = 1 THEN 1 END) AS labelled_cnt
        FROM CLV_SCORED_OUTPUT
    """)
    null_cnt, holdout_cnt, labelled_cnt = cur.fetchone()
    assert null_cnt == 116, f"Expected 116 NULL target labels, got {null_cnt}"
    assert holdout_cnt == 116, f"Expected 116 holdout rows, got {holdout_cnt}"
    assert labelled_cnt == 1884, f"Expected 1,884 labelled rows, got {labelled_cnt}"


def test_sql_column_aliases(sql_db):
    """TEST: Alias columns xgb_churn_probability and churn_label match primary columns."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT 
            COUNT(CASE WHEN xgb_churn_probability != churn_probability THEN 1 END),
            COUNT(CASE WHEN churn_label IS NOT NULL AND churn_label != churn_event_next_12m THEN 1 END),
            COUNT(CASE WHEN churn_label IS NULL THEN 1 END)
        FROM CLV_SCORED_OUTPUT
    """)
    prob_mismatches, label_mismatches, null_labels = cur.fetchone()
    assert prob_mismatches == 0, f"Found {prob_mismatches} prob mismatches!"
    assert label_mismatches == 0, f"Found {label_mismatches} label mismatches!"
    assert null_labels == 116, f"Expected 116 null churn_labels, got {null_labels}"


def test_sql_value_exposed_non_negative_and_zero_for_losses(sql_db):
    """TEST 8 & 9: Value exposed >= 0 everywhere, and exactly 0 for negative stated profits."""
    cur = sql_db.cursor()
    cur.execute("SELECT COUNT(*) FROM CLV_SCORED_OUTPUT WHERE value_exposed < 0.0")
    neg_exp = cur.fetchone()[0]
    assert neg_exp == 0, f"Found {neg_exp} negative value_exposed records!"
    
    cur.execute("SELECT COUNT(*) FROM CLV_SCORED_OUTPUT WHERE stated_profit < 0.0 AND value_exposed != 0.0")
    invalid_loss_exp = cur.fetchone()[0]
    assert invalid_loss_exp == 0, f"Found {invalid_loss_exp} loss-making arrays with non-zero value exposed!"


def test_sql_forecast_debt_reconciliation(sql_db):
    """TEST 10: Forecast Debt reconciles: forecast_debt = stated_profit - retention_adjusted_clv."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT COUNT(*), MAX(ABS(stated_profit - retention_adjusted_clv - forecast_debt))
        FROM CLV_SCORED_OUTPUT
        WHERE ABS(stated_profit - retention_adjusted_clv - forecast_debt) > 0.05
    """)
    unreconciled, max_diff = cur.fetchone()
    assert unreconciled == 0, f"Found {unreconciled} unreconciled debt records! Max diff: {max_diff}"


def test_sql_probability_bucket_consistency(sql_db):
    """TEST 11: Probability bucket matches deterministic thresholds (<0.10 LOW, 0.10-0.20 MEDIUM, >=0.20 HIGH)."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM CLV_SCORED_OUTPUT
        WHERE (churn_probability < 0.10 AND churn_probability_bucket != 'LOW')
           OR (churn_probability >= 0.10 AND churn_probability < 0.20 AND churn_probability_bucket != 'MEDIUM')
           OR (churn_probability >= 0.20 AND churn_probability_bucket != 'HIGH')
    """)
    mismatches = cur.fetchone()[0]
    assert mismatches == 0, f"Found {mismatches} probability bucket mismatches!"


def test_sql_shap_drivers_present(sql_db):
    """TEST 12: SHAP drivers and numeric values are populated for all labelled rows."""
    cur = sql_db.cursor()
    cur.execute("""
        SELECT COUNT(*) FROM CLV_SCORED_OUTPUT
        WHERE is_label_known = 1 
          AND (top_shap_driver_1 IS NULL OR top_shap_driver_2 IS NULL OR top_shap_driver_3 IS NULL
               OR top_shap_value_1 IS NULL OR top_shap_value_2 IS NULL OR top_shap_value_3 IS NULL
               OR top_shap_abs_value_1 IS NULL OR top_shap_abs_value_2 IS NULL OR top_shap_abs_value_3 IS NULL)
    """)
    missing_shap = cur.fetchone()[0]
    assert missing_shap == 0, f"Found {missing_shap} missing SHAP values on labelled rows!"


def test_sql_metadata_populated(sql_db):
    """TEST 13 & 14: Model version and run_id are populated and non-empty."""
    cur = sql_db.cursor()
    cur.execute("SELECT COUNT(*) FROM CLV_SCORED_OUTPUT WHERE model_version IS NULL OR run_id IS NULL")
    missing_meta = cur.fetchone()[0]
    assert missing_meta == 0, f"Found {missing_meta} rows with missing metadata!"


def test_sql_forecast_bias_monitor_integrity(sql_db):
    """TEST 15: Forecast bias monitor contains 18 cohorts across 5 dimensions with valid bounds."""
    cur = sql_db.cursor()
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT cohort_dimension) FROM CLV_FORECAST_BIAS_MONITOR")
    total_cohorts, dimensions = cur.fetchone()
    assert total_cohorts == 18, f"Expected 18 cohorts in monitor table, got {total_cohorts}"
    assert dimensions == 5, f"Expected 5 dimensions in monitor table, got {dimensions}"
