# Snowflake Snowsight UI Evidence & Screenshot Checklist
**Project:** CLV Truth Engine — Phase 4 Production Warehouse Integration  
**Account:** `BCNJRTY-LR91595` (`https://bcnjrty-lr91595.snowflakecomputing.com`)  
**Target Database / Schema:** `CLV_DB.CLV`  
**Warehouse:** `COMPUTE_WH`  
**Role:** `ACCOUNTADMIN`  

---

## 1. Prerequisites in Snowsight UI
1. Open your browser and navigate to: `https://bcnjrty-lr91595.snowflakecomputing.com`
2. Log in with your standard credentials (`AKSHITA190904`).
3. Set your active role to `ACCOUNTADMIN` and warehouse to `COMPUTE_WH`.
4. Open a new SQL Worksheet.

---

## 2. Screenshot Checklist

### Screenshot 1: Object Creation & Stored Procedures
- **Goal:** Prove database, schema, tables, internal stage, and procedures exist.
- **SQL to execute:**
```sql
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;
USE DATABASE CLV_DB;
USE SCHEMA CLV;

SHOW TABLES IN SCHEMA CLV_DB.CLV;
SHOW PROCEDURES IN SCHEMA CLV_DB.CLV;
SHOW STAGES IN SCHEMA CLV_DB.CLV;
```
- **What to capture:**
  - Snowsight left navigation pane showing `CLV_DB` -> `CLV` -> Tables (`CLV_SCORED_OUTPUT`, `CLV_FORECAST_BIAS_MONITOR`, `STG_CLV_SCORES`, `STG_FORECAST_BIAS`).
  - Output pane showing stored procedures `SP_LOAD_CLV_SCORES` and `SP_REFRESH_BIAS_MONITOR`.

---

### Screenshot 2: Stage & Data Ingestion
- **Goal:** Prove staging data is loaded into internal stage and staged tables.
- **SQL to execute:**
```sql
LIST @CLV.STG_CLV_DATA_STAGE;

SELECT COUNT(*) AS stg_scores_count FROM CLV.STG_CLV_SCORES;
SELECT COUNT(*) AS stg_bias_count FROM CLV.STG_FORECAST_BIAS;
```
- **What to capture:**
  - Results showing `stg_scores_count = 2000` and `stg_bias_count = 18`.

---

### Screenshot 3: Stored Procedure Execution
- **Goal:** Prove production stored procedures execute and return structured JSON success messages.
- **SQL to execute:**
```sql
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');
CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4');
```
- **What to capture:**
  - JSON result string containing:
    `{"status": "SUCCESS", "run_id": "run_20261011_phase4", "staging_records": 2000, "production_records": 2000, ...}`.

---

### Screenshot 4: Target Isolation & Unknown Label Handling
- **Goal:** Prove unlabelled holdout rows (116) strictly maintain `NULL` churn labels.
- **SQL to execute:**
```sql
SELECT 
    COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) AS null_target_count,
    COUNT(CASE WHEN is_label_known = 0 THEN 1 END) AS holdout_rows_count,
    COUNT(CASE WHEN is_label_known = 1 THEN 1 END) AS labelled_rows_count,
    CASE 
        WHEN COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) = 116 
             AND COUNT(CASE WHEN is_label_known = 0 THEN 1 END) = 116
        THEN 'PASSED' 
        ELSE 'FAILED' 
    END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;
```
- **What to capture:**
  - Result grid showing `116` null targets, `116` holdout rows, `1884` labelled rows, and `PASSED`.

---

### Screenshot 5: Financial Reconciliation & Value Exposure
- **Goal:** Prove full economic reconciliation of Forecast Debt and Value Exposed.
- **SQL to execute:**
```sql
SELECT 
    COUNT(*) AS total_arrays,
    ROUND(SUM(stated_profit), 2) AS total_stated_profit,
    ROUND(SUM(retention_adjusted_clv), 2) AS total_retention_clv,
    ROUND(SUM(forecast_debt), 2) AS total_forecast_debt,
    ROUND(SUM(value_exposed), 2) AS total_value_exposed,
    ROUND(SUM(forecast_debt) / SUM(stated_profit) * 100, 2) AS forecast_debt_pct
FROM CLV.CLV_SCORED_OUTPUT;
```
- **Expected Results to capture:**
  - `total_arrays`: `2000`
  - `total_stated_profit`: `$4,305,689,564.00`
  - `total_retention_clv`: `$2,348,335,632.31`
  - `total_forecast_debt`: `$1,957,353,931.69`
  - `total_value_exposed`: `$681,507,251.73`
  - `forecast_debt_pct`: `45.46%`

---

### Screenshot 6: TreeSHAP Explainability & Risk Bucketing
- **Goal:** Prove array-level SHAP drivers, numeric contributions, and risk buckets are visible.
- **SQL to execute:**
```sql
SELECT 
    array_id,
    account_id,
    churn_probability,
    churn_probability_bucket,
    top_shap_driver_1,
    top_shap_value_1,
    top_shap_driver_2,
    top_shap_value_2,
    top_shap_driver_3,
    top_shap_value_3,
    forecast_debt
FROM CLV.CLV_SCORED_OUTPUT
ORDER BY forecast_debt DESC
LIMIT 10;
```
- **What to capture:**
  - Top 10 arrays with highest Forecast Debt, showing feature names and actual SHAP values.

---

### Screenshot 7: Cohort Forecast Bias Monitoring (Task 2)
- **Goal:** Prove cohort bias table tracks observed vs baseline churn and Holm-Bonferroni significance.
- **SQL to execute:**
```sql
SELECT 
    cohort_dimension,
    cohort_name,
    cohort_size,
    ROUND(observed_churn_rate * 100, 2) AS observed_churn_pct,
    ROUND(baseline_churn_rate * 100, 2) AS baseline_churn_pct,
    ROUND(xgb_mean_probability * 100, 2) AS model_prob_pct,
    ROUND(baseline_bias * 100, 2) AS static_baseline_bias_pct,
    ROUND(confidence_interval_lower * 100, 2) AS ci_lower_pct,
    ROUND(confidence_interval_upper * 100, 2) AS ci_upper_pct,
    statistically_significant
FROM CLV.CLV_FORECAST_BIAS_MONITOR
ORDER BY cohort_dimension, cohort_name;
```
- **What to capture:**
  - Cohort table showing the massive optimism bias of the 5% baseline (e.g. baseline bias -10.82% overall) and all statistically significant cohorts.

---

### Screenshot 8: Idempotency Verification
- **Goal:** Prove re-calling the stored procedure updates records in place without duplicating rows.
- **SQL to execute:**
```sql
-- Call procedure second time
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');

-- Check row count remains exactly 2,000
SELECT COUNT(*) AS total_rows, COUNT(DISTINCT array_id) AS unique_arrays
FROM CLV.CLV_SCORED_OUTPUT
WHERE run_id = 'run_20261011_phase4';
```
- **What to capture:**
  - Execution summary showing `total_rows = 2000` and `unique_arrays = 2000` after re-execution.
