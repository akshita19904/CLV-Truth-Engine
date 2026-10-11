# Snowflake Snowsight Manual Execution & Screenshot Checklist
**Project:** CLV Truth Engine — Phase 4 Production Warehouse Integration  
**Case Study:** Case Study 2 — Customer Lifetime Value (CLV) Predictive Analytics  
**Account:** `BCNJRTY-LR91595` (`https://bcnjrty-lr91595.snowflakecomputing.com`)  
**User:** `AKSHITA190904`  
**Role:** `ACCOUNTADMIN`  
**Warehouse:** `COMPUTE_WH`  
**Target:** `CLV_DB.CLV`  
**Master SQL Script:** [`sql/08_manual_snowsight_execution.sql`](file:///C:/Users/anura/.gemini/antigravity/scratch/CLV-Truth-Engine/sql/08_manual_snowsight_execution.sql)  

---

## 1. Overview & Execution Workflow

This checklist guides the execution and evidence collection for Phase 4 directly inside Snowflake's web UI (**Snowsight**).

### Execution Sequence Summary:
1. Open Snowsight (`https://bcnjrty-lr91595.snowflakecomputing.com`).
2. Log in with user `AKSHITA190904`.
3. Open a new SQL Worksheet and paste the contents of `sql/08_manual_snowsight_execution.sql`.
4. Execute **Section 1, 2, 3, and 4** to create the database, schema, tables, internal stage, and file format.
5. Upload the two canonical CSV files to the stage or staging tables.
6. Execute **Section 5** to verify staging row counts (2,000 and 18).
7. Execute **Section 6** to create stored procedures.
8. Execute **Section 7** to call the stored procedures.
9. Execute **Section 8** to run the 15 live validation queries.
10. Execute **Section 9** to test idempotency (re-running the procedures).

---

## 2. Required Screenshot Checklist

Capture the following 8 screenshots to provide verifiable evidence of live deployment:

---

### Screenshot 1: Active Context & Warehouse
- **Where to navigate:** Snowsight top bar / worksheet header.
- **What must be visible:**
  - Role: `ACCOUNTADMIN`
  - Warehouse: `COMPUTE_WH` (Status: Active / Running)
  - Database: `CLV_DB`
  - Schema: `CLV`
- **Verification Query (Section 1):**
```sql
SELECT CURRENT_ACCOUNT(), CURRENT_USER(), CURRENT_ROLE(), CURRENT_WAREHOUSE(), CURRENT_DATABASE(), CURRENT_SCHEMA();
```
- **Screenshot Name:** `01_snowsight_context_warehouse.png`

---

### Screenshot 2: Database Object Explorer
- **Where to navigate:** Snowsight left navigation pane: **Data** -> **Databases** -> `CLV_DB` -> `CLV`.
- **What must be visible:**
  - Database: `CLV_DB`
  - Schema: `CLV`
  - Tables:
    - `CLV_SCORED_OUTPUT`
    - `CLV_FORECAST_BIAS_MONITOR`
    - `STG_CLV_SCORES`
    - `STG_FORECAST_BIAS`
  - Stages:
    - `STG_CLV_DATA_STAGE`
  - Procedures:
    - `SP_LOAD_CLV_SCORES(VARCHAR)`
    - `SP_REFRESH_BIAS_MONITOR(VARCHAR)`
- **Screenshot Name:** `02_snowsight_object_explorer.png`

---

### Screenshot 3: `CLV_SCORED_OUTPUT` Production Row Count & Preview
- **Verification Query (Section 5 / 8):**
```sql
SELECT COUNT(*) AS total_rows, COUNT(DISTINCT array_id) AS unique_arrays, COUNT(DISTINCT account_id) AS unique_accounts
FROM CLV.CLV_SCORED_OUTPUT;
```
- **What must be visible in results grid:**
  - `TOTAL_ROWS`: **`2000`**
  - `UNIQUE_ARRAYS`: **`2000`**
  - `UNIQUE_ACCOUNTS`: **`1000`**
- **Screenshot Name:** `03_clv_scored_output_row_count.png`

---

### Screenshot 4: `CLV_FORECAST_BIAS_MONITOR` Row Count & Bias Table
- **Verification Query (Section 5 / 8):**
```sql
SELECT cohort_dimension, cohort_name, cohort_size, observed_churn_rate, baseline_churn_rate, model_bias, baseline_bias, statistically_significant
FROM CLV.CLV_FORECAST_BIAS_MONITOR
ORDER BY cohort_dimension, cohort_name;
```
- **What must be visible in results grid:**
  - Row count: **`18`** cohorts
  - Empirical churn vs 5% baseline gaps clearly displayed
  - `STATISTICALLY_SIGNIFICANT`: `TRUE`
- **Screenshot Name:** `04_clv_forecast_bias_monitor_rows.png`

---

### Screenshot 5: `SP_LOAD_CLV_SCORES` Stored Procedure Execution Result
- **Execution Query (Section 7):**
```sql
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');
```
- **What must be visible in results grid:**
  - JSON response containing:
```json
{
  "status": "SUCCESS",
  "run_id": "run_20261011_phase4",
  "staging_records": 2000,
  "production_records": 2000,
  "message": "Idempotent load into CLV.CLV_SCORED_OUTPUT completed successfully."
}
```
- **Screenshot Name:** `05_sp_load_clv_scores_call.png`

---

### Screenshot 6: `SP_REFRESH_BIAS_MONITOR` Stored Procedure Execution Result
- **Execution Query (Section 7):**
```sql
CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4');
```
- **What must be visible in results grid:**
  - JSON response containing:
```json
{
  "status": "SUCCESS",
  "run_id": "run_20261011_phase4",
  "staging_records": 18,
  "production_records": 18,
  "message": "Idempotent refresh of CLV.CLV_FORECAST_BIAS_MONITOR completed successfully."
}
```
- **Screenshot Name:** `06_sp_refresh_bias_monitor_call.png`

---

### Screenshot 7: Master Validation Query Results (15 Integrity Checks)
- **Execution Query (Section 8 - Master Unified Query):**
```sql
-- Run the Master Unified Validation Query in Section 8 of 08_manual_snowsight_execution.sql
```
- **What must be visible in results grid:**
  - All 15 checks displaying `STATUS = 'PASS'`:
    - `01. Total Rows` -> `2000` -> `PASS`
    - `02. Unique Arrays` -> `2000` -> `PASS`
    - `03. Unique Accounts` -> `1000` -> `PASS`
    - `04. Arrays per Account Violations` -> `0` -> `PASS`
    - `05. Duplicate Logical Keys` -> `0` -> `PASS`
    - `06. Probability Out-of-Bounds` -> `0` -> `PASS`
    - `07. Null Churn Labels (Holdouts)` -> `116` -> `PASS`
    - `08. Negative Value Exposed` -> `0` -> `PASS`
    - `09. Loss Arrays Positive Value Exposed` -> `0` -> `PASS`
    - `10. Forecast Debt Unreconciled` -> `0` -> `PASS`
    - `11. Probability Bucket Mismatches` -> `0` -> `PASS`
    - `12. Missing SHAP on Labelled Rows` -> `0` -> `PASS`
    - `13. Missing Model Version` -> `0` -> `PASS`
    - `14. Missing Run ID` -> `0` -> `PASS`
    - `15. Cohort Monitor Rows` -> `18` -> `PASS`
- **Screenshot Name:** `07_master_validation_results_pass.png`

---

### Screenshot 8: Idempotency Re-Run Verification
- **Execution Queries (Section 9):**
```sql
-- Re-execute both procedures
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');
CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4');

-- Check post-rerun counts
SELECT 'IDEMPOTENCY_TEST_SCORES' AS check_name, '2000' AS expected_row_count, CAST(COUNT(*) AS VARCHAR) AS actual_row_count, CASE WHEN COUNT(*) = 2000 THEN 'PASS' ELSE 'FAIL' END AS status FROM CLV.CLV_SCORED_OUTPUT WHERE run_id = 'run_20261011_phase4'
UNION ALL
SELECT 'IDEMPOTENCY_TEST_BIAS', '18', CAST(COUNT(*) AS VARCHAR), CASE WHEN COUNT(*) = 18 THEN 'PASS' ELSE 'FAIL' END FROM CLV.CLV_FORECAST_BIAS_MONITOR WHERE run_id = 'run_20261011_phase4'
UNION ALL
SELECT 'IDEMPOTENCY_DUPLICATE_KEYS', '0', CAST(COUNT(*) AS VARCHAR), CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END FROM (SELECT array_id, model_version, run_id FROM CLV.CLV_SCORED_OUTPUT WHERE run_id = 'run_20261011_phase4' GROUP BY array_id, model_version, run_id HAVING COUNT(*) > 1);
```
- **What must be visible in results grid:**
  - Row counts remain exactly 2,000 and 18, with 0 duplicates and all `STATUS = 'PASS'`.
- **Screenshot Name:** `08_idempotency_rerun_verification.png`

---

## 3. Storage Location for Captured Screenshots
Save the captured image files in the project repository at:
`reports/screenshots/`

Once captured, they will serve as final auditable submission proof of manual Snowsight live execution.
