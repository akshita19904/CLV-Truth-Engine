-- ==============================================================================
-- 07_validation_queries.sql
-- Project: CLV Truth Engine
-- Description: Production Quality Verification Suite for Snowflake tables:
--   Covers all 14 mandatory validation requirements.
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

-- ------------------------------------------------------------------------------
-- 1. TOTAL PRODUCTION ROW COUNT (Expect: 2,000)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_rows,
    CASE WHEN COUNT(*) = 2000 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- ------------------------------------------------------------------------------
-- 2. UNIQUE ARRAY IDENTIFIERS (Expect: 2,000 unique array_id)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(DISTINCT array_id) AS unique_arrays,
    CASE WHEN COUNT(DISTINCT array_id) = 2000 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- ------------------------------------------------------------------------------
-- 3. UNIQUE ACCOUNT IDENTIFIERS (Expect: 1,000 unique account_id)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(DISTINCT account_id) AS unique_accounts,
    CASE WHEN COUNT(DISTINCT account_id) = 1000 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- ------------------------------------------------------------------------------
-- 4. ARRAYS PER ACCOUNT HIERARCHY (Expect: Exactly 2 arrays for 100% of accounts)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS accounts_with_violations,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM (
    SELECT account_id, COUNT(array_id) AS array_count
    FROM CLV.CLV_SCORED_OUTPUT
    GROUP BY account_id
    HAVING COUNT(array_id) != 2
);

-- ------------------------------------------------------------------------------
-- 5. PRIMARY KEY UNIQUENESS (Expect: 0 duplicate keys)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS duplicate_keys_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM (
    SELECT array_id, model_version, run_id, COUNT(*) AS cnt
    FROM CLV.CLV_SCORED_OUTPUT
    GROUP BY array_id, model_version, run_id
    HAVING COUNT(*) > 1
);

-- ------------------------------------------------------------------------------
-- 6. CHURN PROBABILITY BOUNDS (Expect: 0 rows with p < 0 or p > 1)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS out_of_bounds_count,
    MIN(churn_probability) AS min_prob,
    MAX(churn_probability) AS max_prob,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE churn_probability < 0.0 OR churn_probability > 1.0;

-- ------------------------------------------------------------------------------
-- 7. TARGET LABEL ISOLATION & NULL PRESERVATION (Expect: Exactly 116 NULLs, 1884 non-NULL)
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 8. VALUE EXPOSED NON-NEGATIVE (Expect: 0 negative value_exposed)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS negative_value_exposed_count,
    MIN(value_exposed) AS min_value_exposed,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE value_exposed < 0.0;

-- ------------------------------------------------------------------------------
-- 9. ZERO VALUE EXPOSED FOR NEGATIVE PROFIT ARRAYS (Expect: Exactly 0 violations)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS invalid_loss_exposure_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE stated_profit < 0.0 AND value_exposed != 0.0;

-- ------------------------------------------------------------------------------
-- 10. FORECAST DEBT RECONCILIATION: forecast_debt = stated_profit - retention_adjusted_clv
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS unreconciled_debt_count,
    MAX(ABS(stated_profit - retention_adjusted_clv - forecast_debt)) AS max_discrepancy,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE ABS(stated_profit - retention_adjusted_clv - forecast_debt) > 0.05;

-- ------------------------------------------------------------------------------
-- 11. PROBABILITY BUCKET CONSISTENCY CHECK
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS bucket_mismatch_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE (churn_probability < 0.10 AND churn_probability_bucket != 'LOW')
   OR (churn_probability >= 0.10 AND churn_probability < 0.20 AND churn_probability_bucket != 'MEDIUM')
   OR (churn_probability >= 0.20 AND churn_probability_bucket != 'HIGH');

-- ------------------------------------------------------------------------------
-- 12. SHAP DRIVER POPULATION (Expect: 0 null drivers on labelled rows)
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS missing_shap_drivers_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE is_label_known = 1 
  AND (top_shap_driver_1 IS NULL OR top_shap_driver_2 IS NULL OR top_shap_driver_3 IS NULL);

-- ------------------------------------------------------------------------------
-- 13. METADATA INTEGRITY: MODEL_VERSION POPULATED
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS missing_model_version_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE model_version IS NULL OR TRIM(model_version) = '';

-- ------------------------------------------------------------------------------
-- 14. METADATA INTEGRITY: RUN_ID POPULATED
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS missing_run_id_count,
    CASE WHEN COUNT(*) = 0 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE run_id IS NULL OR TRIM(run_id) = '';

-- ------------------------------------------------------------------------------
-- 15. COHORT BIAS MONITOR INTEGRITY
-- ------------------------------------------------------------------------------
SELECT 
    COUNT(*) AS total_cohorts_monitored,
    COUNT(DISTINCT cohort_dimension) AS monitored_dimensions,
    COUNT(CASE WHEN statistically_significant THEN 1 END) AS statistically_significant_cohorts,
    CASE WHEN COUNT(*) = 18 THEN 'PASSED' ELSE 'FAILED' END AS test_status
FROM CLV.CLV_FORECAST_BIAS_MONITOR;
