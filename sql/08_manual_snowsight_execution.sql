-- ==============================================================================
-- 08_manual_snowsight_execution.sql
-- Project: CLV Truth Engine
-- Title: Behavior-Driven Retention-Adjusted CLV and Forecast Bias Engine
-- Phase: Phase 4 — Production Warehouse Integration (Snowflake Snowsight)
--
-- Account:   BCNJRTY-LR91595 (https://bcnjrty-lr91595.snowflakecomputing.com)
-- User:      AKSHITA190904
-- Role:      ACCOUNTADMIN
-- Warehouse: COMPUTE_WH
-- Database:  CLV_DB
-- Schema:    CLV
-- Run ID:    run_20261011_phase4
--
-- Instructions:
-- Paste this script into a Snowflake Snowsight SQL Worksheet.
-- You can execute sequentially by section or run all statements.
-- Follow the manual CSV upload steps in Section 4 before running Section 5.
-- ==============================================================================


-- ==============================================================================
-- SECTION 1: CONTEXT, DATABASE & SCHEMA SETUP
-- ==============================================================================

USE ROLE ACCOUNTADMIN;
USE WAREHOUSE COMPUTE_WH;

-- Create dedicated production database
CREATE DATABASE IF NOT EXISTS CLV_DB
    COMMENT = 'Everpure CLV Truth Engine Analytics & Scoring Database';

USE DATABASE CLV_DB;

-- Create dedicated production schema
CREATE SCHEMA IF NOT EXISTS CLV
    COMMENT = 'Customer Lifetime Value, Churn Scoring & Forecast Bias Production Schema';

USE SCHEMA CLV;

-- Verify session context
SELECT 
    CURRENT_ACCOUNT()   AS current_account,
    CURRENT_USER()      AS current_user,
    CURRENT_ROLE()      AS current_role,
    CURRENT_WAREHOUSE() AS current_warehouse,
    CURRENT_DATABASE()  AS current_database,
    CURRENT_SCHEMA()    AS current_schema,
    CURRENT_TIMESTAMP() AS session_time;


-- ==============================================================================
-- SECTION 2: CREATE PRODUCTION TABLES
-- ==============================================================================

-- 1. Production Scored Output (Array-Level Scoring & Financial Revaluation)
CREATE TABLE IF NOT EXISTS CLV.CLV_SCORED_OUTPUT (
    -- Run & Model Metadata
    run_id                          VARCHAR(64)       NOT NULL,
    model_version                   VARCHAR(64)       NOT NULL,
    scored_at                       TIMESTAMP_NTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    
    -- Entity Identifiers (Grain: One row per storage array)
    array_id                        VARCHAR(64)       NOT NULL,
    account_id                      VARCHAR(64)       NOT NULL,
    
    -- Model Predictions & Operational Risk Buckets
    churn_probability               NUMBER(8, 5)      NOT NULL,
    xgb_churn_probability           NUMBER(8, 5)      NOT NULL,
    churn_probability_bucket        VARCHAR(16)       NOT NULL,  -- 'LOW', 'MEDIUM', 'HIGH'
    random_forest_probability       NUMBER(8, 5),                -- Comparison benchmark probability
    
    -- Capacity Expansion Signal & Opportunity Score
    expansion_signal                VARCHAR(16),                 -- 'Low', 'Medium', 'High', 'UNKNOWN'
    expansion_opportunity_score     NUMBER(6, 4),                -- Heuristic operational score [0, 1]
    
    -- Hardware & Operational Metrics
    utilization_rate                NUMBER(8, 4)      NOT NULL,
    over_capacity                   NUMBER(1, 0)      NOT NULL,  -- 1 if utilization > 1.0, else 0
    capacity_pressure               NUMBER(12, 2)     NOT NULL,  -- utilization_rate * current_capacity_tb
    current_capacity_tb             NUMBER(12, 2)     NOT NULL,
    
    -- Financial Revaluation Layer (USD)
    stated_profit                   NUMBER(18, 2)     NOT NULL,  -- Baseline economic value (15-yr profit)
    historical_15yr_net_profit_usd  NUMBER(18, 2)     NOT NULL,  -- Preserved audit field
    retention_adjusted_clv          NUMBER(18, 2)     NOT NULL,  -- stated_profit * RF(p_xgb) / RF(0.05)
    forecast_debt                   NUMBER(18, 2)     NOT NULL,  -- stated_profit - retention_adjusted_clv
    value_exposed                   NUMBER(18, 2)     NOT NULL,  -- p_xgb * max(stated_profit, 0)
    
    -- TreeSHAP Explainability Drivers (Feature attribution names and actual numeric/abs values)
    top_shap_driver_1               VARCHAR(64),
    top_shap_value_1                NUMBER(12, 6),
    top_shap_abs_value_1            NUMBER(12, 6),
    top_shap_driver_2               VARCHAR(64),
    top_shap_value_2                NUMBER(12, 6),
    top_shap_abs_value_2            NUMBER(12, 6),
    top_shap_driver_3               VARCHAR(64),
    top_shap_value_3                NUMBER(12, 6),
    top_shap_abs_value_3            NUMBER(12, 6),
    
    -- Evaluation Isolation Flags
    is_label_known                  NUMBER(1, 0)      NOT NULL,  -- 1 if labelled (1,884), 0 if holdout (116)
    churn_event_next_12m            NUMBER(1, 0),                -- 1, 0, or NULL for unknown holdouts!
    churn_label                     NUMBER(1, 0),                -- Alias field preserving NULL
    
    -- Integrity Constraints
    CONSTRAINT pk_clv_scored_output PRIMARY KEY (array_id, model_version, run_id)
)
COMMENT = 'Production scored storage arrays containing calibrated probabilities, Forecast Debt, and TreeSHAP drivers.';

-- 2. Production Forecast Bias Monitor (Cohort-Level Bias Tracking & Statistics)
CREATE TABLE IF NOT EXISTS CLV.CLV_FORECAST_BIAS_MONITOR (
    -- Monitoring Run Metadata
    run_id                          VARCHAR(64)       NOT NULL,
    model_version                   VARCHAR(64)       NOT NULL,
    monitoring_timestamp            TIMESTAMP_NTZ     NOT NULL DEFAULT CURRENT_TIMESTAMP(),
    
    -- Cohort Dimensions
    cohort_dimension                VARCHAR(32)       NOT NULL,  -- 'Overall', 'Utilization', 'Expansion Signal', 'Capacity', 'Historical Profit'
    cohort_name                     VARCHAR(64)       NOT NULL,  -- e.g. 'Q1 [0.35, 0.66)', 'Medium'
    cohort_size                     NUMBER(10, 0)     NOT NULL,
    
    -- Rates & Predictions
    observed_churn_rate             NUMBER(8, 4)      NOT NULL,
    baseline_churn_rate             NUMBER(8, 4)      NOT NULL DEFAULT 0.05,
    xgb_mean_probability            NUMBER(8, 4)      NOT NULL,
    rf_mean_probability             NUMBER(8, 4)      NOT NULL,
    
    -- Quantified Gaps (Bias)
    model_bias                      NUMBER(8, 4)      NOT NULL,  -- mean_p_xgb - observed_churn_rate
    baseline_bias                   NUMBER(8, 4)      NOT NULL,  -- 0.05 - observed_churn_rate
    
    -- Statistical Rigor & Uncertainty Bounds
    confidence_interval_lower       NUMBER(8, 4)      NOT NULL,  -- Wilson 95% CI lower bound
    confidence_interval_upper       NUMBER(8, 4)      NOT NULL,  -- Wilson 95% CI upper bound
    adjusted_p_value                NUMBER(16, 10),              -- Holm-Bonferroni corrected binomial p-value
    statistically_significant       BOOLEAN           NOT NULL,  -- TRUE if adjusted p-value < 0.05
    
    -- Integrity Constraints
    CONSTRAINT pk_clv_bias_monitor PRIMARY KEY (run_id, cohort_dimension, cohort_name)
)
COMMENT = 'Cohort-level forecast bias tracking table comparing model probabilities vs. 5% static baseline.';


-- ==============================================================================
-- SECTION 3: CREATE CONTROLLED STAGING TABLES
-- ==============================================================================

-- Staging table for scored arrays
CREATE TABLE IF NOT EXISTS CLV.STG_CLV_SCORES (
    run_id                          VARCHAR(64),
    model_version                   VARCHAR(64),
    scored_at                       TIMESTAMP_NTZ,
    array_id                        VARCHAR(64),
    account_id                      VARCHAR(64),
    churn_probability               NUMBER(8, 5),
    xgb_churn_probability           NUMBER(8, 5),
    churn_probability_bucket        VARCHAR(16),
    expansion_signal                VARCHAR(16),
    expansion_opportunity_score     NUMBER(6, 4),
    stated_profit                   NUMBER(18, 2),
    retention_adjusted_clv          NUMBER(18, 2),
    forecast_debt                   NUMBER(18, 2),
    value_exposed                   NUMBER(18, 2),
    utilization_rate                NUMBER(8, 4),
    over_capacity                   NUMBER(1, 0),
    capacity_pressure               NUMBER(12, 2),
    current_capacity_tb             NUMBER(12, 2),
    historical_15yr_net_profit_usd  NUMBER(18, 2),
    top_shap_driver_1               VARCHAR(64),
    top_shap_value_1                NUMBER(12, 6),
    top_shap_abs_value_1            NUMBER(12, 6),
    top_shap_driver_2               VARCHAR(64),
    top_shap_value_2                NUMBER(12, 6),
    top_shap_abs_value_2            NUMBER(12, 6),
    top_shap_driver_3               VARCHAR(64),
    top_shap_value_3                NUMBER(12, 6),
    top_shap_abs_value_3            NUMBER(12, 6),
    random_forest_probability       NUMBER(8, 5),
    is_label_known                  NUMBER(1, 0),
    churn_event_next_12m            NUMBER(1, 0),
    churn_label                     NUMBER(1, 0)
)
COMMENT = 'Transient staging table for bulk loading scored array CSV records.';

-- Staging table for cohort forecast bias
CREATE TABLE IF NOT EXISTS CLV.STG_FORECAST_BIAS (
    run_id                          VARCHAR(64),
    model_version                   VARCHAR(64),
    monitoring_timestamp            TIMESTAMP_NTZ,
    cohort_dimension                VARCHAR(32),
    cohort_name                     VARCHAR(64),
    cohort_size                     NUMBER(10, 0),
    observed_churn_rate             NUMBER(8, 4),
    baseline_churn_rate             NUMBER(8, 4),
    xgb_mean_probability            NUMBER(8, 4),
    rf_mean_probability             NUMBER(8, 4),
    model_bias                      NUMBER(8, 4),
    baseline_bias                   NUMBER(8, 4),
    confidence_interval_lower       NUMBER(8, 4),
    confidence_interval_upper       NUMBER(8, 4),
    adjusted_p_value                NUMBER(16, 10),
    statistically_significant       BOOLEAN
)
COMMENT = 'Transient staging table for bulk loading cohort forecast bias metrics.';


-- ==============================================================================
-- SECTION 4: FILE FORMAT, INTERNAL STAGE & CSV INGESTION
-- ==============================================================================

-- Create standard CSV ingestion file format
CREATE OR REPLACE FILE FORMAT CLV.CSV_LOAD_FORMAT
    TYPE = 'CSV'
    FIELD_DELIMITER = ','
    RECORD_DELIMITER = '\n'
    SKIP_HEADER = 1
    FIELD_OPTIONALLY_ENCLOSED_BY = '"'
    NULL_IF = ('', 'NULL', 'null', 'None')
    EMPTY_FIELD_AS_NULL = TRUE
    TRIM_SPACE = TRUE
    ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
    COMMENT = 'Standard CSV ingestion format handling null target labels and quoted text.';

-- Create internal named stage
CREATE OR REPLACE STAGE CLV.STG_CLV_DATA_STAGE
    FILE_FORMAT = CLV.CSV_LOAD_FORMAT
    COMMENT = 'Internal stage for receiving Python-generated scored outputs and bias monitoring CSVs.';

-- ------------------------------------------------------------------------------
-- MANUAL CSV UPLOAD INSTRUCTIONS VIA SNOWSIGHT UI:
-- ------------------------------------------------------------------------------
-- Follow ONE of the two methods below to populate staging:
--
-- METHOD A (RECOMMENDED: Stage Upload + COPY INTO):
-- 1. In the Snowsight left navigation pane, go to:
--    Data -> Databases -> CLV_DB -> CLV -> Stages -> STG_CLV_DATA_STAGE
-- 2. Click the "+ Files" button in the upper right.
-- 3. Upload both files:
--    - data/processed/snowflake_clv_scored_export.csv
--    - data/processed/snowflake_bias_monitor_export.csv
-- 4. After upload, run the two COPY INTO statements below.
--
-- METHOD B (Snowsight Direct Table Load):
-- 1. Go to Data -> Databases -> CLV_DB -> CLV -> Tables -> STG_CLV_SCORES -> Click "Load Data".
--    Upload snowflake_clv_scored_export.csv (Format: CSV, Skip Header: 1).
-- 2. Go to Tables -> STG_FORECAST_BIAS -> Click "Load Data".
--    Upload snowflake_bias_monitor_export.csv (Format: CSV, Skip Header: 1).
-- ------------------------------------------------------------------------------

-- Execute these COPY INTO statements if using Method A:
COPY INTO CLV.STG_CLV_SCORES (
    run_id,
    model_version,
    scored_at,
    array_id,
    account_id,
    churn_probability,
    xgb_churn_probability,
    churn_probability_bucket,
    expansion_signal,
    expansion_opportunity_score,
    stated_profit,
    retention_adjusted_clv,
    forecast_debt,
    value_exposed,
    utilization_rate,
    over_capacity,
    capacity_pressure,
    current_capacity_tb,
    historical_15yr_net_profit_usd,
    top_shap_driver_1,
    top_shap_value_1,
    top_shap_abs_value_1,
    top_shap_driver_2,
    top_shap_value_2,
    top_shap_abs_value_2,
    top_shap_driver_3,
    top_shap_value_3,
    top_shap_abs_value_3,
    random_forest_probability,
    is_label_known,
    churn_event_next_12m,
    churn_label
)
FROM @CLV.STG_CLV_DATA_STAGE/snowflake_clv_scored_export.csv
FILE_FORMAT = (FORMAT_NAME = CLV.CSV_LOAD_FORMAT)
ON_ERROR = 'ABORT_STATEMENT';

COPY INTO CLV.STG_FORECAST_BIAS (
    run_id,
    model_version,
    monitoring_timestamp,
    cohort_dimension,
    cohort_name,
    cohort_size,
    observed_churn_rate,
    baseline_churn_rate,
    xgb_mean_probability,
    rf_mean_probability,
    model_bias,
    baseline_bias,
    confidence_interval_lower,
    confidence_interval_upper,
    adjusted_p_value,
    statistically_significant
)
FROM @CLV.STG_CLV_DATA_STAGE/snowflake_bias_monitor_export.csv
FILE_FORMAT = (FORMAT_NAME = CLV.CSV_LOAD_FORMAT)
ON_ERROR = 'ABORT_STATEMENT';


-- ==============================================================================
-- SECTION 5: VERIFY STAGING ROW COUNTS
-- ==============================================================================

-- Verify STG_CLV_SCORES count (EXPECTED: exactly 2,000)
SELECT 
    'STG_CLV_SCORES' AS staging_table,
    COUNT(*)         AS actual_row_count,
    2000             AS expected_row_count,
    CASE WHEN COUNT(*) = 2000 THEN 'PASS' ELSE 'FAIL' END AS status
FROM CLV.STG_CLV_SCORES;

-- Verify STG_FORECAST_BIAS count (EXPECTED: exactly 18)
SELECT 
    'STG_FORECAST_BIAS' AS staging_table,
    COUNT(*)            AS actual_row_count,
    18                  AS expected_row_count,
    CASE WHEN COUNT(*) = 18 THEN 'PASS' ELSE 'FAIL' END AS status
FROM CLV.STG_FORECAST_BIAS;


-- ==============================================================================
-- SECTION 6: CREATE PRODUCTION STORED PROCEDURES
-- ==============================================================================

-- 1. Stored Procedure: CLV.SP_LOAD_CLV_SCORES(P_RUN_ID)
CREATE OR REPLACE PROCEDURE CLV.SP_LOAD_CLV_SCORES(P_RUN_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
    V_STAGING_COUNT INT DEFAULT 0;
    V_DUPLICATE_COUNT INT DEFAULT 0;
    V_MERGED_COUNT INT DEFAULT 0;
BEGIN
    -- 1. Check if staging contains records for the given RUN_ID
    SELECT COUNT(*)
    INTO :V_STAGING_COUNT
    FROM CLV.STG_CLV_SCORES
    WHERE run_id = :P_RUN_ID;

    IF (:V_STAGING_COUNT = 0) THEN
        RETURN OBJECT_CONSTRUCT(
            'status', 'ERROR',
            'run_id', :P_RUN_ID,
            'message', 'No records found in CLV.STG_CLV_SCORES for the specified run_id.'
        )::VARCHAR;
    END IF;

    -- 2. Validate that staging does not contain duplicate array_id for this run
    SELECT COUNT(*)
    INTO :V_DUPLICATE_COUNT
    FROM (
        SELECT array_id, COUNT(*) AS cnt
        FROM CLV.STG_CLV_SCORES
        WHERE run_id = :P_RUN_ID
        GROUP BY array_id
        HAVING COUNT(*) > 1
    );

    IF (:V_DUPLICATE_COUNT > 0) THEN
        RETURN OBJECT_CONSTRUCT(
            'status', 'ERROR',
            'run_id', :P_RUN_ID,
            'message', 'Integrity check failed: Duplicate array_id detected in staging.'
        )::VARCHAR;
    END IF;

    -- 3. Perform Idempotent MERGE into Production CLV_SCORED_OUTPUT
    MERGE INTO CLV.CLV_SCORED_OUTPUT AS T
    USING (
        SELECT
            run_id,
            model_version,
            COALESCE(scored_at, CURRENT_TIMESTAMP()) AS scored_at,
            array_id,
            account_id,
            churn_probability,
            COALESCE(xgb_churn_probability, churn_probability) AS xgb_churn_probability,
            -- Deterministic operational reporting bucket
            CASE
                WHEN churn_probability < 0.10 THEN 'LOW'
                WHEN churn_probability < 0.20 THEN 'MEDIUM'
                ELSE 'HIGH'
            END AS churn_probability_bucket,
            random_forest_probability,
            expansion_signal,
            expansion_opportunity_score,
            utilization_rate,
            over_capacity,
            capacity_pressure,
            current_capacity_tb,
            stated_profit,
            historical_15yr_net_profit_usd,
            retention_adjusted_clv,
            forecast_debt,
            value_exposed,
            top_shap_driver_1,
            top_shap_value_1,
            top_shap_abs_value_1,
            top_shap_driver_2,
            top_shap_value_2,
            top_shap_abs_value_2,
            top_shap_driver_3,
            top_shap_value_3,
            top_shap_abs_value_3,
            is_label_known,
            -- Strictly preserve NULL if unknown, never convert to zero!
            CASE 
                WHEN is_label_known = 1 THEN churn_event_next_12m 
                ELSE NULL 
            END AS churn_event_next_12m,
            CASE 
                WHEN is_label_known = 1 THEN COALESCE(churn_label, churn_event_next_12m) 
                ELSE NULL 
            END AS churn_label
        FROM CLV.STG_CLV_SCORES
        WHERE run_id = :P_RUN_ID
    ) AS S
    ON (T.array_id = S.array_id 
        AND T.model_version = S.model_version 
        AND T.run_id = S.run_id)
    WHEN MATCHED THEN
        UPDATE SET
            T.scored_at                     = S.scored_at,
            T.churn_probability             = S.churn_probability,
            T.xgb_churn_probability         = S.xgb_churn_probability,
            T.churn_probability_bucket      = S.churn_probability_bucket,
            T.random_forest_probability     = S.random_forest_probability,
            T.expansion_signal              = S.expansion_signal,
            T.expansion_opportunity_score   = S.expansion_opportunity_score,
            T.utilization_rate              = S.utilization_rate,
            T.over_capacity                 = S.over_capacity,
            T.capacity_pressure             = S.capacity_pressure,
            T.current_capacity_tb           = S.current_capacity_tb,
            T.stated_profit                 = S.stated_profit,
            T.historical_15yr_net_profit_usd= S.historical_15yr_net_profit_usd,
            T.retention_adjusted_clv        = S.retention_adjusted_clv,
            T.forecast_debt                 = S.forecast_debt,
            T.value_exposed                 = S.value_exposed,
            T.top_shap_driver_1             = S.top_shap_driver_1,
            T.top_shap_value_1              = S.top_shap_value_1,
            T.top_shap_abs_value_1          = S.top_shap_abs_value_1,
            T.top_shap_driver_2             = S.top_shap_driver_2,
            T.top_shap_value_2              = S.top_shap_value_2,
            T.top_shap_abs_value_2          = S.top_shap_abs_value_2,
            T.top_shap_driver_3             = S.top_shap_driver_3,
            T.top_shap_value_3              = S.top_shap_value_3,
            T.top_shap_abs_value_3          = S.top_shap_abs_value_3,
            T.is_label_known                = S.is_label_known,
            T.churn_event_next_12m          = S.churn_event_next_12m,
            T.churn_label                   = S.churn_label
    WHEN NOT MATCHED THEN
        INSERT (
            run_id,
            model_version,
            scored_at,
            array_id,
            account_id,
            churn_probability,
            xgb_churn_probability,
            churn_probability_bucket,
            random_forest_probability,
            expansion_signal,
            expansion_opportunity_score,
            utilization_rate,
            over_capacity,
            capacity_pressure,
            current_capacity_tb,
            stated_profit,
            historical_15yr_net_profit_usd,
            retention_adjusted_clv,
            forecast_debt,
            value_exposed,
            top_shap_driver_1,
            top_shap_value_1,
            top_shap_abs_value_1,
            top_shap_driver_2,
            top_shap_value_2,
            top_shap_abs_value_2,
            top_shap_driver_3,
            top_shap_value_3,
            top_shap_abs_value_3,
            is_label_known,
            churn_event_next_12m,
            churn_label
        )
        VALUES (
            S.run_id,
            S.model_version,
            S.scored_at,
            S.array_id,
            S.account_id,
            S.churn_probability,
            S.xgb_churn_probability,
            S.churn_probability_bucket,
            S.random_forest_probability,
            S.expansion_signal,
            S.expansion_opportunity_score,
            S.utilization_rate,
            S.over_capacity,
            S.capacity_pressure,
            S.current_capacity_tb,
            S.stated_profit,
            S.historical_15yr_net_profit_usd,
            S.retention_adjusted_clv,
            S.forecast_debt,
            S.value_exposed,
            S.top_shap_driver_1,
            S.top_shap_value_1,
            S.top_shap_abs_value_1,
            S.top_shap_driver_2,
            S.top_shap_value_2,
            S.top_shap_abs_value_2,
            S.top_shap_driver_3,
            S.top_shap_value_3,
            S.top_shap_abs_value_3,
            S.is_label_known,
            S.churn_event_next_12m,
            S.churn_label
        );

    -- 4. Verify post-merge production count for this run
    SELECT COUNT(*)
    INTO :V_MERGED_COUNT
    FROM CLV.CLV_SCORED_OUTPUT
    WHERE run_id = :P_RUN_ID;

    RETURN OBJECT_CONSTRUCT(
        'status', 'SUCCESS',
        'run_id', :P_RUN_ID,
        'staging_records', :V_STAGING_COUNT,
        'production_records', :V_MERGED_COUNT,
        'message', 'Idempotent load into CLV.CLV_SCORED_OUTPUT completed successfully.'
    )::VARCHAR;
END;
$$;

-- 2. Stored Procedure: CLV.SP_REFRESH_BIAS_MONITOR(P_RUN_ID)
CREATE OR REPLACE PROCEDURE CLV.SP_REFRESH_BIAS_MONITOR(P_RUN_ID VARCHAR)
RETURNS VARCHAR
LANGUAGE SQL
EXECUTE AS CALLER
AS
$$
DECLARE
    V_STAGING_COUNT INT DEFAULT 0;
    V_MERGED_COUNT INT DEFAULT 0;
BEGIN
    -- 1. Check if staging contains records for the given RUN_ID
    SELECT COUNT(*)
    INTO :V_STAGING_COUNT
    FROM CLV.STG_FORECAST_BIAS
    WHERE run_id = :P_RUN_ID;

    IF (:V_STAGING_COUNT = 0) THEN
        RETURN OBJECT_CONSTRUCT(
            'status', 'ERROR',
            'run_id', :P_RUN_ID,
            'message', 'No cohort records found in CLV.STG_FORECAST_BIAS for specified run_id.'
        )::VARCHAR;
    END IF;

    -- 2. Idempotent MERGE into CLV.CLV_FORECAST_BIAS_MONITOR
    MERGE INTO CLV.CLV_FORECAST_BIAS_MONITOR AS T
    USING (
        SELECT
            run_id,
            model_version,
            COALESCE(monitoring_timestamp, CURRENT_TIMESTAMP()) AS monitoring_timestamp,
            cohort_dimension,
            cohort_name,
            cohort_size,
            observed_churn_rate,
            baseline_churn_rate,
            xgb_mean_probability,
            rf_mean_probability,
            model_bias,
            baseline_bias,
            confidence_interval_lower,
            confidence_interval_upper,
            adjusted_p_value,
            statistically_significant
        FROM CLV.STG_FORECAST_BIAS
        WHERE run_id = :P_RUN_ID
    ) AS S
    ON (T.run_id = S.run_id 
        AND T.cohort_dimension = S.cohort_dimension 
        AND T.cohort_name = S.cohort_name)
    WHEN MATCHED THEN
        UPDATE SET
            T.monitoring_timestamp      = S.monitoring_timestamp,
            T.cohort_size               = S.cohort_size,
            T.observed_churn_rate       = S.observed_churn_rate,
            T.baseline_churn_rate       = S.baseline_churn_rate,
            T.xgb_mean_probability      = S.xgb_mean_probability,
            T.rf_mean_probability       = S.rf_mean_probability,
            T.model_bias                = S.model_bias,
            T.baseline_bias             = S.baseline_bias,
            T.confidence_interval_lower = S.confidence_interval_lower,
            T.confidence_interval_upper = S.confidence_interval_upper,
            T.adjusted_p_value          = S.adjusted_p_value,
            T.statistically_significant = S.statistically_significant
    WHEN NOT MATCHED THEN
        INSERT (
            run_id,
            model_version,
            monitoring_timestamp,
            cohort_dimension,
            cohort_name,
            cohort_size,
            observed_churn_rate,
            baseline_churn_rate,
            xgb_mean_probability,
            rf_mean_probability,
            model_bias,
            baseline_bias,
            confidence_interval_lower,
            confidence_interval_upper,
            adjusted_p_value,
            statistically_significant
        )
        VALUES (
            S.run_id,
            S.model_version,
            S.monitoring_timestamp,
            S.cohort_dimension,
            S.cohort_name,
            S.cohort_size,
            S.observed_churn_rate,
            S.baseline_churn_rate,
            S.xgb_mean_probability,
            S.rf_mean_probability,
            S.model_bias,
            S.baseline_bias,
            S.confidence_interval_lower,
            S.confidence_interval_upper,
            S.adjusted_p_value,
            S.statistically_significant
        );

    -- 3. Verify post-merge production count for this run
    SELECT COUNT(*)
    INTO :V_MERGED_COUNT
    FROM CLV.CLV_FORECAST_BIAS_MONITOR
    WHERE run_id = :P_RUN_ID;

    RETURN OBJECT_CONSTRUCT(
        'status', 'SUCCESS',
        'run_id', :P_RUN_ID,
        'staging_records', :V_STAGING_COUNT,
        'production_records', :V_MERGED_COUNT,
        'message', 'Idempotent refresh of CLV.CLV_FORECAST_BIAS_MONITOR completed successfully.'
    )::VARCHAR;
END;
$$;


-- ==============================================================================
-- SECTION 7: EXECUTE STORED PROCEDURES (INITIAL LOAD)
-- ==============================================================================

-- Execute procedure for array-level scored output
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');

-- Execute procedure for cohort forecast bias monitoring
CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4');


-- ==============================================================================
-- SECTION 8: LIVE VALIDATION QUERIES (15 INTEGRITY CHECKS)
-- Each query dynamically computes ACTUAL, compares against EXPECTED, and returns PASS/FAIL.
-- ==============================================================================

-- 1. Total Production Row Count (Expected: 2000 rows)
SELECT 
    '1. Production Row Count' AS test_name,
    '2000'                    AS expected,
    CAST(COUNT(*) AS VARCHAR) AS actual,
    CASE WHEN COUNT(*) = 2000 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- 2. Unique Array Identifiers (Expected: 2000 unique array_id)
SELECT 
    '2. Unique Arrays'                     AS test_name,
    '2000'                                 AS expected,
    CAST(COUNT(DISTINCT array_id) AS VARCHAR) AS actual,
    CASE WHEN COUNT(DISTINCT array_id) = 2000 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- 3. Unique Account Identifiers (Expected: 1000 unique account_id)
SELECT 
    '3. Unique Accounts'                     AS test_name,
    '1000'                                   AS expected,
    CAST(COUNT(DISTINCT account_id) AS VARCHAR) AS actual,
    CASE WHEN COUNT(DISTINCT account_id) = 1000 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- 4. Arrays per Account Hierarchy (Expected: 0 accounts with count != 2)
SELECT 
    '4. Arrays per Account Hierarchy' AS test_name,
    '0'                               AS expected,
    CAST(COUNT(*) AS VARCHAR)         AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM (
    SELECT account_id
    FROM CLV.CLV_SCORED_OUTPUT
    GROUP BY account_id
    HAVING COUNT(array_id) != 2
);

-- 5. Primary Key Uniqueness (Expected: 0 duplicate keys)
SELECT 
    '5. Duplicate Logical Keys' AS test_name,
    '0'                         AS expected,
    CAST(COUNT(*) AS VARCHAR)   AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM (
    SELECT array_id, model_version, run_id
    FROM CLV.CLV_SCORED_OUTPUT
    GROUP BY array_id, model_version, run_id
    HAVING COUNT(*) > 1
);

-- 6. Churn Probability Bounds (Expected: 0 rows outside [0, 1])
SELECT 
    '6. Probability Bounds'   AS test_name,
    '0'                       AS expected,
    CAST(COUNT(*) AS VARCHAR) AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE churn_probability < 0.0 OR churn_probability > 1.0;

-- 7. Target Label Isolation & NULL Preservation (Expected: Exactly 116 NULL labels)
SELECT 
    '7. NULL Churn Labels (Holdouts)' AS test_name,
    '116'                             AS expected,
    CAST(COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) AS VARCHAR) AS actual,
    CASE 
        WHEN COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) = 116 
             AND COUNT(CASE WHEN is_label_known = 0 THEN 1 END) = 116
        THEN 'PASS' 
        ELSE 'FAIL' 
    END AS test_status
FROM CLV.CLV_SCORED_OUTPUT;

-- 8. Value Exposed Non-Negative (Expected: 0 rows with value_exposed < 0)
SELECT 
    '8. Negative Value Exposed' AS test_name,
    '0'                         AS expected,
    CAST(COUNT(*) AS VARCHAR)   AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE value_exposed < 0.0;

-- 9. Zero Value Exposed for Losses (Expected: 0 loss rows with value_exposed > 0)
SELECT 
    '9. Loss Arrays with Positive Value Exposed' AS test_name,
    '0'                                          AS expected,
    CAST(COUNT(*) AS VARCHAR)                    AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE stated_profit < 0.0 AND value_exposed != 0.0;

-- 10. Forecast Debt Reconciliation (Expected: 0 rows with discrepancy > 0.05)
SELECT 
    '10. Forecast Debt Reconciliation' AS test_name,
    '0'                                AS expected,
    CAST(COUNT(*) AS VARCHAR)          AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE ABS(stated_profit - retention_adjusted_clv - forecast_debt) > 0.05;

-- 11. Probability Bucket Consistency (Expected: 0 bucket threshold mismatches)
SELECT 
    '11. Probability Bucket Consistency' AS test_name,
    '0'                                  AS expected,
    CAST(COUNT(*) AS VARCHAR)            AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE (churn_probability < 0.10 AND churn_probability_bucket != 'LOW')
   OR (churn_probability >= 0.10 AND churn_probability < 0.20 AND churn_probability_bucket != 'MEDIUM')
   OR (churn_probability >= 0.20 AND churn_probability_bucket != 'HIGH');

-- 12. TreeSHAP Completeness (Expected: 0 missing drivers or numeric values on labelled rows)
SELECT 
    '12. SHAP Completeness (Labelled Rows)' AS test_name,
    '0'                                     AS expected,
    CAST(COUNT(*) AS VARCHAR)               AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE is_label_known = 1 
  AND (top_shap_driver_1 IS NULL OR top_shap_driver_2 IS NULL OR top_shap_driver_3 IS NULL
       OR top_shap_value_1 IS NULL OR top_shap_value_2 IS NULL OR top_shap_value_3 IS NULL);

-- 13. Model Version Completeness (Expected: 0 rows with NULL model_version)
SELECT 
    '13. Model Version Completeness' AS test_name,
    '0'                              AS expected,
    CAST(COUNT(*) AS VARCHAR)        AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE model_version IS NULL OR TRIM(model_version) = '';

-- 14. Run ID Completeness (Expected: 0 rows with NULL run_id)
SELECT 
    '14. Run ID Completeness' AS test_name,
    '0'                       AS expected,
    CAST(COUNT(*) AS VARCHAR) AS actual,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_SCORED_OUTPUT
WHERE run_id IS NULL OR TRIM(run_id) = '';

-- 15. Cohort Bias Monitor Rows (Expected: Exactly 18 cohorts across 5 dimensions)
SELECT 
    '15. Cohort Bias Monitor Integrity' AS test_name,
    '18'                                AS expected,
    CAST(COUNT(*) AS VARCHAR)           AS actual,
    CASE WHEN COUNT(*) = 18 AND COUNT(DISTINCT cohort_dimension) = 5 THEN 'PASS' ELSE 'FAIL' END AS test_status
FROM CLV.CLV_FORECAST_BIAS_MONITOR;

-- ------------------------------------------------------------------------------
-- MASTER UNIFIED VALIDATION SUMMARY TABLE
-- Run this single query to view all 15 validation results in one grid!
-- ------------------------------------------------------------------------------
SELECT 
    '01. Total Rows' AS check_name, 
    '2000' AS expected, 
    CAST(COUNT(*) AS VARCHAR) AS actual, 
    CASE WHEN COUNT(*) = 2000 THEN 'PASS' ELSE 'FAIL' END AS status 
FROM CLV.CLV_SCORED_OUTPUT
UNION ALL
SELECT 
    '02. Unique Arrays', 
    '2000', 
    CAST(COUNT(DISTINCT array_id) AS VARCHAR), 
    CASE WHEN COUNT(DISTINCT array_id) = 2000 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT
UNION ALL
SELECT 
    '03. Unique Accounts', 
    '1000', 
    CAST(COUNT(DISTINCT account_id) AS VARCHAR), 
    CASE WHEN COUNT(DISTINCT account_id) = 1000 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT
UNION ALL
SELECT 
    '04. Arrays per Account Violations', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM (SELECT account_id FROM CLV.CLV_SCORED_OUTPUT GROUP BY account_id HAVING COUNT(array_id) != 2)
UNION ALL
SELECT 
    '05. Duplicate Logical Keys', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM (SELECT array_id, model_version, run_id FROM CLV.CLV_SCORED_OUTPUT GROUP BY array_id, model_version, run_id HAVING COUNT(*) > 1)
UNION ALL
SELECT 
    '06. Probability Out-of-Bounds', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE churn_probability < 0.0 OR churn_probability > 1.0
UNION ALL
SELECT 
    '07. Null Churn Labels (Holdouts)', 
    '116', 
    CAST(COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) AS VARCHAR), 
    CASE WHEN COUNT(CASE WHEN churn_event_next_12m IS NULL THEN 1 END) = 116 AND COUNT(CASE WHEN is_label_known = 0 THEN 1 END) = 116 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT
UNION ALL
SELECT 
    '08. Negative Value Exposed', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE value_exposed < 0.0
UNION ALL
SELECT 
    '09. Loss Arrays Positive Value Exposed', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE stated_profit < 0.0 AND value_exposed != 0.0
UNION ALL
SELECT 
    '10. Forecast Debt Unreconciled', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE ABS(stated_profit - retention_adjusted_clv - forecast_debt) > 0.05
UNION ALL
SELECT 
    '11. Probability Bucket Mismatches', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT 
WHERE (churn_probability < 0.10 AND churn_probability_bucket != 'LOW')
   OR (churn_probability >= 0.10 AND churn_probability < 0.20 AND churn_probability_bucket != 'MEDIUM')
   OR (churn_probability >= 0.20 AND churn_probability_bucket != 'HIGH')
UNION ALL
SELECT 
    '12. Missing SHAP on Labelled Rows', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT 
WHERE is_label_known = 1 
  AND (top_shap_driver_1 IS NULL OR top_shap_driver_2 IS NULL OR top_shap_driver_3 IS NULL 
       OR top_shap_value_1 IS NULL OR top_shap_value_2 IS NULL OR top_shap_value_3 IS NULL)
UNION ALL
SELECT 
    '13. Missing Model Version', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE model_version IS NULL OR TRIM(model_version) = ''
UNION ALL
SELECT 
    '14. Missing Run ID', 
    '0', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_SCORED_OUTPUT WHERE run_id IS NULL OR TRIM(run_id) = ''
UNION ALL
SELECT 
    '15. Cohort Monitor Rows', 
    '18', 
    CAST(COUNT(*) AS VARCHAR), 
    CASE WHEN COUNT(*) = 18 AND COUNT(DISTINCT cohort_dimension) = 5 THEN 'PASS' ELSE 'FAIL' END 
FROM CLV.CLV_FORECAST_BIAS_MONITOR;


-- ==============================================================================
-- SECTION 9: IDEMPOTENCY VERIFICATION (RE-RUN TEST)
-- Re-execute both procedures with the identical RUN_ID to confirm no row duplication.
-- ==============================================================================

-- Re-execute procedure 1
CALL CLV.SP_LOAD_CLV_SCORES('run_20261011_phase4');

-- Re-execute procedure 2
CALL CLV.SP_REFRESH_BIAS_MONITOR('run_20261011_phase4');

-- Verify row counts remain exact (EXPECTED: exactly 2,000 arrays and 18 cohorts)
SELECT 
    'IDEMPOTENCY_TEST_SCORES' AS check_name,
    '2000'                   AS expected_row_count,
    CAST(COUNT(*) AS VARCHAR) AS actual_row_count,
    CASE WHEN COUNT(*) = 2000 THEN 'PASS' ELSE 'FAIL' END AS status
FROM CLV.CLV_SCORED_OUTPUT
WHERE run_id = 'run_20261011_phase4';

SELECT 
    'IDEMPOTENCY_TEST_BIAS'   AS check_name,
    '18'                     AS expected_row_count,
    CAST(COUNT(*) AS VARCHAR) AS actual_row_count,
    CASE WHEN COUNT(*) = 18 THEN 'PASS' ELSE 'FAIL' END AS status
FROM CLV.CLV_FORECAST_BIAS_MONITOR
WHERE run_id = 'run_20261011_phase4';

-- Verify zero duplicate primary keys after re-run
SELECT 
    'IDEMPOTENCY_DUPLICATE_KEYS' AS check_name,
    '0'                          AS expected_duplicates,
    CAST(COUNT(*) AS VARCHAR)    AS actual_duplicates,
    CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END AS status
FROM (
    SELECT array_id, model_version, run_id
    FROM CLV.CLV_SCORED_OUTPUT
    WHERE run_id = 'run_20261011_phase4'
    GROUP BY array_id, model_version, run_id
    HAVING COUNT(*) > 1
);
