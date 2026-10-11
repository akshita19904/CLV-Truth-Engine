-- ==============================================================================
-- 04_load_clv_scores.sql
-- Project: CLV Truth Engine
-- Description: Bulk loading script for staging ingestion via Snowflake stages.
-- Supports:
--   1. Internal Snowflake Stage creation
--   2. File format specification (handling NULLs and headers)
--   3. COPY INTO statements for bulk loading
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

-- ------------------------------------------------------------------------------
-- 1. Create File Format for CSV Ingestion
-- ------------------------------------------------------------------------------
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

-- ------------------------------------------------------------------------------
-- 2. Create Internal Named Stage
-- ------------------------------------------------------------------------------
CREATE OR REPLACE STAGE CLV.STG_CLV_DATA_STAGE
    FILE_FORMAT = CLV.CSV_LOAD_FORMAT
    COMMENT = 'Internal stage for receiving Python-generated scored outputs and bias monitoring CSVs.';

-- ------------------------------------------------------------------------------
-- 3. Bulk Loading Commands (executed via SnowSQL or Python Snowflake Connector)
-- ------------------------------------------------------------------------------

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
