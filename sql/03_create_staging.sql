-- ==============================================================================
-- 03_create_staging.sql
-- Project: CLV Truth Engine
-- Description: DDL for controlled staging ingestion tables:
--   1. CLV.STG_CLV_SCORES (Staging area for scored output ingestion)
--   2. CLV.STG_FORECAST_BIAS (Staging area for cohort bias ingestion)
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

-- ------------------------------------------------------------------------------
-- 1. CLV.STG_CLV_SCORES
-- Ingestion staging table for bulk loading scoring CSV batches.
-- Allows pre-validation before merging into production.
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- 2. CLV.STG_FORECAST_BIAS
-- Ingestion staging table for bulk loading cohort monitor metrics.
-- ------------------------------------------------------------------------------
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
