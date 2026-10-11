-- ==============================================================================
-- 02_create_clv_tables.sql
-- Project: CLV Truth Engine
-- Description: DDL for production tables:
--   1. CLV.CLV_SCORED_OUTPUT (Array-level scores, financial layer & SHAP drivers)
--   2. CLV.CLV_FORECAST_BIAS_MONITOR (Cohort-level bias monitoring & statistics)
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

-- ------------------------------------------------------------------------------
-- 1. CLV.CLV_SCORED_OUTPUT
-- Production table storing validated array-level model scores and financial metrics.
-- Exactly 2,000 array records per scoring run.
-- ------------------------------------------------------------------------------
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


-- ------------------------------------------------------------------------------
-- 2. CLV.CLV_FORECAST_BIAS_MONITOR
-- Production table operationalizing Task 2 cohort forecast bias monitoring.
-- Tracks empirical divergence between actual churn and the static 5% corporate assumption.
-- ------------------------------------------------------------------------------
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
