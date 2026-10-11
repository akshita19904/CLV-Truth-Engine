-- ==============================================================================
-- 06_sp_refresh_bias_monitor.sql
-- Project: CLV Truth Engine
-- Description: Production Stored Procedure: CLV.SP_REFRESH_BIAS_MONITOR(RUN_ID)
--
-- Features:
--   1. Reads from staging table CLV.STG_FORECAST_BIAS for the given run_id.
--   2. Performs idempotent MERGE into CLV.CLV_FORECAST_BIAS_MONITOR.
--   3. Guarantees no duplicate records per run_id + cohort_dimension + cohort_name.
--   4. Preserves verified Wilson confidence intervals and Holm-adjusted p-values.
--   5. Returns JSON summary status.
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

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

    -- 3. Verify total monitor rows for this run
    SELECT COUNT(*)
    INTO :V_MERGED_COUNT
    FROM CLV.CLV_FORECAST_BIAS_MONITOR
    WHERE run_id = :P_RUN_ID;

    RETURN OBJECT_CONSTRUCT(
        'status', 'SUCCESS',
        'run_id', :P_RUN_ID,
        'staging_records', :V_STAGING_COUNT,
        'monitor_records', :V_MERGED_COUNT,
        'message', 'Cohort forecast bias monitor refreshed successfully.'
    )::VARCHAR;
END;
$$;
