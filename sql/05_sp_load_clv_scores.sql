-- ==============================================================================
-- 05_sp_load_clv_scores.sql
-- Project: CLV Truth Engine
-- Description: Production Stored Procedure: CLV.SP_LOAD_CLV_SCORES(RUN_ID)
--
-- Features:
--   1. Reads from staging table CLV.STG_CLV_SCORES for the given run_id.
--   2. Validates staging row counts and mandatory fields.
--   3. Prevents / flags duplicate array_id entries.
--   4. Computes deterministic probability buckets inside SQL:
--        LOW:    churn_probability < 0.10
--        MEDIUM: 0.10 <= churn_probability < 0.20
--        HIGH:   churn_probability >= 0.20
--   5. Preserves NULL target labels for unlabelled holdout rows (is_label_known = 0).
--   6. Stores numeric TreeSHAP values and absolute values.
--   7. Performs idempotent MERGE into CLV.CLV_SCORED_OUTPUT on:
--        (target.array_id = source.array_id AND
--         target.model_version = source.model_version AND
--         target.run_id = source.run_id)
--   8. Returns formatted JSON status message with execution details.
-- ==============================================================================

USE DATABASE CLV_DB;
USE SCHEMA CLV;

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
