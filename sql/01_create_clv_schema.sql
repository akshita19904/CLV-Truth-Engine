-- ==============================================================================
-- 01_create_clv_schema.sql
-- Project: CLV Truth Engine
-- Description: Creates the dedicated CLV database and production schema.
-- Note: Practical production tables directly aligned with assessment requirements.
--       No star schema or unnecessary dimensional layers.
-- ==============================================================================

-- Create Database if not exists
CREATE DATABASE IF NOT EXISTS CLV_DB
    COMMENT = 'Everpure CLV Truth Engine Analytics & Scoring Database';

-- Use Database
USE DATABASE CLV_DB;

-- Create dedicated production schema
CREATE SCHEMA IF NOT EXISTS CLV
    COMMENT = 'Customer Lifetime Value, Churn Scoring & Forecast Bias Production Schema';

-- Use Schema
USE SCHEMA CLV;

SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_TIMESTAMP();
