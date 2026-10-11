# CLV Truth Engine — Case Study 2
## Behavior-Driven Retention-Adjusted CLV & Forecast Bias Engine

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/tests-50%2F50%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. Project Purpose & Executive Overview

The **CLV Truth Engine** is an enterprise predictive analytics and financial engineering system built for **Case Study 2: Customer Lifetime Value (CLV) Predictive Analytics**.

In enterprise hardware renewal planning (e.g., Pure Storage 15-year storage array deployments), corporate financial models traditionally rely on an unvalidated **static 5.0% annual churn assumption**. If true customer churn is higher, multi-year revenue projections compound exponential errors over time, creating hidden balance-sheet **"Forecast Debt"**—the systematic overstatement of expected customer lifetime value.

The CLV Truth Engine resolves this challenge by:
1. Auditing and cleaning 2,000 enterprise storage arrays across 1,000 accounts with zero data leakage.
2. Training, validating, and calibrating supervised machine learning models (**XGBoost vs. Random Forest**) to predict 12-month array renewal churn risk.
3. Quantifying **Forecast Bias** across 18 operational cohorts against the static 5% baseline.
4. Computing **Retention-Adjusted CLV**, exposing **$1.96 Billion (45.46%)** in hidden portfolio Forecast Debt and **$681.51 Million** in immediate 12-month Value Exposed.
5. Packaging the entire pipeline into Snowflake warehouse tables and production stored procedures.

---

## 2. Primary Evaluator Entry Point

The primary evaluator-facing deliverable for this project is:

👉 **[`notebooks/CLV_Truth_Engine_Master.ipynb`](notebooks/CLV_Truth_Engine_Master.ipynb)**

This self-contained Master Notebook guides an interviewer through the complete end-to-end data science lifecycle across 19 clearly structured sections.

---

## 3. Running in Google Colab & GitHub Setup

The Master Notebook is designed to run seamlessly in **Google Colab** without requiring any personal Google Drive mounts or local file uploads:

### Evaluator Colab Workflow:
1. Open [`notebooks/CLV_Truth_Engine_Master.ipynb`](notebooks/CLV_Truth_Engine_Master.ipynb) in Google Colab.
2. In the initial `COLAB SETUP` cell, update the repository URL placeholder if needed:
   ```python
   REPO_URL = "GITHUB_REPOSITORY_URL_HERE"
   ```
3. Run the setup cell. The notebook will automatically:
   - Detect the Colab runtime.
   - Clone the repository into `/content/CLV-Truth-Engine`.
   - Set the working directory to the project root (`%cd /content/CLV-Truth-Engine`).
   - Install required dependencies via `pip install -r requirements.txt -q`.
   - Validate that all canonical datasets, models, and SQL files are present.
4. Run all subsequent cells to view the live execution, plots, and financial reconciliations.

*All file paths throughout the project are strictly repository-relative (e.g., `data/raw/...`, `reports/...`, `sql/...`).*

---

## 4. Repository Structure

```text
CLV-Truth-Engine/
├── .gitignore                          # Clean gitignore excluding caches & secrets
├── requirements.txt                    # Pinned production dependencies (pip compatible)
├── README.md                           # Master project documentation
├── PROJECT_SPEC.md                     # Engineering specifications & business rules
├── DATA_DICTIONARY.md                  # Comprehensive column data dictionary
├── AGENT_PROGRESS.md                   # Chronological development trajectory
│
├── data/
│   ├── raw/
│   │   └── Case_Study_2_CLV_Predictive_Analytics.csv  # Canonical raw dataset (2,000 rows x 9 cols)
│   └── processed/
│       ├── cleaned_data.csv            # Cleaned data with anomaly flags (2,000 rows x 15 cols)
│       ├── model_features.csv          # Feature matrix with sibling features (2,000 rows x 23 cols)
│       ├── snowflake_clv_scored_export.csv  # Scored arrays export for Snowflake (2,000 rows x 32 cols)
│       └── snowflake_bias_monitor_export.csv # Cohort bias export for Snowflake (18 rows x 16 cols)
│
├── notebooks/
│   ├── CLV_Truth_Engine_Master.ipynb   # ★ PRIMARY MASTER EVALUATOR NOTEBOOK (19 sections)
│   ├── 01_data_audit_and_cleaning.ipynb # Supporting: Detailed Phase 1 data audit & cleaning
│   ├── 05_final_task1_models.ipynb     # Supporting: Detailed Task 1 ML modeling & calibration
│   ├── 06_forecast_bias_and_clv.ipynb  # Supporting: Detailed Task 2 back-testing & CLV engine
│   └── 02_features_and_models.ipynb    # [ARCHIVED/EXCLUDED]: Contains exploratory Logistic Regression
│
├── src/
│   ├── cleaning.py                     # Non-destructive data cleaning & defect handling
│   ├── features.py                     # Leakage-safe feature pipeline & sibling relationships
│   ├── task1_models.py                 # XGBoost & Random Forest pipelines with calibration
│   ├── phase3_financial.py             # Retention factor, Forecast Debt, & cohort bias statistics
│   ├── snowflake_export.py             # Standardized CSV generator for Snowflake warehouse
│   ├── validation.py                   # Grouped cross-validation utilities
│   └── config.py                       # Project paths and environment configuration
│
├── sql/
│   ├── 01_create_clv_schema.sql        # Creates CLV_DB and CLV production schema
│   ├── 02_create_clv_tables.sql        # Production tables DDL (CLV_SCORED_OUTPUT & BIAS_MONITOR)
│   ├── 03_create_staging.sql           # Controlled staging tables (STG_CLV_SCORES & STG_FORECAST_BIAS)
│   ├── 04_load_clv_scores.sql          # Named stage and bulk COPY INTO statements
│   ├── 05_sp_load_clv_scores.sql       # Procedure: Idempotent MERGE into CLV_SCORED_OUTPUT
│   ├── 06_sp_refresh_bias_monitor.sql  # Procedure: Idempotent MERGE into CLV_FORECAST_BIAS_MONITOR
│   ├── 07_validation_queries.sql       # 15 automated production verification queries
│   └── 08_manual_snowsight_execution.sql # Master single-worksheet execution script for Snowsight UI
│
├── reports/
│   ├── task1_churn_model_comparison.csv # Final model comparison metrics (XGBoost vs RF vs 5% Baseline)
│   ├── task1_final_scores.csv          # Array-level scores, probabilities, and expansion signals
│   ├── task1_shap_values.csv           # TreeSHAP feature contributions across 1,884 arrays
│   ├── phase_3_cohort_bias_results.csv # 18 cohort back-testing records with Wilson CIs & Holm tests
│   ├── phase_3_forecast_debt.csv       # Array-level Forecast Debt and Value Exposed calculations
│   ├── phase4_manual_snowsight_checklist.md # Step-by-step Snowsight UI execution & screenshot checklist
│   ├── phase4_live_snowflake_validation.md # Engineering audit & Snowflake connection technical report
│   └── figures/                        # Generated charts and diagnostic figures
│
├── scripts/
│   ├── build_master_notebook.py        # Programmatic builder for CLV_Truth_Engine_Master.ipynb
│   ├── run_snowflake_live.py           # Master automated live Snowflake deployment runner
│   └── verify_snowflake_connection.py  # Diagnostic utility for Snowflake connection verification
│
└── tests/
    ├── test_cleaning.py                # Unit tests for data cleaning & defect handling (8 tests)
    ├── test_features.py                # Unit tests for feature matrix construction (5 tests)
    ├── test_split_and_labels.py        # Unit tests for target segregation & leakage safety (5 tests)
    ├── test_task1_models.py            # Unit tests for ML models & calibration (8 tests)
    ├── test_phase3_financial.py        # Unit tests for retention factor & Forecast Debt formulas (12 tests)
    └── test_sql_pipeline.py            # Integration tests for Snowflake SQL logic (12 tests)
```

---

## 5. Dataset & Data Quality Defects

The raw dataset is located at:
`data/raw/Case_Study_2_CLV_Predictive_Analytics.csv` *(2,000 storage arrays, 1,000 accounts)*

### The 4 Injected Quality Defects & Treatment:
1. **Storage Array Utilization > 100% (862 arrays, 43.10%):** Caused by thin provisioning / storage oversubscription. Preserved in raw form; engineered bounded features `utilization_capped = min(u, 1.0)`, `over_capacity = (u > 1.0)`, and `utilization_excess = max(u - 1.0, 0)`.
2. **Negative Net Profit (58 arrays, 2.90%):** Caused by loss-making deployments or discounting. Preserved without deletion; engineered `negative_profit_flag` and `profit_signed_log = sign(profit) * ln(1 + |profit|)`.
3. **Missing Churn Labels (116 arrays, 5.80%):** Unlabelled evaluation holdouts. Flagged via `is_label_known = 0` and **strictly isolated from training/evaluation**. Never converted to 0.
4. **UNKNOWN Expansion Potential Signal (111 arrays, 5.55%):** Missing telemetry. Preserved as an explicit categorical state; never mapped to "Low" or dropped.

---

## 6. Final Model Set & Comparison

The final evaluator-facing model comparison contains **ONLY the required models**:
1. **Calibrated XGBoost (Champion):** Gradient Boosted Decision Trees with nested Platt probability calibration.
2. **Calibrated Random Forest (Comparison):** Bagged Ensembles with nested Platt probability calibration.
3. **Static 5% Annual Churn Baseline:** Corporate financial planning benchmark.

*(Note: Exploratory Logistic Regression from early development is completely excluded from the final presentation layer and Master Notebook. Notebook `02_features_and_models.ipynb` remains archived for historical auditability only.)*

### Model Performance Summary:

| Model | Calibration Status | Log Loss | Brier Score | ROC-AUC | PR-AUC | Mean Predicted Churn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static 5% Baseline** | Uncalibrated Baseline | 0.5170 | 0.1449 | 0.5000 | 0.1582 | 5.00% |
| **Random Forest (Raw)** | Raw | 0.4401 | 0.1342 | 0.5154 | 0.1653 | 14.86% |
| **XGBoost (Raw)** | Raw | 0.4490 | 0.1364 | 0.5206 | 0.1729 | 15.22% |
| **XGBoost (Calibrated)** ★ | **Calibrated (Platt)** | **0.4371** | **0.1333** | **0.5005** | **0.1602** | **15.82%** |
| **Random Forest (Calibrated)** | **Calibrated (Platt)** | **0.4372** | **0.1333** | **0.5045** | **0.1602** | **15.80%** |

*Calibrated XGBoost is selected as the production champion due to superior log loss and exact alignment with empirical churn prevalence (15.82%).*

---

## 7. Financial Revaluation & Forecast Debt Results

Applying the finite-horizon 15-year cumulative retention factor $R(p) = \frac{1 - (1-p)^{15}}{15 \cdot p}$ across all 2,000 storage arrays yields:

| Financial Metric | Mathematical Definition | Total Sum across 2,000 Arrays | % of Stated Profit |
| :--- | :--- | :---: | :---: |
| **Stated Accounting Profit** | $\sum \Pi_i$ (Unadjusted 15-year net profit) | **$4,305,689,564.00** | 100.00% |
| **Retention-Adjusted CLV** | $\sum \Pi_i \times \frac{R(p_{i, \text{xgb}})}{R(0.05)}$ | **$2,348,335,632.55** | 54.54% |
| **Forecast Debt** | $\sum (\Pi_i - \text{CLV}_{i, \text{adj}})$ | **$1,957,353,931.45** | **45.46%** |
| **Immediate Value Exposed** | $\sum p_{i, \text{xgb}} \times \max(\Pi_i, 0)$ | **$681,507,251.63** | 15.83% |

$$\text{Reconciliation Residual}: |\text{Stated Profit} - \text{Retention CLV} - \text{Forecast Debt}| = \$0.00$$

---

## 8. Snowflake Production Warehouse Integration

All SQL scripts are organized in `sql/`:
- **Master Snowsight Script:** [`sql/08_manual_snowsight_execution.sql`](sql/08_manual_snowsight_execution.sql) provides a unified, deterministic deployment script for execution in Snowflake's Snowsight UI worksheet.
- **Evidence Checklist:** [`reports/phase4_manual_snowsight_checklist.md`](reports/phase4_manual_snowsight_checklist.md) outlines the 8 screenshots required to document live Snowsight execution.
- **Production Tables:** `CLV.CLV_SCORED_OUTPUT` (2,000 rows) and `CLV.CLV_FORECAST_BIAS_MONITOR` (18 cohorts).
- **Procedures:** `CLV.SP_LOAD_CLV_SCORES` and `CLV.SP_REFRESH_BIAS_MONITOR` feature duplicate detection, NULL target preservation, deterministic probability bucketing, and idempotent atomic `MERGE` semantics.

---

## 9. Automated Testing & Verification

The project includes an automated test suite verifying data integrity, feature transformations, modeling pipelines, financial math, and SQL logic.

To execute the test suite:
```bash
pytest -v
```

### Test Suite Breakdown (50 / 50 Passing Tests):
- `tests/test_cleaning.py`: 8 tests (Audits, anomaly flags, defect treatments)
- `tests/test_features.py`: 5 tests (Hardware pressure, contract maturity, sibling features)
- `tests/test_split_and_labels.py`: 5 tests (Target segregation, zero holdout leakage, GroupKFold)
- `tests/test_task1_models.py`: 8 tests (Pipelines, OOF probability generation, calibration)
- `tests/test_phase3_financial.py`: 12 tests (Retention factors, Forecast Debt, cohort bias statistics)
- `tests/test_sql_pipeline.py`: 12 tests (Snowflake export schema, 15 validation queries, idempotency)

---

## 10. License & Submission Information
- **Project Name:** CLV Truth Engine
- **Case Study:** Case Study 2 — Customer Lifetime Value (CLV) Predictive Analytics
- **Primary Notebook:** `notebooks/CLV_Truth_Engine_Master.ipynb`
- **License:** MIT License
