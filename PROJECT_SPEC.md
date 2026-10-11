# CLV Truth Engine — Reconciled Project Specification (PROJECT_SPEC.md)

## 1. Executive Overview & Business Objective

The enterprise Customer Lifetime Value (CLV) portal aggregates projected 15-year net profit across tens of thousands of storage arrays. Currently, these forward financial projections rely upon a flat, static **5.0% yearly churn assumption** for every storage array regardless of utilization stress, hardware capacity, financial margin, contract maturity, or customer expansion indicators.

In the actual supplied dataset:
- Exactly **298 out of 1,884 labelled arrays (15.82%)** churned within 12 months.
- Under the portal's static 5.0% assumption, only **~94 churn events** would have been expected.
- Even if all 116 unlabelled arrays remained retained, the population churn rate would be **14.90%**; if all churned, it would be **20.70%**. Both bounds demonstrate that observed enterprise churn is ~3x to 4x higher than the baseline assumption.

### Project Mandate:
1. **Challenge & Back-Test the Baseline**: Rigorously evaluate whether the gap between observed churn and the static 5.0% baseline persists across customer behavioral cohorts.
2. **Leakage-Safe Calibrated ML**: Train, calibrate, and validate supervised machine learning models to predict array-level churn risk without intra-account or future-target data leakage.
3. **Quantify Financial Forecast Debt**: Revalue stated 15-year profits using predicted retention into **Retention-Adjusted CLV**, and calculate the financial distortion as **Forecast Debt**.
4. **Actionable Account Prioritization**: Identify and rank customer accounts by **Value Exposed** to enable targeted intervention economics.
5. **Operationalize in Snowflake**: Deploy production DDL, idempotent stored procedures, views, and automated cohort bias monitoring into the Snowflake data warehouse.

---

## 2. Requirements Reconciliation & Conflict Resolution

### 2.1 Reconciled Directives: Project Documentation vs. Assessment Package Brief

| Assessment Package Brief (Case Study 2) | Full Project Documentation Specification | Reconciled Resolution & Implementation Mandate |
| :--- | :--- | :--- |
| **"Build two distinct ML models (XGBoost vs. Random Forest/Survival)"** | Specifies 3 models: Logistic Regression (interpretable benchmark), Random Forest (non-linear patterns), and XGBoost (gradient boosting), plus Model 0 (5% baseline). Points out survival models cannot be fitted due to lack of time-to-event data. | **Implement 3 distinct ML models**: Logistic Regression, Random Forest, and XGBoost. Evaluate rigorously against the 5% baseline. Do NOT fit a Cox/KM survival model because data lacks time-to-event timestamps; instead, compute 15-year survival scenarios under stated assumptions (constant vs. fading). |
| **"Score 15-year renewal, churn risk, and capacity expansion signals"** | Emphasizes: dataset contains a 12-month binary churn outcome, not a 15-year renewal observation. Expansion signal has no target label. | **Terminology Enforcement**: Score **calibrated 12-month churn risk** ($p$). Project **15-year survival scenarios under stated assumptions** ($S(t)$). Generate a transparent **expansion opportunity score (heuristic)** from signals and capacity pressure. Never claim "15-year renewal probability" or "expansion probability". |
| **"Historical forecast back-testing"** | Specifies that data is a single cross-sectional cohort snapshot without historical scoring logs or repeated temporal snapshots. | **Cohort back-test using out-of-fold predictions**: Group by behavioral bands (utilization, expansion, capacity, profit). Evaluate baseline bias ($p_{\text{actual}} - 0.05$) with Wilson 95% confidence intervals, binomial testing, and Holm-Bonferroni correction. Never call it a "time-based back-test". |
| **"Negative net profit values (down to -$100,000)"** | Actual CSV exhibits 58 negative-profit arrays (down to -$93,780). Recommends signed-log scaling for ML, keeping raw dollars for finance, and separate reporting. | **Dual Treatment**: Use `profit_signed_log = sign(x) * log(1 + abs(x))` and `negative_profit_flag` for ML inputs. In financial calculations, preserve exact dollars. Separate negative-profit arrays in reporting so losses do not generate artificial positive value exposure. |
| **"Storage array utilization exceeding 100%"** | Actual CSV exhibits 862 arrays (43.10%) exceeding 100% (max 1.45). Do not delete. | **Retain & Feature Engineer**: `utilization_capped = min(x, 1.0)`, `over_capacity = (x > 1.0)`, and `utilization_excess = max(x - 1.0, 0.0)`. Run sensitivity analysis comparing models with and without raw utilization. Over-capacity is treated as informative enterprise behavior. |
| **"Missing target labels in churn_event_next_12m"** | Actual CSV exhibits 116 missing labels. Never impute as 0. | **Isolate Cleanly**: Set `is_label_known = False`. Exclude from training, cross-validation, and actual-rate cohort back-testing. Score during final inference for full table population. |
| **"UNKNOWN categorical tags in expansion signals"** | Actual CSV exhibits 111 arrays with `UNKNOWN`. | **Explicit Category**: Treat `UNKNOWN` as an authentic, distinct categorical level across one-hot encodings and cohort monitoring. |
| **"Snowflake DDL, procedures, scores, buckets, SHAP"** | Defines exact schema `CLV`, tables `CLV_SCORED_OUTPUT` and `CLV_FORECAST_BIAS_MONITOR`, staging table `STG_CLV_SCORES`, procedures `SP_LOAD_CLV_SCORES` and `SP_REFRESH_BIAS_MONITOR`. | **Complete Production SQL**: Build full DDL, staging table, idempotent MERGE scripts, verification queries, and explainability payloads (TreeSHAP for tree models, standardized coefficients for Logistic Regression). |

---

## 3. Dataset Audit & Proven Empirical Profile

- **Total Rows**: 2,000
- **Total Customer Accounts**: 1,000 (`ACC_1000` to `ACC_1999`)
- **Total Storage Arrays**: 2,000 (`ARR_5000` to `ARR_6999`)
- **Arrays per Account**: Exactly 2 arrays per account across all 1,000 accounts
- **Target Distribution (`churn_event_next_12m`)**:
  - `0.0` (Stayed): 1,586 (84.18% of labelled)
  - `1.0` (Churned): 298 (15.82% of labelled)
  - `NaN` (Unlabelled): 116 (5.80% of total)
  - **Empirical Churn Rate**: **15.8174%**
  - **Overall Churn Bounds**: [14.90%, 20.70%]
- **Static Baseline**: Constant 0.05 across all 2,000 rows.
- **Utilization Distribution (`utilization_rate_pct`)**:
  - Min: 0.35, Median: 0.93, Mean: 0.917, 75th percentile: 1.18, Max: 1.45
  - Over-capacity (> 1.0): 862 arrays (43.10%)
  - Churn rate in over-capacity: 14.29% (N=812) vs. non-over-capacity: 16.98% (N=1,072)
- **Net Profit Distribution (`historical_15yr_net_profit_usd`)**:
  - Min: -\$93,780, Median: \$2,134,328, Mean: \$2,152,845, Max: \$4,498,824
  - Negative-profit arrays: 58 (2.90%)
  - Churn rate in negative profit: 12.50% (N=56) vs. positive profit: 15.92% (N=1,828)
- **Capacity Distribution (`current_capacity_tb`)**:
  - Min: 53 TB, Median: 1,305.5 TB, Mean: 1,282.8 TB, Max: 2,498 TB (Zero/negative capacity: 0)
- **Expansion Signal (`expansion_potential_signal`)**:
  - Low: 841 (42.05%), Medium: 622 (31.10%), High: 426 (21.30%), UNKNOWN: 111 (5.55%)
- **Contract End Dates**:
  - Array 1 (`ARR_5000`–`ARR_5999`): Dates span `2026-10-01` to `2028-02-12` (2 accounts per day)
  - Array 2 (`ARR_6000`–`ARR_6999`): Dates span `2028-02-13` to `2029-06-26` (2 accounts per day)
  - Correlation with churn: $r = 0.010$ (no significant predictive relationship).

---

## 4. Leakage-Safe Validation Architecture

### 4.1 Account-Level Grouped Splitting
Because each customer account owns exactly two arrays, splitting randomly across arrays would cause severe intra-account data leakage (hardware environment, purchasing behavior, organizational churn decisions).
- **Rule**: All arrays belonging to an `account_id` MUST reside in the same cross-validation fold.
- **Protocol**: 5-fold outer Group Cross-Validation, repeated across 5 distinct random seeds (25 evaluations total).
- **Inner Tuning**: 3-fold inner Group CV for hyperparameter optimization to eliminate optimization leakage.
- **Out-of-Fold (OOF) Scoring**: Out-of-fold predictions are collected strictly from models evaluated on unseen customer folds.

### 4.2 Pipeline Transformations
- Encoders, scalers, imputers, and calibrators are fitted strictly within the training folds.
- The 116 unlabelled rows are excluded from training and validation.
- Sibling churn target values (`churn_event_next_12m`) are strictly forbidden from entering the feature matrix.

---

## 5. Machine Learning Modeling & Calibration Pipeline

### 5.1 Evaluated Model Classes
1. **Model 0 (Static Baseline)**: Flat $p = 0.05$ reference benchmark.
2. **Model 1 (Logistic Regression)**: Interpretable linear benchmark with L2 regularization ($C \in [0.01, 0.1, 1.0, 10.0]$).
3. **Model 2 (Random Forest)**: Non-linear ensemble (300 estimators; max_depth $\in [4, 6, \text{None}]$; min_samples_leaf $\in [5, 10, 20]$).
4. **Model 3 (XGBoost)**: Gradient boosted decision trees (max_depth $\in [2, 3, 4]$; learning_rate $\in [0.05, 0.1]$; subsample = 0.8; early stopping inside folds).

### 5.2 Resampling & Balancing Prohibition
- **SMOTE is strictly prohibited**: Resampling distorts probability estimates and breaks calibration, which directly pollutes downstream financial valuation.

### 5.3 Probability Calibration
- Models are calibrated via **Platt Scaling (Sigmoid)** inside cross-validation.
- Isotonic regression is rejected due to known overfitting on small sample sizes ($N_{\text{positives}} = 298$).

### 5.4 Model Selection Criterion
- Primary metrics: **PR-AUC**, **Brier Score**, **Log Loss**, and **Calibration Curve slope/intercept**.
- Secondary metrics: ROC-AUC, Top-Decile Lift, F1, Recall.
- Selection Decision Rule: Select the simplest model within 1 standard error of the minimum Brier score/log loss, with ROC-AUC as tiebreaker.

---

## 6. Explainability Framework

- **If Tree-Based Model Wins**: TreeSHAP computed on the raw margin (pre-calibration log-odds) of the refitted model.
- **If Logistic Regression Wins**: Feature attribution computed via $\beta_j \times z_{ij}$ (standardized log-odds contribution).
- **Output Storage**: Capture `top_driver_1`, `top_driver_2`, `top_driver_3`, `driver_method`, and full `driver_detail` formatted as a JSON/Snowflake VARIANT string for every scored array.

---

## 7. Cohort Back-Testing & Bias Detection

Cohort back-testing compares model out-of-fold predictions against actual churn and the static 5% baseline across defined cohorts:
1. **Utilization Bands**: `0–50%`, `50–75%`, `75–90%`, `90–100%`, `>100%` (over-capacity).
2. **Expansion Potential Signals**: `Low`, `Medium`, `High`, `UNKNOWN`.
3. **Capacity Bands**: Quartile/tercile capacity divisions.
4. **Profit Bands**: Loss-making (< \$0), Low Margin, Mid Margin, High Margin.

### Statistical Rigor:
- **Baseline Bias**: $\text{Actual Churn Rate} - 0.05$.
- **Uncertainty**: 95% Wilson score interval for binomial proportions.
- **Hypothesis Testing**: Exact two-sided binomial test comparing observed cohort churn against null hypothesis $p_0 = 0.05$.
- **Multiple Comparison Control**: Holm-Bonferroni correction applied across cohorts.

---

## 8. Financial Layer: Retention-Adjusted CLV & Forecast Debt

### 8.1 Retention Factor Formula
For annual churn rate $p$ over a 15-year horizon ($t = 0, \dots, 14$):
$$RF(p) = \frac{1}{15} \sum_{t=0}^{14} (1-p)^t = \frac{1 - (1-p)^{15}}{15 \cdot p} \quad (\text{with } RF(0) = 1.0)$$

### 8.2 Retention-Adjusted CLV
$$\text{Retention-Adjusted CLV} = \text{stated\_profit} \times \frac{RF(p_{\text{model}})}{RF(q)}$$
where $q$ is the churn rate assumed inside the baseline financial valuation (base case $q = 0.05$).

### 8.3 Forecast Debt
$$\text{Forecast Debt} = \text{stated\_profit} - \text{Retention-Adjusted CLV}$$
- **Positive Debt**: Baseline overstates economic value (underestimates churn risk).
- **Negative Debt**: Baseline understates economic value (customer safer than assumed).

### 8.4 15-Year Survival Scenarios (under stated assumptions)
- **Scenario A (Constant Churn)**: $S(t) = (1 - p)^t$
- **Scenario B (Fading Churn)**: $S(1) = 1 - p$; $S(t) = (1 - p)(1 - p/2)^{t-1}$ for $t > 1$.
- Computed at 5, 10, and 15 years.

### 8.5 Profit Sensitivity Analysis
Forecast Debt evaluated across three baseline scenarios:
- **Conservative Baseline**: $q = 0.08$
- **Base Baseline**: $q = 0.05$
- **Aggressive Baseline**: $q = 0.03$

### 8.6 Negative Profit & Value Exposed
- **Negative-Profit Arrays**: Separately reported. Not allowed to generate artificial positive value exposure.
- **Value Exposed**: Prioritization metric calculated as:
$$\text{Value Exposed} = p_{\text{model}} \times \max(\text{stated\_profit}, 0)$$
*(Never multiply Retention-Adjusted CLV by $p$, avoiding double-counting churn).*

---

## 9. Action Layer & Enterprise Interventions

1. **Account-Level Value Exposure**: Aggregate the 2 arrays per customer account to calculate `account_value_exposed`, representative churn risk, and top account-level drivers.
2. **Top 20 Account Call List**: Produce ranked table of accounts with largest dollar value at risk.
3. **Break-Even Intervention Economics**:
$$\text{Break-Even Effectiveness} = \frac{\text{Intervention Cost}}{p_{\text{model}} \times \text{Value Exposed}}$$
Evaluated under flat cost (\$5,000) and capacity-scaled cost (\$5,000 + \$2/TB).

---

## 10. Snowflake Warehouse Integration

- **Database**: `CLV_DB` | **Schema**: `CLV`
- **Tables**:
  - `CLV.CLV_SCORED_OUTPUT` (array-grain predictions, financial adjustments, survival scenarios, drivers)
  - `CLV.CLV_FORECAST_BIAS_MONITOR` (cohort-grain tracking, bias metrics, Wilson intervals, adjusted p-values)
  - `CLV.STG_CLV_SCORES` (staging table matching scored schema)
- **Stored Procedures**:
  - `SP_LOAD_CLV_SCORES(run_id)`: Merges staging into scored table on `(array_id, model_version)`, checks counts, idempotent.
  - `SP_REFRESH_BIAS_MONITOR(run_id)`: Recomputes cohort table from scored output for labelled rows.
- **Views**:
  - `CLV.V_TOP_RISK_ACCOUNTS`: Account-level aggregation ranking top 20 accounts by value exposed.

---

## 11. Testing & Validation Matrix

Automated test suite (`pytest`) verifying:
1. Mathematical identity: $RF(0.05) / RF(0.05) = 1.0$.
2. Zero debt identity: $\text{Forecast Debt} = 0$ when $p_{\text{model}} = q$.
3. Monotonic survival: Survival probabilities non-increasing over time.
4. Non-negative value exposed: Value exposed is strictly $\ge 0$.
5. Negative profit preservation: Negative profits stay negative in financial revaluations.
6. Grouped split isolation: Zero account overlap between train and test folds.
7. Unlabelled data isolation: Unlabelled rows never enter model training or CV.
8. Feature leakage prevention: Sibling churn label is absent from feature matrix.
9. Array hierarchy integrity: Exactly 2 arrays per account.
10. Final scoring completeness: Exactly 2,000 arrays scored.

---

## 12. Assumptions, Limitations & Non-Negotiable Boundaries

1. **Synthetic & 12-Month Target**: The dataset contains a 12-month binary churn label, not time-to-event history. 15-year survival projections are scenarios under stated assumptions, not observed long-term renewal probabilities.
2. **Historical vs. Forward Profit**: The profit field is titled `historical_15yr_net_profit_usd`. We treat it as the stated baseline economic value used by the portal; whether it represents earned past profit or forward contractual value is an acknowledged business limitation.
3. **No Causal Uplift**: Without an intervention control group, intervention analysis is purely break-even scenario math, not causal uplift modeling.
4. **Heuristic Expansion**: Expansion potential signal is transformed into a heuristic score; no predictive expansion model is claimed.
5. **Snowflake Execution**: If live Snowflake credentials are unavailable, production SQL scripts are fully verified locally and execution instructions provided, with the limitation clearly documented.
