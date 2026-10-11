# CLV Truth Engine — Comprehensive Data Dictionary & Governance Catalog

## 1. Raw Dataset Specification

- **File**: `data/raw/Case_Study_2_CLV_Predictive_Analytics.csv`
- **Integrity SHA-256**: `BE100DC8625AA5CE05E937540186A150817F60A5D6A58B50D85804F8F0AA3C31`
- **Total Records**: 2,000 rows
- **Total Columns**: 9
- **Unique Accounts**: 1,000 (`ACC_1000` to `ACC_1999`)
- **Arrays per Account**: Exactly 2 storage arrays per account across all 1,000 accounts

---

## 2. Column Roles & Governance Matrix

| Column Name | Physical Type | Missingness | Analytical Role | Usable as Model Feature? | Data Leakage Concern & Rule | Description & Value Domain |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `account_id` | `VARCHAR` / `str` | 0 (0.0%) | **GROUPING VARIABLE** | **NO** (Grouping only) | High: Identifiers must never enter feature matrix. Used strictly as grouping key in `GroupKFold`. | Unique customer enterprise ID (1,000 unique). Each account owns exactly 2 physical arrays. |
| `array_id` | `VARCHAR` / `str` | 0 (0.0%) | **PRIMARY KEY** | **NO** (ID only) | High: Arbitrary primary key with zero generalizable predictive signal. | Unique storage array identifier (`ARR_5000` to `ARR_6999`). Grain of the dataset. |
| `contract_end_date` | `VARCHAR` / `DATE` | 0 (0.0%) | **METADATA** | **CONDITIONAL** (Temporal only) | Low: Proximity to contract end relative to reference date (`2026-10-01`). Empirical correlation with churn is negligible ($r = 0.010$). | Stated contract expiration date (`2026-10-01` to `2029-06-26`). 2 arrays expire per calendar day. |
| `historical_15yr_net_profit_usd` | `INTEGER` / `int64` | 0 (0.0%) | **FINANCIAL BASELINE** | **YES** (Transformed) | None: Known historical figure at scoring time. Use signed log for ML; keep raw dollars for financial revaluation. | Stated 15-year net profit used by CLV portal. Range: -\$93,780 to +\$4,498,824 (58 negative arrays, 2.90%). |
| `current_capacity_tb` | `INTEGER` / `int64` | 0 (0.0%) | **HARDWARE ATTRIBUTE** | **YES** | None: Provisioned capacity is fully observable prior to churn event. | Provisioned usable storage capacity in Terabytes (Range: 53 to 2,498 TB). Zero non-positive values. |
| `utilization_rate_pct` | `FLOAT` / `float64` | 0 (0.0%) | **BEHAVIORAL ATTRIBUTE** | **YES** (Raw & Capped) | None: Pre-renewal hardware loading metric. | Storage utilization ratio as a decimal (Range: 0.35 to 1.45). 862 arrays (43.10%) exceed 1.0 (100%). |
| `static_baseline_churn_rate` | `FLOAT` / `float64` | 0 (0.0%) | **BASELINE BENCHMARK** | **NO** (Constant baseline) | Zero variance: Constant `0.05` across all rows. | Legacy flat 5.0% annual churn assumption currently embedded in the CLV portal. |
| `churn_event_next_12m` | `FLOAT` / `float64` | 116 (5.8%) | **TARGET (GROUND TRUTH)** | **STRICTLY PROHIBITED** | **CRITICAL TARGET LEAKAGE**: The target variable must NEVER enter the feature matrix. Sibling churn labels are also strictly forbidden! | Binary ground truth (1.0 = churned, 0.0 = stayed, NaN = missing). 1,884 labelled (15.82% churn rate), 116 unlabelled. |
| `expansion_potential_signal` | `VARCHAR` / `str` | 0 (0.0%) | **BEHAVIORAL ATTRIBUTE** | **YES** (One-Hot Encoded) | None: Known categorical sales indicator prior to contract renewal. | Categorical expansion assessment: `Low` (841), `Medium` (622), `High` (426), `UNKNOWN` (111). |

---

## 3. Cleaning & Audit Derived Fields (`data/processed/cleaned_data.csv`)

| Field Name | Physical Type | Calculation / Logic | Analytical Role | Description & Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `is_label_known` | `INTEGER` (`0` or `1`) | `1 if churn_event_next_12m is not null else 0` | **POPULATION ISOLATION** | Separates 1,884 labelled training arrays from 116 unlabelled inference-only arrays. Missing churn labels are never coerced to 0. |
| `utilization_capped` | `FLOAT` | `min(utilization_rate_pct, 1.0)` | **ENGINEERED FEATURE** | Bounds utilization to nominal physical capacity (100%) to stabilize linear and tree splitting behavior. |
| `over_capacity` | `INTEGER` (`0` or `1`) | `1 if utilization_rate_pct > 1.0 else 0` | **ENGINEERED FEATURE** | Behavioral flag indicating hardware thin-provisioning saturation or capacity oversubscription. |
| `utilization_excess` | `FLOAT` | `max(utilization_rate_pct - 1.0, 0.0)` | **ENGINEERED FEATURE** | Quantifies excess committed storage beyond physical threshold (ranges from 0.0 to 0.45). |
| `negative_profit_flag` | `INTEGER` (`0` or `1`) | `1 if historical_15yr_net_profit_usd < 0 else 0` | **FINANCIAL GUARDRAIL** | Identifies loss-making arrays (58 rows) to prevent distortion in positive value exposure calculations. |
| `profit_signed_log` | `FLOAT` | `sign(profit) * ln(1 + abs(profit))` | **ENGINEERED FEATURE** | Variance-stabilizing transformation for machine learning algorithms handling wide dollar variances. |

---

## 4. Analytical & Financial Output Fields (Final Delivery Preview)

| Field Name | Physical Type | Description & Formal Formula |
| :--- | :--- | :--- |
| `churn_probability` | `NUMBER(6,5)` | Final calibrated annual churn probability ($p$) from chosen model refit on all 1,884 labelled rows. |
| `churn_probability_oof` | `NUMBER(6,5)` | Honest out-of-fold calibrated churn probability from 5-fold repeated CV (available for 1,884 labelled arrays; `NULL` for unlabelled). |
| `probability_bucket` | `VARCHAR` | Empirical risk tier: `Low Risk` ($p < 0.10$), `Medium Risk` ($0.10 \le p < 0.25$), `High Risk` ($p \ge 0.25$). |
| `expansion_opportunity_score_heuristic` | `NUMBER(6,5)` | Normalized heuristic score $\in [0, 1]$ combining expansion signal, over-capacity status, and capacity pressure. Strictly labeled as heuristic. |
| `survival_5y_constant`, `10y`, `15y` | `NUMBER(6,5)` | 15-year survival scenario under constant churn: $S(t) = (1 - p)^t$. |
| `survival_5y_fading`, `10y`, `15y` | `NUMBER(6,5)` | 15-year survival scenario under fading churn: $S(1) = 1 - p$; $S(t) = (1 - p)(1 - p/2)^{t-1}$. |
| `stated_profit_usd` | `NUMBER(18,2)` | Raw stated 15-year profit (`historical_15yr_net_profit_usd`). |
| `retention_adjusted_clv_constant` | `NUMBER(18,2)` | Stated profit adjusted by retention factor ratio: $\text{Stated Profit} \times \frac{RF(p)}{RF(q=0.05)}$. |
| `forecast_debt_constant` | `NUMBER(18,2)` | Economic bias in stated value under constant churn: $\text{Stated Profit} - \text{Retention-Adjusted CLV}_{\text{constant}}$. |
| `value_exposed` | `NUMBER(18,2)` | Commercial prioritization metric: $p \times \max(\text{Stated Profit}, 0)$. Zero for negative-profit arrays. |
| `top_driver_1, 2, 3` | `VARCHAR` | Three highest-magnitude explanatory features driving array risk score. |
| `driver_method` | `VARCHAR` | Attribution method (`TreeSHAP` for Tree models; `Standardized Log-Odds Contribution` for Logistic Regression). |
| `driver_detail` | `VARIANT` / `JSON` | Complete JSON breakdown of all feature contributions and raw feature values. |
| `model_version` | `VARCHAR` | Model artifact version identifier (`v1.0.0`). |
| `run_id` | `VARCHAR` | Execution batch identifier (`RUN_20261010_01`). |
| `scored_at` | `TIMESTAMP_NTZ` | Scoring execution timestamp. |
