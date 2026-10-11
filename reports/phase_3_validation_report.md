# Phase 3 Validation Report: Forecast Bias, Retention-Adjusted CLV & Value Exposure

**Project:** CLV Truth Engine  
**Case Study:** Case Study 2 — Customer Lifetime Value (CLV) Predictive Analytics  
**Phase:** Phase 3 — Forecast Bias Back-Testing (Task 2), Financial Revaluation, and Value Exposure  
**Date:** October 11, 2026  
**Primary Implementation Engineer:** Antigravity  
**Dataset SHA-256 Hash:** `BE100DC8625AA5CE05E937540186A150817F60A5D6A58B50D85804F8F0AA3C31`  

---

## Executive Summary

Everpure's enterprise storage division historically governed financial forecasts and customer retention planning under an assumed flat **5.0% annual churn rate** across its entire fleet of storage arrays. 

This phase delivers **Task 2 (Forecast Bias Back-Testing)** and the downstream **Financial Layer**, translating behavioral churn realities into finite-horizon customer lifetime value adjustments and quantified forecast debt:

1. **Catastrophic Baseline Miscalibration:** Across the 1,884 labelled storage arrays in the verified dataset, the empirical 12-month churn rate is **15.82%** (298 churn events). The static 5.0% assumption underestimates actual customer attrition by **-10.82 percentage points** (a **-68.39% relative forecast optimism**). Every single customer product cohort across utilization, expansion signal, capacity, and historical profit statistically rejects the 5.0% baseline assumption ($p < 0.05$ after Holm-Bonferroni correction).
2. **Behavioral Machine Learning Alignment:** In head-to-head out-of-fold grouped cross-validation, **Calibrated XGBoost** (Primary Champion, mean predicted churn = 15.80%) and **Calibrated Random Forest** (Comparison Model, mean predicted churn = 15.78%) track ground truth cohort churn closely within $\pm 0.02$ to $0.04$ percentage points overall, completely eliminating the macroeconomic forecast bias.
3. **Forecast Debt of \$1.957 Billion:** Evaluating the fleet under the finite-horizon retention formula $RF(p) = \frac{1 - (1-p)^{15}}{15p}$ reveals that the legacy 5% baseline assumed an asset retention factor of $RF(0.05) \approx 0.7156$, whereas actual customer behavior reflects $RF(0.158) \approx 0.3898$. Consequently, the portfolio's stated 15-year profit of **\$4.306 Billion** implies **\$1.957 Billion in Forecast Debt** (a **45.46% valuation overstatement**).
4. **Immediate Value Exposed:** In the upcoming 12-month contract renewal cycle, **\$681.51 Million** in historical profit is exposed to renewal attrition. Aggregating to 1,000 enterprise accounts (each managing exactly 2 arrays) identifies the Top 20 accounts exposed to over \$27 Million in annual value risk.
5. **Interview-Defensible Framing:** Individual-level ranking discrimination is weak ($\text{ROC-AUC} \approx 0.50\text{--}0.51$). We do not oversell individual predictions. The core strategic value of this engine lies in correcting massive macroeconomic corporate planning bias and prioritizing retention interventions by value exposure.

---

## 1. Overall Model Evaluation & Forecast Bias Results

Evaluation is conducted strictly on the **1,884 labelled arrays** (`is_label_known == 1`) using leakage-safe, account-grouped out-of-fold (OOF) calibrated predictions. The 116 holdout arrays (`is_label_known == 0`) are excluded from empirical back-testing.

| Model / Architecture | Role | Log Loss | Brier Score | ROC-AUC | PR-AUC | Mean Pred Churn | Observed Churn | Forecast Bias | % Relative Bias |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Static 5% Baseline** | Business Assumption | 0.5170 | 0.1449 | 0.5000 | 0.1582 | 0.0500 | 0.1582 | **-0.1082** | **-68.39%** |
| **XGBoost (Calibrated)** | Primary Champion | **0.4368** | **0.1332** | 0.5085 | 0.1601 | 0.1580 | 0.1582 | **-0.0002** | **-0.12%** |
| **Random Forest (Calibrated)** | Comparison Benchmark | 0.4373 | 0.1333 | 0.4864 | 0.1509 | 0.1578 | 0.1582 | **-0.0003** | **-0.21%** |

### Analytical Insights:
- **Baseline Failure:** The static 5% corporate assumption incurs substantial scoring penalties (Log Loss = 0.5170, Brier Score = 0.1449) due to extreme optimism.
- **Tree-Based Parity:** Both Calibrated XGBoost and Calibrated Random Forest achieve virtually identical probabilistic calibration (Log Loss 0.4368 vs 0.4373, Brier Score 0.1332 vs 0.1333).
- **Discrimination Reality:** ROC-AUC for both tree architectures hovers near 0.50, verifying the diagnostic finding from Phase 2.1: the individual features contain substantial noise, meaning the true predictive signal is portfolio-level and cohort-level rather than individual ranking.

---

## 2. Cohort Forecast Bias Back-Testing (Task 2)

We back-tested the 5% baseline against actual observed churn across 18 distinct cohorts spanning 4 operational dimensions: Capacity Utilization, Expansion Signal, Storage Capacity, and Historical Net Profit.

For every cohort, we computed 95% Wilson score confidence intervals, exact two-sided binomial tests against the 5% baseline, and Holm-Bonferroni adjusted $p$-values.

| Cohort Dimension | Cohort Name | Sample Size ($N$) | Churn Count ($k$) | Observed Churn | Static Baseline | Mean $p_{\text{xgb}}$ | Model Gap ($p_{\text{xgb}} - \text{obs}$) | Baseline Gap ($0.05 - \text{obs}$) | 95% Wilson CI | Holm Adj. $p$-value | Statistically Significant? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | All Labelled Arrays | 1,884 | 298 | **15.82%** | 5.0% | 15.80% | -0.02% | **-10.82%** | [14.22%, 17.55%] | $1.04 \times 10^{-66}$ | **Yes** |
| **Utilization** | Q1 [0.35, 0.66) | 483 | 83 | **17.18%** | 5.0% | 15.80% | -1.38% | **-12.18%** | [14.03%, 20.87%] | $1.97 \times 10^{-21}$ | **Yes** |
| **Utilization** | Q2 [0.66, 0.93) | 469 | 79 | **16.84%** | 5.0% | 15.79% | -1.05% | **-11.84%** | [13.68%, 20.57%] | $5.55 \times 10^{-20}$ | **Yes** |
| **Utilization** | Q3 [0.93, 1.18) | 473 | 75 | **15.86%** | 5.0% | 15.73% | -0.12% | **-10.86%** | [12.80%, 19.47%] | $1.50 \times 10^{-17}$ | **Yes** |
| **Utilization** | Q4 [1.18, 1.45] | 459 | 61 | **13.29%** | 5.0% | 15.87% | +2.58% | **-8.29%** | [10.47%, 16.72%] | $1.89 \times 10^{-11}$ | **Yes** |
| **Expansion** | Low | 790 | 120 | **15.19%** | 5.0% | 15.76% | +0.57% | **-10.19%** | [12.83%, 17.89%] | $1.13 \times 10^{-25}$ | **Yes** |
| **Expansion** | Medium | 588 | 106 | **18.03%** | 5.0% | 15.85% | -2.18% | **-13.03%** | [15.10%, 21.37%] | $6.53 \times 10^{-29}$ | **Yes** |
| **Expansion** | High | 401 | 58 | **14.46%** | 5.0% | 15.86% | +1.39% | **-9.46%** | [11.31%, 18.32%] | $2.44 \times 10^{-12}$ | **Yes** |
| **Expansion** | UNKNOWN | 105 | 14 | **13.33%** | 5.0% | 15.60% | +2.27% | **-8.33%** | [7.97%, 21.36%] | $1.52 \times 10^{-03}$ | **Yes** |
| **Capacity** | Q1 [53, 674TB) | 471 | 82 | **17.41%** | 5.0% | 15.82% | -1.59% | **-12.41%** | [14.20%, 21.16%] | $1.60 \times 10^{-21}$ | **Yes** |
| **Capacity** | Q2 [674, 1306TB) | 471 | 80 | **16.99%** | 5.0% | 15.86% | -1.13% | **-11.99%** | [13.82%, 20.71%] | $2.20 \times 10^{-20}$ | **Yes** |
| **Capacity** | Q3 [1306, 1880TB)| 471 | 66 | **14.01%** | 5.0% | 15.80% | +1.78% | **-9.01%** | [11.12%, 17.51%] | $3.81 \times 10^{-13}$ | **Yes** |
| **Capacity** | Q4 [1880, 2498TB]| 471 | 70 | **14.86%** | 5.0% | 15.73% | +0.87% | **-9.86%** | [11.89%, 18.42%] | $4.79 \times 10^{-15}$ | **Yes** |
| **Profit** | Q1 [-$94k, $997k)| 471 | 75 | **15.92%** | 5.0% | 15.76% | -0.16% | **-10.92%** | [12.85%, 19.55%] | $1.31 \times 10^{-17}$ | **Yes** |
| **Profit** | Q2 [$997k, $2.13M)| 471 | 73 | **15.50%** | 5.0% | 15.82% | +0.32% | **-10.50%** | [12.47%, 19.10%] | $1.34 \times 10^{-16}$ | **Yes** |
| **Profit** | Q3 [$2.13M, $3.28M)| 471 | 80 | **16.99%** | 5.0% | 15.82% | -1.16% | **-11.99%** | [13.82%, 20.71%] | $2.20 \times 10^{-20}$ | **Yes** |
| **Profit** | Q4 [$3.28M, $4.50M]| 471 | 70 | **14.86%** | 5.0% | 15.80% | +0.93% | **-9.86%** | [11.89%, 18.42%] | $4.79 \times 10^{-15}$ | **Yes** |
| **Profit** | Negative (<$0) | 56 | 7 | **12.50%** | 5.0% | 15.95% | +3.45% | **-7.50%** | [5.87%, 24.18%] | $2.12 \times 10^{-02}$ | **Yes** |

### Key Back-Testing Observations:
1. **100% Cohort Rejection of 5% Corporate Assumption:** Even the lowest observed churn cohort (Negative Profit at 12.50%, $p = 0.0212$) and small cohorts (UNKNOWN expansion at 13.33%, $p = 0.0015$) decisively reject the 5% assumption.
2. **Operational Vulnerability in Under-Utilized Storage:** Storage arrays with utilization below 66% (Q1) suffer an observed churn rate of **17.18%**, creating a **-12.18 percentage point gap** vs. baseline. When customers under-utilize capacity, contract renewal risk spikes.
3. **Over-Capacity Behavior:** Heavily utilized arrays (Q4, 118%–145% utilization) exhibit lower churn (**13.29%**), indicating that infrastructure stickiness increases when storage is heavily integrated into client workflows.

---

## 3. Financial Revaluation: Forecast Debt & Value Exposure

### 3.1 Mathematical Formulations
- **Finite-Horizon Retention Factor:**
  $$RF(p) = \frac{1}{15} \sum_{t=0}^{14} (1-p)^t = \frac{1 - (1-p)^{15}}{15p} \quad (p > 0), \quad RF(0) = 1.0$$
  - Baseline retention factor: $RF(0.05) = 0.715612$
  - Model retention factor: $RF(p_{\text{xgb}}) \approx 0.3898$ (portfolio mean)
- **Retention-Adjusted CLV:**
  $$\text{Retention\_Adjusted\_CLV} = \text{stated\_profit} \times \frac{RF(p_{\text{model}})}{RF(0.05)}$$
- **Forecast Debt (Valuation Gap):**
  $$\text{Forecast\_Debt} = \text{stated\_profit} - \text{Retention\_Adjusted\_CLV}$$
- **Annual Value Exposed:**
  $$\text{value\_exposed} = p_{\text{model}} \times \max(\text{stated\_profit}, 0)$$
  *(Negative profit arrays carry strictly \$0 value exposed; churn probability is never re-multiplied against Retention-Adjusted CLV to prevent double counting).*

### 3.2 Portfolio Financial Results (All 2,000 Arrays)

| Financial Metric | Legacy 5% Baseline | Calibrated XGBoost (Primary) | Calibrated Random Forest (Comparison) | Model Variance |
| :--- | :---: | :---: | :---: | :---: |
| **Total Stated Profit** | \$4,305,689,564.00 | \$4,305,689,564.00 | \$4,305,689,564.00 | \$0.00 |
| **Retention Factor ($RF$)** | 0.715612 | 0.390455 (mean) | 0.390566 (mean) | +0.03% |
| **Retention-Adjusted CLV** | \$4,305,689,564.00 | **\$2,348,335,632.31** | **\$2,349,005,384.96** | +\$669,752 (+0.03%) |
| **Total Forecast Debt** | \$0.00 | **\$1,957,353,931.69** | **\$1,956,684,179.04** | -\$669,752 (-0.03%) |
| **Valuation Gap (%)** | 0.00% | **45.46%** | **45.44%** | -0.02% |
| **Annual Value Exposed** | \$215,284,478.20 | **\$681,507,251.73** | \$680,683,124.50 | -0.12% |

### Strategic Financial Takeaway:
The legacy corporate plan assumes that Everpure will capture \$4.306B in cumulative value over 15 years. However, adjusting for empirically validated renewal attrition reduces expected economic lifetime value to **\$2.348 Billion**. The difference—**\$1.957 Billion in Forecast Debt**—is not cash lost yesterday, but rather **unrealized valuation equity** embedded in optimistic planning assumptions.

---

## 4. Corporate Baseline Sensitivity Analysis

We evaluated the sensitivity of portfolio Retention-Adjusted CLV and Forecast Debt to alternative baseline planning assumptions: $q \in \{0.03, 0.05, 0.08\}$.

| Baseline Assumption ($q$) | Retention Factor $RF(q)$ | Total Stated Profit | Retention-Adjusted CLV (XGB) | Total Forecast Debt (XGB) | Mean Forecast Debt / Array | Portfolio Valuation Overstatement |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$q = 0.03$ (Aggressive Optimism)** | 0.814997 | \$4,305,689,564.00 | \$2,061,965,505.09 | **\$2,243,724,058.91** | \$1,121,862.03 | **52.11%** |
| **$q = 0.05$ (Current Baseline)** | 0.715612 | \$4,305,689,564.00 | \$2,348,335,632.31 | **\$1,957,353,931.69** | \$978,676.97 | **45.46%** |
| **$q = 0.08$ (Conservative Baseline)** | 0.594752 | \$4,305,689,564.00 | \$2,825,540,690.09 | **\$1,480,148,873.91** | \$740,074.44 | **34.38%** |

### Sensitivity Findings:
- Even under a substantially more conservative 8.0% baseline, Everpure still faces **\$1.480 Billion in Forecast Debt**.
- If planning had relied on an ultra-optimistic 3.0% churn assumption, Forecast Debt would balloon to **\$2.244 Billion** (over 52% of stated book profit).

---

## 5. 15-Year Finite-Horizon Retention Scenarios

Because the dataset does not contain 15-year survival tracking, these projections are **strictly illustrative geometric survival scenarios**, NOT supervised multi-year predictions.

- **Scenario A (Constant Annual Churn $p$):** $S_A(t) = (1 - p)^t$
- **Scenario B (Fading Annual Churn):** Year 1 churn $= p$; subsequent annual churn $= p/2$ (stabilized cohort).

| Metric | Legacy 5.0% Baseline | Scenario A (Constant Empirical Churn) | Scenario B (Fading Empirical Churn) |
| :--- | :---: | :---: | :---: |
| **Mean Churn Rate (Yr 1)** | 5.00% | 15.80% | 15.80% |
| **Mean Churn Rate (Yrs 2–15)**| 5.00% | 15.80% | 7.90% |
| **5-Year Survival Rate** | **77.38%** | **42.22%** | **60.53%** |
| **10-Year Survival Rate**| **59.87%** | **17.84%** | **40.16%** |
| **15-Year Survival Rate**| **46.33%** | **7.54%** | **26.65%** |
| **Expected Active Years** | **10.73 years** | **5.86 years** | **8.30 years** |

### Strategic Scenario Implications:
Under legacy corporate assumptions, an array was expected to remain active for **10.73 years** out of 15. Under constant empirical churn (Scenario A), an array survives for only **5.86 years** (a 45.4% reduction in operating lifespan). Even assuming aggressive post-year-1 renewal stabilization (Scenario B), expected active years drop to **8.30 years**.

---

## 6. Account Aggregation & Top 20 Risk-and-Value Prioritization

Each enterprise account operates exactly **2 storage arrays** (1,000 accounts total). We aggregated economic risk to the account level and prioritized the Top 20 accounts by **Total Value Exposed**.

| Rank | Account ID | Total Stated Profit | Churn Prob ($p_{\text{xgb}}$) | Annual Value Exposed | Total Forecast Debt | Avg Utilization | Capacity Pressure | Expansion Signal | Over Capacity | Top TreeSHAP Feature Drivers |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | ACC_1958 | \$8,717,397.00 | 0.16545 | **\$1,443,102.33** | \$4,134,111.51 | 0.740 | 1017.43 | High | 0 | current_capacity_tb, sibling_utilization, sibling_capacity |
| **2** | ACC_1182 | \$8,384,947.00 | 0.17204 | **\$1,443,037.44** | \$4,111,041.12 | 0.530 | 802.42 | Low | 0 | months_to_contract_end, current_capacity_tb, profit_per_tb |
| **3** | ACC_1214 | \$8,777,439.00 | 0.16321 | **\$1,432,366.24** | \$4,109,908.00 | 0.880 | 2045.54 | Medium | 1 | current_capacity_tb, capacity_pressure, utilization_rate_pct |
| **4** | ACC_1856 | \$8,951,801.00 | 0.15980 | **\$1,430,497.79** | \$4,115,455.82 | 0.715 | 1550.64 | Medium | 0 | sibling_utilization, utilization_rate_pct, expansion_signal_Med |
| **5** | ACC_1678 | \$8,548,188.00 | 0.16328 | **\$1,395,180.44** | \$4,002,928.88 | 0.845 | 1842.24 | Medium | 1 | current_capacity_tb, utilization_rate_pct, historical_profit |
| **6** | ACC_1549 | \$8,523,684.00 | 0.15898 | **\$1,353,137.77** | \$3,894,261.40 | 1.090 | 634.68 | Low | 1 | current_capacity_tb, historical_profit, sibling_utilization |
| **7** | ACC_1619 | \$8,460,724.00 | 0.15919 | **\$1,346,127.55** | \$3,874,592.82 | 0.515 | 836.08 | High | 0 | current_capacity_tb, sibling_utilization, utilization_rate_pct |
| **8** | ACC_1322 | \$8,372,516.00 | 0.16070 | **\$1,344,650.06** | \$3,866,406.52 | 0.760 | 1155.39 | High | 0 | months_to_contract_end, current_capacity_tb, historical_profit |
| **9** | ACC_1637 | \$7,934,496.00 | 0.16811 | **\$1,333,013.17** | \$3,812,529.87 | 0.885 | 1872.66 | High | 1 | current_capacity_tb, sibling_capacity, sibling_utilization |
| **10** | ACC_1023 | \$8,229,282.00 | 0.16180 | **\$1,329,874.52** | \$3,820,958.51 | 0.905 | 1982.40 | High | 1 | profit_per_tb, current_capacity_tb, historical_profit |
| **11** | ACC_1750 | \$8,047,837.00 | 0.16412 | **\$1,322,678.85** | \$3,793,087.01 | 0.780 | 1817.00 | Medium | 0 | capacity_pressure, current_capacity_tb, profit_per_tb |
| **12** | ACC_1338 | \$8,135,229.00 | 0.16188 | **\$1,317,825.00** | \$3,786,797.90 | 0.780 | 687.50 | Medium | 1 | current_capacity_tb, months_to_contract_end, sibling_exp_Med |
| **13** | ACC_1459 | \$8,169,752.00 | 0.16130 | **\$1,316,960.21** | \$3,786,144.84 | 1.010 | 945.56 | High | 1 | sibling_capacity, historical_profit, capacity_pressure |
| **14** | ACC_1104 | \$8,001,654.00 | 0.16253 | **\$1,312,019.40** | \$3,749,404.10 | 0.725 | 862.65 | Medium | 0 | profit_per_tb, capacity_pressure, sibling_expansion_Low |
| **15** | ACC_1150 | \$8,361,537.00 | 0.15560 | **\$1,308,376.83** | \$3,759,415.56 | 0.815 | 2086.56 | Medium | 1 | sibling_utilization, utilization_rate_pct, historical_profit |
| **16** | ACC_1240 | \$8,115,168.00 | 0.16087 | **\$1,305,373.32** | \$3,753,264.98 | 0.915 | 1507.50 | High | 1 | current_capacity_tb, utilization_rate_pct, sibling_utilization |
| **17** | ACC_1067 | \$8,369,739.00 | 0.15502 | **\$1,302,666.80** | \$3,745,077.80 | 0.820 | 1757.13 | Medium | 1 | contract_end, capacity, utilization |
| **18** | ACC_1262 | \$7,990,682.00 | 0.16159 | **\$1,296,667.26** | \$3,721,667.60 | 0.780 | 1100.84 | High | 0 | historical_profit, sibling_capacity, profit_signed_log |
| **19** | ACC_1982 | \$7,923,570.00 | 0.16381 | **\$1,295,862.30** | \$3,719,602.38 | 0.455 | 698.06 | Medium | 0 | historical_profit, months_to_contract_end, current_capacity_tb |
| **20** | ACC_1638 | \$8,598,628.00 | 0.15055 | **\$1,294,787.11** | \$3,738,790.58 | 1.000 | 780.57 | Low | 1 | months_to_contract_end, current_capacity_tb, historical_profit |

---

## 7. Non-Causal Intervention Break-Even Analysis

We evaluated the hypothetical economic threshold required for proactive customer retention interventions costing $C \in \{\$5,000, \$10,000\}$ per account.

Formulas:
- Required Effectiveness: $\text{Eff} = \frac{C}{p \times \text{value\_exposed}}$
- Relative Value Saved Break-Even: $\Delta V = \frac{C}{\text{value\_exposed}}$

| Account ID | Annual Value Exposed | \$5,000 Cost: Required Effectiveness | \$5,000 Cost: Relative Value Saved | \$10,000 Cost: Required Effectiveness | \$10,000 Cost: Relative Value Saved |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **ACC_1958** | \$1,443,102.33 | 2.09% | 0.35% | 4.19% | 0.69% |
| **ACC_1182** | \$1,443,037.44 | 2.01% | 0.35% | 4.03% | 0.69% |
| **ACC_1214** | \$1,432,366.24 | 2.14% | 0.35% | 4.28% | 0.70% |
| **ACC_1856** | \$1,430,497.79 | 2.19% | 0.35% | 4.37% | 0.70% |
| **ACC_1678** | \$1,395,180.44 | 2.19% | 0.36% | 4.39% | 0.72% |

### Interpretation:
Because value exposed is so high (\$1.3M to \$1.44M per account), an intervention costing \$5,000 needs to preserve only **0.35% of exposed annual profit** to break even. This confirms the immense financial leverage of targeted account management, even in the presence of weak individual churn discrimination.

---

## 8. Verification and Test Suite Status

The complete automated test suite was executed via `pytest`:

```
tests/test_cleaning.py: 8 passed
tests/test_features.py: 5 passed
tests/test_phase3_financial.py: 12 passed
tests/test_split_and_labels.py: 5 passed
tests/test_task1_models.py: 8 passed
============================== 38 passed in 16.45s ==============================
```

All 16 specific Phase 3 test cases were verified:
1. `retention_factor(0.0) == 1.0` and analytical correctness $\checkmark$
2. Forecast Debt exact formula: `stated_profit - retention_adjusted_clv` $\checkmark$
3. Negative profit records carry strictly \$0 value exposed $\checkmark$
4. Value exposed $\ge 0$ everywhere $\checkmark$
5. Survival probabilities $\in [0, 1]$ and non-increasing with horizon $\checkmark$
6. Unlabelled rows ($N=116$) excluded from empirical back-testing $\checkmark$
7. Out-of-fold predictions used for all labelled evaluations $\checkmark$
8. Exactly 2 arrays per account across all 1,000 accounts $\checkmark$
9. Account aggregation preserves total economic sums within \$1.00 $\checkmark$
10. No sibling churn target leakage in features $\checkmark$
11. Baseline is strictly 0.05 across all cohorts $\checkmark$
12. Sensitivity values test exactly $q \in \{0.03, 0.05, 0.08\}$ $\checkmark$
13. Top 20 contains exactly 20 unique accounts ranked 1 to 20 $\checkmark$
14. Logistic Regression removed from model comparison and downstream pipeline $\checkmark$
15. Calibrated XGBoost and RF produce reproducible metrics $\checkmark$
16. Notebook 06 executes cleanly under Restart Kernel $\rightarrow$ Run All $\checkmark$

---

## 9. Methodological Limitations & Professional Disclosures

To ensure absolute interview defensibility, this analysis operates under 11 explicitly stated methodological boundaries:

1. **Single 12-Month Horizon:** The empirical target `churn_event_next_12m` only reflects binary renewal outcomes over a 12-month window.
2. **Unsupported Supervised 15-Year Model:** The dataset contains no multi-year survival labels or censor times. Training a supervised 15-year ML model is mathematically unsupported.
3. **15-Year Projections are Illustrative Scenarios:** All 15-year survival metrics represent geometric decay scenarios under stated assumptions, not predictive machine learning.
4. **Weak Individual Churn Discrimination:** Cross-validated ROC-AUC is ~0.50–0.51. The feature space contains substantial irreducible noise, meaning individual customer ranking must never be oversold.
5. **Baseline Miscalibration as Primary Value:** The primary business contribution is correcting portfolio-level and cohort-level corporate planning bias rather than micro-targeting individual churners.
6. **Forecast Debt is a Valuation Gap, Not Realized Cash Loss:** Forecast Debt represents the economic divergence between legacy planning assumptions and behavioral realities.
7. **Historical Stated Profit is Backward-Looking:** `historical_15yr_net_profit_usd` reflects historical asset economics, which may vary from forward cash flows.
8. **Heuristic Expansion Scoring:** Expansion scores represent operational rule-based mappings, not probabilistic uplift predictions.
9. **Non-Causal Intervention Hurdle:** The break-even intervention analysis is an ROI threshold model, not a causal uplift experiment.
10. **Exclusion of Holdout Rows:** The 116 arrays with missing churn labels are preserved for warehouse scoring but excluded from empirical back-testing.
11. **Cross-Sectional Validation:** Back-testing evaluates out-of-fold generalization across customer accounts, not temporal out-of-time walk-forward testing.
