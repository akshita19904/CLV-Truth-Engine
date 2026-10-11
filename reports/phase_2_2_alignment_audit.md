# Phase 2.2 Alignment Audit: Task 1 Specification vs. Repository State

**Project:** CLV Truth Engine — Behavior-Driven Retention-Adjusted CLV and Forecast Bias Engine  
**Phase:** 2.2 — Correcting the Modeling Layer (Task 1 Alignment)  
**Date:** October 2026  
**Status:** Approved Audit & Implementation Plan  

---

## 1. Executive Summary & Context

The Case Study 2 prompt establishes three core tasks:
* **Task 1 — Machine Learning & Feature Pipeline:** Build and evaluate two distinct machine learning models (**XGBoost vs. Random Forest/Survival**) in Python to score 15-year renewal, churn risk, and capacity expansion.
* **Task 2 — Forecast Bias Back-Testing:** Implement a back-testing framework comparing model predictions against static 5% churn baseline assumptions, quantifying systematic optimism/pessimism across product cohorts.
* **Task 3 — Production Warehouse Integration:** Write optimized SQL procedures to write model feature scores, probability buckets, and SHAP explainability metrics directly into Snowflake warehouse tables.

Before proceeding to Phase 3 (which encompasses Task 2 back-testing and financial CLV modeling), this audit conducts a rigorous alignment review of Task 1 to ensure that the modeling foundation strictly conforms to the assessment guidelines, zero-fabrication rules, and empirical realities of the provided dataset.

---

## 2. Review of Existing Repository State (Phases 1, 2, and 2.1)

### 2.1 Phase 1 Deliverables (Data Audit & Cleaning)
* **Dataset Quality:** Verified 2,000 storage array rows across exactly 1,000 corporate accounts (each account possesses exactly 2 arrays).
* **Missing Label Segregation:** Confirmed that 116 arrays (5.8%) lack churn labels (`churn_event_next_12m` is NaN). These rows are flagged with `is_label_known == 0` and are strictly excluded from model training, validation, and calibration. They are preserved intact for holdout scoring.
* **Labelled Cohort:** 1,884 arrays across 995 accounts, exhibiting an empirical 12-month churn rate of 15.82% (298 churn events, 1,586 active renewals).
* **Data Anomaly Remediation:** Negative profit ($-\$93.8\text{K}$ to $+\$4.5\text{M}$) was transformed via signed log and signed indicator; utilization anomalies (up to 145.0%) were capped at 100.0% with an explicit excess utilization indicator.
* **Contract Dates:** Analysis demonstrated that `contract_end_date` is machine-generated across future dates (2026–2029), exhibiting zero predictive correlation with churn ($r = 0.010, p = 0.66$).

### 2.2 Phase 2 Deliverables (Initial Feature Engineering & Grouped CV)
* **Feature Pipeline:** 15 candidate features constructed across hardware telemetry, economic profitability, contract maturity, and sibling array characteristics.
* **Validation Strategy:** Implemented 5-fold `GroupKFold` grouped strictly by `account_id` (1,000 accounts), preventing sibling array leakage between training and validation splits.
* **Initial Models:** Trained Logistic Regression, Random Forest, and XGBoost on out-of-fold folds.
* **Probability Calibration:** Implemented nested inner `GroupKFold` Platt (sigmoid) scaling. Calibrated Logistic Regression achieved Log Loss of 0.4368 and Brier Score of 0.1332, dramatically outperforming the static 5% baseline (Log Loss 0.5170, Brier 0.1449).

### 2.3 Phase 2.1 Deliverables (Controlled Model Diagnostic)
* **Diagnostic Investigation:** Conducted controlled feature ablation, non-linear kernel testing (RBF SVM), and hyperparameter regularizations to establish whether weak individual discrimination (ROC-AUC $\approx 0.51 - 0.52$) stemmed from model misspecification or intrinsic feature limitations.
* **Conclusion:** Proven conclusively that the 15 available tabular features possess near-zero linear and non-linear correlation with individual 12-month churn ($|r| < 0.04$ for all features). Model complexity (Random Forest, XGBoost, RBF SVM) does not improve AUC over simple regularized logistic models.
* **Core Takeaway:** Probability calibration provides massive value at the portfolio/cohort level (calibrating expected loss from 5% to the true 15.82% base rate), but individual-level ranking power is inherently bounded by feature telemetry noise.

---

## 3. Gap Analysis Against Task 1 Requirements

| Task 1 Requirement | Status Prior to Phase 2.2 | Identified Gap / Required Action |
| :--- | :--- | :--- |
| **Model Comparison: XGBoost vs. Random Forest** | Partially implemented in Phase 2 scripts | Needs formal, dedicated head-to-head comparison with full metrics (Log Loss, Brier, ROC-AUC, PR-AUC) and calibration analysis for both tree architectures, reported in `reports/task1_churn_model_comparison.csv`. Logistic Regression retained as benchmark. |
| **15-Year Renewal Modeling** | Conceptual scenarios proposed | **Data reality check required:** Dataset has no contract start date, no event duration, no censoring, and no multi-year renewal history. A genuine survival model cannot be trained without fabricating data (forbidden by Rule 0). Must formally document why survival modeling is unsupported and provide a disciplined retention scenario $S_{15} = (1 - p)^{15}$. |
| **Capacity Expansion Modeling** | Used as input feature | **Empirical check required:** `expansion_potential_signal` is present on 100% of rows (including unlabelled rows). Diagnostic showed features predict High expansion at ROC-AUC = 0.508 (random). Must document that expansion is an upstream business classification rather than an empirically predictable operational outcome, and provide both an un-leaked secondary model probability and business heuristic score. |
| **TreeSHAP Explainability** | Logistic coefficients used as proxy in Phase 2 | Must implement true `shap.TreeExplainer` on the XGBoost model and export granular instance-level SHAP values (`array_id`, `account_id`, `feature`, `feature_value`, `shap_value`, `model_version`) to `reports/task1_shap_values.csv`. |
| **Final Scoring Artifact** | Fragmented across Phase 2 OOF files | Must generate unified `reports/task1_final_scores.csv` covering all arrays with model probabilities, selected probability, scenario renewal score, and expansion score. |
| **Executable Notebook** | `02_features_and_models.ipynb` exists | Build `notebooks/05_final_task1_models.ipynb` following the 12 prescribed analytical sections. |

---

## 4. Phase 2.2 Implementation Plan

1. **Dedicated Task 1 Engine (`src/task1_models.py`):**
   * Re-execute 5-fold `GroupKFold` across 1,884 labelled arrays for XGBoost, Random Forest, and Logistic Regression.
   * Evaluate raw vs. Platt-calibrated variants using inner grouped splits.
   * Compute Log Loss, Brier Score, ROC-AUC, and PR-AUC.
   * Output `reports/task1_churn_model_comparison.csv`.

2. **15-Year Renewal & Survival Feasibility Audit:**
   * Empirically verify the lack of time-to-event fields (`contract_start_date`, historical renewal logs, censoring indicator).
   * Refuse data fabrication.
   * Define and document the Retention-Adjusted 15-year scenario score $S_{15} = (1 - p_{\text{churn}})^{15}$.

3. **Capacity Expansion Evaluation:**
   * Evaluate `expansion_potential_signal` as an independent classification target.
   * Strictly remove `expansion_potential_signal` and `sibling_expansion` from the feature matrix to eliminate target leakage.
   * Record that OOF ROC-AUC is $0.5082$, confirming expansion is a qualitative business classification, not a learnable downstream outcome.
   * Output both `expansion_probability` (from un-leaked ML) and `expansion_score` (from categorical tier mapping).

4. **TreeSHAP Generation:**
   * Compute exact TreeSHAP values for the full dataset using `shap.TreeExplainer(xgb_pipeline.named_steps["clf"])`.
   * Export tidy SHAP table (`reports/task1_shap_values.csv`).

5. **Final Scoring Dataset:**
   * Generate `reports/task1_final_scores.csv` containing all 2,000 arrays (1,884 labelled rows with OOF scores + 116 holdout rows scored with the full refitted model).

6. **Notebook & Test Suite:**
   * Construct and execute `notebooks/05_final_task1_models.ipynb` (12 sections).
   * Implement unit tests in `tests/test_task1_models.py`.
   * Verify entire test suite passes (`pytest`).
   * Update `reports/technical_report.md`.
