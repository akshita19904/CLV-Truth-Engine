"""CLV Truth Engine - Task 1 Machine Learning Models Module

Executes head-to-head evaluation of XGBoost vs. Random Forest for 12-month churn risk,
with Logistic Regression as reference baseline, leakage-safe GroupKFold validation,
Platt probability calibration, TreeSHAP explainability, capacity expansion analysis,
and final score generation across all 2,000 storage arrays.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score, average_precision_score

from src.config import PATHS, config
from src.features import (
    build_feature_matrix,
    get_modeling_population,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_FEATURE_COLS
)
from src.validation import evaluate_predictions, get_grouped_cv


# Separate feature set for Capacity Expansion prediction (strictly excludes expansion itself to avoid leakage)
EXPANSION_FEATURE_COLS = NUMERICAL_FEATURES


def get_preprocessor(numeric_cols: List[str], cat_cols: List[str], scale_numeric: bool = True) -> ColumnTransformer:
    """Creates a ColumnTransformer for the specified numeric and categorical features."""
    transformers = []
    if numeric_cols:
        num_step = StandardScaler() if scale_numeric else "passthrough"
        transformers.append(("num", num_step, numeric_cols))
    if cat_cols:
        cat_step = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        transformers.append(("cat", cat_step, cat_cols))
    return ColumnTransformer(transformers=transformers)


def build_task1_pipelines(random_state: int = 42) -> Dict[str, Pipeline]:
    """Builds standard Pipelines for XGBoost, Random Forest, and Logistic Regression."""
    lr_pipe = Pipeline([
        ("prep", get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, scale_numeric=True)),
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state))
    ])
    
    rf_pipe = Pipeline([
        ("prep", get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    xgb_pipe = Pipeline([
        ("prep", get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False)),
        ("clf", XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state
        ))
    ])
    
    return {
        "XGBoost": xgb_pipe,
        "Random Forest": rf_pipe,
        "Logistic Regression": lr_pipe
    }


def evaluate_raw_and_calibrated_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Evaluates raw and calibrated versions of all models using nested GroupKFold.
    
    Returns:
        comparison_df: DataFrame of performance metrics across all candidate models.
        oof_dict: Dictionary mapping model names to out-of-fold prediction probability vectors.
    """
    outer_gkf = get_grouped_cv(n_splits=n_outer_splits)
    outer_splits = list(outer_gkf.split(X, y, groups=groups))
    
    base_pipelines = build_task1_pipelines(random_state=random_state)
    n_samples = len(y)
    
    # Store out-of-fold predictions
    raw_oofs = {name: np.zeros(n_samples) for name in base_pipelines}
    cal_oofs = {f"{name} (Calibrated)": np.zeros(n_samples) for name in base_pipelines}
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits, 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. Fit raw models
        for name, pipe in base_pipelines.items():
            pipe.fit(X_train, y_train)
            raw_oofs[name][val_idx] = pipe.predict_proba(X_val)[:, 1]
            
        # 2. Fit calibrated models using inner GroupKFold
        inner_gkf = GroupKFold(n_splits=n_inner_splits)
        inner_splits = list(inner_gkf.split(X_train, y_train, groups=groups_train))
        
        for name, pipe in base_pipelines.items():
            cal_clf = CalibratedClassifierCV(
                estimator=pipe,
                method="sigmoid",
                cv=inner_splits
            )
            cal_clf.fit(X_train, y_train)
            cal_oofs[f"{name} (Calibrated)"][val_idx] = cal_clf.predict_proba(X_val)[:, 1]
            
    # Compile metrics
    records = []
    
    # Add static 5% baseline
    baseline_preds = np.full(n_samples, 0.05)
    records.append({
        "model": "Static 5% Baseline",
        "log_loss": round(log_loss(y, baseline_preds), 4),
        "brier_score": round(brier_score_loss(y, baseline_preds), 4),
        "roc_auc": round(roc_auc_score(y, baseline_preds), 4),
        "pr_auc": round(average_precision_score(y, baseline_preds), 4),
        "calibration_status": "Uncalibrated Baseline"
    })
    
    all_oofs = {**raw_oofs, **cal_oofs}
    for model_name, preds in all_oofs.items():
        is_cal = "Calibrated" if "(Calibrated)" in model_name else "Raw"
        records.append({
            "model": model_name,
            "log_loss": round(log_loss(y, preds), 4),
            "brier_score": round(brier_score_loss(y, preds), 4),
            "roc_auc": round(roc_auc_score(y, preds), 4),
            "pr_auc": round(average_precision_score(y, preds), 4),
            "calibration_status": is_cal
        })
        
    comparison_df = pd.DataFrame(records)
    return comparison_df, all_oofs


def evaluate_expansion_prediction(
    X_raw: pd.DataFrame,
    df_clean: pd.DataFrame,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluates whether expansion_potential_signal can be predicted from telemetry features
    without target leakage (strictly excluding expansion signals from input features).
    """
    # Binary expansion outcome: High expansion (1) vs. Other (0)
    y_exp = (df_clean["expansion_potential_signal"] == "High").astype(int)
    
    # Strictly select only non-expansion features
    X_exp = X_raw[EXPANSION_FEATURE_COLS].copy()
    
    exp_pipe = Pipeline([
        ("prep", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, random_state=random_state))
    ])
    
    gkf = get_grouped_cv(n_splits=n_splits)
    n_samples = len(y_exp)
    oof_exp_probs = np.zeros(n_samples)
    
    for train_idx, val_idx in gkf.split(X_exp, y_exp, groups=groups):
        X_train, y_train = X_exp.iloc[train_idx], y_exp.iloc[train_idx]
        X_val = X_exp.iloc[val_idx]
        exp_pipe.fit(X_train, y_train)
        oof_exp_probs[val_idx] = exp_pipe.predict_proba(X_val)[:, 1]
        
    metrics = {
        "log_loss": round(log_loss(y_exp, oof_exp_probs), 4),
        "brier_score": round(brier_score_loss(y_exp, oof_exp_probs), 4),
        "roc_auc": round(roc_auc_score(y_exp, oof_exp_probs), 4),
        "pr_auc": round(average_precision_score(y_exp, oof_exp_probs), 4),
    }
    return metrics, oof_exp_probs


def generate_treeshap_explanations(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    random_state: int = 42,
    model_version: str = "v1.0"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Computes exact TreeSHAP values for XGBoost fitted on labelled data.
    
    Returns:
        shap_long_df: Tidy instance-level SHAP values dataframe.
        shap_summary_df: Feature-level mean absolute SHAP value summary.
    """
    # 1. Fit preprocessor and XGBoost
    preprocessor = get_preprocessor(NUMERICAL_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False)
    X_trans = preprocessor.fit_transform(X)
    
    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names for readability
    clean_feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]
    
    xgb_clf = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state
    )
    xgb_clf.fit(X_trans, y)
    
    # 2. Compute TreeSHAP
    explainer = shap.TreeExplainer(xgb_clf)
    shap_vals = explainer.shap_values(X_trans)
    
    # 3. Create long-format SHAP dataset
    rows = []
    n_samples, n_feats = shap_vals.shape
    for i in range(n_samples):
        arr_id = metadata["array_id"].iloc[i]
        acc_id = metadata["account_id"].iloc[i]
        for j in range(n_feats):
            feat_name = clean_feature_names[j]
            feat_val = float(X_trans[i, j])
            s_val = float(shap_vals[i, j])
            rows.append({
                "array_id": arr_id,
                "account_id": acc_id,
                "feature": feat_name,
                "feature_value": round(feat_val, 4),
                "shap_value": round(s_val, 6),
                "model_version": model_version
            })
            
    shap_long_df = pd.DataFrame(rows)
    
    # 4. Compute mean absolute SHAP per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_summary_df = pd.DataFrame({
        "feature": clean_feature_names,
        "mean_abs_shap": np.round(mean_abs_shap, 6)
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
    
    return shap_long_df, shap_summary_df


def generate_final_scores(
    df_clean: pd.DataFrame,
    X_full: pd.DataFrame,
    labelled_oofs: Dict[str, np.ndarray],
    exp_oof_probs: np.ndarray,
    random_state: int = 42,
    model_version: str = "v1.0-task1",
    scoring_run_id: str = "run_task1_final"
) -> pd.DataFrame:
    """Creates the unified final scoring dataset for all 2,000 arrays.
    
    For labelled rows (1,884), uses genuine out-of-fold probability predictions.
    For unlabelled holdout rows (116), uses models refitted on the full labelled set.
    """
    is_labelled = df_clean["is_label_known"] == 1
    
    # Fit full models for scoring holdout rows
    X_lab = X_full[is_labelled].copy()
    y_lab = df_clean[is_labelled]["churn_event_next_12m"].astype(int)
    groups_lab = df_clean[is_labelled]["account_id"]
    
    pipelines = build_task1_pipelines(random_state=random_state)
    fitted_models = {}
    
    # Fit full calibrated models
    for name, pipe in pipelines.items():
        gkf = GroupKFold(n_splits=5)
        splits = list(gkf.split(X_lab, y_lab, groups=groups_lab))
        cal_clf = CalibratedClassifierCV(estimator=pipe, method="sigmoid", cv=splits)
        cal_clf.fit(X_lab, y_lab)
        fitted_models[name] = cal_clf
        
    # Fit full expansion model (without expansion features)
    exp_pipe = Pipeline([
        ("prep", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, random_state=random_state))
    ])
    y_exp_lab = (df_clean[is_labelled]["expansion_potential_signal"] == "High").astype(int)
    exp_pipe.fit(X_lab[EXPANSION_FEATURE_COLS], y_exp_lab)
    
    # Pre-allocate score columns for all 2,000 rows
    n_total = len(df_clean)
    p_xgb = np.zeros(n_total)
    p_rf = np.zeros(n_total)
    p_lr = np.zeros(n_total)
    p_exp = np.zeros(n_total)
    
    # Fill labelled rows with honest OOF predictions
    p_xgb[is_labelled] = labelled_oofs["XGBoost (Calibrated)"]
    p_rf[is_labelled] = labelled_oofs["Random Forest (Calibrated)"]
    p_lr[is_labelled] = labelled_oofs["Logistic Regression (Calibrated)"]
    p_exp[is_labelled] = exp_oof_probs
    
    # Fill unlabelled holdout rows using full refitted models
    unlabelled_mask = ~is_labelled
    if unlabelled_mask.sum() > 0:
        X_unlab = X_full[unlabelled_mask]
        p_xgb[unlabelled_mask] = fitted_models["XGBoost"].predict_proba(X_unlab)[:, 1]
        p_rf[unlabelled_mask] = fitted_models["Random Forest"].predict_proba(X_unlab)[:, 1]
        p_lr[unlabelled_mask] = fitted_models["Logistic Regression"].predict_proba(X_unlab)[:, 1]
        p_exp[unlabelled_mask] = exp_pipe.predict_proba(X_unlab[EXPANSION_FEATURE_COLS])[:, 1]
        
    # Selected churn probability: Calibrated Logistic Regression (champion on loss & calibration stability)
    # Calibrated XGBoost and RF are also preserved for comparison
    selected_p = p_lr
    
    # Heuristic expansion tier score mapping
    tier_map = {"High": 1.0, "Medium": 0.5, "Low": 0.2, "UNKNOWN": 0.0}
    exp_scores = df_clean["expansion_potential_signal"].map(tier_map).fillna(0.0).values
    
    # 15-year survival / renewal retention scenario score: S_15 = (1 - p_churn)^15
    # Documented strictly as a long-horizon scenario score under geometric decay
    renewal_survival_scores = np.round((1.0 - selected_p) ** 15, 6)
    
    scores_df = pd.DataFrame({
        "array_id": df_clean["array_id"],
        "account_id": df_clean["account_id"],
        "churn_probability_xgb": np.round(p_xgb, 5),
        "churn_probability_rf": np.round(p_rf, 5),
        "churn_probability_logistic": np.round(p_lr, 5),
        "selected_churn_probability": np.round(selected_p, 5),
        "renewal_survival_score": renewal_survival_scores,
        "expansion_score": np.round(exp_scores, 2),
        "expansion_probability": np.round(p_exp, 5),
        "model_version": model_version,
        "scoring_run_id": scoring_run_id
    })
    
    return scores_df
