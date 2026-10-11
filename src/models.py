"""CLV Truth Engine - Supervised Churn Models Module

Defines candidate ML models (Baseline, Logistic Regression, Random Forest, XGBoost)
within leakage-safe scikit-learn Pipelines, executing GroupKFold cross-validation
and generating out-of-fold probability predictions.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from src.features import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.validation import evaluate_predictions, get_grouped_cv


def get_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Creates a ColumnTransformer for numeric and categorical features."""
    num_step = StandardScaler() if scale_numeric else "passthrough"
    cat_step = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    
    return ColumnTransformer(
        transformers=[
            ("num", num_step, NUMERICAL_FEATURES),
            ("cat", cat_step, CATEGORICAL_FEATURES),
        ]
    )


def build_model_pipelines(random_state: int = 42) -> Dict[str, Any]:
    """Instantiates candidate model pipelines with fixed reproducibility settings."""
    # Model 1: Logistic Regression (interpretable benchmark)
    lr_pipe = Pipeline([
        ("prep", get_preprocessor(scale_numeric=True)),
        ("clf", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=random_state))
    ])
    
    # Model 2: Random Forest (non-linear interactions)
    rf_pipe = Pipeline([
        ("prep", get_preprocessor(scale_numeric=False)),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1
        ))
    ])
    
    # Model 3: XGBoost (gradient boosting)
    xgb_pipe = Pipeline([
        ("prep", get_preprocessor(scale_numeric=False)),
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
        "Logistic Regression": lr_pipe,
        "Random Forest": rf_pipe,
        "XGBoost": xgb_pipe,
    }


def run_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Executes 5-fold GroupKFold cross-validation for Baseline and all ML models.
    
    Returns:
        metrics_df (pd.DataFrame): Comparison table of out-of-fold metrics.
        oof_df (pd.DataFrame): Out-of-fold probability predictions for every labelled row.
        raw_oofs (dict): Dictionary mapping model name to raw OOF probability vector.
    """
    gkf = get_grouped_cv(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups=groups))
    
    pipelines = build_model_pipelines(random_state=random_state)
    n_samples = len(y)
    
    # Baseline 5% constant prediction
    oof_predictions = {
        "5% Baseline": np.full(n_samples, 0.05)
    }
    
    # Initialize OOF prediction arrays for ML models
    for name in pipelines:
        oof_predictions[name] = np.zeros(n_samples)
        
    fold_assignments = np.zeros(n_samples, dtype=int)
    
    # Run GroupKFold loop
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        fold_assignments[val_idx] = fold_idx
        
        for name, pipe in pipelines.items():
            pipe.fit(X_train, y_train)
            oof_predictions[name][val_idx] = pipe.predict_proba(X_val)[:, 1]
            
    # Compute metrics for all models
    metrics_records = []
    for model_name, preds in oof_predictions.items():
        m = evaluate_predictions(y.values, preds)
        m["model"] = model_name
        metrics_records.append(m)
        
    metrics_df = pd.DataFrame(metrics_records)
    # Order columns
    cols = ["model", "log_loss", "brier_score", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    metrics_df = metrics_df[cols]
    
    # Build tidy out-of-fold predictions DataFrame
    oof_rows = []
    for model_name, preds in oof_predictions.items():
        for i in range(n_samples):
            oof_rows.append({
                "account_id": metadata["account_id"].iloc[i],
                "array_id": metadata["array_id"].iloc[i],
                "y_true": int(y.iloc[i]),
                "oof_pred_probability": round(float(preds[i]), 5),
                "fold": int(fold_assignments[i]),
                "model": model_name
            })
            
    oof_df = pd.DataFrame(oof_rows)
    return metrics_df, oof_df, oof_predictions


if __name__ == "__main__":
    from src.features import build_feature_matrix, get_modeling_population
    df_feat = build_feature_matrix()
    X, y, groups, meta = get_modeling_population(df_feat)
    metrics_df, oof_df, _ = run_cross_validation(X, y, groups, meta)
    print("Cross-Validation Metrics Table:")
    print(metrics_df.to_string(index=False))
