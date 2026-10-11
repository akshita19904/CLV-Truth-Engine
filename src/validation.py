"""CLV Truth Engine - Grouped Validation & Evaluation Module

Provides leakage-safe grouped cross-validation splitting on account_id
and comprehensive probability evaluation metrics.
"""

from typing import Generator
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    log_loss,
    brier_score_loss,
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def get_grouped_cv(n_splits: int = 5) -> GroupKFold:
    """Returns GroupKFold generator guaranteeing all arrays for an account stay in one fold."""
    return GroupKFold(n_splits=n_splits)


def verify_grouped_splits_isolation(groups: pd.Series, splits: list[tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """Verifies that no account appears in both train and validation sets across any fold."""
    fold_audit = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits, 1):
        train_accounts = set(groups.iloc[train_idx])
        val_accounts = set(groups.iloc[val_idx])
        overlap = train_accounts.intersection(val_accounts)
        
        if len(overlap) > 0:
            raise ValueError(f"Data leakage detected in Fold {fold_idx}: {len(overlap)} overlapping accounts!")
            
        fold_audit.append({
            "fold": fold_idx,
            "train_rows": len(train_idx),
            "val_rows": len(val_idx),
            "train_accounts": len(train_accounts),
            "val_accounts": len(val_accounts),
            "overlap_accounts": len(overlap),
        })
        
    return fold_audit


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes full suite of classification and probability calibration metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    
    # Clip probabilities for stable log loss computation
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    
    # Probability quality metrics (primary)
    loss = float(log_loss(y_true, y_prob_clipped))
    brier = float(brier_score_loss(y_true, y_prob))
    
    # Discrimination metrics (secondary)
    # Check if predictions are constant
    is_constant = (np.min(y_prob) == np.max(y_prob))
    if is_constant or len(np.unique(y_true)) < 2:
        roc_auc = 0.5  # Constant ranking cannot distinguish classes
        pr_auc = float(np.mean(y_true))  # Baseline PR-AUC equals positive prevalence
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        
    # Hard classification metrics at decision threshold
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    
    return {
        "log_loss": round(loss, 4),
        "brier_score": round(brier, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


if __name__ == "__main__":
    y_test = np.array([0, 0, 1, 1, 0])
    p_test = np.array([0.1, 0.2, 0.8, 0.7, 0.3])
    print("Test evaluation:", evaluate_predictions(y_test, p_test))
