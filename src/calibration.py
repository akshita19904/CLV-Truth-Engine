"""CLV Truth Engine - Probability Calibration Module

Applies leakage-safe Platt (sigmoid) scaling within grouped cross-validation,
evaluates probability reliability, and refits the calibrated model on full labelled data.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import GroupKFold

from src.config import PATHS
from src.models import build_model_pipelines
from src.validation import evaluate_predictions, get_grouped_cv


def run_calibrated_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    metadata: pd.DataFrame,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Runs nested grouped cross-validation producing honest calibrated OOF predictions."""
    outer_gkf = get_grouped_cv(n_splits=n_outer_splits)
    outer_splits = list(outer_gkf.split(X, y, groups=groups))
    
    pipelines = build_model_pipelines(random_state=random_state)
    n_samples = len(y)
    
    calibrated_oofs = {}
    for name in pipelines:
        calibrated_oofs[name] = np.zeros(n_samples)
        
    fold_assignments = np.zeros(n_samples, dtype=int)
    
    for fold_idx, (train_idx, val_idx) in enumerate(outer_splits, 1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        X_val = X.iloc[val_idx]
        fold_assignments[val_idx] = fold_idx
        
        # Inner grouped splits for calibration training
        inner_gkf = GroupKFold(n_splits=n_inner_splits)
        inner_splits = list(inner_gkf.split(X_train, y_train, groups=groups_train))
        
        for name, pipe in pipelines.items():
            cal_clf = CalibratedClassifierCV(
                estimator=pipe,
                method="sigmoid",
                cv=inner_splits
            )
            cal_clf.fit(X_train, y_train)
            calibrated_oofs[name][val_idx] = cal_clf.predict_proba(X_val)[:, 1]
            
    # Evaluation metrics
    records = []
    for model_name, preds in calibrated_oofs.items():
        m = evaluate_predictions(y.values, preds)
        m["model"] = f"{model_name} (Calibrated)"
        records.append(m)
        
    cal_metrics_df = pd.DataFrame(records)
    cols = ["model", "log_loss", "brier_score", "roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]
    cal_metrics_df = cal_metrics_df[cols]
    
    # Format calibrated OOF predictions DataFrame
    oof_rows = []
    for model_name, preds in calibrated_oofs.items():
        for i in range(n_samples):
            oof_rows.append({
                "account_id": metadata["account_id"].iloc[i],
                "array_id": metadata["array_id"].iloc[i],
                "y_true": int(y.iloc[i]),
                "oof_pred_probability": round(float(preds[i]), 5),
                "fold": int(fold_assignments[i]),
                "model": f"{model_name} (Calibrated)"
            })
            
    cal_oof_df = pd.DataFrame(oof_rows)
    return cal_metrics_df, cal_oof_df, calibrated_oofs


def plot_calibration_curves(
    y_true: np.ndarray,
    predictions_dict: Dict[str, np.ndarray],
    output_path: Path = None,
    n_bins: int = 5
):
    """Generates a publication-quality reliability diagram comparing probability calibration."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    
    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated", linewidth=1.5)
    
    colors = {
        "5% Baseline": "#7f7f7f",
        "Logistic Regression": "#1f77b4",
        "Logistic Regression (Calibrated)": "#2ca02c",
        "Random Forest": "#ff7f0e",
        "Random Forest (Calibrated)": "#d62728",
        "XGBoost": "#9467bd",
        "XGBoost (Calibrated)": "#8c564b",
    }
    
    for name, probs in predictions_dict.items():
        if name == "5% Baseline":
            continue
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
        color = colors.get(name, None)
        ax.plot(prob_pred, prob_true, "s-", label=name, color=color, linewidth=1.8, markersize=6)
        
    ax.set_title("Reliability Diagram (Probability Calibration Curve)")
    ax.set_xlabel("Mean Predicted Churn Probability")
    ax.set_ylabel("Fraction of True Churn Events")
    ax.set_xlim([0.0, 0.5])
    ax.set_ylim([0.0, 0.5])
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    if output_path is None:
        output_path = PATHS["figures_dir"] / "08_calibration_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Calibration plot saved to: {output_path}")


def fit_final_calibrated_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    selected_model_name: str = "Logistic Regression",
    n_inner_splits: int = 5,
    random_state: int = 42
) -> CalibratedClassifierCV:
    """Refits the selected pipeline with sigmoid calibration on all labelled rows."""
    pipelines = build_model_pipelines(random_state=random_state)
    base_pipe = pipelines[selected_model_name]
    
    gkf = GroupKFold(n_splits=n_inner_splits)
    inner_splits = list(gkf.split(X, y, groups=groups))
    
    calibrated_model = CalibratedClassifierCV(
        estimator=base_pipe,
        method="sigmoid",
        cv=inner_splits
    )
    calibrated_model.fit(X, y)
    return calibrated_model
