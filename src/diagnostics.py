"""Phase 2.1 diagnostic: reduced features, C values, feature-group ablation.

Same framework as Phase 2: 5-fold GroupKFold on account_id, preprocessing inside
the pipeline, Platt calibration trained on inner grouped splits of each training fold.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PATHS
from src.features import (
    ALL_FEATURE_COLS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
    build_feature_matrix, get_modeling_population,
)
from src.validation import evaluate_predictions

SEED = 42

GROUPS = {
    "operational": ["current_capacity_tb", "utilization_capped", "utilization_excess",
                    "over_capacity", "capacity_pressure"],
    "profit": ["profit_signed_log", "profit_per_tb", "negative_profit_flag"],
    "expansion": ["expansion_potential_signal"],
    "sibling": ["sibling_utilization", "sibling_capacity", "sibling_expansion"],
    "contract": ["months_to_contract_end"],
}


def make_pipe(features, C=1.0):
    num = [f for f in features if f in NUMERICAL_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat),
    ])
    return Pipeline([("prep", prep),
                     ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED))])


def run_experiment(X, y, groups, features, C=1.0):
    """Calibrated OOF predictions using the Phase 2 grouped splits."""
    outer = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
    oof = np.zeros(len(y))
    fold_aucs = []
    for tr, va in outer:
        inner = list(GroupKFold(n_splits=3).split(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr]))
        cal = CalibratedClassifierCV(make_pipe(features, C), method="sigmoid", cv=inner)
        cal.fit(X.iloc[tr][features], y.iloc[tr])
        oof[va] = cal.predict_proba(X.iloc[va][features])[:, 1]
        fold_aucs.append(roc_auc_score(y.iloc[va], oof[va]))
    m = evaluate_predictions(y.values, oof)
    m["roc_auc_fold_sd"] = round(float(np.std(fold_aucs, ddof=1)), 4)
    return m


def main():
    df = build_feature_matrix()
    X, y, groups, _ = get_modeling_population(df)
    assert len(X) == 1884

    full = list(ALL_FEATURE_COLS)
    rows = []

    def add(exp, cfg, feats, C=1.0):
        m = run_experiment(X, y, groups, feats, C)
        rows.append({"experiment": exp, "configuration": cfg, "n_features": len(feats),
                     "log_loss": m["log_loss"], "brier_score": m["brier_score"],
                     "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
                     "roc_auc_fold_sd": m["roc_auc_fold_sd"]})
        print(rows[-1], flush=True)

    add("Reference", "Current Calibrated Logistic", full, 1.0)

    reduced = [f for f in full if f not in ("utilization_rate_pct", "historical_15yr_net_profit_usd")]
    add("A: Reduced features", "Drop raw utilization + raw profit", reduced, 1.0)

    for C in [0.01, 0.1, 1, 10]:
        add("B: Regularization", f"Full features, C={C}", full, C)

    add("C: Ablation", "Full", full)
    for g, cols in GROUPS.items():
        add("C: Ablation", f"Full - {g.capitalize()}", [f for f in full if f not in cols])

    simple = GROUPS["operational"] + ["expansion_potential_signal"]
    add("D: Simple", "Operational + expansion only", simple)

    res = pd.DataFrame(rows)
    out = PATHS["figures_dir"].parent / "model_diagnostic_results.csv"
    res.to_csv(out, index=False)
    print("saved", out)
    make_figures(res)
    return res


def make_figures(res):
    fig_dir = PATHS["figures_dir"]
    ref = res.iloc[0]

    # 1. Overall comparison: ROC-AUC, log loss, Brier for every configuration
    sub = res.drop_duplicates("configuration").reset_index(drop=True)
    labels = sub["configuration"].tolist()
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, col, title in zip(axes, ["roc_auc", "log_loss", "brier_score"],
                              ["ROC-AUC (higher is better)", "Log Loss (lower is better)",
                               "Brier Score (lower is better)"]):
        ax.barh(range(len(sub)), sub[col], color="#4c78a8")
        ax.axvline(ref[col], color="red", linestyle="--", label="Phase 2 champion")
        ax.set_title(title)
        ax.set_xlabel(col.replace("_", " "))
        lo, hi = sub[col].min(), sub[col].max()
        pad = max((hi - lo) * 0.5, 0.002)
        ax.set_xlim(lo - pad, hi + pad)
    axes[0].set_yticks(range(len(sub)))
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Phase 2.1 Diagnostic - grouped OOF, calibrated Logistic Regression\n"
                 "(x-axes zoomed; differences are small)")
    plt.tight_layout()
    fig.savefig(fig_dir / "09_diagnostic_comparison.png", dpi=200)
    plt.close(fig)

    # 2. Ablation: change in ROC-AUC vs full, with fold SD as context
    ab = res[res["experiment"] == "C: Ablation"].reset_index(drop=True)
    full_auc = ab.loc[ab["configuration"] == "Full", "roc_auc"].iloc[0]
    ab = ab[ab["configuration"] != "Full"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(ab["configuration"], ab["roc_auc"] - full_auc, color="#f58518")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Change in ROC-AUC vs Full")
    ax.set_title("Feature-group ablation: change in OOF ROC-AUC")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "10_ablation_roc_auc_change.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
