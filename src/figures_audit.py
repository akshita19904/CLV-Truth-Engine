"""Generates Phase 1 audit and data quality figures."""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.config import PATHS
from src.cleaning import load_raw_data

# Visual styling setup
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

FIG_DIR = PATHS["figures_dir"]
FIG_DIR.mkdir(parents=True, exist_ok=True)


def generate_all_figures():
    df = load_raw_data()
    
    # 1. Churn target distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    churn_counts = df["churn_event_next_12m"].value_counts(dropna=False)
    labels = ["Stayed (0.0)", "Churned (1.0)", "Unlabelled (NaN)"]
    vals = [churn_counts.get(0.0, 0), churn_counts.get(1.0, 0), churn_counts.get(np.nan, 0)]
    colors = ["#2b5c8f", "#d95f02", "#7570b3"]
    
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    ax.set_title("Storage Array 12-Month Churn Target Distribution (N=2,000)")
    ax.set_ylabel("Number of Storage Arrays")
    ax.set_ylim(0, 1850)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_churn_distribution.png", dpi=300)
    plt.close(fig)
    
    # 2. Utilization distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["utilization_rate_pct"], bins=30, kde=True, color="#1f77b4", edgecolor="black", ax=ax)
    ax.axvline(1.0, color="#d62728", linestyle="--", linewidth=2, label="Nominal Capacity Limit (100% / 1.0)")
    ax.axvline(df["utilization_rate_pct"].median(), color="#2ca02c", linestyle=":", linewidth=2, label=f"Median ({df['utilization_rate_pct'].median():.2f})")
    
    n_over = (df["utilization_rate_pct"] > 1.0).sum()
    pct_over = (n_over / len(df)) * 100
    ax.annotate(f"Over-capacity: {n_over} arrays ({pct_over:.1f}%)\nMax: {df['utilization_rate_pct'].max():.2f}",
                xy=(1.05, ax.get_ylim()[1] * 0.75),
                xytext=(1.15, ax.get_ylim()[1] * 0.8),
                arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1, headwidth=6),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffdede", ec="#d62728"))
    
    ax.set_title("Storage Array Utilization Rate Distribution")
    ax.set_xlabel("Utilization Rate (Decimal: 1.0 = 100%)")
    ax.set_ylabel("Array Count")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_utilization_distribution.png", dpi=300)
    plt.close(fig)
    
    # 3. Profit distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    profit_m = df["historical_15yr_net_profit_usd"] / 1e6
    sns.histplot(profit_m, bins=35, kde=True, color="#2b5c8f", edgecolor="black", ax=ax)
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.8, label="Break-Even ($0)")
    ax.axvline(profit_m.median(), color="#2ca02c", linestyle=":", linewidth=2, label=f"Median (${profit_m.median():.2f}M)")
    
    n_neg = (df["historical_15yr_net_profit_usd"] < 0).sum()
    pct_neg = (n_neg / len(df)) * 100
    ax.annotate(f"Negative Profit: {n_neg} arrays ({pct_neg:.1f}%)\nMin: -${abs(df['historical_15yr_net_profit_usd'].min()):,} USD",
                xy=(0.0, ax.get_ylim()[1] * 0.4),
                xytext=(-0.8, ax.get_ylim()[1] * 0.6),
                arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1, headwidth=6),
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#ffdede", ec="#d62728"))
    
    ax.set_title("Stated 15-Year Net Profit Distribution (USD Millions)")
    ax.set_xlabel("Stated 15-Year Net Profit ($ Millions)")
    ax.set_ylabel("Array Count")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_profit_distribution.png", dpi=300)
    plt.close(fig)
    
    # 4. Capacity distribution
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["current_capacity_tb"], bins=30, kde=True, color="#31a354", edgecolor="black", ax=ax)
    ax.axvline(df["current_capacity_tb"].median(), color="#005a32", linestyle="--", linewidth=2, label=f"Median ({df['current_capacity_tb'].median():,.0f} TB)")
    ax.set_title("Storage Array Capacity Distribution (TB)")
    ax.set_xlabel("Provisioned Capacity (Terabytes)")
    ax.set_ylabel("Array Count")
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_capacity_distribution.png", dpi=300)
    plt.close(fig)
    
    # 5. Expansion signal distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    exp_counts = df["expansion_potential_signal"].value_counts()
    colors = ["#74c476", "#31a354", "#006d2c", "#969696"]
    bars = ax.bar(exp_counts.index, exp_counts.values, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Expansion Potential Signal Distribution (N=2,000)")
    ax.set_xlabel("Expansion Category")
    ax.set_ylabel("Array Count")
    ax.set_ylim(0, 1000)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_expansion_signal_distribution.png", dpi=300)
    plt.close(fig)
    
    # 6. Missingness summary
    fig, ax = plt.subplots(figsize=(8, 4))
    miss_series = df.isnull().sum()
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in miss_series.values]
    bars = ax.barh(miss_series.index, miss_series.values, color=colors, height=0.55, edgecolor="black")
    for bar in bars:
        w = bar.get_width()
        if w > 0:
            pct = (w / len(df)) * 100
            ax.annotate(f"{w} missing ({pct:.1f}%)",
                        xy=(w, bar.get_y() + bar.get_height() / 2),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=10, fontweight="bold", color="#d62728")
        else:
            ax.annotate("0 (Complete)",
                        xy=(0, bar.get_y() + bar.get_height() / 2),
                        xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color="#2ca02c")
    ax.set_title("Dataset Missingness Audit across All Features")
    ax.set_xlabel("Number of Missing Values")
    ax.set_xlim(0, 160)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_missingness_summary.png", dpi=300)
    plt.close(fig)
    
    # 7. Observed churn vs 5% baseline
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lab = df[df["churn_event_next_12m"].notnull()]
    obs_rate = lab["churn_event_next_12m"].mean() * 100
    base_rate = 5.0
    
    categories = ["Static 5% Baseline", "Observed Labelled Churn\n(1,884 arrays)"]
    rates = [base_rate, obs_rate]
    bars = ax.bar(categories, rates, color=["#1f77b4", "#d62728"], width=0.45, edgecolor="black", linewidth=0.8)
    
    ax.annotate(f"{base_rate:.1f}%\n(Expected ~94)", xy=(bars[0].get_x() + bars[0].get_width()/2, base_rate),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.annotate(f"{obs_rate:.2f}%\n(Actual: 298)\n+10.82% gap", xy=(bars[1].get_x() + bars[1].get_width()/2, obs_rate),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#d62728")
    
    # Add population bounds band
    min_rate = (298 / 2000) * 100
    max_rate = ((298 + 116) / 2000) * 100
    ax.axhspan(min_rate, max_rate, color="#ff7f0e", alpha=0.18, label=f"Population Bounds [{min_rate:.1f}% - {max_rate:.1f}%]")
    
    ax.set_title("Observed 12-Month Churn vs. Static Baseline Assumption")
    ax.set_ylabel("Annual Churn Rate (%)")
    ax.set_ylim(0, 24)
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_churn_vs_baseline.png", dpi=300)
    plt.close(fig)
    
    print("All 7 figures generated successfully in reports/figures/")


if __name__ == "__main__":
    generate_all_figures()
