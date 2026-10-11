import os
from pathlib import Path
import yaml

# Project root directory (CLV-Truth-Engine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load configuration file
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loads configuration settings from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Helper path references
PATHS = {
    "raw_data": PROJECT_ROOT / config["paths"]["raw_data"],
    "cleaned_data": PROJECT_ROOT / config["paths"]["cleaned_data"],
    "model_features": PROJECT_ROOT / config["paths"]["model_features"],
    "scored_output": PROJECT_ROOT / config["paths"]["scored_output"],
    "model_metrics": PROJECT_ROOT / config["paths"]["model_metrics"],
    "cohort_backtest": PROJECT_ROOT / config["paths"]["cohort_backtest"],
    "forecast_debt_summary": PROJECT_ROOT / config["paths"]["forecast_debt_summary"],
    "top_20_accounts": PROJECT_ROOT / config["paths"]["top_20_accounts"],
    "figures_dir": PROJECT_ROOT / config["paths"]["figures_dir"],
    "sql_dir": PROJECT_ROOT / config["paths"]["sql_dir"],
    "reports_dir": PROJECT_ROOT / "reports",
    "processed_data_dir": PROJECT_ROOT / "data" / "processed",
}
