"""Stage 2: load the raw (synthetic) maintenance data."""
import pandas as pd

from src.utils.paths import DATA_RAW, GROUND_TRUTH


def load_data():
    """Returns (asset_register, maintenance_logs, failure_modes, degradation_episodes).
    The last two are ground truth, used only for evaluation."""
    reg = pd.read_csv(DATA_RAW / "asset_register.csv")
    logs = pd.read_csv(DATA_RAW / "maintenance_logs.csv")
    failures = pd.read_csv(GROUND_TRUTH / "failure_modes.csv", parse_dates=["date"])
    episodes = pd.read_csv(GROUND_TRUTH / "degradation_episodes.csv", parse_dates=["start", "end"])
    return reg, logs, failures, episodes
