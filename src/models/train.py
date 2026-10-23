"""Stage 6: fit the chosen ranker (text + history, LightGBM) and the readable logistic model on every labelled
weekly snapshot, and save them with the learned spelling fixes to models/. The API serves this bundle.
Scores in the README come from the time-based test in src/pipeline.py, not from this model."""
import json
from datetime import date

import joblib
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import learn_typo_fixes
from src.features.snapshots import build_snapshots
from src.models.evaluate import HORIZON
from src.models.rankers import TextLightGBM, TextLogReg
from src.utils.paths import MODELS

MODEL_FILE = MODELS / "breakdown_ranker.joblib"


def train_and_save():
    reg, logs, _, _ = load_data()
    fixes = learn_typo_fixes(logs.text)
    snap = build_snapshots(logs, reg, horizon=HORIZON, fixes=fixes)
    snap = snap[snap.snapshot + pd.Timedelta(days=HORIZON) <= pd.to_datetime(logs.date).max()]
    bundle = dict(ranker=TextLightGBM().fit(snap), explainer=TextLogReg().fit(snap), fixes=fixes)
    joblib.dump(bundle, MODEL_FILE, compress=3)
    (MODELS / "model_card.json").write_text(json.dumps({
        "model": "TF-IDF (in-entry word pairs) -> SVD + history features -> LightGBM; logistic model kept for evidence",
        "trained_on": f"{len(snap)} weekly asset snapshots from synthetic logs",
        "output": "risk score for ranking assets (not a calibrated probability)",
        "trained": date.today().isoformat(),
        "note": "Synthetic data only. Retrain and re-validate on the plant's own history before any use.",
    }, indent=2))
    return MODEL_FILE
