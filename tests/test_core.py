import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocess import clean, learn_typo_fixes  # noqa: E402
from src.features.snapshots import build_snapshots  # noqa: E402
from src.models.rankers import entry_ngrams, keyword_score  # noqa: E402
from src.models.evaluate import HORIZON, split  # noqa: E402

REG = pd.DataFrame([dict(asset_id="P-1001", asset_type="pump", unit="U10", criticality="A", install_year=2010)])


def _logs(rows):
    return pd.DataFrame([dict(log_id=f"L{i}", date=d, asset_id="P-1001", source="operator_round", wo_type=w, text=t)
                         for i, (d, w, t) in enumerate(rows)])


def test_clean_expands_shorthand_and_drops_tags():
    assert clean("BRG TEMP HI on P-1001A, vib incr") == "bearing temperature high on , vibration increasing"


def test_typo_fixes_learned_from_frequent_spelling():
    texts = ["strainer cleaned"] * 40 + ["strainer claened"]
    assert learn_typo_fixes(texts) == {"claened": "cleaned"}
    assert clean("strainer claened", learn_typo_fixes(texts)) == "strainer cleaned"


def test_bigrams_never_cross_log_entries():
    grams = entry_ngrams("pm completed | painting done")
    assert "completed painting" not in grams and "pm completed" in grams and "painting done" in grams


def test_snapshot_uses_only_past_text_and_labels_future_breakdown():
    logs = _logs([
        ("2020-01-01", "", "routine round, all normal."),
        ("2020-03-01", "", "bearing temperature rising."),
        ("2020-03-02", "", "logged on the snapshot day itself"),
        ("2020-03-20", "breakdown", "Breakdown: bearing failed."),
        ("2020-06-01", "", "routine round, all normal."),
    ])
    snap = build_snapshots(logs, REG, horizon=30, freq="1D", recent=14, window=60)
    row = snap[snap.snapshot == pd.Timestamp("2020-03-02")].iloc[0]
    assert "bearing temperature rising" in row.text_recent
    assert "snapshot day" not in row.text_recent + row.text_older  # strictly before t
    assert "breakdown" not in row.text_recent  # the future breakdown text is not visible
    assert row.label == 1 and row.days_to_breakdown == 18
    after = snap[snap.snapshot == pd.Timestamp("2020-03-21")].iloc[0]
    assert after.label == 0 and after.days_since_breakdown == 1


def test_time_split_has_no_label_window_overlap():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="7D")
    snap = pd.DataFrame(dict(snapshot=dates, label=0))
    train, valid, train_full, test = split(snap)
    h = pd.Timedelta(days=HORIZON)
    assert (train.snapshot + h).max() <= valid.snapshot.min()
    assert (train_full.snapshot + h).max() <= test.snapshot.min()


def test_keyword_score_weights_recent_double():
    df = pd.DataFrame(dict(text_recent=["vibration rising"], text_older=["leak"]))
    assert keyword_score(df).tolist() == [5]
