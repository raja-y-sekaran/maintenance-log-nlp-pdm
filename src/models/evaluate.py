"""Stage 7: time-based split, the four rankers scored side by side, and plant-level measures
(breakdowns caught on the weekly top-10 list, lead time, false alarms that were real repaired degradations)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.rankers import TextLightGBM, TextLogReg, history_model, keyword_score

HORIZON, TOP_K = 30, 10
VALID_START, TEST_START = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")


def split(snap):
    """Time-based split. Snapshots whose 30-day label window crosses a boundary are dropped from the earlier set."""
    h = pd.Timedelta(days=HORIZON)
    train = snap[snap.snapshot + h <= VALID_START]
    valid = snap[(snap.snapshot >= VALID_START) & (snap.snapshot + h <= TEST_START)]
    train_full = snap[snap.snapshot + h <= TEST_START]
    test = snap[snap.snapshot >= TEST_START]
    return train, valid, train_full, test


def score_all(train, evalset):
    s = {"Keyword count": keyword_score(evalset)}
    s["History only (no text)"] = history_model().fit(train, train.label).predict_proba(evalset)[:, 1]
    lr = TextLogReg().fit(train)
    s["Text + history, logistic"] = lr.predict_proba(evalset)
    s["Text + history, LightGBM"] = TextLightGBM().fit(train).predict_proba(evalset)
    return s, lr


def weekly_topk(df, score, k=TOP_K):
    d = df.assign(score=score)
    d["rank"] = d.groupby("snapshot").score.rank(ascending=False, method="first")
    return d[d["rank"] <= k]


def failure_level(test, alerts, failures, episodes):
    """For each breakdown in the test period: was the asset on the weekly top-K list in the 30 days before?"""
    f = failures[(failures.date >= TEST_START) & (failures.date <= test.snapshot.max() + pd.Timedelta(days=HORIZON))]
    rows = []
    for r in f.itertuples():
        a = alerts[(alerts.asset_id == r.asset_id) & (alerts.snapshot <= r.date) &
                   (alerts.snapshot > r.date - pd.Timedelta(days=HORIZON))]
        rows.append(dict(asset_id=r.asset_id, date=r.date, failure_mode=r.failure_mode,
                         warned_in_text=r.failure_mode != "sudden", caught=len(a) > 0,
                         lead_days=(r.date - a.snapshot.min()).days if len(a) else np.nan))
    fl = pd.DataFrame(rows)
    # False alarms: alerts not followed by a breakdown. How many sat inside a degradation that was repaired?
    fixed = episodes[episodes.outcome == "fixed"]
    fa = alerts[alerts.label == 0]
    in_fix = [((fixed.asset_id == x.asset_id) & (fixed.start <= x.snapshot) &
               (fixed.end >= x.snapshot)).any() for x in fa.itertuples()]
    return fl, float(np.mean(in_fix)) if len(fa) else np.nan
