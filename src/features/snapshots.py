"""Turn the maintenance log into weekly asset snapshots: recent text, simple history features, and the label.

Leakage rule: a snapshot taken on day t only uses log entries dated strictly before t.
The label is whether a breakdown work order is raised for that asset in [t, t + horizon).
"""
from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd

from src.data.preprocess import clean, learn_typo_fixes


def build_snapshots(logs: pd.DataFrame, register: pd.DataFrame, horizon: int = 30, freq: str = "7D",
                    recent: int = 14, window: int = 60, vocab_until: str | None = None,
                    fixes: dict | None = None) -> pd.DataFrame:
    """vocab_until: learn spelling fixes only from text before this date (keeps the test years unseen)."""
    logs = logs.copy()
    logs["date"] = pd.to_datetime(logs.date)
    src = logs if vocab_until is None else logs[logs.date < pd.Timestamp(vocab_until)]
    fixes = learn_typo_fixes(src.text) if fixes is None else fixes
    logs["clean"] = logs.text.map(lambda t: clean(t, fixes))
    start, end = logs.date.min() + pd.Timedelta(days=window), logs.date.max() - pd.Timedelta(days=horizon)
    dates = pd.date_range(start, end, freq=freq)
    rows = []
    for a in register.itertuples():
        lg = logs[logs.asset_id == a.asset_id].sort_values("date")
        d = lg.date.values
        txt = lg.clean.values
        bd = lg[lg.wo_type == "breakdown"].date.values
        cm = lg[lg.wo_type == "corrective"].date.values
        for t in dates.values:
            i_now = bisect_left(d, t)
            i_rec = bisect_left(d, t - np.timedelta64(recent, "D"))
            i_win = bisect_left(d, t - np.timedelta64(window, "D"))
            past_bd = bd[bd < t]
            past_cm = cm[cm < t]
            nxt = bd[(bd >= t) & (bd < t + np.timedelta64(horizon, "D"))]
            rows.append(dict(
                asset_id=a.asset_id, asset_type=a.asset_type, criticality=a.criticality, snapshot=t,
                text_recent=" | ".join(txt[i_rec:i_now]), text_older=" | ".join(txt[i_win:i_rec]),
                n_logs_recent=i_now - i_rec, n_logs_older=i_rec - i_win,
                days_since_breakdown=min((t - past_bd[-1]) / np.timedelta64(1, "D"), 999) if len(past_bd) else 999,
                days_since_corrective=min((t - past_cm[-1]) / np.timedelta64(1, "D"), 999) if len(past_cm) else 999,
                breakdowns_last_year=int((past_bd >= t - np.timedelta64(365, "D")).sum()),
                equipment_age_yr=pd.Timestamp(t).year - a.install_year,
                label=int(len(nxt) > 0),
                days_to_breakdown=(nxt[0] - t) / np.timedelta64(1, "D") if len(nxt) else np.nan,
            ))
    return pd.DataFrame(rows)


NUMERIC = ["n_logs_recent", "n_logs_older", "days_since_breakdown", "days_since_corrective",
           "breakdowns_last_year", "equipment_age_yr"]


def snapshot_from_entries(entries, fixes: dict, asset_type: str, criticality: str, equipment_age_yr: float,
                          days_since_breakdown: float = 999, days_since_corrective: float = 999,
                          breakdowns_last_year: int = 0, recent: int = 14, window: int = 60) -> pd.DataFrame:
    """One snapshot row for a single asset from (days_ago, text) log entries, same features as training."""
    rec = [clean(t, fixes) for d, t in entries if 0 < d <= recent]
    old = [clean(t, fixes) for d, t in entries if recent < d <= window]
    return pd.DataFrame([dict(asset_type=asset_type, criticality=criticality, text_recent=" | ".join(rec),
                              text_older=" | ".join(old), n_logs_recent=len(rec), n_logs_older=len(old),
                              days_since_breakdown=min(days_since_breakdown, 999),
                              days_since_corrective=min(days_since_corrective, 999),
                              breakdowns_last_year=breakdowns_last_year, equipment_age_yr=equipment_age_yr)])
