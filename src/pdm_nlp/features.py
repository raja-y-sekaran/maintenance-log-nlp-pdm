"""Turn the maintenance log into weekly asset snapshots: recent text, simple history features, and the label.

Leakage rule: a snapshot taken on day t only uses log entries dated strictly before t.
The label is whether a breakdown work order is raised for that asset in [t, t + horizon).
"""
from __future__ import annotations

import re
from bisect import bisect_left

import numpy as np
import pandas as pd

SHORTHAND = {
    r"\btemp\b": "temperature", r"\bvibn?\b": "vibration", r"\bbrg\b": "bearing", r"\bmech\b": "mechanical",
    r"\bpress\b": "pressure", r"\bincr\b": "increasing", r"\brepl\b": "replaced", r"\bnrml\b": "normal",
    r"\bhi\b": "high", r"\blo\b": "low", r"\blkg\b": "leak", r"\bvlv\b": "valve",
}
_ASSET_RE = re.compile(r"\b[a-z]{1,2}-\d{3,4}a?\b")


def clean(text: str, fixes: dict | None = None) -> str:
    t = str(text).lower()
    for pat, rep in SHORTHAND.items():
        t = re.sub(pat, rep, t)
    t = _ASSET_RE.sub(" ", t)  # asset tags carry no meaning for the model
    if fixes:
        t = re.sub(r"[a-z]+", lambda m: fixes.get(m.group(0), m.group(0)), t)
    return re.sub(r"\s+", " ", t).strip()


def learn_typo_fixes(texts, min_ratio: float = 20.0) -> dict:
    """Map a word to a much more frequent word that differs by one swapped pair of adjacent letters.
    Learned from the plant's own text, so local vocabulary ('positioner', 'impeller') is respected."""
    from collections import Counter
    freq = Counter(w for t in texts for w in re.findall(r"[a-z]+", clean(t)))
    fixes = {}
    for w, n in freq.items():
        if len(w) < 5:
            continue
        for i in range(len(w) - 1):
            cand = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if cand != w and freq.get(cand, 0) >= min_ratio * n:
                fixes[w] = cand
                break
    return fixes


def build_snapshots(logs: pd.DataFrame, register: pd.DataFrame, horizon: int = 30, freq: str = "7D",
                    recent: int = 14, window: int = 60, vocab_until: str | None = None) -> pd.DataFrame:
    """vocab_until: learn spelling fixes only from text before this date (keeps the test years unseen)."""
    logs = logs.copy()
    logs["date"] = pd.to_datetime(logs.date)
    src = logs if vocab_until is None else logs[logs.date < pd.Timestamp(vocab_until)]
    fixes = learn_typo_fixes(src.text)
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
