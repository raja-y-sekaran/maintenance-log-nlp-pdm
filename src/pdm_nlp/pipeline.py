"""Build snapshots, choose a model on a validation year, test it on two later years, write the watchlist."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import build_snapshots
from .models import TextLightGBM, TextLogReg, history_model, keyword_score

ROOT = Path(__file__).resolve().parents[2]
DATA, OUT, DOCS = ROOT / "data", ROOT / "outputs", ROOT / "docs"
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


def run():
    OUT.mkdir(exist_ok=True)
    reg = pd.read_csv(DATA / "asset_register.csv")
    logs = pd.read_csv(DATA / "maintenance_logs.csv")
    failures = pd.read_csv(DATA / "ground_truth" / "failure_modes.csv", parse_dates=["date"])
    episodes = pd.read_csv(DATA / "ground_truth" / "degradation_episodes.csv", parse_dates=["start", "end"])
    snap = build_snapshots(logs, reg, horizon=HORIZON, vocab_until=str(VALID_START.date()))
    train, valid, train_full, test = split(snap)

    # Model choice is made on the validation year only.
    vs, _ = score_all(train, valid)
    valid_ap = {k: round(average_precision_score(valid.label, v), 3) for k, v in vs.items()}
    chosen = max(valid_ap, key=valid_ap.get)

    ts, lr = score_all(train_full, test)
    m = dict(snapshots=len(snap), assets=len(reg), log_entries=len(logs),
             test_period=f"{TEST_START.date()} to {test.snapshot.max().date()}",
             test_base_rate=round(test.label.mean(), 4), validation_pr_auc=valid_ap, chosen_model=chosen,
             test_pr_auc={k: round(average_precision_score(test.label, v), 3) for k, v in ts.items()})
    m["test_precision_at_k"] = {k: round(weekly_topk(test, v).label.mean(), 3) for k, v in ts.items()}
    fl_all = {}
    for k, v in ts.items():
        fl, fa_fixed = failure_level(test, weekly_topk(test, v), failures, episodes)
        fl_all[k] = fl
        w = fl[fl.warned_in_text]
        m.setdefault("test_breakdowns_caught", {})[k] = dict(
            all=f"{int(fl.caught.sum())} of {len(fl)}",
            with_warning_in_text=f"{int(w.caught.sum())} of {len(w)}",
            sudden=f"{int(fl[~fl.warned_in_text].caught.sum())} of {int((~fl.warned_in_text).sum())}",
            median_lead_days=float(fl.lead_days.median()),
            false_alarms_inside_repaired_degradation=round(fa_fixed, 3))
    pos, neg = lr.top_terms(12)
    m["top_warning_terms"] = [f"{t} ({c:+.2f})" for t, c in pos]
    m["top_reassuring_terms"] = [f"{t} ({c:+.2f})" for t, c in neg]

    # Watchlist for the latest week, with the log lines that drove each score.
    last = test[test.snapshot == test.snapshot.max()].assign(risk=ts[chosen][test.snapshot == test.snapshot.max()])
    w = lr.recent_term_weights()
    def evidence(text):
        lines = [x.strip() for x in (text or "").split("|") if x.strip()]
        scored = sorted(lines, key=lambda l: -sum(w.get(tok, 0) for tok in l.split()))
        return " / ".join(scored[:3])
    watch = (last.sort_values("risk", ascending=False).head(TOP_K)
             .assign(evidence_last_60_days=lambda d: [evidence(r if r else o) for r, o in zip(d.text_recent, d.text_older)])
             [["asset_id", "asset_type", "criticality", "risk", "days_since_corrective", "evidence_last_60_days",
               "label"]].rename(columns={"label": "actually_broke_down_within_30d"}))
    watch["risk"] = watch.risk.round(3)
    watch.to_csv(OUT / "watchlist_latest_week.csv", index=False)
    fl_all[chosen].to_csv(OUT / "test_breakdowns_caught.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    test.assign(**{f"score_{i}": v for i, v in enumerate(ts.values())})[
        ["asset_id", "snapshot", "label"] + [f"score_{i}" for i in range(len(ts))]
    ].to_csv(OUT / "test_scores.csv", index=False)
    return m, test, ts, failures, episodes, chosen
