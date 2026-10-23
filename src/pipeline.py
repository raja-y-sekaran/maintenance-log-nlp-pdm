"""Stages wired together: snapshots, model choice on a validation year, test on two later years, watchlist."""
from __future__ import annotations

import json

import pandas as pd
from sklearn.metrics import average_precision_score

from src.data.load_data import load_data
from src.features.snapshots import build_snapshots
from src.models.evaluate import HORIZON, TEST_START, TOP_K, VALID_START, failure_level, score_all, split, weekly_topk
from src.utils.paths import DATA_PROCESSED, REPORTS
from src.watchlist.watchlist import build_watchlist
def run():
    reg, logs, failures, episodes = load_data()
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
    watch = build_watchlist(last, lr.recent_term_weights(), TOP_K)
    watch.to_csv(REPORTS / "watchlist_latest_week.csv", index=False)
    fl_all[chosen].to_csv(DATA_PROCESSED / "test_breakdowns_caught.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(m, indent=2, default=str))
    test.assign(**{f"score_{i}": v for i, v in enumerate(ts.values())})[
        ["asset_id", "snapshot", "label"] + [f"score_{i}" for i in range(len(ts))]
    ].to_csv(DATA_PROCESSED / "test_scores.csv", index=False)
    return m, test, ts, failures, episodes, chosen
