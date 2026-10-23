"""Weekly watchlist for the maintenance planner: top-K assets with the log lines that drove each score."""
from __future__ import annotations


def evidence(text: str, weights: dict, n: int = 3) -> str:
    """The log entries whose words carry the most warning weight in the logistic model."""
    lines = [x.strip() for x in (text or "").split("|") if x.strip()]
    scored = sorted(lines, key=lambda l: -sum(weights.get(tok, 0) for tok in l.split()))
    return " / ".join(scored[:n])


def build_watchlist(last_week, weights: dict, k: int):
    watch = (last_week.sort_values("risk", ascending=False).head(k)
             .assign(evidence_last_60_days=lambda d: [evidence(r if r else o, weights)
                                                      for r, o in zip(d.text_recent, d.text_older)])
             [["asset_id", "asset_type", "criticality", "risk", "days_since_corrective", "evidence_last_60_days",
               "label"]].rename(columns={"label": "actually_broke_down_within_30d"}))
    watch["risk"] = watch.risk.round(3)
    return watch
