"""Stage 9: FastAPI service. Send one asset's recent log entries, get a breakdown-risk score and the
entries that drove it.

    uvicorn src.api.app:app --reload        then open http://127.0.0.1:8000/docs
"""
from typing import Literal

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features.snapshots import snapshot_from_entries
from src.models.train import MODEL_FILE
from src.watchlist.watchlist import evidence

app = FastAPI(title="Maintenance-log NLP: breakdown risk", version="1.0",
              description="Synthetic-data demo. Not connected to any CMMS.")
_b = joblib.load(MODEL_FILE)
_weights = _b["explainer"].recent_term_weights()


class LogEntry(BaseModel):
    days_ago: float = Field(..., gt=0, examples=[3])
    text: str = Field(..., examples=["brg temp rising, now 88 C"])


class AssetWeek(BaseModel):
    asset_type: Literal["pump", "compressor", "motor", "control_valve", "transmitter"]
    criticality: Literal["A", "B", "C"] = "B"
    equipment_age_yr: float = 10
    days_since_breakdown: float = 999
    days_since_corrective: float = 999
    breakdowns_last_year: int = 0
    logs: list[LogEntry]


class Risk(BaseModel):
    risk_score: float
    note: str
    evidence: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/risk", response_model=Risk)
def risk(a: AssetWeek):
    row = snapshot_from_entries([(e.days_ago, e.text) for e in a.logs], _b["fixes"], a.asset_type, a.criticality,
                                a.equipment_age_yr, a.days_since_breakdown, a.days_since_corrective,
                                a.breakdowns_last_year)
    s = float(_b["ranker"].predict_proba(row)[0])
    r = row.iloc[0]
    return Risk(risk_score=round(s, 3), note="Score for ranking assets against each other; not a calibrated probability.",
                evidence=evidence(r.text_recent or r.text_older, _weights))
