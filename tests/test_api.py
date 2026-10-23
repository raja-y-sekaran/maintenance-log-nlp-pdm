from fastapi.testclient import TestClient

from src.api.app import app

client = TestClient(app)
BASE = dict(asset_type="pump", criticality="B", equipment_age_yr=12)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_warning_signs_score_higher_than_routine_rounds():
    warn = client.post("/risk", json=dict(BASE, logs=[
        {"days_ago": 2, "text": "bearing temperature rising, now 88 C."},
        {"days_ago": 5, "text": "vibration increasing on DE bearing."},
        {"days_ago": 9, "text": "noise from bearing housing."}])).json()
    calm = client.post("/risk", json=dict(BASE, logs=[
        {"days_ago": 2, "text": "Routine round, all normal."},
        {"days_ago": 9, "text": "Greased bearings as per PM."}])).json()
    assert warn["risk_score"] > calm["risk_score"]
    assert "bearing" in warn["evidence"]
