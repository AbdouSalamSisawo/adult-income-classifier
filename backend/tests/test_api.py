"""API tests. Run from backend/:  python -m pytest"""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
SAMPLE = json.loads((Path(__file__).resolve().parents[2] / "test_request.json").read_text())


def test_health_and_root():
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").json()["model_trained_on_split"] == "90:10"


def test_feature_order_matches_model():
    assert len(main.FEATURE_ORDER) == main.MODEL.n_features_in_ == 41


def test_schema_lists_all_categories():
    s = client.get("/schema").json()
    assert s["education"][0] == "Preschool" and s["education"][-1] == "Doctorate"
    assert "Married-civ-spouse" in s["marital_status"]
    assert len(s["workclass"]) == 8 and len(s["occupation"]) == 14 and len(s["relationship"]) == 6


def test_predict_sample_matches_documented_result():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] == ">50K"
    assert body["probability_above_50k"] == pytest.approx(0.8675)
    assert body["probability_above_50k"] + body["probability_at_or_below_50k"] == pytest.approx(1.0)


def test_predict_low_income_profile():
    payload = {**SAMPLE, "age": 24, "sex": "Female", "education": "HS-grad",
               "maritalStatus": "Never-married", "relationship": "Own-child",
               "occupation": "Adm-clerical", "hoursPerWeek": 20}
    assert client.post("/predict", json=payload).json()["prediction"] == "<=50K"


def test_capital_gain_is_capped_like_training():
    a = client.post("/predict", json={**SAMPLE, "capitalGain": 15024}).json()
    b = client.post("/predict", json={**SAMPLE, "capitalGain": 999999}).json()
    assert a == b


def test_snake_case_field_names_accepted():
    payload = {k: v for k, v in SAMPLE.items() if k != "maritalStatus"}
    payload["marital_status"] = SAMPLE["maritalStatus"]
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("override", [{"age": 5}, {"occupation": "Astronaut"}, {"hoursPerWeek": 0}, {"capitalGain": -1}])
def test_invalid_input_returns_422(override):
    assert client.post("/predict", json={**SAMPLE, **override}).status_code == 422
