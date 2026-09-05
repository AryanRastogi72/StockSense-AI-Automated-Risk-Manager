import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Application" / "Backend"))

from main import app

client = TestClient(app)

def test_explain_regression_endpoint():
    response = client.get("/explain/LT?task=reg")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "LT"
    assert data["task"] == "regression"
    assert data["model"] == "xgboost_tuned"
    assert "base_value" in data
    assert "feature_impacts" in data
    assert "note" in data

def test_explain_classification_endpoint():
    # First, get risk to know what the predicted class is
    risk_resp = client.get("/risk/LT")
    assert risk_resp.status_code == 200
    predicted_class = risk_resp.json()["classification"]["predicted_direction"]
    
    # Now get explanation for that class
    response = client.get(f"/explain/LT?task=class&predicted_class={predicted_class}")
    assert response.status_code == 200
    data = response.json()
    assert data["ticker"] == "LT"
    assert data["task"] == "classification"
    assert data["predicted_class"] == predicted_class
    assert "base_value_logit" in data
    assert "feature_impacts_logit" in data
    assert "lstm_surrogate_fidelity_r2" in data
    assert "low_fidelity_warning" in data
    assert "note" in data
