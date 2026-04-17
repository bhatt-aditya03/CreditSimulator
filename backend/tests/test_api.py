import pytest
from fastapi.testclient import TestClient
from api.index import app

client = TestClient(app)

# MARK: - Health Check

def test_root_returns_ok():
    """GET / should return 200 with status ok."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

# MARK: - Metadata

def test_metadata_returns_expected_keys():
    """GET /metadata should return all required fields."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "algorithm" in data
    assert "roc_auc" in data
    assert "features" in data
    assert "score_mapping" in data
    assert "disclaimer" in data
    
def test_metadata_auc_is_correct():
    """AUC in metadata should match the trained model's performance."""
    response = client.get("/metadata")
    assert response.json()["roc_auc"] == 0.6789

# MARK: - Predict: Valid Input

def test_predict_valid_input_returns_200():
    """POST /predict with valid input should return 200."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 1
    })
    assert response.status_code == 200

def test_predict_response_schema():
    """POST /predict response must contain credit_score, risk_tier, default_prob."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 1
    })
    data = response.json()
    assert "credit_score" in data
    assert "risk_tier" in data
    assert "default_prob" in data

def test_predict_score_in_valid_range():
    """Credit score must be between 300 and 900."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 1
    })
    score = response.json()["credit_score"]
    assert 300 <= score <= 900

def test_predict_risk_tier_is_valid():
    """Risk tier must be one of the five defined buckets."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 1
    })
    valid_tiers = {"Excellent", "Good", "Fair", "Poor", "Very Poor"}
    assert response.json()["risk_tier"] in valid_tiers

def test_predict_default_prob_between_0_and_1():
    """Default probability must be a valid probability value."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 1
    })
    prob = response.json()["default_prob"]
    assert 0.0 <= prob <= 1.0

# MARK: - Predict: Validation Errors

def test_predict_age_below_minimum_returns_422():
    """Age below 18 should fail Pydantic validation."""
    response = client.post("/predict", json={
        "age_years": 15,
        "years_employed": 2,
        "amt_income_total": 50000,
        "amt_credit": 100000,
        "amt_annuity": 5000,
        "cnt_children": 0
    })
    assert response.status_code == 422

def test_predict_negative_income_returns_422():
    """Negative income should fail Pydantic validation."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": -1000,
        "amt_credit": 300000,
        "amt_annuity": 15000,
        "cnt_children": 0
    })
    assert response.status_code == 422

def test_predict_years_employed_exceeds_age_returns_422():
    """years_employed > age - 16 should fail cross-field validation."""
    response = client.post("/predict", json={
        "age_years": 25,
        "years_employed": 20,
        "amt_income_total": 100000,
        "amt_credit": 200000,
        "amt_annuity": 10000,
        "cnt_children": 0
    })
    assert response.status_code == 422

def test_predict_annuity_exceeds_credit_returns_422():
    """amt_annuity > amt_credit should fail cross-field validation."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000,
        "amt_credit": 10000,
        "amt_annuity": 50000,
        "cnt_children": 0
    })
    assert response.status_code == 422

def test_predict_missing_field_returns_422():
    """Missing required field should fail Pydantic validation."""
    response = client.post("/predict", json={
        "age_years": 35,
        "years_employed": 5,
        "amt_income_total": 150000
        # missing amt_credit, amt_annuity, cnt_children
    })
    assert response.status_code == 422