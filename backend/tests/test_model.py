import pytest
import joblib
import json
import numpy as np
import os

# MARK: - Setup

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model_xgb.pkl")
MEDIAN_PATH = os.path.join(BASE_DIR, "model", "train_medians.json")

@pytest.fixture(scope="module")
def model():
    """Load the XGBoost model once for all tests in this module."""
    return joblib.load(MODEL_PATH)

@pytest.fixture(scope="module")
def medians():
    """Load training medians once for all tests in this module."""
    with open(MEDIAN_PATH) as f:
        return json.load(f)

# MARK: - Artifacts

def test_model_file_exists():
    """model_xgb.pkl must exist in backend/model/."""
    assert os.path.exists(MODEL_PATH), "model_xgb.pkl not found"

def test_medians_file_exists():
    """train_medians.json must exist in backend/model/."""
    assert os.path.exists(MEDIAN_PATH), "train_medians.json not found"

def test_medians_has_all_required_keys(medians):
    """train_medians.json must contain all 9 training features."""
    required_keys = [
        "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
        "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
    ]
    for key in required_keys:
        assert key in medians, f"Missing key in train_medians.json: {key}"

# MARK: - Model Predictions

def test_model_predicts_probability(model):
    """Model must return a valid probability array for a known input."""
    features = np.array([[35, 5, 150000, 300000, 15000, 2.0, 0.1, 0.05, 1]])
    proba = model.predict_proba(features)
    assert proba.shape == (1, 2), "Expected shape (1, 2) for binary classifier"

def test_model_output_is_valid_probability(model):
    """Output probability must be between 0 and 1."""
    features = np.array([[35, 5, 150000, 300000, 15000, 2.0, 0.1, 0.05, 1]])
    prob = float(model.predict_proba(features)[0][1])
    assert 0.0 <= prob <= 1.0

def test_score_mapping_formula():
    """Score mapping formula must produce correct values at known points."""
    def probability_to_score(prob: float) -> int:
        return int(900 - (prob * 600))

    assert probability_to_score(0.0) == 900   # no risk → best score
    assert probability_to_score(0.5) == 600   # 50% risk → middle score
    assert probability_to_score(1.0) == 300   # certain default → worst score

def test_high_income_low_credit_scores_better(model):
    """High income + low loan should score better than low income + high loan."""
    # Good profile — high income, small loan
    good = np.array([[40, 10, 500000, 100000, 5000,
                      0.2, 0.01, 0.05, 0]])
    # Risky profile — low income, large loan
    risky = np.array([[25, 2, 50000, 900000, 50000,
                       18.0, 1.0, 0.055, 3]])

    good_prob  = float(model.predict_proba(good)[0][1])
    risky_prob = float(model.predict_proba(risky)[0][1])

    assert good_prob < risky_prob, (
        "Good financial profile should have lower default probability than risky profile"
    )

def test_model_handles_edge_case_minimum_values(model):
    """Model must not crash on minimum valid input values."""
    features = np.array([[18, 0, 30000, 50000, 5000,
                          1.67, 0.167, 0.1, 0]])
    proba = model.predict_proba(features)
    assert proba is not None

def test_model_handles_edge_case_maximum_values(model):
    """Model must not crash on maximum valid input values."""
    features = np.array([[70, 54, 500000, 1000000, 100000,
                          2.0, 0.2, 0.1, 10]])
    proba = model.predict_proba(features)
    assert proba is not None