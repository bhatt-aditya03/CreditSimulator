from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import joblib
import json
import numpy as np
import os

# ── Artifacts ─────────────────────────────────────────────────────────────────
# Paths are resolved relative to this file so they work both locally and on Render
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))  # points to backend/
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model_xgb.pkl")
MEDIAN_PATH = os.path.join(BASE_DIR, "model", "train_medians.json")

# Load once at startup — not on every request
model         = joblib.load(MODEL_PATH)
train_medians = json.load(open(MEDIAN_PATH))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CreditSimulator API", version="1.0.0")

# Allow all origins so the iOS app and Swagger UI can reach the API freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature order must match exactly what the model was trained on in train_model.py
FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

# ── Schema ────────────────────────────────────────────────────────────────────
class CreditInput(BaseModel):
    """6 user-facing inputs. Derived features are computed internally."""
    age_years:        float = Field(..., ge=18, le=70,  description="Age in years")
    years_employed:   float = Field(..., ge=0,  le=50,  description="Years at current/last job")
    amt_income_total: float = Field(..., gt=0,          description="Annual income (INR/USD)")
    amt_credit:       float = Field(..., gt=0,          description="Requested loan amount")
    amt_annuity:      float = Field(..., gt=0,          description="Monthly loan repayment")
    cnt_children:     int   = Field(..., ge=0,  le=10,  description="Number of dependent children")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "CreditInput":
        """Catch logically impossible combinations that pass individual field checks."""
        if self.years_employed > self.age_years - 16:
            raise ValueError("years_employed cannot exceed age minus 16")
        if self.amt_annuity > self.amt_credit:
            raise ValueError("amt_annuity cannot exceed amt_credit")
        if self.amt_credit > self.amt_income_total * 20:
            raise ValueError("amt_credit cannot exceed 20x annual income")
        return self


class CreditOutput(BaseModel):
    credit_score: int    # 300–900 (higher is better)
    risk_tier:    str    # Excellent / Good / Fair / Poor / Very Poor
    default_prob: float  # Raw model output — probability of loan default (0.0–1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def probability_to_score(prob: float) -> int:
    """
    Map default probability → credit score using a linear transform.
    Formula: score = 900 - (prob * 600)
    - 0% default risk  → 900 (best)
    - 50% default risk → 600
    - 100% default risk → 300 (worst)

    Note: Real credit bureaus use PDO (Points to Double the Odds) scaling,
    which is a log-odds calibration. This linear mapping is a simplified
    approximation suitable for demonstration purposes.
    """
    return int(900 - (prob * 600))


def score_to_tier(score: int) -> str:
    """Bucket credit score into human-readable risk tier."""
    if score >= 750:
        return "Excellent"
    elif score >= 700:
        return "Good"
    elif score >= 650:
        return "Fair"
    elif score >= 600:
        return "Poor"
    else:
        return "Very Poor"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "CreditSimulator API is running"}


@app.get("/metadata")
def model_metadata():
    """
    Returns model info, performance metrics, and score mapping rationale.
    Useful for interview demos — shows awareness of model limitations.
    """
    return {
        "model": "XGBoostClassifier",
        "dataset": "Home Credit Default Risk (Kaggle)",
        "training_samples": 246008,
        "test_samples": 61503,
        "roc_auc": 0.6789,
        "features": FEATURES,
        "score_mapping": {
            "formula": "credit_score = int(900 - (default_probability * 600))",
            "rationale": "Linear transform for demo purposes. Real bureau scoring uses calibrated odds-to-score mapping (PDO scaling).",
            "range": "300 (highest risk) to 900 (lowest risk)"
        },
        "disclaimer": (
            "This is a demonstration model built for portfolio purposes. "
            "It is NOT a production credit scoring system and should not "
            "be used for real lending decisions."
        )
    }


@app.post("/predict", response_model=CreditOutput)
def predict(data: CreditInput):
    """
    Predict creditworthiness from 6 user inputs.

    Derived features (ratios) are computed here — not exposed to the user —
    to keep the API surface clean and match the training pipeline exactly.
    Falls back to training medians if a denominator is zero (defensive coding).
    """
    try:
        # Compute derived features — same transformations used during training
        credit_income_ratio  = (data.amt_credit  / data.amt_income_total
                                if data.amt_income_total > 0
                                else train_medians["CREDIT_INCOME_RATIO"])

        annuity_income_ratio = (data.amt_annuity / data.amt_income_total
                                if data.amt_income_total > 0
                                else train_medians["ANNUITY_INCOME_RATIO"])

        credit_term          = (data.amt_annuity / data.amt_credit
                                if data.amt_credit > 0
                                else train_medians["CREDIT_TERM"])

        # Assemble feature vector in the exact order the model expects
        features = np.array([[
            data.age_years,
            data.years_employed,
            data.amt_income_total,
            data.amt_credit,
            data.amt_annuity,
            credit_income_ratio,
            annuity_income_ratio,
            credit_term,
            data.cnt_children
        ]])

        # predict_proba returns [prob_no_default, prob_default] — we want index 1
        default_prob = float(model.predict_proba(features)[0][1])
        credit_score = probability_to_score(default_prob)
        risk_tier    = score_to_tier(credit_score)

        return CreditOutput(
            credit_score=credit_score,
            risk_tier=risk_tier,
            default_prob=round(default_prob, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))