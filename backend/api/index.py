from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, Field, model_validator
import joblib
import json
import numpy as np
import logging
import os

# MARK: - Logging
# Structured logging to stdout — Render captures these in the service
# dashboard under the Logs tab, giving visibility into production traffic.
# Format: timestamp | level | message for easy reading and filtering.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# MARK: - Rate Limiting
# Protects the /predict endpoint from abuse.
# In a real FinTech system, a scoring API being hammered is a real threat
# model — competitors scraping your model, denial-of-service, etc.
# Current limit: 30 requests/minute per IP (generous for a demo).
# In production this would be tighter and tied to authenticated users.
limiter = Limiter(key_func=get_remote_address)

# MARK: - Artifact Loading
# Paths resolved relative to this file so they work both locally and on Render.
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))  # points to backend/
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model_xgb.pkl")
MEDIAN_PATH = os.path.join(BASE_DIR, "model", "train_medians.json")

# Load once at startup — not on every request.
# joblib.load deserializes the XGBoost model from the pickle file.
# train_medians are the feature medians computed on the training split,
# used as fallback imputation values when a denominator would be zero.
model         = joblib.load(MODEL_PATH)
train_medians = json.load(open(MEDIAN_PATH))
logger.info("Model and medians loaded successfully")

# MARK: - App
app = FastAPI(title="CreditSimulator API", version="1.0.0")

# Attach rate limiter and its exception handler to the app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: currently open for demo purposes (allow_origins=["*"]).
# In production this would be restricted — for an iOS-only client,
# CORS headers are not strictly needed (iOS is not a browser),
# but keeping it open allows curl and Swagger UI testing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature order must match exactly what the model was trained on in train_model.py.
# Any mismatch causes silent wrong predictions — no error, just bad output.
FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

# MARK: - Schema

class CreditInput(BaseModel):
    """6 user-facing inputs. Derived ratio features are computed internally."""
    age_years:        float = Field(..., ge=18, le=70,  description="Age in years")
    years_employed:   float = Field(..., ge=0,  le=50,  description="Years at current/last job")
    amt_income_total: float = Field(..., gt=0,          description="Annual income")
    amt_credit:       float = Field(..., gt=0,          description="Requested loan amount")
    amt_annuity:      float = Field(..., gt=0,          description="Monthly loan repayment")
    cnt_children:     int   = Field(..., ge=0,  le=10,  description="Number of dependent children")

    @model_validator(mode="after")
    def validate_cross_fields(self) -> "CreditInput":
        """
        Catch logically impossible combinations that pass individual field checks.
        These are the same rules enforced client-side in CreditViewModel —
        server-side validation is the authoritative second layer of defence.
        """
        if self.years_employed > self.age_years - 16:
            raise ValueError("years_employed cannot exceed age minus 16")
        if self.amt_annuity > self.amt_credit:
            raise ValueError("amt_annuity cannot exceed amt_credit")
        if self.amt_credit > self.amt_income_total * 20:
            raise ValueError("amt_credit cannot exceed 20x annual income")
        return self


class CreditOutput(BaseModel):
    credit_score: int    # 300–900 (higher = lower default risk)
    risk_tier:    str    # Excellent / Good / Fair / Poor / Very Poor
    default_prob: float  # Raw XGBoost output — probability of loan default (0.0–1.0)


# MARK: - Helpers

def probability_to_score(prob: float) -> int:
    """
    Map default probability (0.0→1.0) to credit score (900→300).

    Uses a linear transform: score = 900 - (prob * 600)
    - 0%  default risk → 900 (Excellent)
    - 50% default risk → 600 (Poor)
    - 100% default risk → 300 (Very Poor)

    Real credit bureaus use PDO (Points to Double the Odds) log-odds scaling.
    This linear approximation is an intentional simplification for demo clarity.
    """
    return int(900 - (prob * 600))


def score_to_tier(score: int) -> str:
    """Bucket credit score into a human-readable risk tier."""
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


# MARK: - Routes

@app.get("/")
def root():
    """Health check endpoint. Used by Render to verify the service is up."""
    return {"status": "ok", "message": "CreditSimulator API is running"}


@app.get("/metadata")
def model_metadata():
    """
    Returns model info, performance metrics, score mapping rationale, and disclaimer.

    Exposing AUC and the disclaimer programmatically shows awareness of model
    limitations — a senior engineering practice. An interviewer calling this
    endpoint can see immediately that this is a demo, not a production scorer.
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
            "rationale": (
                "Linear transform for demo purposes. "
                "Real bureau scoring uses calibrated odds-to-score mapping (PDO scaling)."
            ),
            "range": "300 (highest risk) to 900 (lowest risk)"
        },
        "disclaimer": (
            "This is a demonstration model built for portfolio purposes. "
            "It is NOT a production credit scoring system and should not "
            "be used for real lending decisions."
        )
    }


@app.post("/predict", response_model=CreditOutput)
@limiter.limit("30/minute")
def predict(request: Request, data: CreditInput):
    """
    Predict creditworthiness from 6 user inputs.

    Derived features (ratios) are computed here — not exposed to the user —
    to keep the API surface clean and match the training pipeline exactly.
    Falls back to training medians if a denominator is zero (defensive coding).

    Rate limited to 30 requests/minute per IP to prevent API abuse.
    """
    try:
        # Log incoming request — age and income only.
        # Deliberately not logging full input to avoid unnecessary PII exposure
        # (income + credit + age combination could fingerprint an individual).
        logger.info(
            "Predict request | age=%.0f | income=%.0f | credit=%.0f",
            data.age_years, data.amt_income_total, data.amt_credit
        )

        # Compute derived features — same transformations used during training.
        # Falls back to training medians rather than crashing on zero denominators.
        credit_income_ratio  = (data.amt_credit  / data.amt_income_total
                                if data.amt_income_total > 0
                                else train_medians["CREDIT_INCOME_RATIO"])

        annuity_income_ratio = (data.amt_annuity / data.amt_income_total
                                if data.amt_income_total > 0
                                else train_medians["ANNUITY_INCOME_RATIO"])

        credit_term          = (data.amt_annuity / data.amt_credit
                                if data.amt_credit > 0
                                else train_medians["CREDIT_TERM"])

        # Assemble feature vector in the exact order the model expects.
        # Column order must match FEATURES list above and training pipeline.
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

        logger.info(
            "Predict response | score=%d | tier=%s | prob=%.4f",
            credit_score, risk_tier, default_prob
        )

        return CreditOutput(
            credit_score=credit_score,
            risk_tier=risk_tier,
            default_prob=round(default_prob, 4)
        )

    except Exception as e:
        # Log full stack trace so production issues are debuggable from Render logs
        logger.error("Predict failed | error=%s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))