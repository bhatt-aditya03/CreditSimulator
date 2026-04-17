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

# ── Logging ───────────────────────────────────────────────────────────────────
# Structured logging to stdout — Render captures these in the service
# dashboard under the Logs tab, giving real visibility into production traffic.
# Format: timestamp | level | message for easy reading and grep-based filtering.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Protects /predict from abuse. In FinTech, a scoring API being hammered
# is a real threat — competitors scraping the model, DoS attacks, etc.
# Current limit: 30 requests/minute per IP (generous for a demo).
# In production this would be tighter and tied to authenticated API keys.
limiter = Limiter(key_func=get_remote_address)

# ── Artifact Loading ──────────────────────────────────────────────────────────
# Paths resolved relative to this file so they work both locally and on Render.
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))  # points to backend/
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_xgb.pkl")
MEDIAN_PATH = os.path.join(BASE_DIR, "model", "train_medians.json")
META_PATH  = os.path.join(BASE_DIR, "model", "model_metadata.json")

# Load all three artifacts once at startup — not on every request.
# model         — XGBoost classifier (serialized with joblib)
# train_medians — feature medians from training split, used as fallback
#                 imputation values when a denominator would be zero
# model_meta    — version info, AUC, hyperparameters — single source of
#                 truth for everything surfaced at /metadata
model         = joblib.load(MODEL_PATH)
train_medians = json.load(open(MEDIAN_PATH))
model_meta    = json.load(open(META_PATH))
logger.info(
    "Artifacts loaded | model version=%s | AUC=%.4f",
    model_meta["version"],
    model_meta["roc_auc"]
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CreditSimulator API", version="1.0.0")

# Attach rate limiter and its 429 exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: currently open (allow_origins=["*"]) for demo purposes.
# This allows curl, Swagger UI, and any browser to hit the API freely.
# For an iOS-only client, CORS headers are technically unnecessary
# (iOS is not a browser), but keeping it open simplifies local testing.
# In production the allowlist would be restricted to trusted origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Feature order must match exactly what the model was trained on in train_model.py.
# Any mismatch causes silent wrong predictions — no error, just bad output.
# This list is also surfaced in /metadata so API consumers can see what drives the score.
FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

# ── Schema ────────────────────────────────────────────────────────────────────

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
        These mirror the rules enforced client-side in CreditViewModel —
        server-side validation is the authoritative second layer of defence.
        FastAPI returns 422 Unprocessable Entity when any rule fails.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def probability_to_score(prob: float) -> int:
    """
    Map default probability (0.0→1.0) to credit score (900→300).

    Formula: score = 900 - (prob × 600)
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """
    Health check endpoint.
    Used by Render to verify the service is up after deployment.
    Also useful for confirming the API is awake before firing /predict.
    """
    return {"status": "ok", "message": "CreditSimulator API is running"}


@app.get("/metadata")
def metadata():
    """
    Returns model version, performance metrics, hyperparameters,
    score mapping rationale, and disclaimer.

    Data is loaded from model_metadata.json — the single source of truth
    for all model information. This pattern means the /metadata response
    automatically reflects any future model retraining without code changes.

    Exposing AUC and disclaimer programmatically shows awareness of model
    limitations — a senior engineering practice. An interviewer calling
    this endpoint can see immediately that this is a demo, not a
    production credit scorer.
    """
    return {
        **model_meta,
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
        # Log incoming request — age, income, credit amount only.
        # Deliberately not logging the full input to reduce PII exposure:
        # the combination of income + credit + age + children can
        # fingerprint an individual in a small dataset context.
        logger.info(
            "Predict request | age=%.0f | income=%.0f | credit=%.0f",
            data.age_years, data.amt_income_total, data.amt_credit
        )

        # Compute derived features — same transformations used during training.
        # These ratios capture financial stress more effectively than raw amounts:
        # ₹5L loan means very different things on ₹3L vs ₹30L annual income.
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
        # Log full stack trace so production issues are debuggable from Render logs.
        # exc_info=True captures the full traceback, not just the error message.
        logger.error("Predict failed | error=%s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))