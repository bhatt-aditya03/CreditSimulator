from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
import os

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))  # backend/
MODEL_PATH  = os.path.join(BASE_DIR, "model", "model_xgb.pkl")
MEDIAN_PATH = os.path.join(BASE_DIR, "model", "train_medians.json")

model         = joblib.load(MODEL_PATH)
train_medians = json.load(open(MEDIAN_PATH))

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="CreditSimulator API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema ────────────────────────────────────────────────────────────────────
class CreditInput(BaseModel):
    age_years:         float = Field(..., ge=18,  le=70,   description="Age in years")
    years_employed:    float = Field(..., ge=0,   le=50,   description="Years employed")
    amt_income_total:  float = Field(..., ge=0,            description="Annual income")
    amt_credit:        float = Field(..., ge=0,            description="Loan amount")
    amt_annuity:       float = Field(..., ge=0,            description="Loan annuity")
    cnt_children:      int   = Field(..., ge=0,   le=10,   description="Number of children")

class CreditOutput(BaseModel):
    credit_score: int
    risk_tier:    str
    default_prob: float

# ── Helpers ───────────────────────────────────────────────────────────────────
def probability_to_score(prob: float) -> int:
    return int(900 - (prob * 600))

def score_to_tier(score: int) -> str:
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
    return {"status": "ok", "message": "CreditSimulator API is running"}

@app.post("/predict", response_model=CreditOutput)
def predict(data: CreditInput):
    try:
        # Derived features (same as training)
        credit_income_ratio  = data.amt_credit  / data.amt_income_total if data.amt_income_total > 0 else train_medians["CREDIT_INCOME_RATIO"]
        annuity_income_ratio = data.amt_annuity / data.amt_income_total if data.amt_income_total > 0 else train_medians["ANNUITY_INCOME_RATIO"]
        credit_term          = data.amt_annuity / data.amt_credit       if data.amt_credit > 0       else train_medians["CREDIT_TERM"]

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