import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import json
import os
import argparse 

# ── Config ────────────────────────────────────────────────────────────────────
# Update this path to wherever application_train.csv is on your machine
parser = argparse.ArgumentParser(description="Train CreditSimulator XGBoost model")
parser.add_argument(
    "--data",
    type=str,
    default=os.path.expanduser("~/credit-risk-analyzer/application_train.csv"),
    help="Path to application_train.csv"
)
args = parser.parse_args()
DATA_PATH = args.data

OUTPUT_DIR = os.path.dirname(__file__)  # saves into backend/model/

# ── Load & clean ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows")

# Drop columns with >50% missing
df = df[df.columns[df.isnull().mean() < 0.5]]

# Fix sentinel value
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

# ── Feature engineering ───────────────────────────────────────────────────────
df["AGE_YEARS"]             = -df["DAYS_BIRTH"] / 365
df["YEARS_EMPLOYED"]        = -df["DAYS_EMPLOYED"] / 365
df["CREDIT_INCOME_RATIO"]   = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"]  = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"]           = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

X = df[FEATURES]
y = df["TARGET"]

# ── Split BEFORE imputation (no leakage) ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit medians on train only, save them for use at prediction time
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=10,
    random_state=42,
    eval_metric="auc"
)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {auc:.4f}")

# ── Score mapping test ────────────────────────────────────────────────────────
def probability_to_score(default_prob: float) -> int:
    """Map default probability (0→1) to credit score (900→300)."""
    return int(900 - (default_prob * 600))

sample_probs = [0.0, 0.25, 0.5, 0.75, 1.0]
print("\nScore mapping check:")
for p in sample_probs:
    print(f"  {p:.0%} default risk → score {probability_to_score(p)}")

# ── Save artifacts ────────────────────────────────────────────────────────────
model_path   = os.path.join(OUTPUT_DIR, "model_xgb.pkl")
medians_path = os.path.join(OUTPUT_DIR, "train_medians.json")

joblib.dump(model, model_path)
train_medians.to_json(medians_path)

print(f"\n✅ Model saved → {model_path}")
print(f"✅ Medians saved → {medians_path}")