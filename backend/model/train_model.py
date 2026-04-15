import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import json
import os
import argparse

# ── CLI Config ────────────────────────────────────────────────────────────────
# Accepts a custom dataset path via --data flag so this script is portable.
# Default points to the local Home Credit dataset used during development.
# Usage: python train_model.py --data /path/to/application_train.csv
parser = argparse.ArgumentParser(description="Train CreditSimulator XGBoost model")
parser.add_argument(
    "--data",
    type=str,
    default=os.path.expanduser("~/credit-risk-analyzer/application_train.csv"),
    help="Path to application_train.csv"
)
args = parser.parse_args()
DATA_PATH  = args.data
OUTPUT_DIR = os.path.dirname(__file__)  # artifacts saved into backend/model/

# ── Load & Clean ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# Drop columns where more than 50% of values are missing.
# Keeping sparse columns would force heavy imputation and add noise.
df = df[df.columns[df.isnull().mean() < 0.5]]

# DAYS_EMPLOYED uses 365243 as a sentinel for unemployed/pensioners (not a real value).
# Replace with NaN so it gets handled cleanly by median imputation later.
df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

# ── Feature Engineering ───────────────────────────────────────────────────────
# Raw columns store days as negative integers (e.g. -10000 days from today).
# Convert to positive years for interpretability in the API and slider UI.
df["AGE_YEARS"]    = -df["DAYS_BIRTH"] / 365
df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365

# Ratio features capture financial stress more effectively than raw amounts.
# A ₹5L loan means very different things on ₹3L vs ₹30L annual income.
df["CREDIT_INCOME_RATIO"]  = df["AMT_CREDIT"]  / df["AMT_INCOME_TOTAL"]
df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
df["CREDIT_TERM"]          = df["AMT_ANNUITY"] / df["AMT_CREDIT"]

# Final feature set — must match FEATURES list in backend/api/index.py exactly.
# Any mismatch will cause silent wrong predictions (model receives wrong column order).
FEATURES = [
    "AGE_YEARS", "YEARS_EMPLOYED", "AMT_INCOME_TOTAL",
    "AMT_CREDIT", "AMT_ANNUITY", "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CNT_CHILDREN"
]

X = df[FEATURES]
y = df["TARGET"]  # 1 = defaulted, 0 = repaid

# ── Train/Test Split ──────────────────────────────────────────────────────────
# Split BEFORE imputation to prevent data leakage.
# If we imputed first, test set medians would bleed into training statistics.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Compute medians on training data only, then apply to both splits.
# These medians are saved to train_medians.json so the API uses
# the exact same imputation values at prediction time.
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

print(f"Train: {len(X_train):,} samples | Test: {len(X_test):,} samples")

# ── Train XGBoost ─────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    # scale_pos_weight handles class imbalance — ~10x more non-defaults than defaults
    # in the Home Credit dataset. Without this, model would predict 0 for everything.
    scale_pos_weight=10,
    random_state=42,
    eval_metric="auc"
)
model.fit(X_train, y_train)

# ROC-AUC is the right metric here (not accuracy) because the dataset is heavily
# imbalanced. A naive classifier predicting "no default" always gets ~92% accuracy
# but 0.5 AUC — no better than random.
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"XGBoost ROC-AUC: {auc:.4f}")

# ── Score Mapping Verification ────────────────────────────────────────────────
def probability_to_score(default_prob: float) -> int:
    """
    Map default probability (0.0→1.0) to credit score (900→300).

    Uses a linear transform: score = 900 - (prob * 600)
    - 0%  default risk → 900 (Excellent)
    - 50% default risk → 600 (Poor)
    - 100% default risk → 300 (Very Poor)

    Real credit bureaus use PDO (Points to Double the Odds) log-odds scaling.
    This linear approximation is intentional for demo clarity.
    """
    return int(900 - (default_prob * 600))

print("\nScore mapping check:")
for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
    print(f"  {p:.0%} default risk → score {probability_to_score(p)}")

# ── Save Artifacts ────────────────────────────────────────────────────────────
# Both artifacts must be committed to the repo for Render to load them at runtime.
# model_xgb.pkl  — the trained XGBoost model (~415KB, safe to commit)
# train_medians.json — imputation values, must match what the API uses
model_path   = os.path.join(OUTPUT_DIR, "model_xgb.pkl")
medians_path = os.path.join(OUTPUT_DIR, "train_medians.json")

joblib.dump(model, model_path)
train_medians.to_json(medians_path)

print(f"\n✅ Model saved   → {model_path}")
print(f"✅ Medians saved → {medians_path}")