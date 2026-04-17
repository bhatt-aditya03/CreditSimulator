# CreditSimulator

A FinTech "what-if" credit scoring simulator built on 307,511 real loan applications.
Move the sliders → see your predicted credit score update in real time.

---

## Live Demo

**API:** https://creditsimulator.onrender.com

**iOS App:** See [`/ios`](./ios/CreditSimulator/) — SwiftUI, runs on iOS 17+

> First API request may take ~30s if the Render instance is cold (free tier).

---

## Screenshots

| Default Values | Adjusted Profile |
|---|---|
| ![Default](./screenshots/app_default.png) | ![Adjusted](./screenshots/app_adjusted.png) |
## Architecture

```
User (iOS sliders)
        ↓
SwiftUI App — CreditViewModel (MVVM, 400ms debounce)
        ↓  HTTP POST /predict
FastAPI Backend (Render, Python 3.11)
        ↓
XGBoost Model (trained on Home Credit dataset)
        ↓
{ credit_score: 742, risk_tier: "Good", default_prob: 0.26 }
```

---

## Model

| Property | Value |
|---|---|
| Algorithm | XGBoostClassifier |
| Dataset | Home Credit Default Risk (Kaggle) |
| Training samples | 246,008 |
| Test samples | 61,503 |
| ROC-AUC | 0.6789 (baseline: 0.5) |

**Features used:** Age, Years Employed, Annual Income, Loan Amount, Loan Annuity, Children count. Ratio features (Credit/Income, Annuity/Income, Credit Term) are derived server-side — not exposed to the user.

**Score mapping:** `credit_score = int(900 - (default_probability × 600))`
This is a linear transform for demo clarity. Real credit bureaus use PDO (Points to Double the Odds) log-odds calibration. The linear approximation is an intentional simplification.

**Disclaimer:** This is a portfolio/demonstration project. Not a production credit scoring system. Do not use for real lending decisions.

---

## API Endpoints

### `GET /`
Health check.

### `GET /metadata`
Returns model info, AUC, features, score mapping rationale, and disclaimer.

### `POST /predict`
```json
{
  "age_years": 35,
  "years_employed": 5,
  "amt_income_total": 150000,
  "amt_credit": 300000,
  "amt_annuity": 15000,
  "cnt_children": 1
}
```
Returns:
```json
{
  "credit_score": 673,
  "risk_tier": "Fair",
  "default_prob": 0.3777
}
```

Risk tiers: **Excellent** (750+) / **Good** (700+) / **Fair** (650+) / **Poor** (600+) / **Very Poor** (<600)

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | XGBoost, Scikit-learn, Pandas |
| Backend | FastAPI, Python 3.11 |
| Deployment | Render |
| iOS App | SwiftUI, MVVM, Combine |

---

## Project Structure

```
CreditSimulator/
├── backend/
│   ├── api/
│   │   └── index.py              # FastAPI app — /predict, /metadata endpoints
│   ├── model/
│   │   ├── model_xgb.pkl         # Trained XGBoost model (~415KB)
│   │   ├── train_medians.json    # Imputation values saved from training split
│   │   └── train_model.py        # Training script (argparse, no data leakage)
│   ├── requirements.txt
│   └── Procfile                  # Render start command
└── ios/
    └── CreditSimulator/
        ├── AppConfig.swift        # Backend URL + timeout config
        ├── Models.swift           # CreditInput/Output + display helpers
        ├── APIService.swift       # async/await URLSession, typed errors
        ├── CreditViewModel.swift  # MVVM state, debounce, race condition fix
        └── ContentView.swift      # SwiftUI views, accessibility labels
```

---

## iOS Architecture Notes

- **MVVM** — `CreditViewModel` owns all state and API logic. Views are pure UI.
- **400ms debounce** — waits for the user to stop dragging before firing a request.
- **Race condition fix** — in-flight tasks are cancelled before new ones start.
- **Age clamp** — `yearsEmployed` is automatically capped to `age - 16` to prevent 422 validation errors.
- **Cold-start UX** — a hint appears after 4s of loading so users know the server is waking up.

See [`ios/CreditSimulator/README.md`](./ios/CreditSimulator/README.md) for full design decision notes.

---
