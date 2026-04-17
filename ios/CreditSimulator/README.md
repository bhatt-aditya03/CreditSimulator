# CreditSimulator — iOS App

A SwiftUI "what-if" credit score simulator. Move the sliders and watch
your predicted credit score update in real time via a live ML backend.

---

## Architecture

MVVM — clean separation between UI and business logic.

```
ContentView.swift      ← Pure UI, no business logic
↓ observes
CreditViewModel.swift  ← State, debounce, API calls, error handling
↓ calls
APIService.swift       ← URLSession, async/await, typed errors
↓ hits
FastAPI Backend        ← XGBoost model on Render
```
---

## Key Design Decisions

**Why 400ms debounce?**
Median human reaction time for intentional slider pauses is ~250–350ms.
400ms ensures the user has stopped adjusting while still feeling
responsive. Shorter causes too many API calls; longer feels laggy.

**Why cancel in-flight requests?**
Without cancellation, rapid slider changes cause overlapping requests.
Whichever response arrives *last* wins — not the most recent input.
`predictTask?.cancel()` before every new request guarantees only the
latest result is ever displayed.

**Why clamp yearsEmployed to (age - 16)?**
The FastAPI backend validates that `years_employed ≤ age - 16` and
returns 422 if violated. The slider range is capped dynamically, and
`setupAgeClamp()` in the ViewModel clamps the stored value when age
decreases — two layers of defence.

**Why tierColor lives on CreditOutput?**
Single source of truth. Previously duplicated between the model extension
and ScoreCardView — two versions that disagreed. Moving it to the model
means every view that needs a tier color calls `result.tierColor`.

**Why 40s timeout?**
Render free-tier instances spin down after inactivity. Cold starts can
take up to 30s. A 40s timeout with a `isSlowLoad` hint after 4s gives
the user context instead of a confusing spinner.

**PII considerations**
- No user inputs are persisted to UserDefaults or disk
- No client-side logging of request bodies
- HTTPS enforced via `https://` in AppConfig.baseURL
- This is a demo app — no real financial data is processed

---

## File Structure

```
CreditSimulator/
├── AppConfig.swift        — backend URL and timeout (change here only)
├── Models.swift           — CreditInput, CreditOutput, display helpers
├── APIService.swift       — URLSession, async/await, typed APIError enum
├── CreditViewModel.swift  — @Published state, debounce, predict(), errors
└── ContentView.swift      — SwiftUI views, MVVM-clean, accessibility labels
```
---

## Running Locally

1. Open `ios/CreditSimulator.xcodeproj` in Xcode
2. Select any iPhone simulator (iOS 17+)
3. Press `Cmd+R`

The app calls the live backend at `https://creditsimulator.onrender.com`.
First load may take ~30s if the Render instance is cold.

---

## Backend

See [`/backend`](../backend/) for the FastAPI + XGBoost model.  
Live API: https://creditsimulator.onrender.com/metadata