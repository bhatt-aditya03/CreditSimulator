import Foundation
import SwiftUI

// MARK: - API Request

/// The 6 user-facing inputs sent to POST /predict.
///
/// Derived features (CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM)
/// are computed server-side from these values — not sent by the client —
/// to keep the API surface clean and match the training pipeline exactly.
struct CreditInput: Codable {
    var ageYears: Double
    var yearsEmployed: Double
    var amtIncomeTotal: Double
    var amtCredit: Double
    var amtAnnuity: Double
    var cntChildren: Int

    /// Maps Swift camelCase properties to the snake_case keys FastAPI expects.
    enum CodingKeys: String, CodingKey {
        case ageYears       = "age_years"
        case yearsEmployed  = "years_employed"
        case amtIncomeTotal = "amt_income_total"
        case amtCredit      = "amt_credit"
        case amtAnnuity     = "amt_annuity"
        case cntChildren    = "cnt_children"
    }
}

// MARK: - API Response

/// The prediction result returned by POST /predict.
///
/// - creditScore: 300–900 (higher = lower default risk)
/// - riskTier: Excellent / Good / Fair / Poor / Very Poor
/// - defaultProb: raw XGBoost output — probability of loan default (0.0–1.0)
struct CreditOutput: Codable {
    let creditScore: Int
    let riskTier: String
    let defaultProb: Double

    /// Maps snake_case JSON keys from FastAPI to Swift camelCase.
    enum CodingKeys: String, CodingKey {
        case creditScore = "credit_score"
        case riskTier    = "risk_tier"
        case defaultProb = "default_prob"
    }
}

// MARK: - Display Helpers

extension CreditOutput {

    /// Single source of truth for risk tier color used across all views.
    ///
    /// Defined on the model — not in individual views — so color logic
    /// lives in one place and stays consistent everywhere it's used.
    ///
    /// Note: "Good" uses a custom mint-green RGB rather than .mint
    /// because .mint renders too pale on light backgrounds in iOS 17+.
    var tierColor: Color {
        switch riskTier {
        case "Excellent": return .green
        case "Good":      return Color(red: 0.2, green: 0.78, blue: 0.6)
        case "Fair":      return .orange
        case "Poor":      return .red
        default:          return .red  // covers "Very Poor" and unexpected values
        }
    }

    /// Score mapped to 0.0→1.0 for the arc gauge.
    ///
    /// Formula: (score - 300) / 600
    ///   - score 300 → 0.0 (empty arc)
    ///   - score 900 → 1.0 (full arc)
    ///
    /// Clamped to [0, 1] as a defensive measure at the API trust boundary —
    /// if the backend ever returns an out-of-range score, the arc won't
    /// overflow or render incorrectly.
    var gaugeProgress: Double {
        let raw = (Double(creditScore) - 300) / 600
        return min(1.0, max(0.0, raw))
    }

    /// Credit score as a plain display string. e.g. "742"
    var formattedScore: String {
        "\(creditScore)"
    }

    /// Default probability as a percentage string. e.g. "37.8%"
    var formattedDefaultProb: String {
        String(format: "%.1f%%", defaultProb * 100)
    }
}
