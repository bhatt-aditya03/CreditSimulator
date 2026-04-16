import Foundation

/// Central configuration for the CreditSimulator app.
/// All environment-specific values live here so they can be changed
/// in one place without hunting through the codebase.
enum AppConfig {

    /// Base URL of the FastAPI backend deployed on Render.
    /// Defined as a URL (not String) so call sites never deal with
    /// optional URL construction — a hardcoded constant force-unwrap
    /// is safe and idiomatic for compile-time-known values.
    static let baseURL = URL(string: "https://creditsimulator.onrender.com")!

    /// Network timeout in seconds.
    /// Set to 40s to accommodate Render free-tier cold starts,
    /// which can take up to 30s on the first request after inactivity.
    static let requestTimeout: TimeInterval = 40
}
