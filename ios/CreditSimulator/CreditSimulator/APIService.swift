import Foundation

// MARK: - API Errors

/// Typed errors for all failure modes in the predict pipeline.
/// Using a typed enum instead of generic Error means every failure
/// is explicitly handled and mapped to a user-friendly message in
/// CreditViewModel.friendlyError().
enum APIError: Error {
    case encodingFailed
    case networkError(Error)
    case invalidResponse
    case serverError(Int)
    case decodingFailed(Error)
}

// MARK: - API Service

/// Handles all network communication with the CreditSimulator backend.
///
/// Implemented as a singleton so the URLSession (and its configuration)
/// is created once and reused across all requests — avoids the overhead
/// of creating a new session per call.
final class APIService {

    static let shared = APIService()
    private init() {}

    /// Custom URLSession with extended timeout to handle Render free-tier
    /// cold starts (can take up to 30s on the first request after inactivity).
    private let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest  = AppConfig.requestTimeout
        config.timeoutIntervalForResource = AppConfig.requestTimeout
        return URLSession(configuration: config)
    }()

    // MARK: - Predict

    /// Sends a POST /predict request and decodes the CreditOutput response.
    ///
    /// Uses async/await for clean, linear call sites in CreditViewModel.
    /// All errors are wrapped in typed APIError cases so the view model
    /// can map them to user-friendly messages without inspecting raw Error.
    func predict(input: CreditInput) async throws -> CreditOutput {

        // Build URL from AppConfig — force-unwrap is safe for a hardcoded constant
        let url = AppConfig.baseURL.appendingPathComponent("predict")

        // Build request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Encode input to JSON
        do {
            request.httpBody = try JSONEncoder().encode(input)
        } catch {
            throw APIError.encodingFailed
        }

        // Fire network request
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            throw APIError.networkError(error)
        }

        // Validate HTTP status code
        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }
        guard (200...299).contains(httpResponse.statusCode) else {
            // Pass status code through so CreditViewModel can give
            // a specific message for 422 (validation error) vs 5xx (server error)
            throw APIError.serverError(httpResponse.statusCode)
        }

        // Decode response
        do {
            return try JSONDecoder().decode(CreditOutput.self, from: data)
        } catch {
            throw APIError.decodingFailed(error)
        }
    }
}
