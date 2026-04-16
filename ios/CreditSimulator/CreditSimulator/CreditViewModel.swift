import Foundation
import Combine

/// The central view model for CreditSimulator.
///
/// Owns all slider state, fires debounced API calls, and exposes
/// formatted display values to the SwiftUI views.
///
/// All @Published properties are updated on the MainActor to keep
/// UI updates thread-safe without manual DispatchQueue.main calls.
@MainActor
final class CreditViewModel: ObservableObject {

    // MARK: - Slider Inputs (user-facing)

    @Published var ageYears: Double        = 35
    @Published var yearsEmployed: Double   = 5       // Double for Slider binding, cast to Double on send
    @Published var amtIncomeTotal: Double  = 150000
    @Published var amtCredit: Double       = 300000
    @Published var amtAnnuity: Double      = 15000
    @Published var cntChildren: Double     = 0       // Double for Slider binding, cast to Int on send

    // MARK: - Output

    @Published var result: CreditOutput?  = nil
    @Published var isLoading: Bool        = false
    @Published var isSlowLoad: Bool       = false    // true after 4s of loading — triggers cold-start hint
    @Published var errorMessage: String?  = nil

    // MARK: - Slider Ranges

    let ageRange:      ClosedRange<Double> = 18...70
    let employedRange: ClosedRange<Double> = 0...50
    let incomeRange:   ClosedRange<Double> = 30000...500000
    let creditRange:   ClosedRange<Double> = 50000...1000000
    let annuityRange:  ClosedRange<Double> = 5000...100000
    let childrenRange: ClosedRange<Double> = 0...10

    // MARK: - Private State

    private var cancellables = Set<AnyCancellable>()

    /// Holds the current in-flight predict task so it can be cancelled
    /// before a new one starts — prevents stale responses from overwriting
    /// newer results when the user drags sliders rapidly.
    private var predictTask: Task<Void, Never>?

    // MARK: - Init

    init() {
        setupAgeClamp()
        setupDebounce()
        // Fire the initial prediction manually so the app loads with a score.
        // dropFirst() in setupDebounce() ensures the Combine pipeline
        // doesn't also fire on init, avoiding a duplicate launch request.
        predict()
    }

    // MARK: - Age → YearsEmployed Clamp

    /// When the user drags age DOWN, yearsEmployed might exceed the new
    /// maximum (age - 16). This sink clamps it automatically so the API
    /// never receives a logically impossible input combination
    /// (e.g. 24 years employed at age 36 → FastAPI returns 422).
    ///
    /// The SliderRow in ContentView also enforces this visually via
    /// a dynamic range, but the ViewModel clamp is the authoritative fix.
    private func setupAgeClamp() {
        $ageYears
            .sink { [weak self] newAge in
                guard let self else { return }
                let maxEmployed = max(0, newAge - 16)
                if self.yearsEmployed > maxEmployed {
                    self.yearsEmployed = maxEmployed
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Debounce Setup

    /// Watches all 6 slider inputs simultaneously and fires a predict()
    /// call 400ms after the user stops dragging.
    ///
    /// Why 400ms? Median human reaction time for intentional slider pauses
    /// is ~250–350ms. 400ms ensures the user has stopped adjusting while
    /// still feeling responsive. Shorter = too many requests; longer = laggy.
    ///
    /// dropFirst() skips the single initial emit triggered by @Published
    /// property assignments in init — we fire that manually with predict().
    private func setupDebounce() {
        Publishers.CombineLatest3(
            Publishers.CombineLatest(
                $ageYears, $yearsEmployed
            ),
            Publishers.CombineLatest3(
                $amtIncomeTotal, $amtCredit, $amtAnnuity
            ),
            $cntChildren
        )
        .debounce(for: .milliseconds(400), scheduler: RunLoop.main)
        .dropFirst()
        .sink { [weak self] _ in
            self?.predict()
        }
        .store(in: &cancellables)
    }

    // MARK: - Predict

    /// Cancels any in-flight request, then fires a new predict call
    /// with the current slider values.
    ///
    /// Race condition fix: without cancellation, rapid slider changes
    /// could result in two overlapping requests. Whichever arrives LAST
    /// would win — not necessarily the most recent input. Cancellation
    /// ensures only the latest request's response is ever displayed.
    func predict() {
        predictTask?.cancel()

        predictTask = Task { [weak self] in
            guard let self else { return }

            await MainActor.run {
                self.isLoading    = true
                self.isSlowLoad   = false
                self.errorMessage = nil
            }

            // Cold-start hint: after 4s of waiting, show a subtle message
            // so the user knows the server is starting up, not broken.
            // Render free-tier instances spin down after inactivity and
            // can take up to 30s to wake on the first request.
            let slowLoadTimer = Task {
                try? await Task.sleep(nanoseconds: 4_000_000_000) // 4 seconds
                await MainActor.run {
                    if self.isLoading { self.isSlowLoad = true }
                }
            }

            let input = CreditInput(
                ageYears:       self.ageYears,
                yearsEmployed:  self.yearsEmployed,
                amtIncomeTotal: self.amtIncomeTotal,
                amtCredit:      self.amtCredit,
                amtAnnuity:     self.amtAnnuity,
                cntChildren:    Int(self.cntChildren)
            )

            // Bail out if the user already triggered a newer request
            guard !Task.isCancelled else {
                slowLoadTimer.cancel()
                return
            }

            do {
                let output = try await APIService.shared.predict(input: input)

                // Check again — response may have arrived after cancellation
                guard !Task.isCancelled else {
                    slowLoadTimer.cancel()
                    return
                }

                slowLoadTimer.cancel()
                await MainActor.run {
                    self.result      = output
                    self.isLoading   = false
                    self.isSlowLoad  = false
                }

            } catch {
                guard !Task.isCancelled else {
                    slowLoadTimer.cancel()
                    return
                }

                slowLoadTimer.cancel()
                await MainActor.run {
                    self.errorMessage = self.friendlyError(error)
                    self.isLoading    = false
                    self.isSlowLoad   = false
                }
            }
        }
    }

    // MARK: - Error Messaging

    /// Maps typed APIError cases to user-friendly strings.
    ///
    /// Raw localizedDescription exposes cryptic system messages like
    /// "keyNotFound(CodingKeys(...), ...)" — unacceptable in a FinTech UI.
    /// Every error case maps to a clear, actionable message.
    private func friendlyError(_ error: Error) -> String {
        if let apiError = error as? APIError {
            switch apiError {
            case .encodingFailed:
                return "Could not prepare your request. Try again."
            case .networkError:
                return "Network unavailable. Check your connection."
            case .invalidResponse:
                return "Unexpected server response. Try again."
            case .serverError(let code):
                if code == 422 {
                    // FastAPI validation failure — usually an impossible
                    // input combination that slipped past client-side checks
                    return "Invalid input combination. Please adjust your values."
                }
                return "Server error (\(code)). Try again shortly."
            case .decodingFailed:
                return "Could not read the server response. Try again."
            }
        }
        // URLSession timeout — most likely a Render cold start
        if (error as NSError).code == NSURLErrorTimedOut {
            return "Request timed out. The server may be starting up — try again in 30 seconds."
        }
        return "Something went wrong. Please try again."
    }

    // MARK: - Display Helpers

    /// Formatted credit score string for the gauge label. e.g. "742"
    /// Falls back to "---" before the first response arrives.
    var formattedScore: String {
        result?.formattedScore ?? "---"
    }

    /// Risk tier label. e.g. "Good", "Fair".
    var formattedRiskTier: String {
        result?.riskTier ?? "Loading..."
    }

    /// Default probability as a percentage string. e.g. "32.1%"
    var formattedDefaultProb: String {
        result?.formattedDefaultProb ?? "--%"
    }

    /// Score as 0.0→1.0 fraction for the arc gauge.
    /// Delegates to CreditOutput.gaugeProgress which owns the clamped math.
    var scoreProgress: Double {
        result?.gaugeProgress ?? 0
    }
}
