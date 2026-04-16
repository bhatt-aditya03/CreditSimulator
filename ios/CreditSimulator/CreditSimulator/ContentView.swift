import SwiftUI

// MARK: - Root View

struct ContentView: View {
    @StateObject private var vm = CreditViewModel()

    var body: some View {
        ZStack {
            // Subtle gradient background — gives depth without being distracting
            LinearGradient(
                colors: [Color(.systemBackground), Color(.secondarySystemBackground)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            ScrollView {
                VStack(spacing: 20) {
                    HeaderView()
                    ScoreCardView(vm: vm)
                    InputsCardView(vm: vm)

                    // Legal disclaimer — required for any FinTech demo
                    Text("For demonstration purposes only. Not a real credit scoring system.")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                        .padding(.bottom, 8)
                }
                .padding(.horizontal)
                .padding(.top, 8)
            }
        }
    }
}

// MARK: - Header

struct HeaderView: View {
    var body: some View {
        HStack(spacing: 12) {
            // Rupee icon — signals FinTech context immediately
            Image(systemName: "indianrupeesign.circle.fill")
                .font(.system(size: 36))
                .foregroundStyle(.blue)

            VStack(alignment: .leading, spacing: 2) {
                Text("CreditSimulator")
                    .font(.title2)
                    .fontWeight(.bold)
                Text("What-if credit score calculator")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()
        }
        .padding(.top, 8)
    }
}

// MARK: - Score Card

/// Displays the animated arc gauge, risk tier, default probability,
/// cold-start hint, and error messages.
struct ScoreCardView: View {
    @ObservedObject var vm: CreditViewModel

    /// Resolved tier color — falls back to gray before first API response.
    /// Single source of truth lives on CreditOutput.tierColor in Models.swift.
    private var tierColor: Color {
        vm.result?.tierColor ?? .gray
    }

    var body: some View {
        VStack(spacing: 0) {

            // ── Arc Gauge ─────────────────────────────────────────────────
            VStack(spacing: 12) {
                ZStack {
                    // Background track — partial arc (not full circle)
                    // gives a speedometer feel more appropriate for scoring
                    Circle()
                        .trim(from: 0.15, to: 0.85)
                        .stroke(
                            Color(.systemGray5),
                            style: StrokeStyle(lineWidth: 14, lineCap: .round)
                        )
                        .frame(width: 180, height: 180)
                        .rotationEffect(.degrees(90))

                    // Progress arc — animates smoothly on score change
                    Circle()
                        .trim(from: 0.15, to: 0.15 + (vm.scoreProgress * 0.7))
                        .stroke(
                            LinearGradient(
                                colors: [tierColor.opacity(0.7), tierColor],
                                startPoint: .leading,
                                endPoint: .trailing
                            ),
                            style: StrokeStyle(lineWidth: 14, lineCap: .round)
                        )
                        .frame(width: 180, height: 180)
                        .rotationEffect(.degrees(90))
                        .animation(.spring(response: 0.6, dampingFraction: 0.8), value: vm.scoreProgress)

                    // Center label
                    VStack(spacing: 4) {
                        if vm.isLoading {
                            ProgressView()
                                .scaleEffect(1.2)
                        } else {
                            Text(vm.formattedScore)
                                .font(.system(size: 48, weight: .bold, design: .rounded))
                                .foregroundStyle(tierColor)
                                .contentTransition(.numericText())
                                .animation(.spring(response: 0.4), value: vm.formattedScore)

                            Text(vm.formattedRiskTier)
                                .font(.subheadline)
                                .fontWeight(.semibold)
                                .foregroundStyle(tierColor)
                        }
                    }
                }
                .padding(.top, 8)

                // Score range legend
                HStack {
                    Text("300")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                    Spacer()
                    Text("900")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 32)
            }

            Divider()
                .padding(.vertical, 16)

            // ── Stats Row ─────────────────────────────────────────────────
            HStack(spacing: 0) {
                StatCell(title: "Default Risk", value: vm.formattedDefaultProb, color: tierColor)
                Divider().frame(height: 40)
                StatCell(title: "Score Range",  value: "300–900",              color: .secondary)
                Divider().frame(height: 40)
                StatCell(title: "Model",        value: "XGBoost",              color: .blue)
            }
            .padding(.bottom, 16)

            // ── Cold Start Hint ───────────────────────────────────────────
            // Appears after 4s of loading to inform the user the server
            // is waking up (Render free-tier cold start), not broken.
            if vm.isSlowLoad {
                HStack(spacing: 6) {
                    Image(systemName: "clock.arrow.circlepath")
                        .font(.caption)
                    Text("Server is waking up, please wait...")
                        .font(.caption)
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal)
                .padding(.bottom, 8)
            }

            // ── Error Message ─────────────────────────────────────────────
            if let error = vm.errorMessage {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.caption)
                    Text(error)
                        .font(.caption)
                }
                .foregroundStyle(.red)
                .padding(.horizontal)
                .padding(.bottom, 12)
            }
        }
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.06), radius: 12, x: 0, y: 4)
    }
}

// MARK: - Stat Cell

/// A single metric in the stats row below the gauge.
struct StatCell: View {
    let title: String
    let value: String
    let color: Color

    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.subheadline)
                .fontWeight(.bold)
                .foregroundStyle(color)
                .monospacedDigit()
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Inputs Card

/// Contains all 6 sliders. Each slider is color-coded to make the
/// interface visually scannable — a common pattern in FinTech dashboards.
struct InputsCardView: View {
    @ObservedObject var vm: CreditViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {

            // Card header
            HStack {
                Label("Financial Profile", systemImage: "slider.horizontal.3")
                    .font(.subheadline)
                    .fontWeight(.semibold)
                Spacer()
                Text("Drag to simulate")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }

            Divider()

            // Age — dynamic max enforced here visually,
            // but ViewModel.setupAgeClamp() is the authoritative clamp
            SliderRow(
                icon: "person.fill",
                label: "Age",
                value: $vm.ageYears,
                range: vm.ageRange,
                format: "%.0f yrs",
                color: .blue
            )
            .accessibilityLabel("Age slider, currently \(Int(vm.ageYears)) years")

            // Years Employed — range is dynamically capped to (age - 16)
            // so the user can't create an input the API would reject with 422
            SliderRow(
                icon: "briefcase.fill",
                label: "Years Employed",
                value: $vm.yearsEmployed,
                range: 0...max(0, vm.ageYears - 16),
                format: "%.0f yrs",
                color: .indigo
            )
            .accessibilityLabel("Years employed slider, currently \(Int(vm.yearsEmployed)) years")

            SliderRow(
                icon: "banknote.fill",
                label: "Annual Income",
                value: $vm.amtIncomeTotal,
                range: vm.incomeRange,
                format: "₹%.0f",
                color: .green
            )
            .accessibilityLabel("Annual income slider, currently ₹\(Int(vm.amtIncomeTotal))")

            SliderRow(
                icon: "house.fill",
                label: "Loan Amount",
                value: $vm.amtCredit,
                range: vm.creditRange,
                format: "₹%.0f",
                color: .orange
            )
            .accessibilityLabel("Loan amount slider, currently ₹\(Int(vm.amtCredit))")

            SliderRow(
                icon: "calendar.badge.clock",
                label: "Monthly Annuity",
                value: $vm.amtAnnuity,
                range: vm.annuityRange,
                format: "₹%.0f",
                color: .purple
            )
            .accessibilityLabel("Monthly annuity slider, currently ₹\(Int(vm.amtAnnuity))")

            SliderRow(
                icon: "figure.2.and.child.holdinghands",
                label: "Children",
                value: $vm.cntChildren,
                range: vm.childrenRange,
                format: "%.0f",
                color: .pink
            )
            .accessibilityLabel("Number of children slider, currently \(Int(vm.cntChildren))")
        }
        .padding(20)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .shadow(color: .black.opacity(0.06), radius: 12, x: 0, y: 4)
    }
}

// MARK: - Slider Row

/// A single labeled slider with an icon and color-coded value display.
/// Value animates via contentTransition so number changes feel smooth.
struct SliderRow: View {
    let icon: String
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String
    let color: Color

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(color)
                    .frame(width: 16)

                Text(label)
                    .font(.subheadline)
                    .foregroundStyle(.primary)

                Spacer()

                // Animated value label — updates smoothly as slider moves
                Text(String(format: format, value))
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .monospacedDigit()
                    .foregroundStyle(color)
                    .contentTransition(.numericText())
                    .animation(.spring(response: 0.3), value: value)
            }

            Slider(value: $value, in: range)
                .tint(color)
        }
    }
}

#Preview {
    ContentView()
}
