import SwiftUI

struct MaterialQualityReviewPanel: View {
    let diffuseURL: URL
    let mapURL: URL
    let mapType: String
    let purpose: MaterialQualityDecision.Purpose
    var diffuseSHA256: String? = nil
    var mapSHA256: String? = nil
    var diffuseTransform: MapReviewDisplayTransform? = nil
    var mapTransform: MapReviewDisplayTransform? = nil
    var onApply: ((MaterialQualityDecision.Recommendation) -> Void)? = nil
    @State private var adviser = OllamaDecisionService()
    @State private var decision: MaterialQualityDecision?
    @State private var recommendation: MaterialQualityDecision.Recommendation = .review
    @State private var task: Task<Void, Never>?
    @State private var failure: String?
    @State private var isReviewing = false
    private var context: String { [diffuseURL.path, mapURL.path, diffuseSHA256 ?? "", mapSHA256 ?? "", mapType,
        diffuseTransform?.identity ?? "", mapTransform?.identity ?? ""].joined(separator: "|") }

    var body: some View {
        DisclosureGroup("Local quality adviser") {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Button(isReviewing ? "Reviewing…" : "Review Diffuse + \(mapTitle)", systemImage: "sparkle.magnifyingglass") { review() }
                        .disabled(isReviewing)
                    if isReviewing { ProgressView().controlSize(.small); Button("Cancel") { cancel() } }
                    Spacer()
                }
                Text("Clef checks paired previews and aligned native-pixel center crops. Original files stay unchanged. Inspect the full maps before approving; the adviser cannot certify physical accuracy or unseen pixels.")
                    .font(.caption).foregroundStyle(.secondary)
                if let decision {
                    HStack(spacing: 16) {
                        score("Detail", decision.detail)
                        score("Appearance", decision.appeal)
                        score("Alignment", decision.alignment)
                    }
                    Text("No major artifacts: \(Int((decision.artifactFree * 100).rounded()))% · Diffuse ready: \(Int((decision.diffuseReady * 100).rounded()))%")
                        .font(.caption).foregroundStyle(.secondary)
                    HStack {
                        Picker("Suggested review", selection: $recommendation) {
                            ForEach(MaterialQualityDecision.Recommendation.allCases, id: \.self) { Text($0.title).tag($0) }
                        }.frame(maxWidth: 340)
                        if let onApply {
                            Button("Confirm Review") { onApply(recommendation) }
                        }
                    }
                    Text("Recommendation concentration: \(Int((decision.recommendationConfidence * 100).rounded()))%. This describes the model's distribution, not its chance of being correct.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if let failure { Text(failure).font(.caption).foregroundStyle(.secondary).textSelection(.enabled) }
            }.padding(.top, 6)
        }
        .onChange(of: context) { _, _ in cancel(); decision = nil; failure = nil; recommendation = .review }
        .onDisappear { cancel() }
    }
    private var mapTitle: String {
        switch mapType {
        case "height", "depth": "Displacement"
        case "normal": "Normal"
        default: "Roughness"
        }
    }
    private func score(_ name: String, _ value: Double) -> some View {
        Text("\(name): \(value, specifier: "%.1f") / 4").font(.caption.bold())
    }
    private func cancel() { task?.cancel(); adviser.cancel(); task = nil; isReviewing = false }
    private func review() {
        guard !isReviewing else { return }
        failure = nil; decision = nil; isReviewing = true
        let reviewContext = context
        task = Task { @MainActor in
            defer { if context == reviewContext { isReviewing = false; task = nil } }
            do {
                let diffuse = try await ReviewImageLoader.shared.load(diffuseURL, numeric: false, expectedSHA256: diffuseSHA256, displayTransform: diffuseTransform)
                let map = try await ReviewImageLoader.shared.load(mapURL, numeric: true, expectedSHA256: mapSHA256, displayTransform: mapTransform)
                try Task.checkCancellation()
                let result = try await adviser.reviewMaterial(diffuse: diffuse.image, map: map.image, mapType: mapType, purpose: purpose)
                try Task.checkCancellation()
                guard context == reviewContext else { return }
                decision = result; recommendation = result.recommendation
            } catch is CancellationError { }
            catch { if !Task.isCancelled, context == reviewContext { failure = error.localizedDescription } }
        }
    }
}
