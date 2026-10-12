import SwiftUI

/// Numeric source agreement is independent of the display's contrast, gamma,
/// zoom and 8-bit preview. It evaluates the exact paired native grid.
struct MaterialScalarAgreementPanel: View {
    let reference: MapReviewCandidate
    let prediction: MapReviewCandidate
    @State private var agreement: MaterialScalarAgreement.Result?
    @State private var failure: String?

    private var identity: String {
        [reference.id, reference.mapURL.path, reference.displayTransform?.identity ?? "",
         prediction.id, prediction.mapURL.path, prediction.modelIdentity?.checkpointSHA256 ?? ""].joined(separator: "|")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Source agreement · \(prediction.label)").font(.caption.bold()).textSelection(.enabled)
            if let agreement {
                HStack(spacing: 16) {
                    Text("MAE \(number(agreement.mae))")
                        .help("Mean absolute error against the real source's numeric values. Lower is better.")
                    Text("Constant baseline \(number(agreement.medianConstantMAE))")
                        .help("Error of a spatially constant map set to the source median. A useful prediction should beat this reference.")
                    Text("Correlation \(agreement.pearsonCorrelation.map(number) ?? "undefined")")
                        .help("Pearson correlation with the source, from −1 to 1. Constant maps have undefined correlation.")
                }.font(.caption).textSelection(.enabled)
                if let ratio = agreement.standardDeviationRatio {
                    Text("Output variation \(String(format: "%.1f", ratio * 100))% of source · σ \(number(agreement.predictionStandardDeviation)) / \(number(agreement.referenceStandardDeviation))")
                        .font(.caption).textSelection(.enabled)
                    if agreement.nearlyFlat {
                        Label("Very little relief: output variation is below 10% of the source.", systemImage: "waveform.path")
                            .font(.caption).foregroundStyle(.orange)
                    }
                }
                if agreement.mae > agreement.medianConstantMAE {
                    Text("This prediction has more source error than the constant baseline.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("Raw values · \(agreement.width) × \(agreement.height) · source agreement measures this pair, not unseen-material accuracy")
                    .font(.caption2).foregroundStyle(.secondary)
            } else if let failure {
                Text("Source agreement unavailable: \(failure)").font(.caption).foregroundStyle(.secondary)
            } else {
                ProgressView("Measuring source agreement…").controlSize(.small)
            }
        }
        .task(id: identity) {
            agreement = nil; failure = nil
            let referenceURL = reference.mapURL, predictionURL = prediction.mapURL
            let transform = reference.displayTransform
            let mapType = prediction.modelIdentity?.mapType ?? "height"
            do {
                let work = Task.detached(priority: .utility) {
                    try MaterialScalarAgreement.evaluate(referenceURL: referenceURL, predictionURL: predictionURL,
                        mapType: mapType, referenceTransform: transform)
                }
                let result = try await withTaskCancellationHandler {
                    try await work.value
                } onCancel: {
                    work.cancel()
                }
                try Task.checkCancellation()
                agreement = result
            } catch is CancellationError { }
            catch {
                if !Task.isCancelled { failure = error.localizedDescription }
            }
        }
    }

    private func number(_ value: Double) -> String { String(format: "%.5f", value) }
}
