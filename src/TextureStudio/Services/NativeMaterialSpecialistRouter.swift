import CoreImage
import Foundation

/// Routes classifier labels to explicitly approved complete material models.
/// A missing, stale or unapproved specialist remains an ordinary fallback path.
enum NativeMaterialSpecialistRouter {
    struct RouteResult: Sendable {
        let prediction: NativeDiffuseTextureClassifier.Prediction
        let minimumConfidence: Double
        let checkpoints: [String: SelectedMaterialCheckpoint]
        let notices: [String]
        var classLabel: String { prediction.label }
        var confidence: Double { prediction.confidence }
        var probabilities: [String: Double] { prediction.probabilities }
    }

    static func route(diffuse: CIImage, workspaceURL: URL, modelDirectory: String,
                      classifierModelURL: URL? = nil, minimumConfidence: Double = 0.7) async throws -> RouteResult {
        let job = Task.detached(priority: .userInitiated) {
            try routeSynchronously(diffuse: diffuse, workspaceURL: workspaceURL, modelDirectory: modelDirectory,
                classifierModelURL: classifierModelURL, minimumConfidence: minimumConfidence)
        }
        return try await withTaskCancellationHandler {
            try await job.value
        } onCancel: { job.cancel() }
    }

    static func routeSynchronously(diffuse: CIImage, workspaceURL: URL, modelDirectory: String,
                                   classifierModelURL: URL? = nil, minimumConfidence: Double = 0.7) throws -> RouteResult {
        guard minimumConfidence.isFinite, (0...1).contains(minimumConfidence) else {
            throw StudioError("Specialist classifier confidence must be between zero and one.")
        }
        try Task.checkCancellation()
        let classifier = try NativeDiffuseTextureClassifier(modelURL: classifierModelURL)
        let prediction = try classifier.predict(image: diffuse)
        try Task.checkCancellation()
        if prediction.confidence < minimumConfidence {
            return RouteResult(prediction: prediction, minimumConfidence: minimumConfidence, checkpoints: [:],
                notices: ["\(prediction.label) classifier confidence \(prediction.confidence.formatted(.percent.precision(.fractionLength(1)))) is below the specialist threshold."])
        }
        let registry: NativeMaterialSpecialistService.Registry
        do { registry = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspaceURL, classLabel: prediction.label) }
        catch is CancellationError { throw CancellationError() }
        catch {
            return RouteResult(prediction: prediction, minimumConfidence: minimumConfidence, checkpoints: [:],
                notices: ["Specialist registry is unavailable: \(error.localizedDescription)"])
        }
        return try resolve(prediction: prediction, registry: registry, workspaceURL: workspaceURL,
            modelDirectory: modelDirectory, minimumConfidence: minimumConfidence)
    }

    /// Kept independent of image inference so approval and checkpoint contracts
    /// can be verified against real complete checkpoint fixtures.
    static func resolve(prediction: NativeDiffuseTextureClassifier.Prediction,
                        registry: NativeMaterialSpecialistService.Registry, workspaceURL: URL,
                        modelDirectory: String, minimumConfidence: Double = 0.7) throws -> RouteResult {
        try Task.checkCancellation()
        guard minimumConfidence.isFinite, (0...1).contains(minimumConfidence),
              prediction.confidence.isFinite, (0...1).contains(prediction.confidence) else {
            throw StudioError("Specialist classifier confidence is invalid.")
        }
        var notices: [String] = [], checkpoints: [String: SelectedMaterialCheckpoint] = [:]
        guard prediction.confidence >= minimumConfidence else {
            return RouteResult(prediction: prediction, minimumConfidence: minimumConfidence, checkpoints: [:],
                notices: ["Classifier confidence is below the specialist threshold."])
        }
        guard registry.classifierSha256 == prediction.modelSHA256,
              registry.classifierLabels.contains(prediction.label) else {
            return RouteResult(prediction: prediction, minimumConfidence: minimumConfidence, checkpoints: [:],
                notices: ["Refresh specialists for the selected classifier before automatic model routing."])
        }
        // Custom subsets train independently. A parent class prediction must
        // never choose a custom subset merely because it shares the same class.
        let specialists = registry.specialists.filter {
            $0.classLabel == prediction.label && !$0.isCustom && ($0.subtype?.isEmpty ?? true)
        }
        for target in ["height", "normal", "roughness"] {
            var candidates: [(NativeMaterialSpecialistService.Specialist, NativeMaterialSpecialistService.ModelRecord)] = []
            for specialist in specialists {
                guard let current = specialist.targets.first(where: { $0.target == target }),
                      current.materialCount > 0, !current.membershipSha256.isEmpty else { continue }
                candidates += specialist.models.filter {
                    $0.target == target && $0.approved && $0.completedUpdates > 0 &&
                    $0.membershipSha256 == current.membershipSha256 &&
                    ["compact-scalar", "compact-normal"].contains($0.modelFamily)
                }.map { (specialist, $0) }
            }
            candidates.sort { $0.1.completedAt > $1.1.completedAt }
            for (specialist, record) in candidates {
                do {
                    let text = try NativeMaterialCheckpoint.inspect(at: URL(fileURLWithPath: record.checkpointPath),
                        expectedSHA256: record.checkpointSha256)
                    let information = try NativeMaterialTransfer.object(Data(text.utf8))
                    guard information["schema"] as? String == NativeCompactMaterialModel.schema,
                          information["variant"] as? String == "full",
                          information["compatible"] as? Bool == true,
                          information["target"] as? String == target,
                          information["model_family"] as? String == record.modelFamily,
                          information["training_size"] as? Int == record.trainingSize,
                          (information["step"] as? Int ?? 0) >= record.completedUpdates,
                          let checkpointPath = information["checkpoint_path"] as? String,
                          information["sha256"] as? String == record.checkpointSha256 else {
                        throw StudioError("The approved specialist is not a complete trained compact \(target) model.")
                    }
                    checkpoints[target] = SelectedMaterialCheckpoint(checkpointPath: checkpointPath,
                        sha256: record.checkpointSha256, target: target, workspacePath: workspaceURL.path,
                        modelDirectory: modelDirectory, displayName: specialist.name,
                        modelSummary: "\(prediction.label) specialist · \(target) · \(record.checkpointSha256.prefix(12))")
                    break
                } catch {
                    if error is CancellationError { throw error }
                    notices.append("Skipped \(specialist.name) \(target): \(error.localizedDescription)")
                }
            }
        }
        if checkpoints.isEmpty { notices.append("No approved trained specialist currently matches \(prediction.label).") }
        return RouteResult(prediction: prediction, minimumConfidence: minimumConfidence,
            checkpoints: checkpoints, notices: notices)
    }
}
