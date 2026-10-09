import Foundation

extension WorkbenchStore {
    /// Folder and crop selection always display their diffuse image first.
    func selectDatasetSample(_ sample: WorkbenchSample, role: String? = nil) {
        guard !isBusy else { return }
        if selectedSampleId != sample.id { selectedInputVariantId = nil }
        selectedSampleId = sample.id
        selectedRole = role.flatMap { sample.maps[$0] != nil ? $0 : nil }
            ?? ["input", "height", "roughness", "normal"].first(where: { sample.maps[$0] != nil }) ?? "input"
    }

    /// Both library and file-picker refinement use the checkpoint's recorded setup.
    @discardableResult
    func selectCheckpointForRefinement(_ checkpoint: WorkbenchCheckpoint, navigate: Bool = true) -> Bool {
        guard !isBusy else { return false }
        guard checkpoint.supportsTrainingWarmStart else {
            error = "This checkpoint does not support material refinement."
            return false
        }
        configureCheckpointForRefinement(checkpoint)
        if navigate { trainingNavigationRequest = UUID() }
        return true
    }

    func configureCheckpointForRefinement(_ checkpoint: WorkbenchCheckpoint) {
        selectedCheckpointId = checkpoint.id
        training.target = checkpoint.target
        training.scope = checkpoint.scope ?? "final-map"
        training.useWarmStart = true
    }

    var comparisonConfigurationIssue: String? {
        if sourceImageURL == nil && selectedDiffuseMap == nil {
            return "Choose a test photo or select a dataset material with a diffuse map."
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        if selected.isEmpty { return "Choose a checkpoint to compare with its base model." }
        if selected.contains(where: { !$0.supportsStudioInference }) {
            return "Select compatible material checkpoints for this comparison."
        }
        if Set(selected.map(\.target)).count != 1 {
            return "Select checkpoints for the same map: displacement, roughness or normals."
        }
        if selected.count == 1 && !comparisonIncludesBase {
            return "Include the base model or select a second checkpoint."
        }
        return nil
    }
}
