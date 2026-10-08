import Foundation

/// Recorded provenance of the output being reviewed. Unknown architecture stays
/// unknown; a filename is never used to choose a model architecture.
struct MapReviewModelIdentity: Hashable {
    let checkpointPath: String?
    let checkpointSHA256: String?
    let checkpointStep: Int?
    let architecture: String?
    let mapType: String?
    let modelName: String?

    init(checkpointPath: String? = nil, checkpointSHA256: String? = nil,
         checkpointStep: Int? = nil, architecture: String? = nil,
         mapType: String? = nil, modelName: String? = nil) {
        self.checkpointPath = checkpointPath
        self.checkpointSHA256 = checkpointSHA256
        self.checkpointStep = checkpointStep
        self.architecture = architecture
        self.mapType = mapType
        self.modelName = modelName
    }

    var runName: String? {
        guard let checkpointPath else { return nil }
        let directory = URL(fileURLWithPath: checkpointPath).deletingLastPathComponent()
        if ["frozen", "lora", "checkpoints"].contains(directory.lastPathComponent.lowercased()) {
            return directory.deletingLastPathComponent().lastPathComponent + "/" + directory.lastPathComponent
        }
        return directory.lastPathComponent
    }

    var recordedDetails: [String] {
        var parts: [String] = []
        if let architecture { parts.append(architecture) }
        if let checkpointPath { parts.append(URL(fileURLWithPath: checkpointPath).lastPathComponent) }
        if let checkpointStep { parts.append("Step \(checkpointStep.formatted())") }
        if let checkpointSHA256 { parts.append("SHA256 \(checkpointSHA256.prefix(12))") }
        return parts
    }

    var manifestFields: [String: Any] {
        var fields: [String: Any] = [:]
        fields["checkpoint"] = checkpointPath
        fields["checkpoint_sha256"] = checkpointSHA256
        fields["checkpoint_step"] = checkpointStep
        fields["model_architecture"] = architecture
        fields["map_type"] = mapType
        fields["model_name"] = modelName
        return fields
    }
}

struct MapReviewCandidate: Identifiable, Hashable {
    let id: String
    let label: String
    let mapURL: URL
    let numeric: Bool
    let sampleLabel: String?
    let detail: String?
    let role: String
    let modelIdentity: MapReviewModelIdentity?

    init(id: String? = nil, label: String, mapURL: URL, numeric: Bool,
         sampleLabel: String? = nil, detail: String? = nil, role: String = "map",
         modelIdentity: MapReviewModelIdentity? = nil) {
        self.id = id ?? mapURL.path + "|" + label
        self.label = label; self.mapURL = mapURL; self.numeric = numeric
        self.sampleLabel = sampleLabel; self.detail = detail; self.role = role
        self.modelIdentity = modelIdentity
    }

    var accessibleLabel: String { [sampleLabel, label, detail].compactMap { $0 }.joined(separator: " · ") }

    var exportFilename: String {
        guard let sampleLabel else { return mapURL.lastPathComponent }
        let stem = (sampleLabel + "-" + label)
            .replacingOccurrences(of: "[^A-Za-z0-9._-]+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-._"))
        var suffix = ""
        if role == "checkpoint" {
            let token = modelIdentity?.checkpointSHA256 ?? (id.contains("/") ? ReviewImageLoader.hash(Data(id.utf8)) : id)
            suffix = "-" + String(token.prefix(12))
        }
        return String(stem.prefix(180 - suffix.count)) + suffix + "." + mapURL.pathExtension
    }
}
