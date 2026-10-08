import Foundation

/// Provenance of a real photographed material's supplied map, separate from
/// the checkpoint that predicts one. Paths and crop coordinates are recorded,
/// never inferred from an image's appearance or filename.
struct MapReviewSourceIdentity: Hashable, Sendable {
    let path: String?
    let sha256: String?
    let url: String?
    let bits: Int?
    let cropRectangle: [Int]?
    let pixelDimensions: [Int]?

    init(path: String? = nil, sha256: String? = nil, url: String? = nil, bits: Int? = nil,
         cropRectangle: [Int]? = nil, pixelDimensions: [Int]? = nil) {
        self.path = path; self.sha256 = sha256; self.url = url; self.bits = bits
        self.cropRectangle = cropRectangle; self.pixelDimensions = pixelDimensions
    }

    var recordedDetails: [String] {
        var details: [String] = []
        if let bits { details.append("\(bits)-bit original source · linear data") }
        if let pixelDimensions, pixelDimensions.count == 2 {
            details.append("Full source \(pixelDimensions[0]) × \(pixelDimensions[1])")
        }
        if let cropRectangle, cropRectangle.count == 4 {
            details.append("Source crop x\(cropRectangle[0]), y\(cropRectangle[1]), \(cropRectangle[2]) × \(cropRectangle[3])")
        }
        if let url { details.append(url) }
        if let sha256 { details.append("Source SHA256 \(sha256.prefix(12))") }
        return details
    }

    var manifestFields: [String: Any] {
        var fields: [String: Any] = [:]
        fields["source_path"] = path
        fields["source_sha256"] = sha256
        fields["source_url"] = url
        fields["source_bits"] = bits
        fields["source_crop_rectangle"] = cropRectangle
        fields["source_pixel_dimensions"] = pixelDimensions
        return fields
    }

    static func fromDatasetMap(_ map: WorkbenchMap, target: String, sampleID: String?) -> MapReviewSourceIdentity {
        let fallback = MapReviewSourceIdentity(path: map.path, sha256: map.sha256, bits: map.sourceBits,
            pixelDimensions: map.width.flatMap { width in map.height.map { [width, $0] } })
        let folder = map.url.deletingLastPathComponent()
        guard let sampleID,
              let metadataBytes = try? Data(contentsOf: folder.appendingPathComponent("sample.json")),
              let metadata = (try? JSONSerialization.jsonObject(with: metadataBytes)) as? [String: Any],
              metadata["sample_id"] as? String == sampleID,
              let maps = metadata["maps"] as? [String: String], let filename = maps[target],
              folder.appendingPathComponent(filename).resolvingSymlinksInPath().standardizedFileURL == map.url.resolvingSymlinksInPath().standardizedFileURL,
              let mapMetadata = metadata["map_metadata"] as? [String: [String: Any]], let details = mapMetadata[target],
              let recordedHash = details["sample_sha256"] as? String,
              map.sha256 == nil || map.sha256 == recordedHash,
              let bytes = try? Data(contentsOf: map.url, options: .mappedIfSafe), ReviewImageLoader.hash(bytes) == recordedHash,
              let source = details["original_source"] as? [String: Any] ?? details["source"] as? [String: Any],
              let path = source["path"] as? String, !path.isEmpty,
              let width = source["width"] as? Int, let height = source["height"] as? Int, width > 0, height > 0,
              let crop = metadata["crop_rectangle_top_left_xywh"] as? [Int], crop.count == 4,
              crop.allSatisfy({ $0 >= 0 }), crop[2] > 0, crop[3] > 0,
              crop[0] <= width - crop[2], crop[1] <= height - crop[3],
              (map.width == nil || map.width == crop[2]), (map.height == nil || map.height == crop[3]) else { return fallback }
        let original = path.hasPrefix("/") ? URL(fileURLWithPath: path) : folder.appendingPathComponent(path)
        return MapReviewSourceIdentity(path: original.path, sha256: source["file_sha256"] as? String,
            url: source["published_url"] as? String ?? metadata["source_url"] as? String,
            bits: source["sample_bits"] as? Int, cropRectangle: crop, pixelDimensions: [width, height])
    }
}

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
    let sourceIdentity: MapReviewSourceIdentity?

    init(id: String? = nil, label: String, mapURL: URL, numeric: Bool,
         sampleLabel: String? = nil, detail: String? = nil, role: String = "map",
         modelIdentity: MapReviewModelIdentity? = nil, sourceIdentity: MapReviewSourceIdentity? = nil) {
        self.id = id ?? mapURL.path + "|" + label
        self.label = label; self.mapURL = mapURL; self.numeric = numeric
        self.sampleLabel = sampleLabel; self.detail = detail; self.role = role
        self.modelIdentity = modelIdentity
        self.sourceIdentity = sourceIdentity
    }

    var accessibleLabel: String { [sampleLabel, label, detail].compactMap { $0 }.joined(separator: " · ") }

    var isReference: Bool { role == "target" || (role == "map" && label.lowercased() == "target") }
    var referenceMapName: String { modelIdentity?.mapType == "height" ? "displacement" : modelIdentity?.mapType ?? "map" }

    /// The full original target can be inspected without generating a new
    /// prediction. A comparison crop is never described as a full-frame result.
    var fullSourceReference: MapReviewCandidate? {
        guard isReference, let sourceIdentity, let path = sourceIdentity.path, !path.isEmpty else { return nil }
        let source = URL(fileURLWithPath: path)
        guard source.standardizedFileURL != mapURL.standardizedFileURL else { return nil }
        return MapReviewCandidate(id: "full-source|" + source.path, label: "Full source \(referenceMapName) · reference",
            mapURL: source, numeric: true, sampleLabel: sampleLabel,
            detail: (["Full original source map · not a model output"] + sourceIdentity.recordedDetails).joined(separator: " · "),
            role: "target", modelIdentity: MapReviewModelIdentity(mapType: modelIdentity?.mapType ?? "height", modelName: "Real source reference"),
            sourceIdentity: sourceIdentity)
    }

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
