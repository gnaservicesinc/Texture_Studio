import Foundation
import Darwin

enum MaterialTool: String, CaseIterable, Identifiable {
    case review, compare, dataset, train
    var id: String { rawValue }
    var title: String {
        switch self {
        case .review: "Material Review"
        case .compare: "Checkpoint Compare"
        case .dataset: "Material Dataset"
        case .train: "Material Trainer"
        }
    }
    var symbol: String {
        switch self { case .review: "photo.on.rectangle"; case .compare: "rectangle.split.2x1"; case .dataset: "square.stack.3d.up"; case .train: "cpu" }
    }
    static var launchRole: MaterialTool? {
        if let i = CommandLine.arguments.firstIndex(of: "--tool"), CommandLine.arguments.indices.contains(i + 1) {
            return MaterialTool(rawValue: CommandLine.arguments[i + 1])
        }
        return (Bundle.main.object(forInfoDictionaryKey: "MaterialToolRole") as? String).flatMap(MaterialTool.init(rawValue:))
    }
}

struct WorkbenchMap: Decodable, Identifiable, Sendable {
    let path: String
    let sha256: String?
    let sourceBits: Int?
    let encoding: String?
    var id: String { path }
    var url: URL { URL(fileURLWithPath: path) }
}

struct WorkbenchSample: Decodable, Identifiable, Sendable {
    let sampleId: String
    let status: String
    let split: String
    let width: Int
    let height: Int
    let maps: [String: WorkbenchMap]
    let note: String?
    var id: String { sampleId }
}

struct WorkbenchMaterial: Decodable, Identifiable, Sendable {
    let materialId: String
    let samples: [WorkbenchSample]
    var id: String { materialId }
}

struct WorkbenchDataset: Decodable, Sendable {
    let datasetPath: String
    let indexSha256: String
    let materials: [WorkbenchMaterial]
    let validationScope: String?
    let crossSizeValidationNotice: String?
    let preparation: WorkbenchDatasetPreparation?
    let automaticValidation: WorkbenchAutomaticValidation?
    var samples: [WorkbenchSample] { materials.flatMap(\.samples) }
    func readyForTraining(size: Int, material: String?) -> Bool {
        guard hasNativeSize(size), let policy = automaticValidation,
              policy.policy == "automatic-material-check-5pct-v1" else { return false }
        if let material { return policy.materialIds.contains(material) }
        return policy.quickFitMaterialId == nil
    }
    func hasNativeSize(_ size: Int) -> Bool {
        !samples.isEmpty && samples.allSatisfy { $0.width == size && $0.height == size }
    }
}

struct WorkbenchAutomaticValidation: Decodable, Sendable {
    let policy: String
    let materialIds: [String]
    let quickFitMaterialId: String?
}

struct WorkbenchDatasetPreparation: Decodable, Sendable {
    let sourceDatasetPath: String
    let sourceIndexSha256: String
    let preparedDatasetPath: String
    let cropSize: Int
    let reused: Bool
    let targetResized: Bool
    let originalDatasetModified: Bool
    let splitLineageChanged: Bool?
    let crossSizeValidationNotice: String?
}

struct WorkbenchCheckpoint: Decodable, Identifiable, Sendable {
    let checkpointPath: String
    let sha256: String
    let schema: String
    let target: String
    let step: Int
    let compatible: Bool
    let variant: String?
    let warmStartSupported: Bool?
    let refinementPolicy: String?
    enum CodingKeys: String, CodingKey {
        case checkpointPath, sha256, schema, target, step, compatible, variant, refinementPolicy
        case warmStartSupported = "supportsTrainingWarmStart"
    }
    var supportsTrainingWarmStart: Bool { compatible && (warmStartSupported ?? (variant != "lora")) }
    var id: String { sha256 }
    var url: URL { URL(fileURLWithPath: checkpointPath) }
    var title: String { url.deletingLastPathComponent().lastPathComponent + " · " + url.lastPathComponent }
}

struct MaterialTrainingOptions: Equatable, Sendable {
    var target = "height"
    var size = 1024
    var updatesPerCrop = 100
    var maxMinutes = 30.0
    /// GiB throughout the UI and CLI; retain the field name for existing callers.
    var memoryGB = MachineResources.current.defaultTrainingGiB
    var allowUnreviewed = true
    var maskTransparency = true
    var useSelectedMaterialOnly = false
    var useWarmStart = false
}

enum MaterialWorkbenchRuntime {
    /// Reuse a located Studio runtime when a distribution has no repository
    /// environment. Existence is only discovery; the backend verifies imports.
    static func defaultPython(workspace: URL, registry: URL? = nil) -> String {
        let repositoryPython = workspace.appendingPathComponent(".venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: repositoryPython.path) { return repositoryPython.path }
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let registryURL = registry ?? support.appendingPathComponent("Texture Studio/Runtimes/python-runtime.json")
        if let data = try? Data(contentsOf: registryURL),
           let record = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let path = record["path"] as? String, path.hasPrefix("/"),
           FileManager.default.isExecutableFile(atPath: path) { return path }
        return repositoryPython.path
    }
}

struct MaterialInferenceResponse: Decodable, Sendable {
    struct Output: Decodable, Sendable { let path: String }
    let outputs: [String: Output]
    let checkpointSha256: String
}

struct SelectedMaterialCheckpoint: Codable, Sendable {
    let checkpointPath: String
    let sha256: String
    let target: String
    let pythonPath: String
    let workspacePath: String
    let modelDirectory: String
    let codeDirectory: String
    static var registryURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/selected-material-checkpoint.json")
    }
    static func read() throws -> Self { try JSONDecoder().decode(Self.self, from: Data(contentsOf: registryURL)) }

    func save(to url: URL = registryURL) throws {
        try Self.withRegistryLock(at: url) {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(self).write(to: url, options: .atomic)
        }
    }

    /// Runtime changes reconnect the existing selected model. They never select
    /// a library row or change its checkpoint identity as a side effect.
    static func refreshRuntime(pythonPath: String, workspacePath: String, modelDirectory: String,
                               codeDirectory: String, at url: URL = registryURL) throws {
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        try withRegistryLock(at: url) {
            let raw = try Data(contentsOf: url)
            let selected = try JSONDecoder().decode(Self.self, from: raw)
            guard selected.pythonPath != pythonPath || selected.workspacePath != workspacePath
                    || selected.modelDirectory != modelDirectory || selected.codeDirectory != codeDirectory else { return }
            guard var document = try JSONSerialization.jsonObject(with: raw) as? [String: Any] else {
                throw CocoaError(.fileReadCorruptFile)
            }
            // Preserve any newer metadata fields alongside the exact selected
            // checkpoint path, SHA256 and target.
            document["pythonPath"] = pythonPath
            document["workspacePath"] = workspacePath
            document["modelDirectory"] = modelDirectory
            document["codeDirectory"] = codeDirectory
            try JSONSerialization.data(withJSONObject: document, options: [.sortedKeys])
                .write(to: url, options: .atomic)
        }
    }

    private static func withRegistryLock(at url: URL, body: () throws -> Void) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let descriptor = Darwin.open(url.appendingPathExtension("lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { Darwin.close(descriptor) }
        guard flock(descriptor, LOCK_EX) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { flock(descriptor, LOCK_UN) }
        try body()
    }
}
