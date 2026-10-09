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
    let width: Int?
    let height: Int?
    var originalSourcePath: String? = nil
    var originalSourceSha256: String? = nil
    var originalSourceWidth: Int? = nil
    var originalSourceHeight: Int? = nil
    var originalNormalConvention: String? = nil
    var sourceNormalConvention: String? = nil
    var cropRectangle: [Int]? = nil
    var variantId: String? = nil
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
    var sourceFamilyId: String? = nil
    var sourceSetId: String? = nil
    var inputVariants: [WorkbenchMap]? = nil
    var availableTargets: [String]? = nil
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
    var supportedTrainingSizes: [Int]? = nil
    var samples: [WorkbenchSample] { materials.flatMap(\.samples) }
    func readyForTraining(size: Int, material: String?, target: String? = nil) -> Bool {
        guard hasNativeSize(size), let policy = automaticValidation,
              policy.policy == "source-family-native-regions-v1" else { return false }
        if let target, let preparedTarget = policy.target, preparedTarget != target { return false }
        if let material { return policy.quickFitMaterialId == material }
        return policy.quickFitMaterialId == nil
    }
    func hasNativeSize(_ size: Int) -> Bool {
        !samples.isEmpty && samples.allSatisfy { sample in
            sample.width == size && sample.height == size && !sample.maps.isEmpty &&
                sample.maps.values.allSatisfy { $0.width == size && $0.height == size }
        }
    }
}

struct WorkbenchAutomaticValidation: Decodable, Sendable {
    let policy: String
    let materialIds: [String]
    let quickFitMaterialId: String?
    var target: String? = nil
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
    var targetCropped: Bool? = nil
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
    var architecture: String? = nil
    var scope: String? = nil
    enum CodingKeys: String, CodingKey {
        case checkpointPath, sha256, schema, target, step, compatible, variant, refinementPolicy, architecture, scope
        case warmStartSupported = "supportsTrainingWarmStart"
    }
    var supportsTrainingWarmStart: Bool { compatible && warmStartSupported == true }
    var supportsStudioInference: Bool {
        compatible && ["texture-studio-material-lora-v1", "texture-studio-material-checkpoint-v1"].contains(schema)
    }
    var availabilityLabel: String { variant == "full" ? "Full material checkpoint" : "Material LoRA" }
    var id: String { sha256 }
    var url: URL { URL(fileURLWithPath: checkpointPath) }
    var title: String { url.deletingLastPathComponent().lastPathComponent + " · " + url.lastPathComponent }
    var trainingBaseLabel: String {
        architecture ?? "PBRnxt material model"
    }
    var modelSummary: String {
        let map = target == "height" ? "Surface height / displacement" : target.capitalized
        return "\(map) · \(trainingBaseLabel) · step \(step.formatted())"
    }
}

struct MaterialTrainingOptions: Codable, Equatable, Sendable {
    var target = "height"
    var scope = "final-map"
    var size = 1024
    var updatesPerCrop = 100
    var maxMinutes = 30.0
    /// GiB throughout the UI and CLI; retain the field name for existing callers.
    var memoryGB = MachineResources.current.defaultTrainingGiB
    var automaticMemory = true
    var useSelectedMaterialOnly = false
    var useWarmStart = false
    var loraRank = 8
    var loraAlpha = 8.0
    var cacheGB = 0.5

    init() {}

    enum CodingKeys: String, CodingKey {
        case target, scope, size, updatesPerCrop, maxMinutes, memoryGB, automaticMemory, useSelectedMaterialOnly, useWarmStart, loraRank, loraAlpha, cacheGB
    }

    init(from decoder: Decoder) throws {
        self.init()
        let values = try decoder.container(keyedBy: CodingKeys.self)
        target = try values.decodeIfPresent(String.self, forKey: .target) ?? target
        scope = try values.decodeIfPresent(String.self, forKey: .scope) ?? scope
        size = try values.decodeIfPresent(Int.self, forKey: .size) ?? size
        updatesPerCrop = try values.decodeIfPresent(Int.self, forKey: .updatesPerCrop) ?? updatesPerCrop
        maxMinutes = try values.decodeIfPresent(Double.self, forKey: .maxMinutes) ?? maxMinutes
        memoryGB = try values.decodeIfPresent(Double.self, forKey: .memoryGB) ?? memoryGB
        automaticMemory = try values.decodeIfPresent(Bool.self, forKey: .automaticMemory) ?? true
        useSelectedMaterialOnly = try values.decodeIfPresent(Bool.self, forKey: .useSelectedMaterialOnly) ?? useSelectedMaterialOnly
        useWarmStart = try values.decodeIfPresent(Bool.self, forKey: .useWarmStart) ?? useWarmStart
        loraRank = try values.decodeIfPresent(Int.self, forKey: .loraRank) ?? loraRank
        loraAlpha = try values.decodeIfPresent(Double.self, forKey: .loraAlpha) ?? loraAlpha
        cacheGB = try values.decodeIfPresent(Double.self, forKey: .cacheGB) ?? cacheGB
    }

    /// A stored memory setting can come from a different Mac. Preserve every
    /// supported choice; adapt only values outside this machine's actual limits.
    func restored(for resources: MachineResources) -> Self {
        var result = self
        if !["height", "roughness", "normal"].contains(result.target) { result.target = "height" }
        if !["final-map", "map-decoder"].contains(result.scope) { result.scope = "final-map" }
        if ![256, 512, 1024, 2048].contains(result.size) { result.size = 1024 }
        result.loraRank = min(64, max(1, result.loraRank))
        result.loraAlpha = result.loraAlpha.isFinite ? min(128, max(0.01, result.loraAlpha)) : 8
        result.cacheGB = result.cacheGB.isFinite ? min(4, max(0, result.cacheGB)) : 0.5
        result.updatesPerCrop = min(10_000, max(1, result.updatesPerCrop))
        result.maxMinutes = result.maxMinutes.isFinite ? min(240, max(1, result.maxMinutes)) : 30
        if resources.trainingMemoryIssue(result.memoryGB) != nil {
            result.memoryGB = result.memoryGB.isFinite
                ? min(resources.maximumTrainingGiB, max(resources.trainingMemoryRange.lowerBound, result.memoryGB))
                : resources.defaultTrainingGiB
        }
        return result
    }
}

struct WorkbenchPreferences: Codable {
    var training: MaterialTrainingOptions?
    var selectedSampleId: String?
    var selectedRole: String?
    var selectedInputVariantId: String?
    var selectedCheckpointId: String?
    var comparisonCheckpointIds: Set<String>?
    var comparisonIncludesBase: Bool?
    var sourceImagePath: String?
    var lastOutputPath: String?
    var lastLogPath: String?
    var lastPackagePath: String?
    var lastPackageCheckpointId: String?
    static let key = "workbenchSettings.v1"

    static func load(from defaults: UserDefaults) -> Self {
        guard let data = defaults.data(forKey: key), let value = try? JSONDecoder().decode(Self.self, from: data) else { return Self() }
        return value
    }
    func save(to defaults: UserDefaults) {
        guard let data = try? JSONEncoder().encode(self) else { return }
        defaults.set(data, forKey: Self.key)
    }
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
    var displayName: String? = nil
    var modelSummary: String? = nil
    var supportsStudioInference: Bool { ["height", "normal", "roughness"].contains(target) && URL(fileURLWithPath: checkpointPath).pathExtension == "safetensors" }
    var selectionIdentity: String { checkpointPath + "|" + sha256 }
    var title: String {
        displayName ?? (URL(fileURLWithPath: checkpointPath).deletingLastPathComponent().lastPathComponent
            + " · " + URL(fileURLWithPath: checkpointPath).lastPathComponent)
    }
    static let changeNotification = Notification.Name("org.ipde.texture-studio.material-selection-changed")
    static var registryURL: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Texture Studio/selected-material-checkpoint.json")
    }
    static func registryURL(for target: String, heightRegistryURL: URL = registryURL) -> URL {
        target == "height" ? heightRegistryURL : heightRegistryURL.deletingPathExtension().appendingPathExtension(target + ".json")
    }
    static func readAll(heightRegistryURL: URL = registryURL) -> [String: Self] {
        Dictionary(uniqueKeysWithValues: ["height", "roughness", "normal"].compactMap { target in
            guard let checkpoint = try? read(from: registryURL(for: target, heightRegistryURL: heightRegistryURL)),
                  checkpoint.target == target, checkpoint.supportsStudioInference else { return nil }
            return (target, checkpoint)
        })
    }
    static func read(from url: URL = registryURL) throws -> Self { try JSONDecoder().decode(Self.self, from: Data(contentsOf: url)) }

    func save(to requestedURL: URL? = nil) throws {
        let url = requestedURL ?? Self.registryURL(for: target)
        try Self.withRegistryLock(at: url) {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            try encoder.encode(self).write(to: url, options: .atomic)
        }
        if ["height", "roughness", "normal"].contains(where: { url.standardizedFileURL == Self.registryURL(for: $0).standardizedFileURL }) {
            let information = ["selectionID": UUID().uuidString, "target": target]
            NotificationCenter.default.post(name: Self.changeNotification, object: nil, userInfo: information)
            DistributedNotificationCenter.default().postNotificationName(Self.changeNotification, object: nil, userInfo: information, deliverImmediately: true)
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

struct WorkbenchTrainingCapabilities: Decodable, Sendable {
    let trainingSizes: [Int]
    var memoryPlans: [String: WorkbenchMemoryPlan]? = nil
}
struct WorkbenchMemoryPlan: Decodable, Sendable {
    let requiredMemoryGib: Double
    let recommendedMemoryGib: Double
}
struct WorkbenchTrainingResponse: Decodable, Sendable {
    let checkpointPath: String
    let packagePath: String?
}
struct WorkbenchAdapterWeight: Identifiable {
    let id = UUID()
    var path: String
    var weight = 1.0
}
struct WorkbenchHubModel: Decodable, Identifiable, Sendable {
    let repository: String
    let revision: String?
    let target: String?
    var id: String { repository }
}
struct WorkbenchHubModels: Decodable, Sendable {
    let models: [WorkbenchHubModel]
}

struct WorkbenchDatasetCleanup: Decodable {
    let datasetPath: String
    let sourceDatasetPath: String?
    let removed: Bool
}
