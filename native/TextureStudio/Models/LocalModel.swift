import Foundation

struct ModelDownloadArtifact: Codable, Sendable, Equatable {
    let relativePath: String
    let url: URL
    let byteCount: Int64
    let sha256: String
}

enum LocalModelBackend: String, Codable, Sendable { case coreML, pytorchDA3 }

struct LocalModelDescriptor: Identifiable, Sendable {
    let id: String
    let name: String
    let summary: String
    let license: String
    let licenseURL: URL
    let sourceURL: URL
    let packageName: String
    let artifacts: [ModelDownloadArtifact]
    var downloadBytes: Int64 { artifacts.reduce(0) { $0 + $1.byteCount } }
    var downloadable: Bool { !artifacts.isEmpty }

    static let depthAnythingSmallID = "depth-anything-v2-small"
    static let customDepthID = "custom-coreml-depth"

    static let da3GiantID = "da3-giant-1.1"
    static let da3Revision = "72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19"
    static let da3WeightsSHA256 = "1e47a08338ca73a6d6a21d37fd060b26b993b672bc6ddf6295fe474df2592001"
    var backend: LocalModelBackend = .coreML

    static let catalog: [LocalModelDescriptor] = {
        let repository = "https://huggingface.co/depth-anything/DA3-GIANT-1.1"
        func artifact(_ path: String, _ bytes: Int64, _ hash: String) -> ModelDownloadArtifact {
            ModelDownloadArtifact(relativePath: path,
                url: URL(string: "\(repository)/resolve/\(da3Revision)/\(path)")!,
                byteCount: bytes, sha256: hash)
        }
        return [
            LocalModelDescriptor(id: da3GiantID, name: "Depth Anything 3 GIANT 1.1",
                summary: "The exact GIANT 1.1 checkpoint, 5.42 GB. Local PyTorch on Apple Metal. Relative camera-Z depth; default inference edge 1036 pixels. CC BY-NC 4.0 permits noncommercial use only.",
                license: "CC BY-NC 4.0 — noncommercial", licenseURL: URL(string: "https://creativecommons.org/licenses/by-nc/4.0/")!,
                sourceURL: URL(string: repository)!, packageName: "DA3-GIANT-1.1",
                artifacts: [
                    artifact("config.json", 1880, "74626a50d6dee2a11820291a4305c1a34aa5adc4f7260908bbdbc9a939ba8e93"),
                    artifact("model.safetensors", 5422814644, da3WeightsSHA256),
                    artifact("README.md", 4881, "b939f8754d5147d42e926f20073ed59912f751eeb7c727cd4697b0f771c1c487")
                ], backend: .pytorchDA3),
            LocalModelDescriptor(id: customDepthID, name: "Your Core ML depth model",
                summary: "Optional image-to-depth .mlpackage, .mlmodel, or .mlmodelc. Choose the output explicitly if needed. Your original file stays in place.",
                license: "Your model's license", licenseURL: URL(string: "https://developer.apple.com/documentation/coreml")!,
                sourceURL: URL(string: "https://developer.apple.com/documentation/coreml")!,
                packageName: "Custom.mlpackage", artifacts: [])
        ]
    }()
}

struct ModelOutputDescriptor: Codable, Identifiable, Sendable, Equatable {
    let name: String
    let width: Int
    let height: Int
    let storage: String
    var id: String { name }
}

struct ModelInterface: Codable, Sendable, Equatable {
    let inputName: String
    let inputWidth: Int
    let inputHeight: Int
    let outputs: [ModelOutputDescriptor]
    let interpretation: String
    var unambiguousOutput: String? { outputs.count == 1 ? outputs.first?.name : nil }
}

struct InstalledLocalModel: Codable, Sendable, Equatable {
    let path: String
    let isManaged: Bool
    let interface: ModelInterface
    var selectedOutput: String?
    let installedAt: Date
}

enum LocalModelStatus: Equatable {
    case missing
    case ready
    case downloading(Double)
    case validating
    case failed(String)

    var isBusy: Bool {
        switch self { case .downloading, .validating: true; default: false }
    }
    var progress: Double? { if case let .downloading(value) = self { value } else { nil } }
    var message: String {
        switch self {
        case .missing: "Model missing. Locate it or download the optional model."
        case .ready: "Ready on this Mac"
        case .downloading(let progress): "Downloading \(Int(progress * 100))%"
        case .validating: "Checking the model files…"
        case .failed(let message): message
        }
    }
}

enum LocalModelError: LocalizedError {
    case unknownModel
    case missingModel
    case busy
    case unsupported(String)
    case invalidDownload(String)
    case unsafeRemoval
    case chooseOutput([String])

    var errorDescription: String? {
        switch self {
        case .unknownModel: "This model is not in the local catalog."
        case .missingModel: "The model could not be found. Locate its new path, download it, or continue without ML."
        case .busy: "The model is in use by an installation. Cancel or wait for it to finish."
        case .unsupported(let reason): "Unsupported local model: \(reason)"
        case .invalidDownload(let reason): "Model download failed validation: \(reason). Retry the download."
        case .unsafeRemoval: "The model's managed path changed. Unlink it instead; no file was deleted."
        case .chooseOutput(let names): "Choose the depth output before running this model: \(names.joined(separator: ", "))."
        }
    }
}

struct ModelDepthProvenance: Codable, Sendable, Equatable {
    let modelID: String
    let revision: String
    let checkpointSHA256: String
    let upstreamRevision: String
    let backend: String
    let device: String
    let precision: String
    let inputWidth: Int
    let inputHeight: Int
    let processResolution: Int
    let fullSourceFieldOfView: Bool
    let rowOrder: String
    let elapsedSeconds: Double
    var residentPeakBytes: UInt64? = nil
    var metalAllocatedBytes: UInt64? = nil
    var metalDriverAllocatedBytes: UInt64? = nil
}

struct ModelDepthResult: Sendable {
    let width: Int
    let height: Int
    let values: [Float]
    let outputName: String
    let interpretation: String
    var provenance: ModelDepthProvenance? = nil
}
