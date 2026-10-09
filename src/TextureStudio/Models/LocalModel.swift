import Foundation

struct ModelDownloadArtifact: Codable, Sendable, Equatable {
    let relativePath: String
    let url: URL
    let byteCount: Int64
    let sha256: String
}

enum LocalModelBackend: String, Codable, Sendable { case coreML }

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

    static let customDepthID = "custom-coreml-depth"
    var backend: LocalModelBackend = .coreML
    static let catalog: [LocalModelDescriptor] = [
        LocalModelDescriptor(id: customDepthID, name: "Your Core ML depth model",
            summary: "Developer image-to-depth .mlpackage, .mlmodel, or .mlmodelc. Choose the output explicitly if needed. Your original file stays in place.",
            license: "Your model's license", licenseURL: URL(string: "https://developer.apple.com/documentation/coreml")!,
            sourceURL: URL(string: "https://developer.apple.com/documentation/coreml")!,
            packageName: "Custom.mlpackage", artifacts: [])
    ]

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
