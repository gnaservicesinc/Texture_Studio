import Foundation

enum MaterialPreview: String, CaseIterable, Identifiable {
    case source = "Photo"
    case diffuse = "Diffuse"
    case roughness = "Roughness"
    case normal = "Normal"
    case height = "Displacement"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .source: "photo"
        case .diffuse: "paintpalette"
        case .roughness: "circle.lefthalf.filled"
        case .normal: "cube.transparent"
        case .height: "mountain.2"
        }
    }
}

enum DepthChoice: String, Codable, CaseIterable, Identifiable {
    case photoDetail = "Flat surface"
    case attached = "Attached depth"
    case model = "Local ML depth"
    case materialCheckpoint = "Material checkpoint"
    var id: String { rawValue }
    static let studioChoices: [DepthChoice] = [.photoDetail, .attached, .materialCheckpoint]

    var title: String {
        switch self {
        case .photoDetail: "Flat surface"
        case .attached: "Attached height / depth map"
        case .model: "Custom depth model"
        case .materialCheckpoint: "Trained material model"
        }
    }


}

struct TextureRecipe: Codable {
    var version = 3
    var photoPath: String
    var depthPath: String?
    var depthChoice: DepthChoice
    var modelID: String
    var customInverseDepth: Bool
    var settings: TextureSettings
    var materialCheckpoint: MaterialCheckpointIdentity? = nil
    var materialMapCheckpoints: [String: MaterialCheckpointIdentity]? = nil
}

/// Recipes identify model data, independently of the native runtime.
struct MaterialCheckpointIdentity: Codable, Sendable {
    let checkpointPath: String
    let sha256: String
    let displayName: String?
    let modelSummary: String?
    let target: String

    init(_ checkpoint: SelectedMaterialCheckpoint) {
        checkpointPath = checkpoint.checkpointPath
        sha256 = checkpoint.sha256
        displayName = checkpoint.displayName
        modelSummary = checkpoint.modelSummary
        target = checkpoint.target
    }

    func resolve(using runtime: SelectedMaterialCheckpoint) -> SelectedMaterialCheckpoint {
        SelectedMaterialCheckpoint(checkpointPath: checkpointPath, sha256: sha256, target: target,
            workspacePath: runtime.workspacePath,
            modelDirectory: runtime.modelDirectory,
            displayName: displayName, modelSummary: modelSummary)
    }
}

struct StudioNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}
