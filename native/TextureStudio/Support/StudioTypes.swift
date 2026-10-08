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

    init(from decoder: Decoder) throws {
        let value = try decoder.singleValueContainer().decode(String.self)
        switch value {
        case "Photo detail", "Flat surface": self = .photoDetail
        case "Embedded depth": self = .model
        case "Attached depth": self = .attached
        case "Local ML depth": self = .model
        case "Material checkpoint": self = .materialCheckpoint
        default: throw DecodingError.dataCorruptedError(in: try decoder.singleValueContainer(), debugDescription: "Unknown depth source")
        }
    }
}

struct TextureRecipe: Codable {
    var version = 2
    var photoPath: String
    var depthPath: String?
    var depthChoice: DepthChoice
    var modelID: String
    var customInverseDepth: Bool
    var settings: TextureSettings
}

struct StudioNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}
