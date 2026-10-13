import Foundation

enum MaterialPreview: String, CaseIterable, Identifiable {
    case source = "Photo", diffuse = "Diffuse", roughness = "Roughness", normal = "Normal", height = "Displacement"
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
    case textureLoom = "Texture Loom", attached = "Attached height / depth map", photoDetail = "Flat surface"
    var id: String { rawValue }
    var title: String { rawValue }
    static let studioChoices = allCases
}

struct TextureRecipe: Codable {
    var version = 4
    var photoPath: String
    var depthPath: String?
    var depthChoice: DepthChoice
    var settings: TextureSettings
    var textureLoomID: String?
    var textureLoomVersion: String?
    var enabledLoRAs: [String]
}

struct StudioNotice: Identifiable {
    let id = UUID()
    let title: String
    let message: String
}

struct StudioError: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
