import Foundation

struct TextureLoomManifest: Codable, Sendable {
    struct File: Codable, Sendable { let path: String; let sha256: String }
    struct Label: Codable, Sendable {
        let id: String; let name: String; let sourceLabel: String?
        var classifierValue: String { sourceLabel ?? id }
    }
    struct Classifier: Codable, Sendable {
        let format: String; let path: String; let weightsPath: String?; let architecture: String
        let inputName: String?; let outputName: String?
        let inputWidth: Int?; let inputHeight: Int?; let inputColorSpace: String?
        let inputType: String?; let inputShape: [Int]?
        let labels: [Label]
    }
    struct Model: Codable, Sendable {
        let map: String; let format: String; let path: String; let weightsPath: String; let architecture: String
    }
    struct Category: Codable, Sendable, Identifiable {
        let id: String; let name: String; let version: String; let models: [Model]
    }
    let schema: String; let id: String; let name: String; let version: String
    let classifier: Classifier; let categories: [Category]; let files: [File]
}

struct TextureLoomAdapterManifest: Codable, Sendable {
    struct Base: Codable, Sendable { let id: String; let version: String }
    struct Adapter: Codable, Sendable {
        let map: String; let format: String; let path: String; let weightsPath: String; let baseSHA256: String
    }
    struct Category: Codable, Sendable { let id: String; let version: String; let adapters: [Adapter] }
    let schema: String; let id: String; let name: String; let version: String
    let baseModel: Base; let categories: [Category]; let files: [TextureLoomManifest.File]
}
