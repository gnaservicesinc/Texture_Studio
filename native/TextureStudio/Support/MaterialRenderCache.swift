import CoreImage
import Foundation

struct MaterialRenderKey: Equatable, Sendable {
    let sourceIdentity: String
    let depthIdentity: String
    var settings: TextureSettings

    init(sourceIdentity: String, depthIdentity: String, settings: TextureSettings) {
        self.sourceIdentity = sourceIdentity
        self.depthIdentity = depthIdentity
        self.settings = settings
        // Storage precision changes export encoding, not the rendered maps.
        self.settings.exrPrecision = .float32
        self.settings.useEmbeddedDepth = false
    }
}

/// One generated material, shared by Studio and any open inspection windows.
/// Float map files preserve native samples; only display images are 8-bit.
final class MaterialRenderCache: @unchecked Sendable {
    let folder: URL
    let key: MaterialRenderKey
    let material: MaterialResult

    static func make(material: MaterialResult, key: MaterialRenderKey, engine: TextureEngine) async throws -> MaterialRenderCache {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("texture-material-cache-\(UUID().uuidString)", isDirectory: true)
        var complete = false
        defer { if !complete { try? FileManager.default.removeItem(at: folder) } }
        _ = try await engine.export(material, to: folder, precision: .float32)
        try Task.checkCancellation()
        func image(_ name: String, numeric: Bool) throws -> CIImage {
            let options: [CIImageOption: Any] = numeric ? [.colorSpace: NSNull()] : [:]
            guard let image = CIImage(contentsOf: folder.appendingPathComponent(name), options: options) else {
                throw TextureError.processing("Could not retain the generated \(name) map.")
            }
            return image
        }
        // Reading the completed files freezes the lazy processing graph. Later
        // inspection/encoding cannot accidentally rerun lighting or ML work.
        let frozen = try MaterialResult(diffuse: image("diffuse.png", numeric: false),
            roughness: image("roughness.exr", numeric: true), normal: image("normal.exr", numeric: true),
            height: image("displacement.exr", numeric: true), crop: material.crop, warnings: material.warnings,
            outputSize: material.outputSize, depthOrigin: material.depthOrigin, settings: material.settings,
            sourceURL: material.sourceURL, camera: material.camera)
        let cache = MaterialRenderCache(folder: folder, key: key, material: frozen)
        complete = true
        return cache
    }

    private init(folder: URL, key: MaterialRenderKey, material: MaterialResult) {
        self.folder = folder; self.key = key; self.material = material
    }

    func mapURL(_ selection: MaterialPreview) -> URL? {
        let name: String
        switch selection {
        case .source: return nil
        case .diffuse: name = "diffuse.png"
        case .roughness: name = "roughness.exr"
        case .normal: name = "normal.exr"
        case .height: name = "displacement.exr"
        }
        return folder.appendingPathComponent(name)
    }

    func export(to destination: URL, precision: EXRPrecision, settings: TextureSettings, engine: TextureEngine) async throws {
        if precision == .float32 {
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            for name in ["diffuse.png", "roughness.exr", "normal.exr", "displacement.exr", "material.json", "Blender-setup.txt"] {
                try Task.checkCancellation()
                try FileManager.default.copyItem(at: folder.appendingPathComponent(name), to: destination.appendingPathComponent(name))
            }
        } else {
            // Encode existing Float32 maps as half floats; no inference or
            // material processing is repeated for this storage choice.
            _ = try await engine.export(material, to: destination, precision: precision)
        }
        let metadata = destination.appendingPathComponent("material.json")
        if var document = try JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as? [String: Any] {
            document["settings"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(settings))
            document["precision"] = precision.rawValue
            try JSONSerialization.data(withJSONObject: document, options: [.prettyPrinted, .sortedKeys]).write(to: metadata, options: .atomic)
        }
    }

    deinit { try? FileManager.default.removeItem(at: folder) }
}
