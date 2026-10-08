import Foundation
import CoreImage
import ImageIO

enum StudioSmoke {
    static func run() async throws {
        if let directory = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_SMOKE_IMPORT_DIR"] {
            try await StudioSurfaceValidation.importDirectory(directory)
            return
        }
        if ProcessInfo.processInfo.environment["TEXTURE_STUDIO_SMOKE_DA3"] == "1" {
            try await StudioSurfaceValidation.run()
            return
        }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("texture-studio-smoke-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let engine = TextureEngine()
        let source: TextureSource
        if let path = ProcessInfo.processInfo.environment["TEXTURE_STUDIO_SMOKE_PHOTO"] {
            source = try await engine.importPhoto(URL(fileURLWithPath: path))
        } else {
            let image = CIImage(color: CIColor(red: 0.28, green: 0.4, blue: 0.55)).cropped(to: CGRect(x: 0,y: 0,width: 1300,height: 1100))
            source = TextureSource(url: temporary.appendingPathComponent("test-photo.png"), orientedImage: image,
                embeddedDepth: nil, camera: CameraMetadata(), pixelWidth: 1300, pixelHeight: 1100)
        }
        var settings = TextureSettings()
        settings.rotationX = 7.93
        settings.rotationZ = 0.22
        settings.lightingStrength = 0
        settings.heightDetail = 0
        settings.useEmbeddedDepth = false
        let material = try await engine.process(source: source, settings: settings)
        let preview = try await engine.preview(material.diffuse)
        guard preview.width == 1024, preview.height == 1024 else { throw StudioError("Unexpected preview dimensions") }
        for precision in EXRPrecision.allCases {
            let folder = temporary.appendingPathComponent(precision.rawValue)
            let exported = try await engine.export(material, to: folder, precision: precision)
            try BlenderMaterialScript.write(to: folder, settings: settings)
            guard exported.count >= 4 else { throw StudioError("Missing material export maps") }
            for name in ["roughness.exr", "normal.exr", "displacement.exr"] {
                let url = folder.appendingPathComponent(name)
                try FloatEXRWriter.verifyChannelPrecision(at: url, expected: precision)
                guard let image = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let properties = CGImageSourceCopyPropertiesAtIndex(image, 0, nil) as? [String: Any],
                      (properties[kCGImagePropertyPixelWidth as String] as? NSNumber)?.intValue == 1024,
                      (properties[kCGImagePropertyPixelHeight as String] as? NSNumber)?.intValue == 1024 else {
                    throw StudioError("Exported EXR failed ImageIO decode")
                }
            }
        }
    }
}
