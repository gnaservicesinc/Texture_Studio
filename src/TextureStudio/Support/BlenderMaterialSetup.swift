import Foundation

enum BlenderMaterialSetup {
    struct Document: Codable, Sendable {
        struct Image: Codable, Sendable {
            let filename: String
            let colorSpace: String
            let destination: String
            let encoding: String
        }
        let schema: String
        let images: [String: Image]
        let normalConvention: String
        let materialWidthMeters: Double
        let displacementScaleMeters: Double
        let displacementMidlevel: Double
        let reliefMode: String
        let geometricDisplacementNormalStrength: Double
    }
    static func write(to folder: URL, settings: TextureSettings) throws {
        let document = Document(schema: "texture-studio-material-nodes-v1", images: [
            "diffuse": .init(filename: "diffuse.png", colorSpace: "sRGB", destination: "Principled BSDF.Base Color", encoding: "16-bit sRGB PNG"),
            "roughness": .init(filename: "roughness.exr", colorSpace: "Non-Color", destination: "Principled BSDF.Roughness", encoding: "linear scalar"),
            "normal": .init(filename: "normal.exr", colorSpace: "Non-Color", destination: "Normal Map.Color → Principled BSDF.Normal", encoding: "tangent-space OpenGL +Y, encoded 0...1"),
            "height": .init(filename: "displacement.exr", colorSpace: "Non-Color", destination: "Displacement.Height → Material Output.Displacement", encoding: "linear relative surface height")
        ], normalConvention: "OpenGL +Y", materialWidthMeters: settings.materialWidthMeters,
            displacementScaleMeters: settings.displacementScaleMeters, displacementMidlevel: 0.5,
            reliefMode: "normal", geometricDisplacementNormalStrength: 0)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(document).write(to: folder.appendingPathComponent("material-nodes.json"), options: .withoutOverwriting)
        let guide = """
        Texture Studio → Blender material

        Add a Principled BSDF connected to Material Output.Surface. Add Image Texture
        nodes for the four files below. Use the same UV coordinates for every map.

        diffuse.png: 16-bit sRGB PNG. Color → Principled BSDF.Base Color. Set sRGB.
        roughness.exr: linear scalar. Color → Principled BSDF.Roughness. Set Non-Color.
        normal.exr: OpenGL +Y tangent normal encoded 0...1. Set Non-Color.
        displacement.exr: linear relative height. Set Non-Color. Midlevel: 0.5.

        Choose how to apply the relief:
        • Normal shading: normal.exr → Normal Map.Color → Principled BSDF.Normal.
          Set Normal Map to Tangent Space, Strength 1. Leave geometric displacement disconnected.
        • Geometric displacement: displacement.exr → Displacement.Height →
          Material Output.Displacement. Use Cycles displacement and sufficient mesh subdivision.
          Set the equivalent height-derived Normal Map Strength to 0, so relief is applied once.

        Material width: \(settings.materialWidthMeters) meters.
        Displacement scale: \(settings.displacementScaleMeters) meters.
        Set these physical scales to match the object's dimensions. Height is relative;
        these artistic estimates do not establish measured real-world geometry.

        Numeric EXR values retain their exported precision. Display contrast never changes
        these files. Illumination correction cannot recover clipped or fully hidden photo
        detail. Inspect relief, border artifacts and tiling before production use.

        material-nodes.json records the same filenames, color spaces and physical scales
        for tools that read this material package.
        """
        try guide.write(to: folder.appendingPathComponent("BLENDER.txt"), atomically: false, encoding: .utf8)
    }
}
