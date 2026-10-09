import Foundation
import XCTest
@testable import TextureStudio

final class BlenderMaterialSetupTests: XCTestCase {
    func testNativeExportGuideAndNodeMetadataPreserveMapPrecisionAndPhysicalScales() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("native-material-setup-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: folder) }
        var settings = TextureSettings()
        settings.materialWidthMeters = 2.25; settings.displacementScaleMeters = 0.017
        try BlenderMaterialSetup.write(to: folder, settings: settings)
        let document = try JSONDecoder().decode(BlenderMaterialSetup.Document.self, from: Data(contentsOf: folder.appendingPathComponent("material-nodes.json")))
        XCTAssertEqual(document.images["diffuse"]?.filename, "diffuse.png")
        XCTAssertEqual(document.images["diffuse"]?.colorSpace, "sRGB")
        XCTAssertEqual(document.images["diffuse"]?.encoding, "16-bit sRGB PNG")
        for role in ["roughness", "normal", "height"] { XCTAssertEqual(document.images[role]?.colorSpace, "Non-Color") }
        XCTAssertEqual(document.images["height"]?.filename, "displacement.exr")
        XCTAssertEqual(document.normalConvention, "OpenGL +Y")
        XCTAssertEqual(document.materialWidthMeters, 2.25)
        XCTAssertEqual(document.displacementScaleMeters, 0.017)
        XCTAssertEqual(document.displacementMidlevel, 0.5)
        XCTAssertEqual(document.reliefMode, "normal")
        XCTAssertEqual(document.geometricDisplacementNormalStrength, 0)
        let guide = try String(contentsOf: folder.appendingPathComponent("BLENDER.txt"), encoding: .utf8)
        XCTAssertTrue(guide.contains("16-bit sRGB PNG")); XCTAssertTrue(guide.contains("Strength to 0"))
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: folder.path).contains { $0.hasSuffix(".py") })
    }
}
