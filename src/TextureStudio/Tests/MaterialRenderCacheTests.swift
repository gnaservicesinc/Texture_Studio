import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialRenderCacheTests: XCTestCase {
    func testRenderKeyIgnoresStoragePrecisionOnly() {
        let settings = TextureSettings()
        let original = MaterialRenderKey(sourceIdentity: "photo-a", depthIdentity: "checkpoint-a", settings: settings)
        var storage = settings
        storage.exrPrecision = .float16
        XCTAssertEqual(original, MaterialRenderKey(sourceIdentity: "photo-a", depthIdentity: "checkpoint-a", settings: storage))
        for updated in [changed(settings, { $0.outputSize = 2048 }), changed(settings, { $0.lightingStrength = 0.3 }),
                        changed(settings, { $0.rotationX = 3 }), changed(settings, { $0.heightInvert = true }),
                        changed(settings, { $0.useSupportingViews = false }), changed(settings, { $0.roughnessBase = 0.8 })] {
            XCTAssertNotEqual(original, MaterialRenderKey(sourceIdentity: "photo-a", depthIdentity: "checkpoint-a", settings: updated))
        }
        XCTAssertNotEqual(original, MaterialRenderKey(sourceIdentity: "photo-b", depthIdentity: "checkpoint-a", settings: settings))
        XCTAssertNotEqual(original, MaterialRenderKey(sourceIdentity: "photo-a", depthIdentity: "checkpoint-b", settings: settings))
    }

    func testCacheFreezesNativeFloatMapsAndRemainsIndependentOfOriginalGraphFile() async throws {
        let fixture = try RenderCacheFixture()
        defer { fixture.remove() }
        let original = fixture.root.appendingPathComponent("original-height.exr")
        try FloatEXRWriter.write(fixture.material.height, to: original, context: fixture.context, color: false)
        let originalHeight = try XCTUnwrap(CIImage(contentsOf: original, options: [.colorSpace: NSNull()]))
        let graph = originalHeight.applyingFilter("CIColorMatrix", parameters: ["inputRVector": CIVector(x: 1, y: 0, z: 0, w: 0)])
        let material = fixture.withHeight(graph)
        let expectedHeight = fixture.samples(material.height)
        let expectedRoughness = fixture.samples(material.roughness)
        let expectedNormal = fixture.samples(material.normal)
        let cache = try await MaterialRenderCache.make(material: material, key: fixture.key, engine: fixture.engine)
        try FileManager.default.removeItem(at: original)
        XCTAssertFalse(FileManager.default.fileExists(atPath: original.path))
        XCTAssertEqual(cache.material.outputSize, 2)
        XCTAssertEqual(cache.material.crop, material.crop)
        XCTAssertEqual(cache.material.sourceURL, material.sourceURL)
        XCTAssertEqual(cache.material.depthOrigin, material.depthOrigin)
        XCTAssertEqual(cache.material.warnings, material.warnings)
        XCTAssertEqual(cache.material.settings, material.settings)
        XCTAssertEqual(fixture.samples(cache.material.height).map(\.bitPattern), expectedHeight.map(\.bitPattern))
        XCTAssertEqual(fixture.samples(cache.material.roughness).map(\.bitPattern), expectedRoughness.map(\.bitPattern))
        XCTAssertEqual(fixture.samples(cache.material.normal).map(\.bitPattern), expectedNormal.map(\.bitPattern))
        XCTAssertNil(cache.mapURL(.source))
        for choice in [MaterialPreview.diffuse, .roughness, .normal, .height] {
            let url = try XCTUnwrap(cache.mapURL(choice))
            XCTAssertEqual(url.deletingLastPathComponent(), cache.folder)
            XCTAssertTrue(FileManager.default.fileExists(atPath: url.path))
            if choice != .diffuse { try FloatEXRWriter.verifyChannelPrecision(at: url, expected: .float32) }
        }
    }

    func testFloat32MaterialExportCopiesCachedImageBytesExactlyAndUpdatesMetadata() async throws {
        let fixture = try RenderCacheFixture()
        defer { fixture.remove() }
        let cache = try await MaterialRenderCache.make(material: fixture.material, key: fixture.key, engine: fixture.engine)
        let destination = fixture.root.appendingPathComponent("export-32")
        var settings = fixture.material.settings
        settings.exrPrecision = .float32
        try await cache.export(to: destination, precision: .float32, settings: settings, engine: fixture.engine)
        for name in ["diffuse.png", "roughness.exr", "normal.exr", "displacement.exr", "Blender-setup.txt"] {
            XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent(name)),
                           try Data(contentsOf: cache.folder.appendingPathComponent(name)), name)
        }
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: destination.appendingPathComponent("material.json"))) as? [String: Any])
        XCTAssertEqual(document["precision"] as? String, "float32")
        let restoredSettings = try JSONDecoder().decode(TextureSettings.self,
            from: JSONSerialization.data(withJSONObject: try XCTUnwrap(document["settings"])))
        XCTAssertEqual(restoredSettings, settings)
        let exportedHeight = try XCTUnwrap(CIImage(contentsOf: destination.appendingPathComponent("displacement.exr"), options: [.colorSpace: NSNull()]))
        XCTAssertEqual(fixture.samples(exportedHeight).map(\.bitPattern), fixture.samples(cache.material.height).map(\.bitPattern))
    }

    func testHalfFloatExportConvertsExistingMapsWithoutChangingFloat32Cache() async throws {
        let fixture = try RenderCacheFixture()
        defer { fixture.remove() }
        let cache = try await MaterialRenderCache.make(material: fixture.material, key: fixture.key, engine: fixture.engine)
        let cachedBytes = try Data(contentsOf: try XCTUnwrap(cache.mapURL(.height)))
        let originalSamples = fixture.samples(cache.material.height)
        var settings = fixture.material.settings
        settings.exrPrecision = .float16
        let destination = fixture.root.appendingPathComponent("export-16")
        try await cache.export(to: destination, precision: .float16, settings: settings, engine: fixture.engine)
        for name in ["roughness.exr", "normal.exr", "displacement.exr"] {
            try FloatEXRWriter.verifyChannelPrecision(at: destination.appendingPathComponent(name), expected: .float16)
        }
        XCTAssertEqual(try Data(contentsOf: try XCTUnwrap(cache.mapURL(.height))), cachedBytes)
        try FloatEXRWriter.verifyChannelPrecision(at: try XCTUnwrap(cache.mapURL(.height)), expected: .float32)
        let half = try XCTUnwrap(CIImage(contentsOf: destination.appendingPathComponent("displacement.exr"), options: [.colorSpace: NSNull()]))
        let actual = fixture.samples(half)
        for (expected, converted) in zip(originalSamples, actual) {
            let rounded = Float16(expected)
            XCTAssertEqual(converted, Float(Float16(converted)), "The stored number is a half-float value")
            XCTAssertEqual(converted, expected, accuracy: Float(rounded.ulp),
                           "Core Image half encoding may truncate or round within one half ULP; it must retain linear values, sign and range")
        }
        XCTAssertTrue(zip(originalSamples, actual).contains { pair in pair.0.bitPattern != pair.1.bitPattern }, "Non-half-representable data must demonstrate the requested precision conversion")
        let document = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: destination.appendingPathComponent("material.json"))) as? [String: Any])
        XCTAssertEqual(document["precision"] as? String, "float16")
        XCTAssertEqual((document["settings"] as? [String: Any])?["exrPrecision"] as? String, "float16")
    }

    func testReleasingLastOwnerRemovesOnlyItsTemporaryCache() async throws {
        let fixture = try RenderCacheFixture()
        defer { fixture.remove() }
        let independent = fixture.root.appendingPathComponent("user-source.exr")
        let originalBytes = Data("User data must remain untouched".utf8)
        try originalBytes.write(to: independent)
        var cache: MaterialRenderCache? = try await MaterialRenderCache.make(material: fixture.material, key: fixture.key, engine: fixture.engine)
        let folder = try XCTUnwrap(cache?.folder)
        weak var released = cache
        var inspectorOwner = cache
        cache = nil
        XCTAssertNotNil(inspectorOwner)
        XCTAssertTrue(FileManager.default.fileExists(atPath: folder.path), "An open inspector keeps the original maps alive")
        inspectorOwner = nil
        XCTAssertNil(released)
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.path))
        XCTAssertEqual(try Data(contentsOf: independent), originalBytes)
    }

    private func changed(_ source: TextureSettings, _ update: (inout TextureSettings) -> Void) -> TextureSettings {
        var result = source
        update(&result)
        return result
    }
}

private struct RenderCacheFixture {
    let root: URL
    let engine = TextureEngine()
    let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf])
    let material: MaterialResult
    let key: MaterialRenderKey
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("render-cache-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var settings = TextureSettings()
        settings.outputSize = 2
        let height = Self.image([0.12345679, 0.12345679, 0.12345679, 1, 0.87654322, 0.87654322, 0.87654322, 1,
                                 -0.12345679, -0.12345679, -0.12345679, 1, 1.23456789, 1.23456789, 1.23456789, 1])
        let roughness = Self.image([0.23456790, 0.23456790, 0.23456790, 1, 0.76543210, 0.76543210, 0.76543210, 1,
                                    0.34567891, 0.34567891, 0.34567891, 1, 0.65432109, 0.65432109, 0.65432109, 1])
        let normal = Self.image([0.12345679, 0.23456790, 0.34567891, 1, 0.45678912, 0.56789123, 0.67891234, 1,
                                 0.78912345, 0.89123456, 0.91234567, 1, 0.91234567, 0.23456790, 0.12345679, 1])
        material = MaterialResult(diffuse: CIImage(color: CIColor(red: 0.4, green: 0.5, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2)),
            roughness: roughness, normal: normal, height: height, crop: CGRect(x: 3, y: 4, width: 2, height: 2),
            warnings: ["Retain this source warning"], outputSize: 2, depthOrigin: "Selected trained surface-height model",
            settings: settings, sourceURL: root.appendingPathComponent("photo.png"), camera: CameraMetadata())
        key = MaterialRenderKey(sourceIdentity: material.sourceURL.path, depthIdentity: "exact-checkpoint", settings: settings)
    }
    func withHeight(_ image: CIImage) -> MaterialResult {
        MaterialResult(diffuse: material.diffuse, roughness: material.roughness, normal: material.normal, height: image,
            crop: material.crop, warnings: material.warnings, outputSize: material.outputSize, depthOrigin: material.depthOrigin,
            settings: material.settings, sourceURL: material.sourceURL, camera: material.camera)
    }
    func samples(_ image: CIImage) -> [Float] {
        var result = [Float](repeating: 0, count: 16)
        result.withUnsafeMutableBytes { context.render(image, toBitmap: $0.baseAddress!, rowBytes: 32, bounds: image.extent, format: .RGBAf, colorSpace: nil) }
        return result
    }
    func remove() { try? FileManager.default.removeItem(at: root) }
    private static func image(_ samples: [Float]) -> CIImage {
        CIImage(bitmapData: samples.withUnsafeBytes { Data($0) }, bytesPerRow: 32,
                size: CGSize(width: 2, height: 2), format: .RGBAf, colorSpace: nil)
    }
}
