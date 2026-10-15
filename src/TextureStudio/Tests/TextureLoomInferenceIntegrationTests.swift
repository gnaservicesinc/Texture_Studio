import CoreImage
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class TextureLoomInferenceIntegrationTests: XCTestCase {
    func testPreparedModelInputRetainsFloat32DifferencesThroughImageFilters() async throws {
        let width = 8, height = 4
        let srgb = try XCTUnwrap(CGColorSpace(name: CGColorSpace.sRGB))
        var source = [Float](repeating: 1, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let base = (y * width + x) * 4
                // All red values round to the same half float, but remain
                // distinct Float32 values in the model's prepared diffuse input.
                source[base] = 0.375 + Float(x + 1) / 262_144
                source[base + 1] = 0.625 - Float(y + 1) / 262_144
                source[base + 2] = 0.25 + Float(x + y + 1) / 262_144
            }
        }
        let image = CIImage(bitmapData: source.withUnsafeBytes { Data($0) }, bytesPerRow: width * 16,
            size: CGSize(width: width, height: height), format: .RGBAf, colorSpace: srgb)
        // Blur evaluates intermediate buffers: requesting Float32 only for the
        // final render cannot restore values already rounded in those buffers.
        let filtered = image.clampedToExtent().applyingGaussianBlur(sigma: 0.7).cropped(to: image.extent)
        let referenceContext = CIContext(options: [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: srgb, .cacheIntermediates: false])
        var reference = [Float](repeating: 0, count: source.count)
        reference.withUnsafeMutableBytes {
            referenceContext.render(filtered, toBitmap: $0.baseAddress!, rowBytes: width * 16,
                bounds: filtered.extent, format: .RGBAf, colorSpace: srgb)
        }
        let firstRed = reference[0], lastRed = reference[(width - 1) * 4]
        XCTAssertGreaterThan(lastRed - firstRed, 1e-6)
        XCTAssertEqual(Float16(firstRed), Float16(lastRed),
                       "The fixture must exercise distinctions that half-float intermediates discard")
        XCTAssertNotEqual(firstRed, Float(Float16(firstRed)))

        let tensor = try await TextureLoomService().imageTensor(filtered, width: width, height: height)
        XCTAssertEqual(tensor.dataType, .float32)
        XCTAssertEqual(tensor.shape.map(\.intValue), [1, 3, height, width])
        let strides = tensor.strides.map(\.intValue)
        let values = tensor.dataPointer.assumingMemoryBound(to: Float.self)
        for channel in 0..<3 {
            for y in 0..<height {
                for x in 0..<width {
                    XCTAssertEqual(values[channel * strides[1] + y * strides[2] + x * strides[3]],
                        reference[(y * width + x) * 4 + channel], accuracy: 2e-7)
                }
            }
        }
        XCTAssertGreaterThan(values[(width - 1) * strides[3]] - values[0], 1e-6,
                             "Preparing the real model tensor must retain the fine diffuse differences")
    }

    func testRealWorkshopAdapterImportsEnablesNative2048OverrideAndRestoresBaseWhenDisabled() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let basePath = environment["TEXTURE_LOOM_TEST_ADAPTER_BASE"],
              let adapterPath = environment["TEXTURE_LOOM_TEST_ADAPTER"],
              let photoPath = environment["TEXTURE_LOOM_TEST_ADAPTER_PHOTO"] else {
            throw XCTSkip("Provide the fresh Workshop LoRA, its exact installed base and a paired training photo for native adapter integration.")
        }
        let base = URL(fileURLWithPath: basePath, isDirectory: true)
        let photo = URL(fileURLWithPath: photoPath)
        let originalHash = try TextureLoomPackage.sha256(photo)
        let package = try TextureLoomPackage(root: base)
        let baseHashes = try Dictionary(uniqueKeysWithValues: package.manifest.files.map {
            ($0.path, try TextureLoomPackage.sha256(TextureLoomPackage.resolve($0.path, root: base)))
        })
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent("real-workshop-import-\(UUID().uuidString)", isDirectory: true)
        let suite = "real-workshop-import-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { try? FileManager.default.removeItem(at: storage); preferences.removePersistentDomain(forName: suite) }
        let store = TextureLoomStore(bundleRoot: base, storage: storage, preferences: preferences)
        let diffuse = try XCTUnwrap(CIImage(contentsOf: photo, options: [.applyOrientationProperty: true]))
        XCTAssertEqual(diffuse.extent.size, CGSize(width: 2048, height: 2048))
        let service = TextureLoomService()
        let baseline = try await service.predict(diffuse: diffuse, package: package, overrides: [:])
        let baselineMap = try XCTUnwrap(baseline.maps["height"])
        let baselineValues = scalarValues(baselineMap.image)
        let adapterID = try store.importAdapter(URL(fileURLWithPath: adapterPath))
        XCTAssertEqual(store.adapters.count, 1)
        XCTAssertTrue(try store.activeOverrides().isEmpty)
        try store.setEnabled(adapterID, true)
        let overrides = try store.activeOverrides()
        XCTAssertEqual(overrides.count, 1)
        let adapted = try await service.predict(diffuse: diffuse, package: package, overrides: overrides)
        XCTAssertEqual(adapted.categoryID, baseline.categoryID)
        let adaptedMap = try XCTUnwrap(adapted.maps["height"])
        XCTAssertEqual(adaptedMap.image.extent.size, diffuse.extent.size)
        let adaptedValues = scalarValues(adaptedMap.image)
        XCTAssertTrue(adaptedValues.allSatisfy { $0.isFinite && (0...1).contains($0) })
        let changedBy = zip(baselineValues, adaptedValues).map { abs($0 - $1) }.max() ?? 0
        XCTAssertGreaterThan(changedBy, 1e-6, "Enabling a learned adapter must change the native prediction.")
        try store.setEnabled(adapterID, false)
        XCTAssertTrue(try store.activeOverrides().isEmpty)
        let restored = try await service.predict(diffuse: diffuse, package: package, overrides: store.activeOverrides())
        let restoredValues = scalarValues(try XCTUnwrap(restored.maps["height"]).image)
        let restorationDifference = zip(baselineValues, restoredValues).map { abs($0 - $1) }.max() ?? 0
        XCTAssertLessThanOrEqual(restorationDifference, 1e-6)
        XCTAssertEqual(try TextureLoomPackage.sha256(photo), originalHash)
        for (path, hash) in baseHashes {
            XCTAssertEqual(try TextureLoomPackage.sha256(TextureLoomPackage.resolve(path, root: base)), hash)
        }
        if let output = environment["TEXTURE_LOOM_TEST_OUTPUT"] {
            let receipt: [String: Any] = ["schema": "texture-studio-workshop-native-adapter-proof-v1",
                "base_id": package.manifest.id, "base_version": package.manifest.version,
                "adapter_id": adapterID, "category_id": adapted.categoryID, "grid": [2048, 2048],
                "native_map": "depth", "maximum_enabled_difference": changedBy,
                "maximum_disabled_difference": restorationDifference, "all_values_finite": true,
                "originals_preserved": true, "installed_base_preserved": true, "quality_qualification": false,
                "note": "Actual archive import, enable/disable and native prediction routing; not unseen-material quality."]
            try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output).appendingPathComponent("texture-studio-workshop-native-adapter-proof.json"), options: .atomic)
        }
    }

    private func scalarValues(_ image: CIImage) -> [Float] {
        let width = Int(image.extent.width), height = Int(image.extent.height)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        var result = [Float](repeating: 0, count: width * height)
        result.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 4, bounds: image.extent, format: .Rf, colorSpace: nil)
        }
        return result
    }

    func testFreshFactoryModelsRouteReal2048PhotoThroughCoreMLAndPreserveOriginal() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["TEXTURE_LOOM_TEST_PACKAGE"], let photoPath = environment["TEXTURE_LOOM_TEST_PHOTO"] else {
            throw XCTSkip("Provide the fresh Loom Factory proof package and original photo to run native deployment integration.")
        }
        let package = try TextureLoomPackage(root: URL(fileURLWithPath: path))
        let photo = URL(fileURLWithPath: photoPath)
        let original = try TextureLoomPackage.sha256(photo)
        let image = try XCTUnwrap(CIImage(contentsOf: photo, options: [.applyOrientationProperty: true]))
        XCTAssertEqual(image.extent.size, CGSize(width: 2048, height: 2048))
        let prediction = try await TextureLoomService().predict(diffuse: image, package: package, overrides: [:])
        XCTAssertEqual(prediction.categoryName, "Bricks")
        XCTAssertTrue(prediction.confidence > 0.5)
        let height = try XCTUnwrap(prediction.maps["height"])
        XCTAssertEqual(height.image.extent.size, image.extent.size)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        var values = [Float](repeating: 0, count: 2048 * 2048)
        values.withUnsafeMutableBytes { context.render(height.image, toBitmap: $0.baseAddress!, rowBytes: 2048 * 4, bounds: height.image.extent, format: .Rf, colorSpace: nil) }
        XCTAssertTrue(values.allSatisfy { $0.isFinite && (0...1).contains($0) })
        XCTAssertEqual(try TextureLoomPackage.sha256(photo), original)
        if let output = environment["TEXTURE_LOOM_TEST_OUTPUT"] {
            let receipt: [String: Any] = ["schema": "texture-studio-fresh-coreml-proof-v1", "package_id": package.manifest.id,
                "package_version": package.manifest.version, "category_id": prediction.categoryID, "category_name": prediction.categoryName,
                "confidence": prediction.confidence, "grid": [2048, 2048], "map": "depth", "all_values_finite": true,
                "original_sha256": original, "originals_preserved": true, "quality_qualification": false,
                "note": "Native routing and inference execution on a classifier training example; not unseen-material quality."]
            try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: output).appendingPathComponent("texture-studio-real-coreml-proof.json"), options: .atomic)
        }
    }
}
