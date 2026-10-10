import AppKit
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import TextureStudio

final class MapInspectionTests: XCTestCase {
    func testTrainingGridReconstructionPurgesTemporaryBytesAndExportsOriginalSource() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try pngBytes(size: 4)
        let reconstructed = try pngBytes(size: 2)
        let source = directory.appendingPathComponent("original.png")
        try original.write(to: source)
        let recorder = ReviewReconstructionRecorder()
        let loader = ReviewImageLoader(reconstruct: { _, _, destination in
            await recorder.record(destination)
            try reconstructed.write(to: destination)
        })
        let transform = MapReviewDisplayTransform(size: 2, sourceSHA256: ReviewImageLoader.hash(original), algorithm: MapReviewDisplayTransform.exactCrop)
        let loaded = try await loader.load(source, numeric: true, displayTransform: transform)
        XCTAssertEqual([loaded.pixelWidth, loaded.pixelHeight], [2, 2])
        XCTAssertEqual(loaded.sourceURL, source)
        XCTAssertEqual(loaded.sourceSHA256, ReviewImageLoader.hash(original))
        let recordedOutput = await recorder.output
        let output = try XCTUnwrap(recordedOutput)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
        let exported = directory.appendingPathComponent("export.png")
        try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: exported)
        XCTAssertEqual(try Data(contentsOf: exported), original)
        XCTAssertEqual(try Data(contentsOf: source), original)
    }

    func testMissingReconstructedPreviewDoesNotClassifyExistingOriginalAsMissing() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try pngBytes(size: 4)
        let source = directory.appendingPathComponent("original.png")
        try original.write(to: source)
        let recorder = ReviewReconstructionRecorder()
        let loader = ReviewImageLoader(reconstruct: { _, _, destination in await recorder.record(destination) })
        let transform = MapReviewDisplayTransform(size: 2, sourceSHA256: ReviewImageLoader.hash(original), algorithm: MapReviewDisplayTransform.exactCrop)
        do {
            _ = try await loader.load(source, numeric: true, displayTransform: transform)
            XCTFail("A missing reconstruction was displayed")
        } catch ReviewImageError.invalidImage { }
        let recordedOutput = await recorder.output
        let output = try XCTUnwrap(recordedOutput)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.deletingLastPathComponent().path))
        XCTAssertEqual(try Data(contentsOf: source), original)
    }

    private func pngBytes(size: Int) throws -> Data {
        let data = Data(repeating: 64, count: size * size)
        let image = try XCTUnwrap(CGImage(width: size, height: size, bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: size, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [],
            provider: CGDataProvider(data: data as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let encoded = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(encoded, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return encoded as Data
    }
    func testOnlyMissingSourcesAreClassifiedForRemoval() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let missing = directory.appendingPathComponent("missing.exr")
        do {
            _ = try await ReviewImageLoader.shared.load(missing, numeric: true)
            XCTFail("A missing source was accepted")
        } catch ReviewImageError.missingSource { }
        let existing = directory.appendingPathComponent("unsupported.exr")
        let original = Data([0, 255, 7, 9])
        try original.write(to: existing)
        do {
            _ = try await ReviewImageLoader.shared.load(existing, numeric: true)
            XCTFail("Invalid preview data was accepted")
        } catch ReviewImageError.invalidImage { }
        XCTAssertEqual(try Data(contentsOf: existing), original, "Preview failure must not remove or alter a source")
        XCTAssertThrowsError(try ReviewImageLoader.readSource(directory)) { error in
            if case ReviewImageError.missingSource = error { XCTFail("A read failure is not a missing source") }
        }
    }
    @MainActor func testNestedInspectorCacheMatchingHonorsFolderBoundaries() {
        let folder = URL(fileURLWithPath: "/tmp/texture-cache", isDirectory: true)
        XCTAssertTrue(ReviewWindowController.mapURL(folder.appendingPathComponent("normal.exr"), isInCacheFolder: folder))
        XCTAssertTrue(ReviewWindowController.mapURL(folder.appendingPathComponent("maps/height.exr"), isInCacheFolder: folder))
        XCTAssertFalse(ReviewWindowController.mapURL(URL(fileURLWithPath: "/tmp/texture-cache-other/normal.exr"), isInCacheFolder: folder))
        XCTAssertFalse(ReviewWindowController.mapURL(folder, isInCacheFolder: folder))
        XCTAssertFalse(ReviewWindowController.mapURL(folder.appendingPathComponent("../outside.exr"), isInCacheFolder: folder))
    }

    func testOriginalAndBatchExportsDoNotRequireDisplayDecodingAndKeepExactBytes() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appendingPathComponent("first.exr")
        let second = directory.appendingPathComponent("second.exr")
        // Arbitrary binary data proves this path copies original bytes without
        // depending on a successful display decode or applying image changes.
        let firstBytes = Data([0, 1, 255, 32, 128, 0])
        let secondBytes = Data([255, 0, 129, 7])
        try firstBytes.write(to: first); try secondBytes.write(to: second)
        let direct = directory.appendingPathComponent("direct.exr")
        try ReviewImageLoader.exportOriginal(first, to: direct)
        XCTAssertEqual(try Data(contentsOf: direct), firstBytes)
        let batch = directory.appendingPathComponent("exports", isDirectory: true)
        try ReviewImageLoader.exportOriginals([
            ReviewOriginalExport(sourceURL: first, filename: "soil.exr", expectedSHA256: nil),
            ReviewOriginalExport(sourceURL: second, filename: "soil.exr", expectedSHA256: ReviewImageLoader.hash(secondBytes))], to: batch)
        XCTAssertEqual(try Data(contentsOf: batch.appendingPathComponent("soil.exr")), firstBytes)
        XCTAssertEqual(try Data(contentsOf: batch.appendingPathComponent("soil-2.exr")), secondBytes)
        XCTAssertEqual(try Data(contentsOf: first), firstBytes)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginals([], to: batch))
        XCTAssertEqual(try Data(contentsOf: batch.appendingPathComponent("soil.exr")), firstBytes)
    }

    func testBatchExportRejectsChangedSourcesAndRemovesOnlyItsNewPartialFolder() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("source.png")
        let bytes = Data([10, 20, 30])
        try bytes.write(to: source)
        let batch = directory.appendingPathComponent("exports", isDirectory: true)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginals([
            ReviewOriginalExport(sourceURL: source, filename: "first.png", expectedSHA256: nil),
            ReviewOriginalExport(sourceURL: source, filename: "second.png", expectedSHA256: "changed")], to: batch))
        XCTAssertFalse(FileManager.default.fileExists(atPath: batch.path))
        XCTAssertEqual(try Data(contentsOf: source), bytes)
    }

    @MainActor func testReviewDisplayChoicesSurviveReopeningWithoutInventedCaps() throws {
        let name = "MapInspectionPreferences-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: name))
        defer { preferences.removePersistentDomain(forName: name) }
        let context = "reviewDisplay.exact-comparison-fixture"
        let viewport = InspectionViewport(preferences: preferences, preferenceKey: context)
        viewport.setZoom(2)
        viewport.displayContrast = 12
        viewport.displayMidpoint = 0.42
        let reopened = InspectionViewport(preferences: preferences, preferenceKey: context)
        XCTAssertEqual(reopened.zoom, 2)
        XCTAssertFalse(reopened.fitToView)
        XCTAssertEqual(reopened.displayContrast, 12)
        XCTAssertEqual(reopened.displayMidpoint, 0.42)
        reopened.fit()
        XCTAssertTrue(InspectionViewport(preferences: preferences, preferenceKey: context).fitToView)
        preferences.set(["zoom": -20, "contrast": 99, "midpoint": -1], forKey: context)
        let restored = InspectionViewport(preferences: preferences, preferenceKey: context)
        XCTAssertEqual(restored.zoom, 1, "An invalid zoom uses the neutral default.")
        XCTAssertEqual(restored.displayContrast, 99, "Valid contrast must not be silently capped.")
        XCTAssertEqual(restored.displayMidpoint, 0)
        for zoom in [0.001, 100.0] {
            restored.setZoom(zoom)
            restored.displayContrast = zoom
            let reopened = InspectionViewport(preferences: preferences, preferenceKey: context)
            XCTAssertEqual(reopened.zoom, zoom)
            XCTAssertEqual(reopened.displayContrast, zoom)
        }
    }

    @MainActor func testDisplaySettingsStayWithExactComparisonAndNewFullSourceStartsNeutral() throws {
        let suite = "scoped-review-display-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let legacy: [String: Any] = ["zoom": 8.0, "fit": true, "contrast": 32.0, "midpoint": 0.9]
        preferences.set(legacy, forKey: "reviewDisplay")
        let reference = MapReviewCandidate(id: "real-reference", label: "Source displacement · reference",
            mapURL: URL(fileURLWithPath: "/crop/reference.exr"), numeric: true, role: "target",
            sourceIdentity: MapReviewSourceIdentity(path: "/original/source-displacement-4k.png"))
        let first = [reference, MapReviewCandidate(id: "checkpoint-first-sha", label: "First model",
            mapURL: URL(fileURLWithPath: "/run-1/height.exr"), numeric: true, role: "checkpoint")]
        let second = [reference, MapReviewCandidate(id: "checkpoint-second-sha", label: "Second model",
            mapURL: URL(fileURLWithPath: "/run-2/height.exr"), numeric: true, role: "checkpoint")]
        let firstKey = ReviewWorkbenchView.displayPreferenceKey(first)
        let secondKey = ReviewWorkbenchView.displayPreferenceKey(second)
        let fullKey = ReviewWorkbenchView.displayPreferenceKey([try XCTUnwrap(reference.fullSourceReference)])
        XCTAssertNotEqual(firstKey, secondKey)
        XCTAssertNotEqual(firstKey, fullKey)
        XCTAssertEqual(firstKey, ReviewWorkbenchView.displayPreferenceKey(Array(first.reversed())))
        let viewport = InspectionViewport(preferences: preferences, preferenceKey: firstKey)
        XCTAssertEqual(viewport.displayContrast, 1, "Unbound global contrast must not burn a new reference display")
        XCTAssertEqual(viewport.displayMidpoint, 0.5)
        XCTAssertEqual(viewport.zoom, 1)
        XCTAssertFalse(viewport.fitToView)
        viewport.displayContrast = 32; viewport.displayMidpoint = 0.73; viewport.setZoom(2)
        let firstSaved = try XCTUnwrap(preferences.dictionary(forKey: firstKey)) as NSDictionary
        let reopened = InspectionViewport(preferences: preferences, preferenceKey: firstKey)
        XCTAssertEqual(reopened.displayContrast, 32)
        XCTAssertEqual(reopened.displayMidpoint, 0.73)
        XCTAssertEqual(reopened.zoom, 2)
        let full = InspectionViewport(preferences: preferences, preferenceKey: fullKey)
        XCTAssertEqual(full.displayContrast, 1)
        XCTAssertEqual(full.displayMidpoint, 0.5)
        XCTAssertNil(preferences.dictionary(forKey: fullKey), "Merely loading defaults must not write a context")
        viewport.useDisplayContext(secondKey)
        XCTAssertEqual(viewport.displayContrast, 1)
        XCTAssertEqual(viewport.displayMidpoint, 0.5)
        XCTAssertEqual(viewport.zoom, 1)
        XCTAssertTrue(firstSaved.isEqual(try XCTUnwrap(preferences.dictionary(forKey: firstKey)) as NSDictionary))
        XCTAssertNil(preferences.dictionary(forKey: secondKey))
        viewport.displayContrast = 8; viewport.displayMidpoint = 0.42
        viewport.useDisplayContext(firstKey)
        XCTAssertEqual(viewport.displayContrast, 32)
        XCTAssertEqual(viewport.displayMidpoint, 0.73)
        XCTAssertEqual(viewport.zoom, 2)
        viewport.useDisplayContext(secondKey)
        XCTAssertEqual(viewport.displayContrast, 8)
        XCTAssertEqual(viewport.displayMidpoint, 0.42)
        XCTAssertTrue((legacy as NSDictionary).isEqual(try XCTUnwrap(preferences.dictionary(forKey: "reviewDisplay")) as NSDictionary),
                      "Legacy global preferences are preserved without assigning them to unrelated images")
    }

    @MainActor func testVisibleMapPreferencesAreScopedToExactCandidateIdentitiesAndFiles() {
        let original = MapReviewCandidate(id: "checkpoint", label: "Model", mapURL: URL(fileURLWithPath: "/first/model.exr"), numeric: true)
        let second = MapReviewCandidate(id: "target", label: "Reference", mapURL: URL(fileURLWithPath: "/first/target.png"), numeric: true)
        let unrelated = MapReviewCandidate(id: "checkpoint", label: "Model", mapURL: URL(fileURLWithPath: "/second/model.exr"), numeric: true)
        XCTAssertEqual(ReviewWorkbenchView.selectionPreferenceKey([original, second]), ReviewWorkbenchView.selectionPreferenceKey([second, original]))
        XCTAssertNotEqual(ReviewWorkbenchView.selectionPreferenceKey([original]), ReviewWorkbenchView.selectionPreferenceKey([unrelated]))
        XCTAssertNotEqual(ReviewWorkbenchView.selectionPreferenceKey([original]), ReviewWorkbenchView.selectionPreferenceKey([original, second]))
    }

    func testPNGFullDimensionsAndOriginalBytesRemainIntact() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("original.png")
        let width = 1537, height = 1025
        let samples = [UInt16](repeating: UInt16(32769).bigEndian, count: width * height)
        let data = samples.withUnsafeBytes { Data($0) }
        let image = try XCTUnwrap(CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16,
            bytesPerRow: width * 2, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: [.byteOrder16Big],
            provider: CGDataProvider(data: data as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(source as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let original = try Data(contentsOf: source)
        let loaded = try await ReviewImageLoader.shared.load(source, numeric: true)
        XCTAssertEqual(loaded.pixelWidth, width)
        XCTAssertEqual(loaded.pixelHeight, height)
        XCTAssertEqual(loaded.storageDescription, "16-bit integer PNG")
        XCTAssertEqual(loaded.image.bitsPerComponent, 8, "Display precision must not be confused with the stored source precision")
        let copy = directory.appendingPathComponent("copy.png")
        try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy)
        XCTAssertEqual(try Data(contentsOf: copy), original)
        XCTAssertEqual(try Data(contentsOf: source), original)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy))
    }

    func testFloatEXRContrastIsAppliedBeforeDisplayQuantizationAndExportUsesPinnedHash() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("height.exr")
        let values: [Float] = [0.4995,0.4995,0.4995,1, 0.5005,0.5005,0.5005,1]
        let image = CIImage(bitmapData: values.withUnsafeBytes { Data($0) }, bytesPerRow: 32,
            size: CGSize(width: 2, height: 1), format: .RGBAf, colorSpace: nil)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf])
        try FloatEXRWriter.write(image, to: source, context: context, color: true)
        let original = try Data(contentsOf: source)
        let loaded = try await ReviewImageLoader.shared.load(source, numeric: true)
        let contrast = try await ReviewImageLoader.shared.load(source, numeric: true, contrast: 128, midpoint: 0.5,
            expectedSHA256: loaded.sourceSHA256)
        XCTAssertEqual(loaded.pixelWidth, 2)
        XCTAssertEqual(loaded.pixelHeight, 1)
        XCTAssertEqual(loaded.storageDescription, "32-bit float EXR")
        XCTAssertEqual(contrast.sourceSHA256, loaded.sourceSHA256)
        let codes = rgbaCodes(contrast.image)
        XCTAssertGreaterThan(Int(codes[4]) - Int(codes[0]), 20, "Contrast above 32 must be applied before display conversion, without a hidden cap.")
        let compressed = try await ReviewImageLoader.shared.load(source, numeric: true, contrast: 0.5, midpoint: 0,
            expectedSHA256: loaded.sourceSHA256)
        XCTAssertLessThan(rgbaCodes(compressed.image)[0], 70, "Contrast below one must remain a valid display choice.")
        XCTAssertEqual(try Data(contentsOf: source), original)
        let copy = directory.appendingPathComponent("copy.exr")
        try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256, to: copy)
        XCTAssertEqual(try Data(contentsOf: copy), original)
        try Data("changed".utf8).write(to: source)
        XCTAssertThrowsError(try ReviewImageLoader.exportOriginal(source, expectedSHA256: loaded.sourceSHA256,
            to: directory.appendingPathComponent("refused.exr")))
        do {
            _ = try await ReviewImageLoader.shared.load(source, numeric: true, expectedSHA256: loaded.sourceSHA256)
            XCTFail("A contrast refresh must reject changed originals")
        } catch ReviewImageError.sourceChanged { }
    }

    @MainActor func testLinkedViewportUsesNormalizedPanAndRetinaDevicePixelZoom() {
        let viewport = InspectionViewport()
        XCTAssertFalse(viewport.fitToView)
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 2048, height: 2048),
            viewSize: CGSize(width: 500, height: 500), backingScale: 2), 0.5)
        viewport.pan(dx: 0.125, dy: -0.25)
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 0.625, y: 0.25))
        viewport.setZoom(2)
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 1024, height: 1024),
            viewSize: CGSize(width: 500, height: 500), backingScale: 2), 1)
        viewport.fit()
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 0.5, y: 0.5))
        XCTAssertEqual(viewport.imageScale(imageSize: CGSize(width: 2048, height: 1024),
            viewSize: CGSize(width: 512, height: 512), backingScale: 2), 0.25)
        viewport.pan(dx: 99, dy: -99)
        XCTAssertEqual(viewport.normalizedCenter, CGPoint(x: 1, y: 0))
        viewport.setZoom(.nan)
        XCTAssertTrue(viewport.zoom.isFinite)
        viewport.setZoom(100)
        XCTAssertEqual(viewport.zoom, 100)
        viewport.setZoom(0)
        XCTAssertEqual(viewport.zoom, 100, "Zero zoom would have no visible scale and must retain the last valid value.")
    }

    @MainActor func testReferenceAndAllExplicitModelCandidatesAreInitiallyVisible() {
        let candidates = [("diffuse", "diffuse"), ("target", "target"), ("base", "base"), ("trained_2k", "checkpoint")].map { name, role in
            MapReviewCandidate(id: name, label: name, mapURL: URL(fileURLWithPath: "/\(name).exr"), numeric: role != "diffuse", role: role)
        }
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(candidates).map(\.label), ["target", "base", "trained_2k"])
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates([candidates[0], candidates[1]]).map(\.label), ["target"])
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates([candidates[2], candidates[3]]).map(\.label), ["base", "trained_2k"])
    }

    @MainActor func testSavedVisibilityCannotHideTheRealSourceReferenceOrReplaceItWithABasePrediction() {
        let candidates = [
            MapReviewCandidate(id: "photo", label: "Source photo", mapURL: URL(fileURLWithPath: "/source.png"), numeric: false, role: "source"),
            MapReviewCandidate(id: "reference", label: "Source displacement · reference", mapURL: URL(fileURLWithPath: "/reference.exr"), numeric: true, role: "target"),
            MapReviewCandidate(id: "base", label: "PBRnxt base", mapURL: URL(fileURLWithPath: "/base.exr"), numeric: true, role: "base"),
            MapReviewCandidate(id: "refined", label: "PBRnxt refined", mapURL: URL(fileURLWithPath: "/refined.exr"), numeric: true, role: "checkpoint")]
        XCTAssertEqual(ReviewWorkbenchView.resolvedCandidates(candidates, visibleIDs: ["base", "refined"]).map(\.id),
                       ["reference", "base", "refined"])
        XCTAssertEqual(ReviewWorkbenchView.resolvedCandidates(candidates, visibleIDs: ["refined"]).map(\.id), ["reference", "refined"])
        XCTAssertEqual(ReviewWorkbenchView.resolvedCandidates(candidates, visibleIDs: []).map(\.id), ["reference", "base", "refined"])
        XCTAssertEqual(ReviewWorkbenchView.resolvedCandidates([candidates[2], candidates[3]], visibleIDs: []).map(\.id), ["base", "refined"],
                       "A review without a recorded reference must not relabel a prediction as one")
        XCTAssertFalse(MapReviewCandidate(label: "target", mapURL: URL(fileURLWithPath: "/prediction.exr"), numeric: true,
            role: "model").isReference, "An explicit prediction role overrides a misleading name")
    }

    func testPhotoDisplayHonorsWideGamutProfileWhileNumericCodesStayUnmanaged() async throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = directory.appendingPathComponent("wide-gamut.png")
        let colors: [UInt8] = [179, 77, 26, 255, 179, 77, 26, 255]
        let space = try XCTUnwrap(CGColorSpace(name: CGColorSpace.displayP3))
        let image = try XCTUnwrap(CGImage(width: 2, height: 1, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 8, space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: CGDataProvider(data: Data(colors) as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let destination = try XCTUnwrap(CGImageDestinationCreateWithURL(source as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        let managed = try await ReviewImageLoader.shared.load(source, numeric: false)
        let raw = try await ReviewImageLoader.shared.load(source, numeric: true)
        let numericCodes = try XCTUnwrap(raw.image.dataProvider?.data) as Data
        let photoCodes = try XCTUnwrap(managed.image.dataProvider?.data) as Data
        XCTAssertEqual(Array(numericCodes.prefix(3)), Array(colors.prefix(3)))
        XCTAssertGreaterThan(abs(Int(photoCodes[0]) - Int(numericCodes[0])), 5)
        XCTAssertEqual(managed.sourceSHA256, raw.sourceSHA256)
    }

    private func temporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("MapInspectionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
    private func rgbaCodes(_ image: CGImage) -> [UInt8] {
        var values = [UInt8](repeating: 0, count: image.width * image.height * 4)
        values.withUnsafeMutableBytes { bytes in
            let context = CGContext(data: bytes.baseAddress, width: image.width, height: image.height,
                bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return values
    }
}

private actor ReviewReconstructionRecorder {
    var output: URL?
    func record(_ url: URL) { output = url }
}
