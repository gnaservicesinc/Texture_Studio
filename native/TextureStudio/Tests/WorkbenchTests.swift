import XCTest
import CoreImage
import ImageIO
@testable import TextureStudio

final class WorkbenchTests: XCTestCase {
    @MainActor func testReviewRemembersSampleAndCandidateOnlyInsideTheirOwnManifest() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let suite = "ReviewSelection-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { preferences.removePersistentDomain(forName: suite) }
        let store = ReviewSessionStore(preferences: preferences)
        store.groups = ["soil", "stucco"].map { id in
            MaterialReviewGroup(id: id, candidates: ["base", "trained"].map {
                MapReviewCandidate(id: "\(id)/\($0)", label: $0, mapURL: directory.appendingPathComponent("\(id)-\($0).exr"), numeric: true)
            })
        }
        let manifest = directory.appendingPathComponent("first.json")
        try store.writeReview(to: manifest)
        store.load(manifest)
        store.selectedGroupId = "stucco"
        store.selectedCandidateId = "stucco/trained"
        let reopened = ReviewSessionStore(preferences: preferences)
        reopened.load(manifest)
        XCTAssertEqual(reopened.selectedGroupId, "stucco")
        XCTAssertEqual(reopened.selectedCandidateId, "stucco/trained")
        let unrelated = directory.appendingPathComponent("second.json")
        try FileManager.default.copyItem(at: manifest, to: unrelated)
        reopened.load(unrelated)
        XCTAssertEqual(reopened.selectedGroupId, "soil")
        XCTAssertNil(reopened.selectedCandidateId)
        reopened.load(manifest)
        reopened.selectedGroupId = "soil"
        XCTAssertNil(reopened.selectedCandidateId, "Changing sample clears a candidate that belongs to the previous sample")
    }

    @MainActor func testStudioInferencePhotoRetainsSixteenBitColorAndExplicitSize() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("InferencePhoto-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let image = CIImage(color: CIColor(red: 0.499, green: 0.501, blue: 0.503))
            .cropped(to: CGRect(x: 0, y: 0, width: 128, height: 64))
        try MaterialCheckpointService.writeInferencePhoto(image, maximumSize: 64, to: url)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil))
        let decoded = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(decoded.width, 64)
        XCTAssertEqual(decoded.height, 32)
        XCTAssertEqual(decoded.bitsPerComponent, 16)
    }

    @MainActor func testSavedReviewsReopenDecisionsAndExactMapPaths() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let savedPreference = UserDefaults(suiteName: "org.ipde.material-tools")!.object(forKey: "reviewManifest")
        defer { UserDefaults(suiteName: "org.ipde.material-tools")!.set(savedPreference, forKey: "reviewManifest") }
        let original = directory.appendingPathComponent("normal.float32.exr")
        let photo = directory.appendingPathComponent("diffuse.png")
        let blend = directory.appendingPathComponent("geometry.blend")
        try Data("unchanged numeric map".utf8).write(to: original)
        try Data().write(to: blend)
        let store = ReviewSessionStore()
        store.groups = [MaterialReviewGroup(id: "soil", candidates: [
            MapReviewCandidate(id: "exact-checkpoint-sha", label: "Normal candidate", mapURL: original, numeric: true),
            MapReviewCandidate(id: "photo", label: "Photo", mapURL: photo, numeric: false)])]
        store.decisions["exact-checkpoint-sha"] = "needs_work"
        store.notes["exact-checkpoint-sha"] = "Inverted pebble; inspect displacement"
        store.blendURL = blend
        let manifest = directory.appendingPathComponent("review.json")
        try store.writeReview(to: manifest)
        let reopened = ReviewSessionStore()
        reopened.load(manifest)
        XCTAssertNil(reopened.error)
        XCTAssertEqual(reopened.groups.first?.candidates.first?.mapURL, original)
        XCTAssertEqual(reopened.groups.first?.candidates.last?.numeric, false)
        XCTAssertEqual(reopened.decisions["exact-checkpoint-sha"], "needs_work")
        XCTAssertEqual(reopened.notes["exact-checkpoint-sha"], "Inverted pebble; inspect displacement")
        XCTAssertEqual(reopened.blendURL, blend)
        XCTAssertEqual(try Data(contentsOf: original), Data("unchanged numeric map".utf8))
    }

    func testProtocolBindsSelectedDatasetAndCheckpoint() throws {
        let text = """
        {"dataset_path":"/second/dataset.json","index_sha256":"abc","materials":[{"material_id":"soil","samples":[{"sample_id":"soil_2","status":"approved","split":"train","width":2048,"height":2048,"maps":{"height":{"path":"/second/height.png","sha256":"123","source_bits":16,"encoding":"linear_data"}}}]}]}
        """
        let dataset = try WorkbenchProcess.decode(WorkbenchDataset.self, output: text)
        XCTAssertEqual(dataset.datasetPath, "/second/dataset.json")
        XCTAssertEqual(dataset.samples.first?.maps["height"]?.sourceBits, 16)
        let result = try WorkbenchProcess.decode(MaterialInferenceResponse.self, output:
            "{\"outputs\":{\"height\":{\"path\":\"/native/height.exr\"}},\"checkpoint_sha256\":\"selected\"}")
        XCTAssertEqual(result.checkpointSha256, "selected")
        XCTAssertThrowsError(try WorkbenchProcess.decode(MaterialInferenceResponse.self, output: "{\"ok\":false}"))
    }

    func testMaterialCheckpointRecipeIsAnExplicitSource() throws {
        let choice = try JSONDecoder().decode(DepthChoice.self, from: Data("\"Material checkpoint\"".utf8))
        XCTAssertEqual(choice, .materialCheckpoint)
    }

    func testMaterialHeadHeightKeepsAmplitudeAndOutOfRangeSamples() async throws {
        let engine = TextureEngine()
        let photo = CIImage(color: CIColor(red: 0.3, green: 0.4, blue: 0.5)).cropped(to: CGRect(x: 0, y: 0, width: 1024, height: 1024))
        let source = TextureSource(url: URL(fileURLWithPath: "/test/photo.png"), orientedImage: photo,
            embeddedDepth: nil, camera: CameraMetadata(), pixelWidth: 1024, pixelHeight: 1024)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        for value: Float in [0.25, 1.125] {
            let depth = try TextureDepth(width: 16, height: 16, values: [Float](repeating: value, count: 256), sourceLabel: "Selected material head", interpretation: .surfaceHeight)
            var settings = TextureSettings()
            settings.heightDetail = 0
            settings.surfacePlaneRemoval = 1
            settings.depthCleanup = 1
            let result = try await engine.process(source: source, settings: settings, attachedDepth: depth)
            var pixel = Float.zero
            withUnsafeMutableBytes(of: &pixel) { context.render(result.height, toBitmap: $0.baseAddress!, rowBytes: 4, bounds: CGRect(x: 512, y: 512, width: 1, height: 1), format: .Rf, colorSpace: nil) }
            XCTAssertEqual(pixel, value, accuracy: 0.00001, "Material height must not be camera-depth normalized or clipped")
        }
    }
}
