import CoreImage
import Foundation
import Metal
import XCTest
@testable import TextureStudio

@MainActor
final class DepthProCheckpointProductionTests: XCTestCase {
    func testRefinerProductionReadbackMatchesDirectInferenceOnPrepared2KSource() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_DEPTHPRO_PRODUCTION_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in DepthPro refiner production verification with retained trial weights.")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        func requiredURL(_ name: String) throws -> URL {
            URL(fileURLWithPath: try XCTUnwrap(environment[name], "Set \(name) for production verification."))
        }
        let checkpointURL = try requiredURL("TEXTURE_STUDIO_DEPTHPRO_PRODUCTION_CHECKPOINT")
        let modelURL = try requiredURL("TEXTURE_STUDIO_DEPTHPRO_PRODUCTION_MODEL")
        let inputURL = try requiredURL("TEXTURE_STUDIO_DEPTHPRO_PRODUCTION_INPUT")
        let configured = NativeDepthProContract.configuredModelURL.standardizedFileURL.resolvingSymlinksInPath()
        guard configured == modelURL.standardizedFileURL.resolvingSymlinksInPath() else {
            throw XCTSkip("The production DepthPro preference resolves to \(configured.path), rather than the requested model. This test does not change shared model preferences.")
        }
        let sharedModelPath = StudioPreferences.defaults.string(forKey: NativeDepthProContract.modelPathKey)
        let sharedSelection = try? Data(contentsOf: SelectedMaterialCheckpoint.registryURL)
        let checkpointDigest = try NativeMaterialTransfer.hash(checkpointURL)
        let inputDigest = try NativeMaterialTransfer.hash(inputURL)
        let upstreamDigest = try NativeDepthProService.modelIdentity(at: modelURL)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("depthpro-production-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "depthpro-production-selection-\(UUID().uuidString)"
        let preferences = try XCTUnwrap(UserDefaults(suiteName: suite))
        preferences.set(modelURL.path, forKey: NativeDepthProContract.modelPathKey)
        defer { preferences.removePersistentDomain(forName: suite) }
        let registry = root.appendingPathComponent("selected.json")
        let store = WorkbenchStore(preferences: preferences, managedWorkspaceURL: root,
            selectedCheckpointRegistryURL: registry)
        store.modelDirectory = root.appendingPathComponent("absent-pbr-base").path
        let imported = try await MaterialCheckpointImporter.activate(checkpointURL, using: store)
        XCTAssertEqual(imported.modelFamily, .depthProHeightRefiner)
        XCTAssertEqual(imported.target, "height")
        let selected = try SelectedMaterialCheckpoint.read(from: registry)
        XCTAssertEqual(selected.sha256, checkpointDigest)
        let inspection = try NativeMaterialTransfer.object(Data(NativeMaterialCheckpoint.inspect(
            at: checkpointURL, expectedSHA256: checkpointDigest).utf8))
        XCTAssertEqual(inspection["depthpro_model_sha256"] as? String, upstreamDigest)
        XCTAssertEqual(inspection["depthpro_preparation_contract"] as? String, NativeDepthProContract.identifier)

        let engine = TextureEngine()
        let source = try await engine.importPhoto(inputURL)
        XCTAssertEqual([source.pixelWidth, source.pixelHeight], [2048, 2048])
        var settings = TextureSettings()
        settings.outputSize = 256
        settings.useSupportingViews = false
        let prepared = try await engine.prepareDiffuse(source: source, settings: settings)
        let grid = CGRect(x: 0, y: 0, width: 256, height: 256)
        XCTAssertEqual(prepared.diffuse.extent, grid)
        let preparedURL = root.appendingPathComponent("prepared.png")
        try MaterialCheckpointService.writeInferencePhoto(prepared.diffuse, maximumSize: 256, to: preparedURL)
        let header = try NativePNG.inspect(preparedURL)
        XCTAssertEqual([header.width, header.height], [256, 256])
        XCTAssertEqual(header.bits, 16)
        let preparedDigest = try NativeMaterialTransfer.hash(preparedURL)
        let directOutput = root.appendingPathComponent("direct-inference")
        let inference = try NativeMaterialTransfer.object(Data(await NativeMaterialTrainer.run(arguments: ["infer",
            "--checkpoint", selected.checkpointPath, "--expected-sha256", selected.sha256,
            "--model-family", "depthpro-height-refiner", "--depthpro-model", modelURL.path,
            "--image", preparedURL.path, "--output", directOutput.path]).utf8))
        XCTAssertEqual(inference["checkpoint_sha256"] as? String, checkpointDigest)
        XCTAssertEqual(inference["checkpoint_step"] as? Int, imported.step)
        XCTAssertEqual(inference["image_sha256"] as? String, preparedDigest)
        XCTAssertEqual(inference["input_kind"] as? String, "depthpro")
        XCTAssertEqual(inference["input_channels"] as? Int, 1)
        XCTAssertEqual(inference["output_channels"] as? Int, 1)
        XCTAssertEqual(inference["native_dimensions"] as? [Int], [256, 256])
        XCTAssertEqual(inference["depthpro_model_sha256"] as? String, upstreamDigest)
        XCTAssertEqual(inference["depthpro_preparation_contract"] as? String, NativeDepthProContract.identifier)
        let output = try XCTUnwrap((inference["outputs"] as? [String: [String: Any]])?["height"])
        XCTAssertEqual(output["storage"] as? String, "FLOAT32")
        XCTAssertEqual(output["encoding"] as? String, "linear_data")
        let exr = URL(fileURLWithPath: try XCTUnwrap(output["path"] as? String))
        try Self.verifyScalarFloat32EXR(exr)
        let stored = try XCTUnwrap(CIImage(contentsOf: exr, options: [.colorSpace: NSNull()]))
        XCTAssertEqual(stored.extent, grid)
        let directValues = Self.values(stored)
        XCTAssertTrue(directValues.allSatisfy(\.isFinite))

        let prediction = try await MaterialCheckpointService().predictMap(diffuse: prepared.diffuse, checkpoint: selected, size: 256)
        XCTAssertEqual(prediction.target, "height")
        XCTAssertEqual(prediction.image.extent, grid)
        XCTAssertTrue(prediction.sourceLabel.contains(String(checkpointDigest.prefix(12))))
        let productionValues = Self.values(prediction.image)
        XCTAssertTrue(productionValues.allSatisfy(\.isFinite))
        XCTAssertEqual(productionValues.map(\.bitPattern), directValues.map(\.bitPattern),
            "Production must preserve the same one-channel Float32 result as identical prepared-PNG inference")
        XCTAssertEqual(try NativeMaterialTransfer.hash(checkpointURL), checkpointDigest)
        XCTAssertEqual(try NativeMaterialTransfer.hash(inputURL), inputDigest)
        XCTAssertEqual(try NativeMaterialTransfer.hash(preparedURL), preparedDigest)
        XCTAssertEqual(StudioPreferences.defaults.string(forKey: NativeDepthProContract.modelPathKey), sharedModelPath)
        XCTAssertEqual(try? Data(contentsOf: SelectedMaterialCheckpoint.registryURL), sharedSelection)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.modelDirectory))

        if let path = environment["TEXTURE_STUDIO_DEPTHPRO_PRODUCTION_REPORT"], !path.isEmpty {
            let receipt: [String: Any] = ["checkpoint_sha256": checkpointDigest, "checkpoint_step": imported.step,
                "source_path": inputURL.path, "source_sha256": inputDigest, "source_dimensions": [2048, 2048],
                "prepared_sha256": preparedDigest, "model_family": "depthpro-height-refiner",
                "depthpro_model_sha256": upstreamDigest, "depthpro_preparation_contract": NativeDepthProContract.identifier,
                "input_channels": 1, "output_channels": 1, "output_dimensions": [256, 256], "output_storage": "FLOAT32",
                "output_sha256": try NativeMaterialTransfer.hash(exr), "numeric_readback_bitwise_equal": true,
                "model_and_source_unchanged": true, "shared_model_preference_and_selection_unchanged": true,
                "minimum": productionValues.min()!, "maximum": productionValues.max()!, "quality_qualification": false,
                "fixture": "original white_stucco_02 2K diffuse prepared through TextureEngine at256"]
            let report = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: report.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: report)
        }
    }

    private static func values(_ image: CIImage) -> [Float] {
        var values = [Float](repeating: 0, count: 256 * 256)
        let numeric = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        values.withUnsafeMutableBytes { numeric.render(image, toBitmap: $0.baseAddress!, rowBytes: 256 * 4,
            bounds: image.extent, format: .Rf, colorSpace: nil) }
        return values
    }

    private static func verifyScalarFloat32EXR(_ url: URL) throws {
        let bytes = try Data(contentsOf: url)
        var cursor = 8
        func word(_ offset: Int) throws -> UInt32 {
            guard offset + 4 <= bytes.count else { throw StudioError("Truncated EXR word.") }
            return bytes.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        func string() throws -> String {
            let start = cursor
            while cursor < bytes.count && bytes[cursor] != 0 { cursor += 1 }
            guard cursor < bytes.count else { throw StudioError("Truncated EXR string.") }
            defer { cursor += 1 }
            return String(decoding: bytes[start..<cursor], as: UTF8.self)
        }
        XCTAssertEqual(try word(0), 20_000_630)
        var channels: Data?
        while cursor < bytes.count && bytes[cursor] != 0 {
            let name = try string(), type = try string(), count = Int(try word(cursor))
            cursor += 4
            guard count <= bytes.count - cursor else { throw StudioError("Truncated EXR attribute.") }
            if name == "channels" {
                XCTAssertEqual(type, "chlist")
                channels = bytes.subdata(in: cursor..<cursor + count)
            }
            cursor += count
        }
        let list = try XCTUnwrap(channels)
        XCTAssertEqual(list.prefix(2), Data([0x59, 0]))
        XCTAssertEqual(list.count, 19, "Exactly one grayscale Y channel")
        guard list.count >= 6 else { throw StudioError("Truncated EXR channel declaration.") }
        let storage = list.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 2, as: UInt32.self)) }
        XCTAssertEqual(storage, 2, "EXR pixel type2 is Float32")
    }
}
