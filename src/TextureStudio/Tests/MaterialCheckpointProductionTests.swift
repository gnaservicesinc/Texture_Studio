import CoreImage
import Foundation
import Metal
import XCTest
@testable import TextureStudio

@MainActor
final class MaterialCheckpointProductionTests: XCTestCase {
    /// Optional verification of an existing trained model through the production
    /// import, photo preparation, native inference and numeric readback paths.
    func testLocalCompactHeightCheckpointOnAnImported2KPhoto() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_COMPACT_PRODUCTION_VERIFY"] == "1" else {
            throw XCTSkip("Opt-in production verification with an existing compact checkpoint.")
        }
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let checkpointPath = try XCTUnwrap(environment["TEXTURE_STUDIO_COMPACT_PRODUCTION_CHECKPOINT"],
            "Set the existing checkpoint path when enabling production verification.")
        let checkpointURL = URL(fileURLWithPath: checkpointPath)
        let checkpointDigest = try NativeMaterialTransfer.hash(checkpointURL)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("compact-production-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let inputURL: URL
        if let path = environment["TEXTURE_STUDIO_COMPACT_PRODUCTION_INPUT"] {
            inputURL = URL(fileURLWithPath: path)
        } else {
            inputURL = root.appendingPathComponent("new-2k-photo.png")
            try Self.writeNewPhoto(to: inputURL)
        }
        let inputDigest = try NativeMaterialTransfer.hash(inputURL)
        let suite = "compact-production-selection-\(UUID().uuidString)"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let registry = root.appendingPathComponent("selected.json")
        let store = WorkbenchStore(preferences: preferences, managedWorkspaceURL: root,
            selectedCheckpointRegistryURL: registry)
        store.modelDirectory = root.appendingPathComponent("absent-pbr-base").path
        let imported = try await MaterialCheckpointImporter.activate(checkpointURL, using: store)
        XCTAssertEqual(imported.modelFamily, .compactScalar)
        XCTAssertEqual(imported.target, "height")
        let selected = try SelectedMaterialCheckpoint.read(from: registry)
        XCTAssertEqual(selected.sha256, checkpointDigest)
        XCTAssertEqual(selected.checkpointPath, checkpointURL.standardizedFileURL.resolvingSymlinksInPath().path)

        let engine = TextureEngine()
        let source = try await engine.importPhoto(inputURL)
        XCTAssertGreaterThanOrEqual(max(source.pixelWidth, source.pixelHeight), 2048)
        var settings = TextureSettings()
        settings.outputSize = 512
        settings.useSupportingViews = false
        let prepared = try await engine.prepareDiffuse(source: source, settings: settings)
        let grid = CGRect(x: 0, y: 0, width: 512, height: 512)
        XCTAssertEqual(prepared.diffuse.extent, grid)
        let preparedURL = root.appendingPathComponent("prepared.png")
        try MaterialCheckpointService.writeInferencePhoto(prepared.diffuse, maximumSize: 512, to: preparedURL)
        let header = try NativePNG.inspect(preparedURL)
        XCTAssertEqual(header.width, 512); XCTAssertEqual(header.height, 512)
        XCTAssertEqual(header.bits, 16)
        let preparedDigest = try NativeMaterialTransfer.hash(preparedURL)
        let output = root.appendingPathComponent("inference")
        let inferenceText = try await NativeMaterialTrainer.run(arguments: ["infer",
            "--checkpoint", selected.checkpointPath, "--expected-sha256", selected.sha256,
            "--image", preparedURL.path, "--output", output.path,
            "--input-kind", "diffuse", "--input-encoding", "srgb"])
        let inference = try NativeMaterialTransfer.object(Data(inferenceText.utf8))
        XCTAssertEqual(inference["checkpoint_sha256"] as? String, checkpointDigest)
        XCTAssertEqual(inference["checkpoint_step"] as? Int, imported.step)
        XCTAssertEqual(inference["image_sha256"] as? String, preparedDigest)
        XCTAssertEqual(inference["native_dimensions"] as? [Int], [512, 512])
        XCTAssertEqual(inference["output_channels"] as? Int, 1)
        let height = try XCTUnwrap((inference["outputs"] as? [String: [String: Any]])?["height"])
        XCTAssertEqual(height["storage"] as? String, "FLOAT32")
        XCTAssertEqual(height["encoding"] as? String, "linear_data")
        let exrURL = URL(fileURLWithPath: try XCTUnwrap(height["path"] as? String))
        XCTAssertEqual(exrURL.lastPathComponent, "height.float32.exr")
        XCTAssertEqual(height["sha256"] as? String, try NativeMaterialTransfer.hash(exrURL))
        try Self.verifyScalarFloat32EXR(exrURL)
        let stored = try XCTUnwrap(CIImage(contentsOf: exrURL, options: [.colorSpace: NSNull()]))
        XCTAssertEqual(stored.extent, grid)
        let storedValues = Self.values(stored)
        XCTAssertTrue(storedValues.allSatisfy(\.isFinite))

        let prediction = try await MaterialCheckpointService().predictMap(diffuse: prepared.diffuse, checkpoint: selected, size: 512)
        XCTAssertEqual(prediction.target, "height")
        XCTAssertEqual(prediction.image.extent, grid)
        XCTAssertTrue(prediction.sourceLabel.contains(String(checkpointDigest.prefix(12))))
        let productionValues = Self.values(prediction.image)
        XCTAssertTrue(productionValues.allSatisfy(\.isFinite))
        XCTAssertEqual(productionValues.map(\.bitPattern), storedValues.map(\.bitPattern),
            "The app's numeric readback must preserve the native model's Float32 prediction.")
        XCTAssertEqual(try NativeMaterialTransfer.hash(checkpointURL), checkpointDigest)
        XCTAssertEqual(try NativeMaterialTransfer.hash(inputURL), inputDigest)
        XCTAssertEqual(try NativeMaterialTransfer.hash(preparedURL), preparedDigest)
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.modelDirectory))

        if let receiptPath = environment["TEXTURE_STUDIO_COMPACT_PRODUCTION_REPORT"] {
            let receipt: [String: Any] = ["checkpoint_sha256": checkpointDigest, "checkpoint_step": imported.step,
                "source_sha256": inputDigest, "source_dimensions": [source.pixelWidth, source.pixelHeight],
                "prepared_sha256": preparedDigest, "output_dimensions": [512, 512], "output_storage": "FLOAT32",
                "output_sha256": try NativeMaterialTransfer.hash(exrURL), "numeric_readback_bitwise_equal": true,
                "model_and_source_unchanged": true, "minimum": productionValues.min()!, "maximum": productionValues.max()!,
                "quality_qualification": false, "fixture": environment["TEXTURE_STUDIO_COMPACT_PRODUCTION_INPUT"] == nil
                    ? "new procedural imported 2K RGB PNG" : "user-selected imported photo"]
            let receiptURL = URL(fileURLWithPath: receiptPath)
            try FileManager.default.createDirectory(at: receiptURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys]).write(to: receiptURL)
        }
    }

    private static func writeNewPhoto(to url: URL) throws {
        let size = 2048
        var pixels = Data(count: size * size * 3)
        pixels.withUnsafeMutableBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self)
            for y in 0..<size {
                for x in 0..<size {
                    let checker = ((x / 37 + y / 61) % 2) * 73
                    let index = (y * size + x) * 3
                    bytes[index] = UInt8((x * 3 + y + checker) % 211 + 22)
                    bytes[index + 1] = UInt8((x + y * 2 + checker) % 193 + 29)
                    bytes[index + 2] = UInt8((x * 2 + y * 3 + checker) % 173 + 31)
                }
            }
        }
        try NativePNG(header: .init(width: size, height: size, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: pixels, colorChunks: []).encoded().write(to: url)
    }

    private static func values(_ image: CIImage) -> [Float] {
        var result = [Float](repeating: 0, count: 512 * 512)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        result.withUnsafeMutableBytes { context.render(image, toBitmap: $0.baseAddress!, rowBytes: 512 * 4,
            bounds: image.extent, format: .Rf, colorSpace: nil) }
        return result
    }

    private static func verifyScalarFloat32EXR(_ url: URL) throws {
        let data = try Data(contentsOf: url)
        guard data.count > 8 else { throw StudioError("Truncated EXR header.") }
        func word(_ offset: Int) -> UInt32 {
            data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
        }
        var cursor = 8
        func string() throws -> String {
            let start = cursor
            while cursor < data.count && data[cursor] != 0 { cursor += 1 }
            guard cursor < data.count else { throw StudioError("Unterminated EXR attribute.") }
            defer { cursor += 1 }
            return String(decoding: data[start..<cursor], as: UTF8.self)
        }
        XCTAssertEqual(word(0), 20_000_630)
        var channels: Data?
        while cursor < data.count && data[cursor] != 0 {
            let name = try string(), type = try string()
            guard cursor + 4 <= data.count else { throw StudioError("Truncated EXR attribute size.") }
            let count = Int(word(cursor)); cursor += 4
            guard count <= data.count - cursor else { throw StudioError("Truncated EXR attribute.") }
            if name == "channels" {
                XCTAssertEqual(type, "chlist")
                channels = data.subdata(in: cursor..<cursor + count)
            }
            cursor += count
        }
        let list = try XCTUnwrap(channels)
        XCTAssertEqual(list.prefix(2), Data([0x59, 0]))
        XCTAssertEqual(list.count, 19)
        guard list.count >= 6 else { throw StudioError("Truncated EXR channel declaration.") }
        let storage = list.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 2, as: UInt32.self)) }
        XCTAssertEqual(storage, 2, "EXR pixel type 2 is full Float32 storage.")
    }
}
