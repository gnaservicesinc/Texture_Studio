import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialTrainerTests: XCTestCase {
    func testControlKeepsSnapshotSaveAndAbortSemanticsIndependent() throws {
        let control = NativeMaterialTrainingControl()
        XCTAssertFalse(control.consumeCheckpoint()); XCTAssertFalse(control.shouldStopAndSave)
        control.saveCheckpoint(); XCTAssertTrue(control.consumeCheckpoint()); XCTAssertFalse(control.consumeCheckpoint())
        control.stopAndSave(); XCTAssertTrue(control.shouldStopAndSave); XCTAssertNoThrow(try control.check())
        control.stop(); XCTAssertThrowsError(try control.check()) { XCTAssertTrue($0 is CancellationError) }
    }
    func testEXRStoresEveryFloat32BitAndTopLeftChannelWithoutNormalization() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let words: [UInt32] = [0x80000000, 1, 0x3f800001, 0xbf000000, 0x40000001, 0x00800001,
            0x3e000003, 0xbe000004, 0x3f400005, 0x3e800006, 0x3fa00007, 0xbf800008,
            0x3f000009, 0xbe80000a, 0x3fc0000b, 0x3d00000c, 0x4020000d, 0xc010000e]
        let values = words.map(Float.init(bitPattern:)), url = root.appendingPathComponent("numeric.exr")
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 3, values: values), to: url)
        let bytes = try Data(contentsOf: url)
        XCTAssertEqual(u32(bytes, 0), 20_000_630); XCTAssertEqual(u32(bytes, 4), 2)
        var cursor = 8, attributes: [String: (String, Data)] = [:]
        while bytes[cursor] != 0 {
            let name = cString(bytes, &cursor), type = cString(bytes, &cursor), count = Int(u32(bytes, cursor)); cursor += 4
            attributes[name] = (type, bytes.subdata(in: cursor..<cursor + count)); cursor += count
        }
        cursor += 1
        XCTAssertEqual(attributes["compression"]?.1, Data([0])); XCTAssertEqual(attributes["channels"]?.0, "chlist")
        let channelData = try XCTUnwrap(attributes["channels"]?.1)
        var channelCursor = 0, names: [String] = []
        while channelData[channelCursor] != 0 {
            names.append(cString(channelData, &channelCursor)); XCTAssertEqual(u32(channelData, channelCursor), 2); channelCursor += 16
        }
        XCTAssertEqual(names, ["B", "G", "R"])
        for row in 0..<2 {
            let offset = Int(u64(bytes, cursor + row * 8))
            XCTAssertEqual(u32(bytes, offset), UInt32(row)); XCTAssertEqual(u32(bytes, offset + 4), 36)
            for (stored, original) in [2, 1, 0].enumerated() {
                for column in 0..<3 { XCTAssertEqual(u32(bytes, offset + 8 + (stored * 3 + column) * 4), words[original * 6 + row * 3 + column]) }
            }
        }
    }
    func testRealTrainerUpdatesValidatesSnapshotsAndStopsWithCompleteNativePackage() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset)
        let fixture = NativeMaterialModelFixture(), original = try fixture.model(adapter: true)
        var configuration: [String: Any] = ["schema": "texture-studio-material-lora-v1", "architecture": "pbrnxt-native-v1",
            "target": "height", "scope": "final-map", "step": 0, "training_size": 256,
            "image_padding": false, "image_resizing": false, "base": ["sha256": String(repeating: "a", count: 64)]]
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original.layers))
        let model = try NativeMaterialModel(baseWeights: original.baseWeights, adapterWeights: original.adapterWeights,
            layers: original.layers, configuration: configuration, baseSHA256: String(repeating: "a", count: 64), architecture: .test)
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "5", "--validation-every", "2", "--learning-rate", "0.001"])
        let control = NativeMaterialTrainingControl(), events = Recorder()
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if line.contains("\"event\":\"update\"") {
                if events.updates == 1 { control.saveCheckpoint() } else { control.stopAndSave() }
            }
        }, control: control, model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped"); XCTAssertEqual(result["completed_updates"] as? Int, 2)
        let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("export/adapter.safetensors"))
        XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original.adapterWeights["ups.3.model.10.lora_B"]!.bytes)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000001.safetensors").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000002.safetensors").path))
        XCTAssertEqual(events.updates, 2)
        XCTAssertFalse(events.executedOnMain)
        let final = try XCTUnwrap(result["final_validation"] as? [String: Any])
        XCTAssertEqual(final["scope"] as? String, "full"); XCTAssertEqual(final["sample_count"] as? Int, 1)
    }
    func testAbortBeforeTrainingDoesNotCreateOutput() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", root.path, "--output", root.appendingPathComponent("cancelled").path])
        let control = NativeMaterialTrainingControl(); control.stop()
        XCTAssertThrowsError(try NativeMaterialTrainer.train(options, onEvent: { _ in }, control: control)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: options.output.path))
    }
    private final class Recorder: @unchecked Sendable {
        let lock = NSLock(); private var lines: [String] = [], mainThread = false
        func append(_ line: String) { lock.withLock { lines.append(line); mainThread = mainThread || Thread.isMainThread } }
        var updates: Int { lock.withLock { lines.filter { $0.contains("\"event\":\"update\"") }.count } }
        var executedOnMain: Bool { lock.withLock { mainThread } }
    }
    private func datasetFixture(_ root: URL) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        var entries: [[String: Any]] = []
        for split in ["train", "validation"] {
            let folder = root.appendingPathComponent("samples/" + split)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let input = try NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data(repeating: 127, count: 256 * 256 * 3), colorChunks: []).encoded()
            let target = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: Data(repeating: 128, count: 256 * 256 * 2), colorChunks: []).encoded()
            try input.write(to: folder.appendingPathComponent("diffuse.png")); try target.write(to: folder.appendingPathComponent("height.png"))
            let entry: [String: Any] = ["sample_id": split, "material_id": split, "status": "approved", "split": split, "path": "samples/" + split]
            entries.append(entry)
            let sample: [String: Any] = ["sample_id": split, "material_id": split, "status": "approved", "split": split,
                "sample_pixel_dimensions": [256, 256], "maps": ["input": "diffuse.png", "height": "height.png"],
                "map_metadata": ["input": ["filename": "diffuse.png", "encoding": "srgb", "sample_sha256": NativeMaterialTrainer.checksum(input)],
                    "height": ["filename": "height.png", "sample_sha256": NativeMaterialTrainer.checksum(target)]]]
            try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
        }
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "samples": entries]).write(to: root.appendingPathComponent("dataset.json"))
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-training-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root
    }
    private func cString(_ bytes: Data, _ cursor: inout Int) -> String {
        let start = cursor; while bytes[cursor] != 0 { cursor += 1 }
        defer { cursor += 1 }; return String(decoding: bytes[start..<cursor], as: UTF8.self)
    }
    private func u32(_ bytes: Data, _ offset: Int) -> UInt32 { (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) } }
    private func u64(_ bytes: Data, _ offset: Int) -> UInt64 { (0..<8).reduce(UInt64(0)) { $0 | UInt64(bytes[offset + $1]) << ($1 * 8) } }
}
