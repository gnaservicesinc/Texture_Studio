import CryptoKit
import Foundation
import XCTest
@testable import TextureStudio

final class NativeSafetensorsTests: XCTestCase {
    func testNativeWriterRoundTripsExactBitsAndPublishesAtomically() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("native.safetensors")
        let payload = encoded([UInt32(0x80000000), 1, 0x3f800001])
        let tensors = ["weight": NativeTensor(dtype: "F32", shape: [3], bytes: payload),
            "empty": NativeTensor(dtype: "I64", shape: [0], bytes: Data())]
        let metadata = ["source": "native Swift", "configuration": "{}"]
        let first = try NativeSafetensors.encoded(tensors: tensors, metadata: metadata)
        XCTAssertEqual(first, try NativeSafetensors.encoded(tensors: tensors, metadata: metadata))
        try NativeSafetensors.write(tensors: tensors, metadata: metadata, to: url)
        let saved = try NativeSafetensors(contentsOf: url)
        XCTAssertEqual(saved.bytes, first); XCTAssertEqual(saved.metadata, metadata)
        XCTAssertEqual(try saved.tensorBytes(named: "weight"), payload)
        XCTAssertEqual(try saved.nativeTensors()["empty"]?.bytes, Data())
        let bad = ["weight": NativeTensor(dtype: "F32", shape: [3], bytes: encoded([UInt32(0x7fc00001), 1, 0]))]
        XCTAssertThrowsError(try NativeSafetensors.write(tensors: bad, metadata: metadata, to: url))
        XCTAssertEqual(try Data(contentsOf: url), first)
    }
    func testAllFloatingPointBitsRemainExact() throws {
        let cases: [(String, Data)] = [
            ("F16", encoded([UInt16(0x8000), 1, 0x3c01])),
            ("BF16", encoded([UInt16(0x8000), 1, 0x3f81])),
            ("F32", encoded([UInt32(0x80000000), 1, 0x3f800001])),
            ("F64", encoded([UInt64(0x8000000000000000), 1, 0x3ff0000000000001]))]
        for (dtype, payload) in cases {
            let bytes = file(header: ["weight": descriptor(dtype: dtype, shape: [3], offsets: [0, payload.count])], payload: payload)
            let snapshot = try NativeSafetensors(bytes: bytes)
            XCTAssertEqual(snapshot.bytes, bytes)
            XCTAssertEqual(try snapshot.tensorBytes(named: "weight"), payload)
            XCTAssertEqual(snapshot.sha256, SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined())
            XCTAssertNoThrow(try snapshot.validateFiniteFloatingPoint())
        }
    }

    func testScalarAndEmptyTensorsAndTrailingHeaderPadding() throws {
        let payload = encoded([UInt32(0x3f800000)])
        let snapshot = try NativeSafetensors(bytes: file(header: [
            "empty": descriptor(shape: [0, 12], offsets: [0, 0]),
            "scalar": descriptor(shape: [], offsets: [0, 4])], payload: payload, padding: 7))
        XCTAssertEqual(try snapshot.tensorBytes(named: "empty"), Data())
        XCTAssertEqual(try snapshot.tensorBytes(named: "scalar"), payload)
    }

    func testInvalidRangesDimensionsAndDtypesAreRejected() throws {
        let cases: [[String: Any]] = [
            descriptor(shape: [1], offsets: [1, 5]),
            descriptor(shape: [2], offsets: [0, 4]),
            descriptor(shape: [-1], offsets: [0, 4]),
            descriptor(shape: [Int.max, 2], offsets: [0, 4]),
            descriptor(shape: [true], offsets: [0, 4]),
            descriptor(dtype: "unknown", shape: [1], offsets: [0, 4]),
            descriptor(shape: [1], offsets: [0, 5]),
            descriptor(shape: [1], offsets: [0]),
        ]
        for item in cases {
            XCTAssertThrowsError(try NativeSafetensors(bytes: file(header: ["weight": item], payload: encoded([UInt32(0)]))))
        }
        // JSONSerialization canonicalizes an integral Double to an integer;
        // retain the original floating lexeme to exercise the wire contract.
        for raw in [#"{"weight":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#,
                    #"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0.0,4.0]}}"#] {
            XCTAssertThrowsError(try NativeSafetensors(bytes: file(rawHeader: Data(raw.utf8), payload: encoded([UInt32(0)]))))
        }
        let overlap = ["one": descriptor(shape: [1], offsets: [0, 4]), "two": descriptor(shape: [1], offsets: [0, 4])]
        XCTAssertThrowsError(try NativeSafetensors(bytes: file(header: overlap, payload: encoded([UInt32(0)]))))
        XCTAssertThrowsError(try NativeSafetensors(bytes: file(header: ["weight": descriptor(shape: [1], offsets: [0, 4])], payload: Data(repeating: 0, count: 5))))
        XCTAssertThrowsError(try NativeSafetensors(bytes: file(header: ["__metadata__": ["not-a-string": 1]], payload: Data())))
    }

    func testDuplicateKeysIncludingEscapedNamesAreRejected() {
        let duplicate = Data(#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"\u0077eight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.utf8)
        XCTAssertThrowsError(try NativeSafetensors(bytes: file(rawHeader: duplicate, payload: encoded([UInt32(0)]))))
        let malformed = Data(#"{"weight":{"dtype":"F32","shape":[1],"shape":[2],"data_offsets":[0,4]}}"#.utf8)
        XCTAssertThrowsError(try NativeSafetensors(bytes: file(rawHeader: malformed, payload: encoded([UInt32(0)]))))
    }

    func testTruncationAndIncorrectSelectedHashAreRejected() {
        XCTAssertThrowsError(try NativeSafetensors(bytes: Data()))
        XCTAssertThrowsError(try NativeSafetensors(bytes: encoded([UInt64.max])))
        XCTAssertThrowsError(try NativeSafetensors(bytes: encoded([UInt64(101)])))
        let bytes = file(header: ["weight": descriptor(shape: [1], offsets: [0, 4])], payload: encoded([UInt32(0)]))
        XCTAssertThrowsError(try NativeSafetensors(bytes: bytes, expectedSHA256: String(repeating: "0", count: 64)))
    }

    func testNonfiniteFloatEncodingsAreRejectedWithoutChangingBits() throws {
        let cases: [(String, Data)] = [
            ("F16", encoded([UInt16(0x7c00)])), ("F16", encoded([UInt16(0x7e01)])),
            ("BF16", encoded([UInt16(0xff80)])), ("BF16", encoded([UInt16(0x7fc1)])),
            ("F32", encoded([UInt32(0xff800000)])), ("F32", encoded([UInt32(0x7fc00001)])),
            ("F64", encoded([UInt64(0x7ff0000000000000)])), ("F64", encoded([UInt64(0x7ff8000000000001)]))]
        for (dtype, payload) in cases {
            let snapshot = try NativeSafetensors(bytes: file(header: ["weight": descriptor(dtype: dtype, shape: [1], offsets: [0, payload.count])], payload: payload))
            XCTAssertThrowsError(try snapshot.validateFiniteFloatingPoint())
            XCTAssertEqual(try snapshot.tensorBytes(named: "weight"), payload)
        }
    }

    func testFullCheckpointInspectionAndImmutableSnapshot() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("model.safetensors")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("source"), withIntermediateDirectories: false)
        let configuration = config()
        let payload = encoded([UInt32(0x80000000), 1])
        let bytes = file(header: ["weight": descriptor(shape: [2], offsets: [0, 8]), "__metadata__": metadata(configuration)], payload: payload)
        try bytes.write(to: url)
        let snapshot = try NativeSafetensors(contentsOf: url)
        let information = try JSONSerialization.jsonObject(with: Data(NativeMaterialCheckpoint.inspect(at: root).utf8)) as! [String: Any]
        XCTAssertEqual(information["checkpoint_path"] as? String, url.path)
        XCTAssertEqual(information["variant"] as? String, "full")
        XCTAssertEqual(information["sha256"] as? String, snapshot.sha256)
        XCTAssertNil(information["code_directory"])
        XCTAssertNil(information["model_directory"])
        XCTAssertEqual(information["supports_training_warm_start"] as? Bool, true)
        try Data("replacement".utf8).write(to: url, options: .atomic)
        XCTAssertEqual(try snapshot.tensorBytes(named: "weight"), payload)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url, expectedSHA256: snapshot.sha256))
    }

    func testLoRAInspectionChecksIdentitiesShapesTargetsAndPrecision() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("adapter.safetensors")
        var configuration = config()
        configuration["schema"] = "texture-studio-material-lora-v1"
        configuration["scope"] = "final-map"
        configuration["layers"] = ["ups.3.conv_last": ["weight_shape": [1, 2], "rank": 1, "alpha": 1]]
        var header: [String: Any] = [
            "ups.3.conv_last.lora_A": descriptor(shape: [1, 2], offsets: [0, 8]),
            "ups.3.conv_last.lora_B": descriptor(shape: [1, 1], offsets: [8, 12]),
            "__metadata__": metadata(configuration)]
        let payload = encoded([UInt32(0x3f800000), 0x3f800001, 0x80000000])
        try file(header: header, payload: payload).write(to: url)
        let result = try JSONSerialization.jsonObject(with: Data(NativeMaterialCheckpoint.inspect(at: root).utf8)) as! [String: Any]
        XCTAssertEqual(result["variant"] as? String, "lora")
        configuration["target"] = "roughness"
        header["__metadata__"] = metadata(configuration)
        try file(header: header, payload: payload).write(to: url)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url))
        configuration["target"] = "height"
        header["__metadata__"] = metadata(configuration)
        header["ups.3.conv_last.lora_A"] = descriptor(dtype: "I32", shape: [1, 2], offsets: [0, 8])
        try file(header: header, payload: payload).write(to: url)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url))
        header["ups.3.conv_last.lora_A"] = descriptor(shape: [2, 1], offsets: [0, 8])
        try file(header: header, payload: payload).write(to: url)
        XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url))
    }

    func testMalformedNativeMaterialConfigurationIsRejected() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appendingPathComponent("model.safetensors")
        let modifications: [(String, Any)] = [("step", true), ("step", -1), ("training_size", 1025),
            ("image_padding", true), ("image_resizing", true), ("architecture", "another-network"),
            ("base", ["sha256": String(repeating: "f", count: 63)]), ("target", "diffuse")]
        for (key, value) in modifications {
            var configuration = config(); configuration[key] = value
            let header: [String: Any] = ["weight": descriptor(shape: [1], offsets: [0, 4]), "__metadata__": metadata(configuration)]
            try file(header: header, payload: encoded([UInt32(0)])).write(to: url)
            XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url), key)
        }
        let original = try XCTUnwrap(metadata(config())["configuration"])
        for (old, new) in [("\"step\":12", "\"step\":12.0"), ("\"training_size\":1024", "\"training_size\":1024e0")] {
            let text = original.replacingOccurrences(of: old, with: new)
            XCTAssertNotEqual(text, original)
            let header: [String: Any] = ["weight": descriptor(shape: [1], offsets: [0, 4]), "__metadata__": ["configuration": text]]
            try file(header: header, payload: encoded([UInt32(0)])).write(to: url)
            XCTAssertThrowsError(try NativeMaterialCheckpoint.inspect(at: url))
        }
    }

    func testCancelledCheckpointInspectionDoesNotReadMissingFile() async throws {
        let gate = StartGate()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        let request = Task {
            await gate.wait()
            return try NativeMaterialCheckpoint.inspect(at: url)
        }
        await gate.waitUntilRequested()
        request.cancel()
        await gate.release()
        do {
            _ = try await request.value
            XCTFail("Cancellation must be observed before reading the checkpoint")
        } catch is CancellationError { }
    }

    private actor StartGate {
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async { await withCheckedContinuation { continuation = $0 } }
        func waitUntilRequested() async { while continuation == nil { await Task.yield() } }
        func release() { continuation?.resume(); continuation = nil }
    }

    private func config() -> [String: Any] {
        ["schema": "texture-studio-material-checkpoint-v1", "architecture": "pbrnxt-native-v1", "target": "height",
         "step": 12, "training_size": 1024, "image_padding": false, "image_resizing": false,
         "base": ["sha256": String(repeating: "a", count: 64)]]
    }
    private func metadata(_ config: [String: Any]) -> [String: String] {
        ["configuration": String(decoding: try! JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]), as: UTF8.self)]
    }
    private func descriptor(dtype: String = "F32", shape: [Any], offsets: [Int]) -> [String: Any] {
        ["dtype": dtype, "shape": shape, "data_offsets": offsets]
    }
    private func file(header: [String: Any], payload: Data, padding: Int = 0) -> Data {
        var raw = try! JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        raw.append(Data(repeating: 32, count: padding))
        return file(rawHeader: raw, payload: payload)
    }
    private func file(rawHeader: Data, payload: Data) -> Data {
        var bytes = encoded([UInt64(rawHeader.count)])
        bytes.append(rawHeader); bytes.append(payload)
        return bytes
    }
    private func encoded<T: FixedWidthInteger>(_ values: [T]) -> Data {
        values.map(\.littleEndian).withUnsafeBytes { Data($0) }
    }
    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-checkpoint-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
}
