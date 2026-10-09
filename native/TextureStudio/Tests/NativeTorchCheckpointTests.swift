import CryptoKit
import Foundation
import XCTest
import zlib
@testable import TextureStudio

final class NativeTorchCheckpointTests: XCTestCase {
    func testStoredStateDictionaryPreservesFloatAndIntegerBitsAndModuleMetadata() throws {
        let float = Data([0, 0, 0, 128, 1, 0, 0, 0, 255, 255, 127, 63, 0, 0, 128, 63])
        let integer = Data([224, 0, 0, 0, 0, 0, 0, 0])
        let archive = try fixture(tensors: [Tensor("gen.weight", "FloatStorage", 4, 0, [2, 2], [2, 1], "0"),
                                         Tensor("gen.index", "LongStorage", 1, 0, [1], [1], "1")],
                                  storages: ["0": float, "1": integer], metadata: true)
        let result = try NativeTorchCheckpoint.load(bytes: archive)
        XCTAssertEqual(result.sha256, digest(archive))
        XCTAssertEqual(result.tensors["gen.weight"]?.dtype, "F32")
        XCTAssertEqual(result.tensors["gen.weight"]?.shape, [2, 2])
        XCTAssertEqual(result.tensors["gen.weight"]?.bytes, float, "Signed zero and subnormal codes are untouched")
        XCTAssertEqual(result.tensors["gen.index"]?.dtype, "I64")
        XCTAssertEqual(result.tensors["gen.index"]?.bytes, integer)
    }
    func testDeflatedZIPTensorViewGathersStridedOriginalStorageCodes() throws {
        let storage = Data([0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64])
        let archive = try fixture(tensors: [Tensor("weight", "FloatStorage", 5, 1, [2, 2], [1, 2], "0")], storages: ["0": storage], deflated: true)
        let result = try NativeTorchCheckpoint.load(bytes: archive)
        XCTAssertEqual(result.tensors["weight"]?.bytes, Data([0, 0, 128, 63, 0, 0, 64, 64, 0, 0, 0, 64, 0, 0, 128, 64]))
    }
    func testEmptyTensorRetainsShapeAndNoFabricatedBytes() throws {
        let archive = try fixture(tensors: [Tensor("empty", "FloatStorage", 0, 0, [0, 3], [3, 1], "0")], storages: ["0": Data()])
        let result = try NativeTorchCheckpoint.load(bytes: archive)
        XCTAssertEqual(result.tensors["empty"]?.shape, [0, 3])
        XCTAssertEqual(result.tensors["empty"]?.bytes, Data())
    }
    func testRejectsArbitraryPickleGlobalsAndStorageViewOutsideBounds() throws {
        var malicious = Data([0x80, 2]); global("os", "system", to: &malicious); malicious.append(contentsOf: [0x29, 0x52, 0x2e])
        let archive = try zip([("archive/data.pkl", malicious), ("archive/byteorder", Data("little".utf8))])
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: archive)) { error in
            XCTAssertTrue(error.localizedDescription.contains("not a data-only tensor operation"))
        }
        let outside = try fixture(tensors: [Tensor("weight", "FloatStorage", 1, 1, [1], [1], "0")], storages: ["0": Data(repeating: 0, count: 4)])
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: outside))
    }
    func testRejectsChangedChecksumExpectedDigestAndTruncatedArchive() throws {
        let valid = try fixture(tensors: [Tensor("weight", "FloatStorage", 1, 0, [1], [1], "0")], storages: ["0": Data([1, 2, 3, 4])])
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: valid, expectedSHA256: String(repeating: "0", count: 64)))
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: valid.dropLast()))
        var corrupted = valid
        let location = try XCTUnwrap(corrupted.range(of: Data([1, 2, 3, 4])))
        corrupted[location.lowerBound] ^= 1
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: corrupted)) { error in
            XCTAssertTrue(error.localizedDescription.contains("checksum changed"))
        }
    }
    func testRejectsDuplicateStateKeysAndArchiveTraversal() throws {
        let duplicate = try fixture(tensors: [Tensor("weight", "FloatStorage", 1, 0, [1], [1], "0"),
                                             Tensor("weight", "FloatStorage", 1, 0, [1], [1], "0")], storages: ["0": Data(repeating: 0, count: 4)])
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: duplicate))
        let traversal = try zip([("archive/../data.pkl", Data([0x80, 2, 0x7d, 0x2e]))])
        XCTAssertThrowsError(try NativeTorchCheckpoint.load(bytes: traversal))
    }
    func testCancelledImportDoesNotParseOrReturnWeights() async throws {
        let archive = try fixture(tensors: [Tensor("weight", "FloatStorage", 1, 0, [1], [1], "0")], storages: ["0": Data(repeating: 0, count: 4)])
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try NativeTorchCheckpoint.load(bytes: archive)
        }
        do { _ = try await task.value; XCTFail("Cancelled import must fail") }
        catch { XCTAssertTrue(error is CancellationError) }
    }

    private struct Tensor {
        let name: String, storageType: String, count: Int, offset: Int, shape: [Int], stride: [Int], key: String
        init(_ name: String, _ storageType: String, _ count: Int, _ offset: Int, _ shape: [Int], _ stride: [Int], _ key: String) {
            self.name = name; self.storageType = storageType; self.count = count; self.offset = offset; self.shape = shape; self.stride = stride; self.key = key
        }
    }
    /// Independent bytes follow torch.save's protocol-2 OrderedDict/tensor
    /// descriptors and ZIP storage convention; no interpreter creates fixtures.
    private func fixture(tensors: [Tensor], storages: [String: Data], deflated: Bool = false, metadata: Bool = false) throws -> Data {
        var pickle = Data([0x80, 2])
        global("collections", "OrderedDict", to: &pickle); pickle.append(contentsOf: [0x71, 0, 0x29, 0x52, 0x71, 1, 0x28])
        for tensor in tensors {
            text(tensor.name, to: &pickle); global("torch._utils", "_rebuild_tensor_v2", to: &pickle); pickle.append(contentsOf: [0x28, 0x28])
            text("storage", to: &pickle); global("torch", tensor.storageType, to: &pickle); text(tensor.key, to: &pickle); text("cpu", to: &pickle); number(tensor.count, to: &pickle)
            pickle.append(contentsOf: [0x74, 0x51]); number(tensor.offset, to: &pickle)
            tuple(tensor.shape, to: &pickle); tuple(tensor.stride, to: &pickle)
            pickle.append(contentsOf: [0x89, 0x68, 0, 0x29, 0x52, 0x74, 0x52])
        }
        pickle.append(0x75)
        if metadata {
            pickle.append(0x7d); text("_metadata", to: &pickle)
            global("collections", "OrderedDict", to: &pickle); pickle.append(contentsOf: [0x29, 0x52])
            text("gen", to: &pickle); pickle.append(0x7d); text("version", to: &pickle); number(1, to: &pickle)
            pickle.append(contentsOf: [0x73, 0x73, 0x73, 0x62])
        }
        pickle.append(0x2e)
        var entries = [("archive/data.pkl", pickle), ("archive/byteorder", Data("little".utf8)), ("archive/version", Data("3\n".utf8))]
        entries += storages.keys.sorted().map { ("archive/data/" + $0, storages[$0]!) }
        return try zip(entries, deflated: deflated)
    }
    private func global(_ module: String, _ name: String, to bytes: inout Data) { bytes.append(0x63); bytes.append(Data("\(module)\n\(name)\n".utf8)) }
    private func text(_ text: String, to bytes: inout Data) { let payload = Data(text.utf8); bytes.append(0x58); bytes.little(UInt32(payload.count)); bytes.append(payload) }
    private func number(_ value: Int, to bytes: inout Data) { bytes.append(0x4a); bytes.little(UInt32(bitPattern: Int32(value))) }
    private func tuple(_ values: [Int], to bytes: inout Data) { bytes.append(0x28); for value in values { number(value, to: &bytes) }; bytes.append(0x74) }
    private func digest(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    private func zip(_ entries: [(String, Data)], deflated: Bool = false) throws -> Data {
        var output = Data(), directory = Data()
        for (name, payload) in entries {
            let filename = Data(name.utf8), offset = output.count
            let crc = payload.withUnsafeBytes { UInt32(crc32_z(0, $0.bindMemory(to: Bytef.self).baseAddress, payload.count)) }
            let compressed: Data
            if deflated {
                var stream = z_stream()
                XCTAssertEqual(deflateInit2_(&stream, 1, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)), Z_OK)
                defer { deflateEnd(&stream) }
                let capacity = Int(deflateBound(&stream, uLong(payload.count)))
                var bytes = Data(count: capacity)
                let status = bytes.withUnsafeMutableBytes { destination in payload.withUnsafeBytes { source in
                    stream.next_in = UnsafeMutablePointer(mutating: source.bindMemory(to: Bytef.self).baseAddress)
                    stream.avail_in = uInt(payload.count); stream.next_out = destination.bindMemory(to: Bytef.self).baseAddress; stream.avail_out = uInt(capacity)
                    return deflate(&stream, Z_FINISH)
                } }
                XCTAssertEqual(status, Z_STREAM_END); bytes.count = Int(stream.total_out); compressed = bytes
            } else { compressed = payload }
            output.little(UInt32(0x04034b50)); output.little(UInt16(20)); output.little(UInt16(0)); output.little(UInt16(deflated ? 8 : 0))
            output.little(UInt16(0)); output.little(UInt16(0)); output.little(crc); output.little(UInt32(compressed.count)); output.little(UInt32(payload.count)); output.little(UInt16(filename.count)); output.little(UInt16(0)); output.append(filename); output.append(compressed)
            directory.little(UInt32(0x02014b50)); directory.little(UInt16(20)); directory.little(UInt16(20)); directory.little(UInt16(0)); directory.little(UInt16(deflated ? 8 : 0)); directory.little(UInt16(0)); directory.little(UInt16(0)); directory.little(crc); directory.little(UInt32(compressed.count)); directory.little(UInt32(payload.count)); directory.little(UInt16(filename.count)); directory.little(UInt16(0)); directory.little(UInt16(0)); directory.little(UInt16(0)); directory.little(UInt16(0)); directory.little(UInt32(0)); directory.little(UInt32(offset)); directory.append(filename)
        }
        let directoryStart = output.count
        output.append(directory); output.little(UInt32(0x06054b50)); output.little(UInt16(0)); output.little(UInt16(0)); output.little(UInt16(entries.count)); output.little(UInt16(entries.count)); output.little(UInt32(directory.count)); output.little(UInt32(directoryStart)); output.little(UInt16(0))
        return output
    }
}

private extension Data {
    mutating func little<T: FixedWidthInteger>(_ number: T) { var value = number.littleEndian; append(Swift.withUnsafeBytes(of: &value) { Data($0) }) }
}
