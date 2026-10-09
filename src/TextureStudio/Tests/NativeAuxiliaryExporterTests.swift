import CoreVideo
import CoreImage
import XCTest
@testable import TextureStudio

final class NativeAuxiliaryExporterTests: XCTestCase {
    func testFloat32SpecialBitPatternsSurvivePaddingRemovalAndNPY() throws {
        let words: [UInt32] = [0x80000000, 0x7fc12345, 0x7f800000, 0xff800000]
        var raw = Data()
        for row in 0..<2 {
            for word in words[(row * 2)..<(row * 2 + 2)] {
                var encoded = word.littleEndian
                withUnsafeBytes(of: &encoded) { raw.append(contentsOf: $0) }
            }
            raw.append(contentsOf: [0xa5, 0xa5, 0xa5, 0xa5])
        }
        let original = raw
        let array = try XCTUnwrap(NativeAuxiliaryExporter.scalarArray(raw: raw, description:
            ["PixelFormat": kCVPixelFormatType_DepthFloat32, "Width": 2, "Height": 2, "BytesPerRow": 12]))
        XCTAssertEqual(raw, original)
        XCTAssertEqual(array.descriptor, "<f4")
        let recovered = array.bytes.withUnsafeBytes { data in (0..<4).map {
            UInt32(littleEndian: data.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))
        } }
        XCTAssertEqual(recovered, words)
        let npy = try array.npy()
        let headerSize = Int(npy[8]) + Int(npy[9]) * 256
        XCTAssertEqual((10 + headerSize) % 64, 0)
        XCTAssertEqual(npy.suffix(array.bytes.count), array.bytes)
        XCTAssertTrue(String(decoding: npy[10..<(10 + headerSize)], as: UTF8.self).contains("'shape': (2, 2)"))
    }

    func testHalfPrecisionAndIntegerCodesAreNeverConverted() throws {
        for (pixelFormat, descriptor, bytes) in [
            (kCVPixelFormatType_DisparityFloat16, "<f2", Data([0, 128, 1, 126, 0, 124, 255, 123])),
            (kCVPixelFormatType_OneComponent16, "<u2", Data([0, 0, 1, 0, 255, 255, 0, 128])),
            (kCVPixelFormatType_OneComponent8, "|u1", Data([0, 1, 255, 128]))
        ] {
            let array = try XCTUnwrap(NativeAuxiliaryExporter.scalarArray(raw: bytes, description:
                ["PixelFormat": pixelFormat, "Width": 2, "Height": 2, "BytesPerRow": bytes.count / 2]))
            XCTAssertEqual(array.bytes, bytes)
            XCTAssertEqual(array.descriptor, descriptor)
        }
    }

    func testUnknownFormatsRemainOpaqueAndMalformedKnownBuffersAreRejected() throws {
        XCTAssertNil(try NativeAuxiliaryExporter.scalarArray(raw: Data([1]), description: ["PixelFormat": 0]))
        for fields: [String: Any] in [
            ["Width": 2, "Height": 2, "BytesPerRow": 4],
            ["Width": 2, "Height": 2, "BytesPerRow": 8],
            ["Width": -1, "Height": 2, "BytesPerRow": 8],
            ["Width": 1.5, "Height": 2, "BytesPerRow": 8],
            ["Width": true, "Height": 2, "BytesPerRow": 8],
            ["Width": Int.max, "Height": Int.max, "BytesPerRow": Int.max]
        ] {
            var description = fields
            description["PixelFormat"] = kCVPixelFormatType_DepthFloat32
            XCTAssertThrowsError(try NativeAuxiliaryExporter.scalarArray(raw: Data(repeating: 0, count: 4), description: description))
        }
    }

    func testNativeExportIsTransactionalAndLeavesSourceUntouched() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let photo = root.appendingPathComponent("photo.png")
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        try CIContext().writePNGRepresentation(of: image, to: photo, format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let original = try Data(contentsOf: photo)
        let output = root.appendingPathComponent("auxiliary")
        let report = try NativeAuxiliaryExporter.exportSynchronously(sourceURL: photo, to: output)
        XCTAssertEqual(report.auxiliaryCount, 0)
        XCTAssertEqual(report.sourceSHA256, NativeAuxiliaryExporter.checksum(original))
        XCTAssertEqual(try Data(contentsOf: photo), original)
        let manifest = try Data(contentsOf: output.appendingPathComponent("auxiliary.json"))
        XCTAssertThrowsError(try NativeAuxiliaryExporter.exportSynchronously(sourceURL: photo, to: output))
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("auxiliary.json")), manifest)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix(".auxiliary-export-") })
    }

    func testInvalidSourceDoesNotPublishOrLeaveStagingArtifacts() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("invalid.heic")
        let original = Data("invalid image data".utf8)
        try original.write(to: source)
        let output = root.appendingPathComponent("auxiliary")
        XCTAssertThrowsError(try NativeAuxiliaryExporter.exportSynchronously(sourceURL: source, to: output))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        XCTAssertEqual(try Data(contentsOf: source), original)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["invalid.heic"])
    }

    func testCancelledStartDoesNotReadSourceOrPublishFolder() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("missing.heic")
        let output = root.appendingPathComponent("auxiliary")
        let gate = StartGate()
        let request = Task {
            await gate.wait()
            return try await NativeAuxiliaryExporter.export(sourceURL: source, to: output)
        }
        await gate.waitUntilRequested()
        request.cancel()
        await gate.release()
        do {
            _ = try await request.value
            XCTFail("Cancelled export must not read source or publish a result")
        } catch is CancellationError { }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), [])
    }

    private actor StartGate {
        var continuation: CheckedContinuation<Void, Never>?
        func wait() async { await withCheckedContinuation { continuation = $0 } }
        func waitUntilRequested() async { while continuation == nil { await Task.yield() } }
        func release() { continuation?.resume(); continuation = nil }
    }
}
