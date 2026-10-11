import CoreGraphics
import CoreML
import Foundation
import XCTest
@testable import TextureStudio

final class NativeDepthProServiceTests: XCTestCase {
    func testOriginalWidthUsesSourceWidthAndPublishedFloat16Contract() throws {
        let value = try NativeDepthProService.originalWidth(1024, dataType: .float16)
        XCTAssertEqual(value.shape.map(\.intValue), [1, 1, 1, 1])
        XCTAssertEqual(value.dataType, .float16)
        XCTAssertEqual(value[0].intValue, 1024)
        XCTAssertEqual(try NativeDepthProService.originalWidth(4001, dataType: .float16)[0].intValue, 4000)
        XCTAssertEqual(try NativeDepthProService.originalWidth(4001, dataType: .float32)[0].intValue, 4001)
        XCTAssertThrowsError(try NativeDepthProService.originalWidth(0, dataType: .float16))
        XCTAssertThrowsError(try NativeDepthProService.originalWidth(100_000, dataType: .float16))
        XCTAssertThrowsError(try NativeDepthProService.originalWidth(1024, dataType: .int32))
    }

    func testPackageIdentityIncludesWeightsAndSurvivesRelocation() throws {
        let root = try temporaryDirectory()
        let first = root.appendingPathComponent("First.mlpackage"), second = root.appendingPathComponent("Second.mlpackage")
        let weights = first.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        try FileManager.default.createDirectory(at: weights.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("specification".utf8).write(to: first.appendingPathComponent("model.mlmodel"))
        try Data("weights A".utf8).write(to: weights)
        let identity = try NativeDepthProService.modelIdentity(at: first)
        try FileManager.default.copyItem(at: first, to: second)
        XCTAssertEqual(try NativeDepthProService.modelIdentity(at: second), identity)
        try Data("weights B".utf8).write(to: weights)
        XCTAssertNotEqual(try NativeDepthProService.modelIdentity(at: first), identity)
        let original = second.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin")
        try FileManager.default.moveItem(at: original, to: original.deletingLastPathComponent().appendingPathComponent("different.bin"))
        XCTAssertNotEqual(try NativeDepthProService.modelIdentity(at: second), identity)
    }

    func testDedicatedAdapterSuppliesWidthAndPreservesAsymmetricSourceOrientation() throws {
        let directory = try temporaryDirectory(), modelURL = directory.appendingPathComponent("Fixture.mlmodel")
        try Self.fixture().write(to: modelURL)
        let service = try NativeDepthProService(modelURL: modelURL)
        // Top-left, top-right, bottom-left, bottom-right have different brightness.
        let bytes = Data([30, 30, 30, 255, 90, 90, 90, 255, 150, 150, 150, 255, 210, 210, 210, 255])
        let provider = try XCTUnwrap(CGDataProvider(data: bytes as CFData))
        let image = try XCTUnwrap(CGImage(width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32,
            bytesPerRow: 8, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: provider,
            decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        let result = try service.predict(image: image), size = NativeDepthProService.inputSize
        XCTAssertEqual(result.width, size); XCTAssertEqual(result.height, size)
        // Fixture returns RGB mean plus originalWidth (2, not the 1536 model input).
        XCTAssertEqual(result.values[128 * size + 128], 32, accuracy: 1)
        XCTAssertEqual(result.values[128 * size + size - 129], 92, accuracy: 1)
        XCTAssertEqual(result.values[(size - 129) * size + 128], 152, accuracy: 1)
        XCTAssertEqual(result.values[(size - 129) * size + size - 129], 212, accuracy: 1)
        let repeated = try service.predict(image: image)
        XCTAssertEqual(result.values, repeated.values)
        XCTAssertEqual(result.provenance?.checkpointSHA256, service.modelSHA256)
        XCTAssertEqual(result.provenance?.rowOrder, "top-to-bottom")
        XCTAssertEqual(result.provenance?.modelID, "developer-depthpro-coreml")
        XCTAssertEqual(result.provenance?.upstreamRevision, "", "A matching interface does not establish upstream weight provenance.")
    }

    func testDedicatedAdapterRejectsWrongInterface() throws {
        let directory = try temporaryDirectory(), modelURL = directory.appendingPathComponent("Wrong.mlmodel")
        try Self.fixture(widthName: "width").write(to: modelURL)
        XCTAssertThrowsError(try NativeDepthProService(modelURL: modelURL)) { error in
            XCTAssertTrue(error.localizedDescription.contains("originalWidth"))
        }
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("NativeDepthProTests-\(UUID())")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    /// Tiny native Core ML fixture: mean RGB channels then add the width scalar.
    /// Encode Apple's protobuf format directly; no Python runtime/model weights.
    private static func fixture(widthName: String = "originalWidth") -> Data {
        let imageType = Wire.message(4, Wire.number(1, 1536) + Wire.number(2, 1536) + Wire.number(3, 20))
        func array(_ shape: [UInt64]) -> Data {
            Wire.message(5, Wire.message(1, shape.reduce(into: Data()) { $0.append(Wire.varint($1)) }) + Wire.number(2, 65568))
        }
        func feature(_ name: String, _ type: Data) -> Data { Wire.string(1, name) + Wire.message(3, type) }
        let description = Wire.message(1, feature("image", imageType)) + Wire.message(1, feature(widthName, array([1, 1, 1, 1])))
            + Wire.message(10, feature("depthMeters", array([1, 1, 1536, 1536])))
        let mean = Wire.string(1, "mean") + Wire.string(2, "image") + Wire.string(3, "gray")
            + Wire.message(1280, Wire.number(1, 1) + Wire.number(2, 1))
        let add = Wire.string(1, "width") + Wire.string(2, "gray") + Wire.string(2, widthName)
            + Wire.string(3, "depthMeters") + Wire.message(880, Data())
        let network = Wire.message(1, mean) + Wire.message(1, add) + Wire.number(5, 1) + Wire.number(6, 1)
        return Wire.number(1, 4) + Wire.message(2, description) + Wire.message(500, network)
    }
}

private enum Wire {
    static func varint(_ value: UInt64) -> Data {
        var value = value, bytes = Data()
        while value > 127 { bytes.append(UInt8(value & 127) | 128); value >>= 7 }
        bytes.append(UInt8(value))
        return bytes
    }
    static func number(_ field: UInt64, _ value: UInt64) -> Data { varint(field << 3) + varint(value) }
    static func message(_ field: UInt64, _ value: Data) -> Data { varint(field << 3 | 2) + varint(UInt64(value.count)) + value }
    static func string(_ field: UInt64, _ value: String) -> Data { message(field, Data(value.utf8)) }
}
