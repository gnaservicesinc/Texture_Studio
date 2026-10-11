import CoreGraphics
import Foundation
import XCTest
@testable import TextureStudio

final class NativeDepthProCacheAugmentationTests: XCTestCase {
    func testDatasetCacheReusesVerifiedLegacyEntryAndPublishesNewEntriesLocally() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let hash = try fixture.writePhoto(Array(0..<12))
        let input = try fixture.cache.input(imageURL: fixture.imageURL, expectedSHA256: hash, width: 4, height: 3)
        let legacyURL = fixture.entryURL(sourceSHA256: hash), bytes = try Data(contentsOf: legacyURL)
        let local = fixture.root.appendingPathComponent("dataset/.training-data/depthpro-inputs")
        var predictions = 0
        let cache = NativeDepthProInputCache(modelSHA256: fixture.cache.modelSHA256, directory: local,
            fallbackDirectories: [fixture.cache.directory]) { _ in
                predictions += 1
                return ModelDepthResult(width: 3, height: 2, values: Fixture.meters,
                    outputName: "depthMeters", interpretation: "Camera depth in meters")
            }
        XCTAssertEqual(try cache.input(imageURL: fixture.imageURL, expectedSHA256: hash, width: 4, height: 3).map(\.bitPattern), input.map(\.bitPattern))
        XCTAssertEqual(predictions, 0)
        XCTAssertEqual(cache.legacyReusedCount, 1)
        XCTAssertEqual(cache.reusedCount, 1)
        let localURL = local.appendingPathComponent(legacyURL.lastPathComponent)
        XCTAssertEqual(try Data(contentsOf: localURL), bytes)
        let legacyAttributes = try FileManager.default.attributesOfItem(atPath: legacyURL.path)
        let localAttributes = try FileManager.default.attributesOfItem(atPath: localURL.path)
        XCTAssertEqual(legacyAttributes[.systemFileNumber] as? NSNumber, localAttributes[.systemFileNumber] as? NSNumber,
                       "A lazily adopted entry shares an inode rather than duplicating its numeric payload")
        let changed = try fixture.writePhoto(Array((0..<12).reversed()))
        _ = try cache.input(imageURL: fixture.imageURL, expectedSHA256: changed, width: 4, height: 3)
        XCTAssertEqual(predictions, 1)
        XCTAssertEqual(cache.generatedCount, 1)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: local.path).count, 2)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: fixture.cache.directory.path).count, 1)
        XCTAssertEqual(try Data(contentsOf: legacyURL), bytes)
    }

    func testLegacyCacheCorruptionDoesNotPublishOrRegenerateInput() throws {
        let fixture = try Fixture(); defer { fixture.remove() }
        let hash = try fixture.writePhoto(Array(0..<12))
        _ = try fixture.cache.input(imageURL: fixture.imageURL, expectedSHA256: hash, width: 4, height: 3)
        let snapshot = try fixture.entry(sourceSHA256: hash)
        var tensors = try snapshot.nativeTensors()
        tensors["input"] = .floats([Float](repeating: 0.5, count: 12), shape: [1, 1, 3, 4])
        try NativeSafetensors.write(tensors: tensors, metadata: snapshot.metadata, to: fixture.entryURL(sourceSHA256: hash))
        let local = fixture.root.appendingPathComponent("local-cache")
        let cache = NativeDepthProInputCache(modelSHA256: fixture.cache.modelSHA256, directory: local,
            fallbackDirectories: [fixture.cache.directory]) { _ in
                XCTFail("A corrupt legacy cache must remain an actual error")
                return ModelDepthResult(width: 3, height: 2, values: Fixture.meters, outputName: "depthMeters", interpretation: "Meters")
            }
        XCTAssertThrowsError(try cache.input(imageURL: fixture.imageURL, expectedSHA256: hash, width: 4, height: 3))
        XCTAssertFalse(FileManager.default.fileExists(atPath: local.path))
        XCTAssertEqual(cache.legacyReusedCount, 0)
    }

    func testProgressivePairedFlipsPreserveBothNumericGridsWithoutPrediction() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let originalIDs = Array(0..<12)
        let originalHash = try fixture.writePhoto(originalIDs)
        let originalInput = try fixture.cache.input(imageURL: fixture.imageURL,
            expectedSHA256: originalHash, width: 4, height: 3)
        let originalMeters = Fixture.meters
        let steps: [(horizontal: Bool, vertical: Bool, photoIDs: [Int], meterIDs: [Int])] = [
            (true, false, [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8], [2, 1, 0, 5, 4, 3]),
            (false, true, [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [5, 4, 3, 2, 1, 0]),
            (true, false, [8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3], [3, 4, 5, 0, 1, 2]),
            (false, true, originalIDs, Array(0..<6))
        ]
        var previousHash = originalHash
        for step in steps {
            let nextHash = try fixture.writePhoto(step.photoIDs)
            XCTAssertTrue(try fixture.cache.remapFlips(fromSourceSHA256: previousHash, toSourceSHA256: nextHash,
                width: 4, height: 3, horizontal: step.horizontal, vertical: step.vertical))
            let input = try fixture.cache.input(imageURL: fixture.imageURL,
                expectedSHA256: nextHash, width: 4, height: 3)
            XCTAssertEqual(input.map(\.bitPattern), step.photoIDs.map { originalInput[$0].bitPattern })
            let entry = try fixture.entry(sourceSHA256: nextHash)
            XCTAssertEqual(entry.tensors["depth_meters"]?.shape, [1, 1, 2, 3],
                           "Native DepthPro grid is flipped independently of the paired 4 × 3 training grid")
            let meters = try NativeTensor(dtype: "F32", shape: [1, 1, 2, 3],
                bytes: entry.tensorBytes(named: "depth_meters")).floatValues()
            XCTAssertEqual(meters.map(\.bitPattern), step.meterIDs.map { originalMeters[$0].bitPattern })
            XCTAssertEqual(input.min()?.bitPattern, originalInput.min()?.bitPattern)
            XCTAssertEqual(input.max()?.bitPattern, originalInput.max()?.bitPattern)
            XCTAssertEqual(entry.metadata["source_sha256"], nextHash)
            XCTAssertEqual(entry.metadata["remapped_from_source_sha256"], previousHash)
            XCTAssertEqual(entry.metadata["preparation_contract"], NativeDepthProContract.identifier)
            XCTAssertEqual(entry.metadata["depthpro_model_sha256"], fixture.cache.modelSHA256)
            previousHash = nextHash
        }
        XCTAssertEqual(previousHash, originalHash)
        XCTAssertEqual(fixture.predictions, 1)
        XCTAssertEqual(fixture.cache.generatedCount, 1)
        XCTAssertEqual(fixture.cache.remappedCount, 4)
        XCTAssertEqual(fixture.cache.reusedCount, 4)
    }

    func testIdentityFlipPreservesCacheBytesAndUnmappedSourceRequiresPrediction() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let sourceHash = try fixture.writePhoto(Array(0..<12))
        _ = try fixture.cache.input(imageURL: fixture.imageURL, expectedSHA256: sourceHash, width: 4, height: 3)
        let original = try fixture.entry(sourceSHA256: sourceHash)
        XCTAssertTrue(try fixture.cache.remapFlips(fromSourceSHA256: sourceHash, toSourceSHA256: sourceHash,
            width: 4, height: 3, horizontal: false, vertical: false))
        XCTAssertEqual(try fixture.entry(sourceSHA256: sourceHash).bytes, original.bytes)
        XCTAssertEqual(fixture.cache.remappedCount, 0)

        let newSourceHash = try fixture.writePhoto([1, 5, 9, 0, 4, 8, 3, 7, 11, 2, 6, 10])
        XCTAssertFalse(try fixture.cache.remapFlips(fromSourceSHA256: String(repeating: "f", count: 64),
            toSourceSHA256: newSourceHash, width: 4, height: 3, horizontal: true, vertical: true))
        XCTAssertEqual(fixture.predictions, 1)
        _ = try fixture.cache.input(imageURL: fixture.imageURL, expectedSHA256: newSourceHash, width: 4, height: 3)
        XCTAssertEqual(fixture.predictions, 2, "A rotated/re-cropped source hash cannot reuse an untransformed prediction")
        XCTAssertEqual(fixture.cache.generatedCount, 2)
        XCTAssertEqual(fixture.cache.remappedCount, 0)
    }

    func testFlipUsesExactPermutationsAndRejectsInvalidGridOrCancellation() throws {
        let values = [Float(bitPattern: 0x8000_0000), Float(bitPattern: 1), 0.5, 1, 0.125, 0.75]
        XCTAssertEqual(try NativeDepthProContract.flipped(values, width: 3, height: 2,
            horizontal: true, vertical: true).map(\.bitPattern), values.reversed().map(\.bitPattern))
        XCTAssertEqual(try NativeDepthProContract.flipped(values, width: 3, height: 2,
            horizontal: false, vertical: false).map(\.bitPattern), values.map(\.bitPattern))
        XCTAssertThrowsError(try NativeDepthProContract.flipped(values, width: 2, height: 2,
            horizontal: true, vertical: false))
        XCTAssertThrowsError(try NativeDepthProContract.flipped(values, width: 0, height: 2,
            horizontal: false, vertical: true))
        XCTAssertThrowsError(try NativeDepthProContract.flipped(values, width: 3, height: 2,
            horizontal: true, vertical: false, checkCancellation: { throw CancellationError() })) {
                XCTAssertTrue($0 is CancellationError)
            }
    }

    func testRemapRejectsCorruptMetersAndCancellationWithoutPublishingDestination() throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let sourceHash = try fixture.writePhoto(Array(0..<12))
        _ = try fixture.cache.input(imageURL: fixture.imageURL, expectedSHA256: sourceHash, width: 4, height: 3)
        let newSourceHash = try fixture.writePhoto([3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8])
        XCTAssertThrowsError(try fixture.cache.remapFlips(fromSourceSHA256: sourceHash, toSourceSHA256: newSourceHash,
            width: 4, height: 3, horizontal: true, vertical: false, checkCancellation: { throw CancellationError() })) {
                XCTAssertTrue($0 is CancellationError)
            }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: fixture.cache.directory,
            includingPropertiesForKeys: nil).count, 1)

        let snapshot = try fixture.entry(sourceSHA256: sourceHash)
        var tensors = try snapshot.nativeTensors()
        tensors["depth_meters"] = .floats([2, 2.25, 4.5, 8.125, 16.25, 32.5], shape: [1, 1, 2, 3])
        try NativeSafetensors.write(tensors: tensors, metadata: snapshot.metadata,
            to: fixture.entryURL(sourceSHA256: sourceHash))
        XCTAssertThrowsError(try fixture.cache.remapFlips(fromSourceSHA256: sourceHash, toSourceSHA256: newSourceHash,
            width: 4, height: 3, horizontal: true, vertical: false)) {
                XCTAssertTrue($0 is NativeDepthProRuntimeError)
            }
        XCTAssertEqual(fixture.predictions, 1)
        XCTAssertEqual(fixture.cache.remappedCount, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.entryURL(sourceSHA256: newSourceHash).path))
    }

    private final class Fixture {
        static let meters: [Float] = [1.125, 2.25, 4.5, 8.125, 16.25, 32.5]
        let root: URL, imageURL: URL
        private(set) var predictions = 0
        lazy var cache = NativeDepthProInputCache(modelSHA256: String(repeating: "a", count: 64),
            directory: root.appendingPathComponent("cache")) { [unowned self] image in
                XCTAssertEqual(image.width, 4); XCTAssertEqual(image.height, 3)
                self.predictions += 1
                return ModelDepthResult(width: 3, height: 2, values: Self.meters,
                    outputName: "depthMeters", interpretation: "Camera depth in meters")
            }

        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("depthpro-cache-flips-" + UUID().uuidString)
            imageURL = root.appendingPathComponent("source.png")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        func remove() { try? FileManager.default.removeItem(at: root) }
        func writePhoto(_ pixelIDs: [Int]) throws -> String {
            let pixels = Data(pixelIDs.flatMap { [UInt8($0 * 7), UInt8($0 * 11), UInt8($0 * 13)] })
            let bytes = try NativePNG(header: .init(width: 4, height: 3, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: pixels, colorChunks: []).encoded()
            try bytes.write(to: imageURL)
            return NativeMaterialTrainer.checksum(bytes)
        }
        func entryURL(sourceSHA256: String) -> URL {
            let binding = [NativeDepthProContract.identifier, sourceSHA256, cache.modelSHA256, "4", "3"].joined(separator: "|")
            return cache.directory.appendingPathComponent(NativeMaterialTrainer.checksum(Data(binding.utf8)) + ".safetensors")
        }
        func entry(sourceSHA256: String) throws -> NativeSafetensors { try NativeSafetensors(contentsOf: entryURL(sourceSHA256: sourceSHA256)) }
    }
}
