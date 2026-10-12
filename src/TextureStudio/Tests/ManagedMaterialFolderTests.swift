import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import TextureStudio

@MainActor
final class ManagedMaterialFolderTests: XCTestCase {
    func testManagedFoldersAccountForUnpairedAncillaryUnknownAndUnreadableImages() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), folder = root.appendingPathComponent("Pictures")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Managed photographs"])
        try png(channels: 3, bits: 8).write(to: folder.appendingPathComponent("BaseColor.png"))
        try png(channels: 1, bits: 8).write(to: folder.appendingPathComponent("AO.png"))
        try jpeg().write(to: folder.appendingPathComponent("holiday.jpg"))
        try Data([1, 2, 3]).write(to: folder.appendingPathComponent("broken.png"))
        let added = try await managed(dataset, folder: folder)
        XCTAssertEqual(added["added_material_count"] as? Int, 0)
        XCTAssertEqual(added["added_file_count"] as? Int, 4)
        XCTAssertEqual(added["training_size"] as? Int, 2048)
        let items = try XCTUnwrap(added["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(items.count, 4)
        XCTAssertEqual(items.first { $0["filename"] as? String == "BaseColor.png" }?["role"] as? String, "input")
        XCTAssertEqual(items.first { $0["filename"] as? String == "AO.png" }?["role"] as? String, "ancillary")
        let photo = try XCTUnwrap(items.first { $0["filename"] as? String == "holiday.jpg" })
        XCTAssertEqual(photo["role"] as? String, "unknown")
        XCTAssertEqual(photo["width"] as? Int, 256)
        XCTAssertEqual(photo["state"] as? String, "needs_assignment")
        XCTAssertEqual(items.first { $0["filename"] as? String == "broken.png" }?["state"] as? String, "header_error")
        XCTAssertEqual(try object(dataset.appendingPathComponent("dataset.json"))["source_folders"] as? [String], [folder.path])
        let before = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual((reopened["source_inventory"] as? [Any])?.count, 4)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), before)
        let repeated = try await managed(dataset, folder: folder)
        XCTAssertEqual(repeated["added_file_count"] as? Int, 0)
    }

    func testStandaloneAndNamedRolesPairTogetherWithoutHidingCompleteSiblings() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        for (name, input, target) in [("wood", "BaseColor.png", "wood_height_2k.png"),
                                      ("brick", "brick_baseColor_2k.png", "Height.png")] {
            let folder = sources.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try png(channels: 3, bits: 8).write(to: folder.appendingPathComponent(input))
            try png(channels: 1, bits: 16).write(to: folder.appendingPathComponent(target))
        }
        let mixed = sources.appendingPathComponent("mixed")
        try FileManager.default.createDirectory(at: mixed, withIntermediateDirectories: false)
        for (name, channels, bits) in [("BaseColor.png", 3, 8), ("Height.png", 1, 16),
                                      ("cloth_baseColor_2k.png", 3, 8), ("cloth_height_2k.png", 1, 16)] {
            try png(channels: channels, bits: bits).write(to: mixed.appendingPathComponent(name))
        }
        let imported = try await managed(dataset, folder: sources)
        XCTAssertEqual(imported["added_material_count"] as? Int, 4)
        XCTAssertEqual((imported["source_inventory"] as? [Any])?.count, 8)
        XCTAssertEqual(imported["warnings"] as? [String], [])
        let originals = try originalSamples(imported)
        XCTAssertEqual(originals.count, 4)
        XCTAssertTrue(originals.allSatisfy { Set(($0["maps"] as? [String: Any] ?? [:]).keys) == ["input", "height"] })
    }

    func testManagedRegistrationPreservesOriginalsAndAlternateColorPathsWithoutPrematureChecksums() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset, size: 2048)
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let inputA = sources.appendingPathComponent("cloth_col_1_1k.png"), inputB = sources.appendingPathComponent("cloth_col_2_1k.png")
        try png(channels: 3, bits: 8, code: 19).write(to: inputA)
        try png(channels: 3, bits: 8, code: 93).write(to: inputB)
        try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent("cloth_disp_1k.png"))
        let imported = try await managed(dataset, folder: sources)
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        XCTAssertEqual((imported["materials"] as? [Any])?.count, 0, "Undersized sources are not training crops")
        let sample = try XCTUnwrap(try originalSamples(imported).first)
        XCTAssertEqual(sample["split"] as? String, "original")
        XCTAssertEqual(sample["source_region_id"] as? String, "original")
        let variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
        XCTAssertEqual(Set(variants.compactMap { $0["path"] as? String }), [inputA.path, inputB.path])
        let record = try durableSamples(dataset).first!
        XCTAssertEqual(record["source_verification"] as? String, "header_inspected")
        XCTAssertEqual(record["source_precision_verified"] as? Bool, false)
        for detail in (record["map_metadata"] as? [String: [String: Any]] ?? [:]).values {
            XCTAssertNil((detail["source"] as? [String: Any])?["file_sha256"])
        }
        XCTAssertThrowsError(try NativeMaterialDatasetService.trainingSamples(datasetURL: dataset, size: 256, target: "height"))
        let bytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        try FileManager.default.removeItem(at: inputB)
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), bytes)
        XCTAssertEqual(try originalSamples(reopened).count, 1)
        XCTAssertEqual((reopened["source_inventory"] as? [[String: Any]])?.first { $0["path"] as? String == inputB.path }?["state"] as? String, "missing")
    }

    func testPrepareVerifiesSelectedMapsAndKeepsValidSiblingsWhenAnotherSourceIsCorrupt() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        for family in ["good", "bad"] {
            var input = try png(channels: 3, bits: 8)
            if family == "bad" { input[45] ^= 1 }
            try input.write(to: sources.appendingPathComponent(family + "_diff_1k.png"))
            try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent(family + "_disp_1k.png"))
            try png(channels: 1, bits: 8).write(to: sources.appendingPathComponent(family + "_rough_1k.png"))
        }
        _ = try await managed(dataset, folder: sources)
        let events = ManagedVerificationEvents()
        let text = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path,
            "--size", "256", "--target", "height", "--skip-source-recovery"], onEvent: { events.record($0) })
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy { $0.id.hasPrefix("good_") })
        let records = try durableSamples(dataset)
        let good = try XCTUnwrap(records.first { ($0["material_id"] as? String)?.hasPrefix("good_") == true })
        let maps = try XCTUnwrap(good["map_metadata"] as? [String: [String: Any]])
        XCTAssertEqual(((maps["input"]?["source"] as? [String: Any])?["file_sha256"] as? String)?.count, 64)
        XCTAssertEqual(((maps["height"]?["source"] as? [String: Any])?["file_sha256"] as? String)?.count, 64)
        XCTAssertNil((maps["roughness"]?["source"] as? [String: Any])?["file_sha256"], "Unselected targets stay deferred")
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(try originalSamples(reopened).count, 2)
        let failure = (reopened["source_inventory"] as? [[String: Any]])?.first { $0["filename"] as? String == "bad_diff_1k.png" }
        XCTAssertEqual(failure?["state"] as? String, "header_error")
        XCTAssertFalse((failure?["issue"] as? String ?? "").isEmpty)
        XCTAssertTrue(events.text.contains("source_verification_progress"))
        XCTAssertTrue(events.text.contains("source_verification_completed"))
    }

    func testManagedReaddingVerifiedFolderPreservesItsProofAndReviews() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        try png(channels: 3, bits: 8).write(to: sources.appendingPathComponent("wood_diff_1k.png"))
        try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent("wood_disp_1k.png"))
        _ = try await command(["import-folder", "--dataset", dataset.path, "--folder", sources.path])
        let sample = try XCTUnwrap(try durableSamples(dataset).first)
        let id = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await command(["curate", "--dataset", dataset.path, "--sample", id, "--status", "approved",
                               "--original-source", "true", "--note", "Original approved"])
        let record = try durableSamples(dataset).first!
        let result = try await managed(dataset, folder: sources)
        XCTAssertEqual(result["added_material_count"] as? Int, 0)
        XCTAssertEqual(result["duplicate_material_count"] as? Int, 1)
        XCTAssertTrue(NSDictionary(dictionary: record).isEqual(to: try durableSamples(dataset).first!))
        let original = try XCTUnwrap(try originalSamples(result).first)
        XCTAssertEqual(original["status"] as? String, "approved")
        XCTAssertEqual(original["note"] as? String, "Original approved")
    }

    func testCancellationDuringDeferredVerificationLeavesRegistrationAndSourcesIntact() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await create(dataset)
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = try png(channels: 3, bits: 8), height = try png(channels: 1, bits: 16)
        let inputURL = sources.appendingPathComponent("wood_diff_1k.png"), heightURL = sources.appendingPathComponent("wood_disp_1k.png")
        try input.write(to: inputURL); try height.write(to: heightURL)
        _ = try await managed(dataset, folder: sources)
        let indexBefore = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let recordsBefore = try durableSamples(dataset)
        let cancellation = ManagedVerificationCancellation()
        let operation = Task {
            try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path,
                "--size", "256", "--target", "height", "--skip-source-recovery"], onEvent: { line in
                    if line.contains("source_verification_started") { cancellation.cancel() }
                })
        }
        cancellation.install(operation)
        do { _ = try await operation.value; XCTFail("Cancelling deferred verification must stop preparation") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBefore)
        XCTAssertTrue(NSArray(array: recordsBefore).isEqual(to: try durableSamples(dataset)))
        XCTAssertEqual(try Data(contentsOf: inputURL), input)
        XCTAssertEqual(try Data(contentsOf: heightURL), height)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent(".training-data").path))
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("managed-material-folders-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func create(_ dataset: URL, size: Int = 256) async throws -> [String: Any] {
        try await command(["create-dataset", "--dataset", dataset.path, "--name", "Managed source test", "--training-size", String(size)])
    }
    private func managed(_ dataset: URL, folder: URL) async throws -> [String: Any] {
        try await command(["import-folder", "--dataset", dataset.path, "--folder", folder.path, "--managed-import", "true"])
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let output = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(output).utf8)) as? [String: Any])
    }
    private func png(channels: Int, bits: Int, code: UInt8 = 71) throws -> Data {
        try NativePNG(header: .init(width: 256, height: 256, bits: bits, channels: channels,
                                    color: channels == 1 ? 0 : 2, interlace: 0),
                      pixels: Data(repeating: code, count: 256 * 256 * channels * bits / 8), colorChunks: []).encoded()
    }
    private func jpeg() throws -> Data {
        let source = try XCTUnwrap(CGImageSourceCreateWithData(try png(channels: 3, bits: 8) as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let bytes = NSMutableData()
        let destination = try XCTUnwrap(CGImageDestinationCreateWithData(bytes, UTType.jpeg.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return bytes as Data
    }
    private func object(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }
    private func durableSamples(_ dataset: URL) throws -> [[String: Any]] {
        try (object(dataset.appendingPathComponent("dataset.json"))["samples"] as? [[String: Any]] ?? []).map {
            try object(dataset.appendingPathComponent(try XCTUnwrap($0["path"] as? String)).appendingPathComponent("sample.json"))
        }
    }
    private func originalSamples(_ info: [String: Any]) throws -> [[String: Any]] {
        try XCTUnwrap(info["original_materials"] as? [[String: Any]]).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
    }
}

private final class ManagedVerificationEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var stored = ""
    func record(_ line: String) { lock.lock(); defer { lock.unlock() }; stored += line }
    var text: String { lock.lock(); defer { lock.unlock() }; return stored }
}

private final class ManagedVerificationCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var operation: Task<String?, Error>?
    func install(_ operation: Task<String?, Error>) { lock.lock(); defer { lock.unlock() }; self.operation = operation }
    func cancel() { lock.lock(); defer { lock.unlock() }; operation?.cancel() }
}
