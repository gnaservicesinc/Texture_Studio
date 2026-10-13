import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest
@testable import TextureStudio

@MainActor
final class ManagedMaterialFolderTests: XCTestCase {
    func testSourceFoldersAccountForUnpairedAncillaryUnknownAndUnreadableImages() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), folder = dataset
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--name", "Source photographs", "--source-folder", "true"])
        try png(channels: 3, bits: 8).write(to: folder.appendingPathComponent("BaseColor.png"))
        try png(channels: 1, bits: 8).write(to: folder.appendingPathComponent("AO.png"))
        try jpeg().write(to: folder.appendingPathComponent("holiday.jpg"))
        try Data([1, 2, 3]).write(to: folder.appendingPathComponent("broken.png"))
        let added = try await inspect(dataset)
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
        XCTAssertEqual(added["source_folders"] as? [String], [folder.path])
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual((reopened["source_inventory"] as? [Any])?.count, 4)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent("dataset.json").path))
        let repeated = try await inspect(dataset)
        XCTAssertEqual((repeated["source_inventory"] as? [Any])?.count, 4)
    }

    func testStandaloneAndNamedRolesPairTogetherWithoutHidingCompleteSiblings() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
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
        let imported = try await inspect(dataset)
        XCTAssertEqual(imported["source_set_count"] as? Int, 4)
        XCTAssertEqual((imported["source_inventory"] as? [Any])?.count, 8)
        XCTAssertEqual(imported["warnings"] as? [String], [])
        let originals = try originalSamples(imported)
        XCTAssertEqual(originals.count, 4)
        XCTAssertTrue(originals.allSatisfy { Set(($0["maps"] as? [String: Any] ?? [:]).keys) == ["input", "height"] })
    }

    func testFolderReadsPreserveAlternateColorsWithoutPrematureChecksumsAndReflectDeletion() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset, size: 2048)
        let inputA = sources.appendingPathComponent("cloth_col_1_1k.png"), inputB = sources.appendingPathComponent("cloth_col_2_1k.png")
        try png(channels: 3, bits: 8, code: 19).write(to: inputA)
        try png(channels: 3, bits: 8, code: 93).write(to: inputB)
        try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent("cloth_disp_1k.png"))
        let imported = try await inspect(dataset)
        XCTAssertEqual(imported["source_set_count"] as? Int, 1)
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
        try FileManager.default.removeItem(at: inputB)
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(try originalSamples(reopened).count, 1)
        XCTAssertFalse((reopened["source_inventory"] as? [[String: Any]] ?? []).contains { $0["path"] as? String == inputB.path })
        XCTAssertEqual((try originalSamples(reopened).first?["input_variants"] as? [Any])?.count, 1, "Deleted physical originals disappear from membership")
    }

    func testPrepareVerifiesSelectedMapsAndKeepsValidSiblingsWhenAnotherSourceIsCorrupt() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        for family in ["good", "bad"] {
            var input = try png(channels: 3, bits: 8)
            if family == "bad" { input[45] ^= 1 }
            try input.write(to: sources.appendingPathComponent(family + "_diff_1k.png"))
            try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent(family + "_disp_1k.png"))
            try png(channels: 1, bits: 8).write(to: sources.appendingPathComponent(family + "_rough_1k.png"))
        }
        _ = try await inspect(dataset)
        let events = ManagedVerificationEvents()
        let text = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path,
            "--size", "256", "--target", "height", "--skip-source-recovery"], onEvent: { events.record($0) })
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(text).utf8)) as? [String: Any])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(samples.isEmpty)
        XCTAssertTrue(samples.allSatisfy { $0.id.hasPrefix("good_") })
        XCTAssertEqual(NativeSourceFileTags.read(sources.appendingPathComponent("good_diff_1k.png"))?.verifiedSHA256?.count, 64)
        XCTAssertEqual(NativeSourceFileTags.read(sources.appendingPathComponent("good_disp_1k.png"))?.verifiedSHA256?.count, 64)
        XCTAssertNil(NativeSourceFileTags.read(sources.appendingPathComponent("good_rough_1k.png"))?.verifiedSHA256, "Unselected targets stay deferred")
        let reopened = try await command(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(try originalSamples(reopened).count, 2)
        let failure = (reopened["source_inventory"] as? [[String: Any]])?.first { $0["filename"] as? String == "bad_diff_1k.png" }
        XCTAssertEqual(failure?["state"] as? String, "header_error")
        XCTAssertFalse((failure?["issue"] as? String ?? "").isEmpty)
        XCTAssertTrue(events.text.contains("source_verification_progress"))
        XCTAssertTrue(events.text.contains("source_verification_completed"))
    }

    func testReopeningVerifiedFolderPreservesItsProofAndReviews() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        try png(channels: 3, bits: 8).write(to: sources.appendingPathComponent("wood_diff_1k.png"))
        try png(channels: 1, bits: 16).write(to: sources.appendingPathComponent("wood_disp_1k.png"))
        _ = try await command(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let sample = try XCTUnwrap(try durableSamples(dataset).first)
        let id = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await command(["curate", "--dataset", dataset.path, "--sample", id, "--status", "approved",
                               "--original-source", "true", "--note", "Original approved"])
        let record = try durableSamples(dataset).first!
        let result = try await inspect(dataset)
        XCTAssertEqual(result["source_set_count"] as? Int, 1, "Repeated folder reads never add duplicate membership")
        XCTAssertTrue(NSDictionary(dictionary: record).isEqual(to: try durableSamples(dataset).first!))
        let original = try XCTUnwrap(try originalSamples(result).first)
        XCTAssertEqual(original["status"] as? String, "approved")
        XCTAssertEqual(original["note"] as? String, "Original approved")
    }

    func testCancellationDuringDeferredVerificationLeavesSourceMembershipAndBytesIntact() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = dataset
        _ = try await create(dataset)
        let input = try png(channels: 3, bits: 8), height = try png(channels: 1, bits: 16)
        let inputURL = sources.appendingPathComponent("wood_diff_1k.png"), heightURL = sources.appendingPathComponent("wood_disp_1k.png")
        try input.write(to: inputURL); try height.write(to: heightURL)
        _ = try await inspect(dataset)
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
        XCTAssertTrue(NSArray(array: recordsBefore).isEqual(to: try durableSamples(dataset)))
        XCTAssertEqual(try Data(contentsOf: inputURL), input)
        XCTAssertEqual(try Data(contentsOf: heightURL), height)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent("dataset.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent(".training-data").path))
    }

    func testAddingSourceCopiesPhysicalOriginalsAndTeachingThenReaddingInsideFolderDoesNotDuplicate() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), external = root.appendingPathComponent("Incoming Wood")
        _ = try await create(dataset)
        try FileManager.default.createDirectory(at: external, withIntermediateDirectories: false)
        let input = external.appendingPathComponent("photo.png"), height = external.appendingPathComponent("height.png")
        let inputBytes = try png(channels: 3, bits: 8), heightBytes = try png(channels: 1, bits: 16)
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        try NativeSourceFileTags.update(input, role: "input", classLabel: "Wood", classifierIdentity: "user")
        let added = try await command(["add-source", "--dataset", dataset.path, "--source", external.path])
        let copied = dataset.appendingPathComponent(external.lastPathComponent), copiedInput = copied.appendingPathComponent(input.lastPathComponent)
        XCTAssertEqual(try Data(contentsOf: copiedInput), inputBytes)
        XCTAssertEqual(try Data(contentsOf: copied.appendingPathComponent(height.lastPathComponent)), heightBytes)
        XCTAssertEqual(NativeSourceFileTags.read(copiedInput)?.role, "input")
        XCTAssertEqual(NativeSourceFileTags.read(copiedInput)?.classLabel, "Wood")
        XCTAssertEqual(added["source_set_count"] as? Int, 1)
        XCTAssertTrue((added["source_inventory"] as? [[String: Any]] ?? []).allSatisfy { ($0["path"] as? String)?.hasPrefix(dataset.path + "/") == true })
        let again = try await command(["add-source", "--dataset", dataset.path, "--source", copied.path])
        XCTAssertEqual(again["source_set_count"] as? Int, 1)
        XCTAssertEqual((again["source_inventory"] as? [Any])?.count, 2)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes, "The incoming original remains unchanged after copying")
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("managed-material-folders-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func create(_ dataset: URL, size: Int = 256) async throws -> [String: Any] {
        _ = try await command(["create-dataset", "--dataset", dataset.path, "--source-folder", "true"])
        return try await command(["edit-dataset", "--dataset", dataset.path,
            "--training-size", String(size), "--subject-validation", "disabled"])
    }
    private func inspect(_ dataset: URL) async throws -> [String: Any] {
        try await command(["dataset", "--dataset", dataset.path])
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
    private func durableSamples(_ dataset: URL) throws -> [[String: Any]] {
        try NativeMaterialDatasetService.sourceFolderRecords(dataset).map(\.sample)
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
    private var requested = false
    func install(_ operation: Task<String?, Error>) { lock.lock(); defer { lock.unlock() }; self.operation = operation; if requested { operation.cancel() } }
    func cancel() { lock.lock(); defer { lock.unlock() }; requested = true; operation?.cancel() }
}
