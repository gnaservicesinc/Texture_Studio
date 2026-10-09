import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialDatasetTests: XCTestCase {
    func testPNGPreservesNativeChannelsAndCodesThroughExactCrop() throws {
        for bits in [8, 16] { for (color, channels) in [(UInt8(0), 1), (UInt8(2), 3), (UInt8(4), 2), (UInt8(6), 4)] {
            let pixels = Data((0..<3 * 2 * channels * bits / 8).map { UInt8(truncatingIfNeeded: $0 * 97) })
            let source = NativePNG(header: .init(width: 3, height: 2, bits: bits, channels: channels, color: color, interlace: 0), pixels: pixels, colorChunks: [("gAMA", Data([0, 0, 177, 143]))])
            let decoded = try NativePNG.decode(source.encoded())
            XCTAssertEqual(decoded.header, source.header); XCTAssertEqual(decoded.pixels, pixels)
            XCTAssertEqual(decoded.colorChunks.first?.1, source.colorChunks.first?.1)
            let selected = try decoded.crop([1, 1, 2, 1])
            XCTAssertEqual(selected.pixels, pixels.subdata(in: 4 * source.header.bytesPerPixel..<6 * source.header.bytesPerPixel))
            XCTAssertEqual(try NativePNG.decode(selected.encoded()).pixels, selected.pixels)
        } }
    }
    func testPNGNormalConventionChangesOnlyGreenIntegerCodes() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 16, channels: 4, color: 6, interlace: 0), pixels: Data([0, 1, 0, 0, 255, 255, 0, 19, 1, 0, 128, 1, 0, 0, 2, 0]), colorChunks: [])
        let result = try source.crop([0, 0, 2, 1], flipGreen: true)
        XCTAssertEqual(result.pixels, Data([0, 1, 255, 255, 255, 255, 0, 19, 1, 0, 127, 254, 0, 0, 2, 0]))
        XCTAssertEqual(try result.crop([0, 0, 2, 1], flipGreen: true).pixels, source.pixels)
        XCTAssertThrowsError(try source.crop([-1, 0, 2, 1]))
    }
    func testIndependentPNGFixturesDecodeAllFiltersAndAdam7WithoutChangingCodes() throws {
        // Independent zlib fixtures cover filter bytes 0...4 and all seven
        // Adam7 passes. Values are deliberately asymmetric and span both bytes.
        let filtered = try XCTUnwrap(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAMAAAAFEAIAAABfgx22AAAARElEQVR4nGNgSDyk3PLUbTlH5intnrd+6wUKGS8ZT/katt0NCTBdwgDMKldZzZ6J8/Tw9AARD4hmgchIXJIAoktuIBoASx8sjjtVTA0AAAAASUVORK5CYII="))
        let adam7 = try XCTUnwrap(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAEEAAAAAFEz0ZJAAAAO0lEQVR4nAEwAM//AABhAAhpAITlAJT1GHmc/QDCI0anAFa32jsAyiuM7U6vEHHSMwBevyCB4kOkBWbHt1wTjZvigdQAAAAASUVORK5CYII="))
        XCTAssertEqual(try NativePNG.decode(filtered).pixels, Data((0..<90).map { UInt8(truncatingIfNeeded: $0 * 97) }))
        XCTAssertEqual(try NativePNG.decode(adam7).pixels, Data((0..<40).map { UInt8(truncatingIfNeeded: $0 * 97) }))
    }
    func testNativePNGRejectsAmbiguousHeadersAndUnsupportedTransparency() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 0, count: 6), colorChunks: [("tRNS", Data(repeating: 0, count: 6))])
        XCTAssertThrowsError(try NativePNG.decode(source.encoded()))
        let plain = NativePNG(header: source.header, pixels: source.pixels, colorChunks: [])
        var duplicated = try plain.encoded(); duplicated.insert(contentsOf: duplicated.subdata(in: 8..<33), at: 33)
        XCTAssertThrowsError(try NativePNG.decode(duplicated))
    }
    func testPNGRejectsChecksumChangesAndIncompleteStreams() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 1, color: 0, interlace: 0), pixels: Data([0, 255]), colorChunks: [])
        var bytes = try source.encoded(); bytes[29] ^= 1
        XCTAssertThrowsError(try NativePNG.decode(bytes))
        XCTAssertThrowsError(try NativePNG.decode(source.encoded().dropLast()))
    }
    func testNativeCreateEditConflictAndOriginalPreservation() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Native Dataset")
        let created = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native Dataset", "--training-size", "256"])
        XCTAssertEqual(created["material_count"] as? Int, 0)
        let hash = try XCTUnwrap(created["index_sha256"] as? String)
        let edited = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Renamed", "--description", "Exact originals", "--expected-index-sha256", hash])
        XCTAssertEqual(edited["name"] as? String, "Renamed")
        do { _ = try await output(["edit-dataset", "--dataset", dataset.path, "--validation-subject", "missing", "--subject-validation", "automatic"]); XCTFail("A stale subject cannot be reset") }
        catch { XCTAssertTrue(error.localizedDescription.contains("no longer in this dataset")) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent(".material-workbench-journal.json").path))
        do { _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Stale", "--expected-index-sha256", hash]); XCTFail("A stale index must not edit metadata") }
        catch { XCTAssertTrue(error.localizedDescription.contains("changed since selection")) }
        let bytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let delete = try await output(["validate-delete", "--dataset", dataset.path, "--expected-index-sha256", try XCTUnwrap(edited["index_sha256"] as? String)])
        XCTAssertEqual(delete["safe_to_trash_folder"] as? Bool, true)
        try Data([1, 2, 3]).write(to: dataset.appendingPathComponent("original.png"))
        let conservative = try await output(["validate-delete", "--dataset", dataset.path])
        XCTAssertEqual(conservative["safe_to_trash_folder"] as? Bool, false)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), bytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("original.png")), Data([1, 2, 3]))
    }
    func testNativeReadReconstructsSharedCropPlansAndRejectsPathEscape() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Dataset", "--training-size", "256", "--validation-percent", "25"])
        var index = try object(dataset.appendingPathComponent("dataset.json"))
        var entries: [[String: Any]] = [], sourceBytes: [URL: Data] = [:]
        for number in 0..<4 {
            let material = "material_\(number)", id = material + "_full"
            let folder = root.appendingPathComponent("source_\(number)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            var maps: [String: String] = [:], metadata: [String: Any] = [:]
            for (role, bits) in [("input", 8), ("height", 16)] {
                let file = folder.appendingPathComponent(role + ".png")
                let png = NativePNG(header: .init(width: 512, height: 512, bits: bits, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: UInt8(number * 31), count: 512 * 512 * bits / 8), colorChunks: [])
                let data = try png.encoded(); try data.write(to: file); sourceBytes[file] = data
                let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                maps[role] = file.path
                metadata[role] = ["storage": "source_reference", "sample_sha256": hash, "sample_bits": bits, "source": ["path": file.path, "width": 512, "height": 512, "sample_bits": bits, "file_sha256": hash]]
            }
            let entry: [String: Any] = ["sample_id": id, "material_id": material, "status": "approved", "split": "train", "path": "samples/" + id]
            entries.append(entry)
            let sample: [String: Any] = ["sample_id": id, "material_id": material, "status": "approved", "split": "train", "source_directory": folder.path, "source_pixel_dimensions": [512, 512], "sample_pixel_dimensions": [512, 512], "maps": maps, "map_metadata": metadata]
            let metadataFolder = dataset.appendingPathComponent("samples/" + id)
            try FileManager.default.createDirectory(at: metadataFolder, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: sample).write(to: metadataFolder.appendingPathComponent("sample.json"))
        }
        index["samples"] = entries
        try JSONSerialization.data(withJSONObject: index).write(to: dataset.appendingPathComponent("dataset.json"))
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--target", "height"])
        XCTAssertEqual(inspected["material_count"] as? Int, 4); XCTAssertEqual(inspected["sample_count"] as? Int, 5)
        let plans = try XCTUnwrap(inspected["training_plans"] as? [String: [String: Any]])
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 4)
        XCTAssertEqual(plans["height"]?["validation_count"] as? Int, 1)
        XCTAssertEqual(plans["roughness"]?["unavailable_target_count"] as? Int, 5)
        for (file, bytes) in sourceBytes { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        entries[0]["path"] = "../escaped"; index["samples"] = entries
        try JSONSerialization.data(withJSONObject: index).write(to: dataset.appendingPathComponent("dataset.json"))
        do { _ = try await output(["dataset", "--dataset", dataset.path]); XCTFail("Metadata path escapes must fail") }
        catch { XCTAssertTrue(error.localizedDescription.contains("inside its directory")) }
    }
    func testCancellationBeforeNativeCommandDoesNotCreateDataset() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Cancelled")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", dataset.path, "--name", "Cancelled"])
        }
        do { _ = try await task.value; XCTFail("Cancelled command should fail") } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.path))
    }
    func testCurrentSourceInventoryRefreshIsStableAcrossRepeatedReads() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native", "--training-size", "256"])
        let folder = dataset.appendingPathComponent("sources/Brick")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let input = folder.appendingPathComponent("brick_diff_1k.png"), height = folder.appendingPathComponent("brick_disp_1k.png")
        let rgb = NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 127, count: 256 * 256 * 3), colorChunks: [])
        let numeric = NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: 97, count: 256 * 256 * 2), colorChunks: [])
        let inputBytes = try rgb.encoded(), heightBytes = try numeric.encoded()
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        let first = try await output(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(first["material_count"] as? Int, 1)
        let indexURL = dataset.appendingPathComponent("dataset.json"), metadata = try object(indexURL)
        XCTAssertEqual(metadata["source_discovery"] as? String, "registered-resolution-and-color-sets-v2")
        XCTAssertEqual((metadata["source_inventory_sha256"] as? String)?.count, 64)
        let indexBytes = try Data(contentsOf: indexURL)
        var before = stat(); XCTAssertEqual(stat(indexURL.path, &before), 0)
        let second = try await output(["dataset", "--dataset", dataset.path])
        var after = stat(); XCTAssertEqual(stat(indexURL.path, &after), 0)
        XCTAssertEqual(second["index_sha256"] as? String, first["index_sha256"] as? String)
        XCTAssertEqual(try Data(contentsOf: indexURL), indexBytes)
        XCTAssertEqual(before.st_mtimespec.tv_sec, after.st_mtimespec.tv_sec)
        XCTAssertEqual(before.st_mtimespec.tv_nsec, after.st_mtimespec.tv_nsec)
        XCTAssertEqual(before.st_ctimespec.tv_sec, after.st_ctimespec.tv_sec)
        XCTAssertEqual(before.st_ctimespec.tv_nsec, after.st_ctimespec.tv_nsec)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        var retired = metadata; retired["storage_policy"] = "copied-normalized-legacy"
        try JSONSerialization.data(withJSONObject: retired).write(to: indexURL)
        do { _ = try await output(["dataset", "--dataset", dataset.path]); XCTFail("Retired storage cannot be silently migrated") }
        catch { XCTAssertTrue(error.localizedDescription.contains("retired material schema")) }
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
    }
    func testNativeImportPrepareCleanupAndMissingSourceLifecycleKeepsOriginalCodes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Walls"), dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("diffuse.png"), height = sources.appendingPathComponent("height.png")
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 127, count: 512 * 512 * 3), colorChunks: [])
        let rawHeight = Data((0..<512 * 512 * 2).map { UInt8(truncatingIfNeeded: $0 * 97) })
        let numeric = NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0), pixels: rawHeight, colorChunks: [])
        let inputBytes = try rgb.encoded(), heightBytes = try numeric.encoded()
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native", "--training-size", "256"])
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String), "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "256"])
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let materials = try XCTUnwrap(imported["materials"] as? [[String: Any]])
        let sample = try XCTUnwrap((materials[0]["samples"] as? [[String: Any]])?.first)
        let sampleID = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--status", "approved", "--note", "Keep relief", "--expected-review-sha256", try XCTUnwrap(imported["review_sha256"] as? String)])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex, "Crop reviews only update their source-bound sidecar")
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        let preparedPath = try XCTUnwrap(prepared["dataset_path"] as? String), preparedURL = URL(fileURLWithPath: preparedPath)
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        let descriptor = try XCTUnwrap(descriptors.first)
        let preparedHeight = try NativePNG.decode(Data(contentsOf: descriptor.targetURL))
        XCTAssertEqual(preparedHeight.pixels, try numeric.crop([128, 128, 256, 256]).pixels)
        XCTAssertEqual(try preparedHeight.modelFloatSamples(role: "height").count, 256 * 256)
        let cropBytes = try Data(contentsOf: descriptor.targetURL)
        let reviewBytes = try Data(contentsOf: dataset.appendingPathComponent(".material-size-reviews.json"))
        try Data([1, 2, 3]).write(to: descriptor.targetURL)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Modified crops need inspection before cleanup") }
        catch { XCTAssertTrue(error.localizedDescription.contains("crop contents changed")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent(".material-size-reviews.json")), reviewBytes)
        try cropBytes.write(to: descriptor.targetURL)
        let unrelated = preparedURL.appendingPathComponent("Unrelated")
        try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: false)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Empty unrelated folders cannot be removed with crops") }
        catch { XCTAssertTrue(error.localizedDescription.contains("unrelated folder")) }
        try FileManager.default.removeItem(at: unrelated)
        let pipe = preparedURL.appendingPathComponent("unrelated-pipe")
        XCTAssertEqual(mkfifo(pipe.path, S_IRUSR | S_IWUSR), 0)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Special files cannot be removed with crops") }
        catch { XCTAssertTrue(error.localizedDescription.contains("special file")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: pipe.path))
        try FileManager.default.removeItem(at: pipe)
        let reused = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true)
        _ = try await output(["cleanup-size", "--dataset", preparedPath])
        XCTAssertFalse(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        let inspected = try await output(["dataset", "--dataset", dataset.path])
        let inspectedMaterials = inspected["materials"] as? [[String: Any]]
        let inspectedSamples = inspectedMaterials?.first?["samples"] as? [[String: Any]]
        XCTAssertEqual(inspectedSamples?.first?["note"] as? String, "Keep relief")
        _ = try await output(["remove-missing", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--path", height.path])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex, "Present originals cannot be removed by a missing-source operation")
        try FileManager.default.removeItem(at: height)
        let removed = try await output(["remove-missing", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--path", height.resolvingSymlinksInPath().path])
        XCTAssertEqual(removed["removed"] as? Bool, true)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
    }
    func testModelBoundaryUsesPlanarFloatCodesAndKeepsIntegerSourceUntouched() throws {
        let codes = Data([0, 128, 255, 255, 64, 192, 0, 255])
        let png = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 4, color: 6, interlace: 0), pixels: codes, colorChunks: [])
        XCTAssertEqual(try png.modelFloatSamples(role: "input", encoding: "source_srgb_assumed"), [0, Float(64) / 255, Float(128) / 255, Float(192) / 255, 1, 0])
        let normal = try png.modelFloatSamples(role: "normal", normalConvention: "DirectX")
        XCTAssertEqual(normal[2], Float(127) / 255); XCTAssertEqual(normal[3], Float(63) / 255)
        XCTAssertEqual(png.pixels, codes)
        XCTAssertThrowsError(try png.modelFloatSamples(role: "height"))
        XCTAssertThrowsError(try png.modelFloatSamples(role: "input", encoding: "unknown"))
    }
    private func output(_ arguments: [String]) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments)
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
    private func temporary() throws -> URL { let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-material-\(UUID().uuidString)"); try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root }
}
