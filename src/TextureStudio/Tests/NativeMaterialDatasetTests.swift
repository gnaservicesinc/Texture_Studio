import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialDatasetTests: XCTestCase {
    func testPreparationWorkersRespectCPUsAndWorkingMemory() throws {
        let gib = MachineResources.gibibyte
        let policy = NativeMaterialDatasetService.PreparationPolicy(resources: .init(physicalBytes: 8 * gib), processorCount: 12)
        XCTAssertEqual(policy.memoryBudgetBytes, gib)
        XCTAssertEqual(try policy.workerCount(jobCount: 20, estimatedPeakBytes: gib / 3), 3)
        XCTAssertEqual(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib / 3), 2)
        XCTAssertThrowsError(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib + 1))
        let cores = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 8 * gib, maximumWorkers: 2)
        XCTAssertEqual(try cores.workerCount(jobCount: 8, estimatedPeakBytes: gib), 2)
    }

    func testPreparationKeepsByteIdenticalColorVariantsAtDistinctPaths() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Bricks"), dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let canonical = sources.appendingPathComponent("brick_diff_1k.png")
        let alternate = sources.appendingPathComponent("brick_diff2_1k.png")
        let height = sources.appendingPathComponent("brick_disp_1k.png")
        let pixels = Data((0..<512 * 512 * 3).map { UInt8(truncatingIfNeeded: $0 * 97) })
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0), pixels: pixels, colorChunks: [])
        let inputBytes = try rgb.encoded()
        let heightBytes = try NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0),
                                        pixels: Data(repeating: 79, count: 512 * 512 * 2), colorChunks: []).encoded()
        try inputBytes.write(to: canonical); try inputBytes.write(to: alternate); try heightBytes.write(to: height)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Identical variants", "--training-size", "256"])
        let plan = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        _ = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path,
                              "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
                              "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "256"])
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sourceSampleURL = dataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
        var sourceSample = try object(sourceSampleURL)
        let originalVariants = try XCTUnwrap(sourceSample["input_variants"] as? [[String: Any]])
        XCTAssertEqual(originalVariants.count, 2)
        // Move the noncanonical source first: hash equality or list order must
        // never replace the explicitly recorded canonical input source path.
        sourceSample["input_variants"] = Array(originalVariants.reversed())
        let sourceMetadata = try JSONSerialization.data(withJSONObject: sourceSample, options: [.sortedKeys])
        try sourceMetadata.write(to: sourceSampleURL)
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let preparedIndex = try object(preparedURL.appendingPathComponent("dataset.json"))
        let records = try XCTUnwrap(preparedIndex["samples"] as? [[String: Any]])
        XCTAssertFalse(records.isEmpty)
        let expectedPixels = try rgb.crop([128, 128, 256, 256]).pixels
        for record in records {
            let folder = preparedURL.appendingPathComponent(try XCTUnwrap(record["path"] as? String))
            let sample = try object(folder.appendingPathComponent("sample.json"))
            let variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
            let names = variants.compactMap { $0["filename"] as? String }
            XCTAssertEqual(names, ["input-variant-1.png", "diffuse.png"])
            XCTAssertEqual(Set(names).count, 2)
            XCTAssertEqual(variants.compactMap { $0["variant_id"] as? String }, ["color_2", "color_default"])
            XCTAssertEqual(((variants[0]["source"] as? [String: Any])?["path"] as? String), alternate.resolvingSymlinksInPath().path)
            let maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual(((maps["input"]?["source"] as? [String: Any])?["path"] as? String), canonical.resolvingSymlinksInPath().path)
            XCTAssertEqual((sample["maps"] as? [String: String])?["input"], "diffuse.png")
            for name in names { XCTAssertEqual(try NativePNG.decode(Data(contentsOf: folder.appendingPathComponent(name))).pixels, expectedPixels) }
        }
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, records.count * 2)
        XCTAssertEqual(Set(descriptors.map { $0.inputURL.path }).count, records.count * 2)
        XCTAssertEqual(try Data(contentsOf: canonical), inputBytes); XCTAssertEqual(try Data(contentsOf: alternate), inputBytes)
        XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        XCTAssertEqual(try Data(contentsOf: sourceSampleURL), sourceMetadata)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
    }

    func testParallelPreparationKeepsOrderedMetadataAndExactCropBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 1)
        let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
        let serial = try await preparationOutput(arguments, policy: policy)
        let serialRoot = URL(fileURLWithPath: try XCTUnwrap(serial["dataset_path"] as? String))
        let firstInventory = try preparedInventory(serialRoot)
        let firstIndex = try object(serialRoot.appendingPathComponent("dataset.json"))
        XCTAssertEqual((firstIndex["native_size_preparation"] as? [String: Any])?["preparation_workers"] as? Int, 1)
        _ = try await output(["cleanup-size", "--dataset", serialRoot.path])
        // Restore identical review inputs; cleanup deliberately persists the
        // generated crop review snapshot as a new per-size sidecar.
        try FileManager.default.removeItem(at: dataset.appendingPathComponent(".material-size-reviews.json"))
        let concurrent = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let events = PreparationEventRecorder()
        let parallel = try await preparationOutput(arguments, policy: concurrent, events: events)
        let parallelRoot = URL(fileURLWithPath: try XCTUnwrap(parallel["dataset_path"] as? String))
        let secondIndex = try object(parallelRoot.appendingPathComponent("dataset.json"))
        let binding = try XCTUnwrap(secondIndex["native_size_preparation"] as? [String: Any])
        XCTAssertEqual(binding["preparation_workers"] as? Int, 4)
        let peak = try XCTUnwrap(binding["estimated_worker_peak_bytes"] as? NSNumber).uint64Value
        XCTAssertLessThanOrEqual(peak * 4, concurrent.memoryBudgetBytes)
        XCTAssertEqual(try preparedInventory(parallelRoot), firstInventory)
        XCTAssertEqual(try JSONSerialization.data(withJSONObject: firstIndex["samples"]!, options: [.sortedKeys]),
                       try JSONSerialization.data(withJSONObject: secondIndex["samples"]!, options: [.sortedKeys]))
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        let recorded = try events.objects()
        XCTAssertEqual(recorded.first?["event"] as? String, "preparation_started")
        XCTAssertEqual(recorded.last?["event"] as? String, "preparation_completed")
        XCTAssertEqual(recorded.last?["completed"] as? Int, 4)
        XCTAssertTrue(recorded.allSatisfy { $0["worker_count"] as? Int == 4 && $0["total"] as? Int == 4 })
        let counts = recorded.compactMap { $0["completed"] as? Int }
        XCTAssertEqual(counts, counts.sorted())
        XCTAssertTrue(recorded.contains { $0["event"] as? String == "preparation_progress" })
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: parallelRoot, size: 256, target: "height")
        XCTAssertEqual(samples.count, 4)
        for sample in samples {
            let details = try object(parallelRoot.appendingPathComponent("samples/" + sample.id + "/sample.json"))
            let maps = try XCTUnwrap(details["map_metadata"] as? [String: [String: Any]])
            let source = try XCTUnwrap(maps["height"]?["source"] as? [String: Any])
            let original = try NativePNG.decode(Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))))
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels, try original.crop([128, 128, 256, 256]).pixels)
        }
        _ = try await output(["cleanup-size", "--dataset", parallelRoot.path])
    }

    func testCancelledParallelPreparationJoinsWritersAndRemovesOnlyItsStage() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 1024)
        let original = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let operation = Task {
            try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path, "--size", "512", "--target", "height"], preparationPolicy: policy)
        }
        let staging = dataset.appendingPathComponent(".training-data")
        var found = false
        for _ in 0..<400 {
            if let names = try? FileManager.default.contentsOfDirectory(atPath: staging.path), names.contains(where: {
                $0.hasPrefix(".preparing-") && ((try? FileManager.default.contentsOfDirectory(atPath: staging.appendingPathComponent($0 + "/samples").path).isEmpty) == false)
            }) { found = true; break }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(found, "Cancellation exercises live crop writers")
        operation.cancel()
        do { _ = try await operation.value; XCTFail("Cancellation must prevent publication") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), [])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), original)
        // Once the cancelled worker join returns, no background writer can
        // recreate files or keep the source dataset locked.
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--description", "After cancellation"])
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), [])
    }

    func testFailedParallelDecodePreservesSourcesAndNeverPublishesPartialDataset() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 3, dimension: 512)
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let first = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sampleURL = dataset.appendingPathComponent(try XCTUnwrap(first["path"] as? String)).appendingPathComponent("sample.json")
        var sample = try object(sampleURL), maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
        var details = try XCTUnwrap(maps["height"]), source = try XCTUnwrap(details["source"] as? [String: Any])
        let original = URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))
        // Valid, checksum-bound IHDR reaches the worker; missing IDAT must
        // fail decoding while sibling materials may already be writing crops.
        let truncated = try Data(contentsOf: original).prefix(33)
        try truncated.write(to: original)
        let digest = SHA256.hash(data: truncated).map { String(format: "%02x", $0) }.joined()
        source["file_sha256"] = digest; source["file_bytes"] = truncated.count
        details["source"] = source; details["sample_sha256"] = digest; maps["height"] = details; sample["map_metadata"] = maps
        try JSONSerialization.data(withJSONObject: sample, options: [.sortedKeys]).write(to: sampleURL)
        let indexBytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), metadataBytes = try Data(contentsOf: sampleURL)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 3)
        do {
            _ = try await preparationOutput(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"], policy: policy)
            XCTFail("An incomplete source cannot publish generated samples")
        } catch { XCTAssertFalse(error is CancellationError) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dataset.appendingPathComponent(".training-data").path), [])
        XCTAssertEqual(try Data(contentsOf: original), truncated)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBytes)
        XCTAssertEqual(try Data(contentsOf: sampleURL), metadataBytes)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--description", "After failed workers"])
    }

    private func preparationOutput(_ arguments: [String], policy: NativeMaterialDatasetService.PreparationPolicy, events: PreparationEventRecorder? = nil) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, preparationPolicy: policy, onEvent: { events?.append($0) })
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func preparationFixture(_ root: URL, materials: Int, dimension: Int) async throws -> URL {
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Parallel exact maps", "--training-size", "256"])
        for number in 0..<materials {
            let input = root.appendingPathComponent("input-\(number).png"), height = root.appendingPathComponent("height-\(number).png")
            var seed = UInt64(number + 1)
            func pixels(_ count: Int) -> Data {
                var result = Data(count: count)
                result.withUnsafeMutableBytes { bytes in
                    let values = bytes.bindMemory(to: UInt8.self)
                    for index in 0..<count { seed = seed &* 6364136223846793005 &+ 1442695040888963407; values[index] = UInt8(truncatingIfNeeded: seed >> 32) }
                }
                return result
            }
            try NativePNG(header: .init(width: dimension, height: dimension, bits: 8, channels: 3, color: 2, interlace: 0), pixels: pixels(dimension * dimension * 3), colorChunks: []).encoded().write(to: input)
            try NativePNG(header: .init(width: dimension, height: dimension, bits: 16, channels: 1, color: 0, interlace: 0), pixels: pixels(dimension * dimension * 2), colorChunks: []).encoded().write(to: height)
            _ = try await output(["add-material", "--dataset", dataset.path, "--name", "Material \(number)", "--input", input.path, "--height", height.path])
        }
        return dataset
    }
    private func preparedInventory(_ root: URL) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for case let file as URL in FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])! {
            if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true, file.lastPathComponent != "dataset.json", file.lastPathComponent != ".material-workbench.lock" {
                let path = file.resolvingSymlinksInPath().standardizedFileURL.path, prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
                result[String(path.dropFirst(prefix.count))] = try Data(contentsOf: file)
            }
        }
        return result
    }

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

private final class PreparationEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []
    func append(_ value: String) { lock.withLock { values.append(value) } }
    func objects() throws -> [[String: Any]] {
        try lock.withLock { try values.map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] } }
    }
}
