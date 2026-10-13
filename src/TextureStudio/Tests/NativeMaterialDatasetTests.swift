import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialDatasetTests: XCTestCase {
    func testFolderMembershipIgnoresLegacyManifestReferencesAndTombstones() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 1, dimension: 256)
        let manifest = dataset.appendingPathComponent("dataset.json")
        let legacy: [String: Any] = ["schema_version": 2, "dataset_management": [:],
            "samples": [["sample_id": "ghost", "material_id": "ghost", "path": "../external/ghost"]],
            "removed_materials": [["material_id": "material_0_256x256"]]]
        let bytes = try JSONSerialization.data(withJSONObject: legacy)
        try bytes.write(to: manifest)
        let info = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(info["source_set_count"] as? Int, 1, "Legacy lists cannot hide images currently inside the folder")
        let inventory = try XCTUnwrap(info["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(Set(inventory.compactMap { $0["path"] as? String }), [
            dataset.appendingPathComponent("Material 0/BaseColor.png").path,
            dataset.appendingPathComponent("Material 0/Height.png").path])
        XCTAssertFalse((info["materials"] as? [[String: Any]] ?? []).contains { $0["material_id"] as? String == "ghost" })
        XCTAssertEqual(try Data(contentsOf: manifest), bytes)
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: "height")
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 1)
    }

    func testFolderKeepsLocalResolutionTiersWithoutAutomaticProviderDownloads() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Stone")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        var originals: [URL: Data] = [:]
        for (dimension, label) in [(256, "1k"), (512, "2k"), (1024, "4k")] {
            for (suffix, bits, channels) in [("diff", 8, 3), ("disp", 16, 1), ("rough", 8, 1), ("nor_gl", 8, 3)] {
                let file = dataset.appendingPathComponent("stone_\(suffix)_\(label).png")
                let bytes = try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                try bytes.write(to: file); originals[file] = bytes
            }
        }
        let probe = DatasetRecoveryMetadataProbe()
        let localOnly = NativeMaterialSourceRecovery(transport: .init(fetchMetadata: { url in
            await probe.record(url); throw StudioError("Opening source folders does not download tiers")
        }, download: { _, _ in throw StudioError("Opening source folders does not download tiers") }))
        let read = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"], sourceRecovery: localOnly)
        XCTAssertEqual(read["source_set_count"] as? Int, 3)
        for target in ["height", "roughness", "normal"] {
            let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target], sourceRecovery: localOnly)
            let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: target)
            XCTAssertEqual(samples.count, 3, "Every adequate tier contributes once for the requested map")
            for sample in samples {
                XCTAssertEqual(try NativePNG.inspect(sample.targetURL).width, 256)
                XCTAssertEqual(try NativePNG.inspect(sample.targetURL).height, 256)
            }
            _ = try await output(["cleanup-size", "--dataset", prepared.path])
        }
        let requests = await probe.urls(); XCTAssertEqual(requests, [])
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dataset.path).count, 12)
    }

    func testCurrentSourceInventoryIsStableAcrossRepeatedReadsWithoutWritingMetadata() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 2, dimension: 512)
        let originals = try sourceSnapshot(dataset)
        let first = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let second = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(first["index_sha256"] as? String, second["index_sha256"] as? String)
        XCTAssertEqual(first["source_set_count"] as? Int, 2)
        XCTAssertEqual(try sourceSnapshot(dataset), originals)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent("dataset.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent("samples").path))
    }

    func testFolderReadReconstructsSharedCropPlansAndPreparedMetadataRejectsPathEscape() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--validation-enabled", "yes", "--validation-percent", "25"])
        let originalSources = try sourceSnapshot(dataset)
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--target", "height", "--review-size", "256"])
        let plans = try XCTUnwrap(inspected["training_plans"] as? [String: [String: Any]])
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 4)
        XCTAssertEqual(plans["height"]?["validation_count"] as? Int, 1)
        XCTAssertEqual(plans["roughness"]?["unavailable_target_count"] as? Int, 5)
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: "height")
        let indexURL = prepared.appendingPathComponent("dataset.json"), originalIndex = try Data(contentsOf: indexURL)
        defer { try? originalIndex.write(to: indexURL); _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        var index = try object(indexURL), entries = try XCTUnwrap(index["samples"] as? [[String: Any]])
        entries[0]["path"] = "../escaped"; index["samples"] = entries
        try JSONSerialization.data(withJSONObject: index).write(to: indexURL)
        do { _ = try await output(["dataset", "--dataset", prepared.path]); XCTFail("Owned prepared metadata cannot escape its directory") }
        catch { XCTAssertTrue(error.localizedDescription.contains("inside its directory")) }
        XCTAssertEqual(try sourceSnapshot(dataset), originalSources)
    }

    func testPreparationExportsOnlySelectedTargetWithoutDependingOnOtherMaps() async throws {
        let dataset = try temporary(); defer { try? FileManager.default.removeItem(at: dataset) }
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--training-size", "256", "--validation-enabled", "no"])
        var paths: [String: URL] = [:]
        for (role, channels, bits) in [("input", 3, 8), ("height", 1, 16), ("roughness", 1, 8), ("normal", 3, 8)] {
            let file = dataset.appendingPathComponent(role + ".png")
            try fixturePNG(width: 512, height: 512, bits: bits, channels: channels).write(to: file)
            try NativeSourceFileTags.update(file, role: role)
            paths[role] = file
        }
        for target in ["roughness", "normal", "height"] {
            if target == "height" { for role in ["roughness", "normal"] { try FileManager.default.removeItem(at: paths[role]!) } }
            let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
            let index = try object(prepared.appendingPathComponent("dataset.json"))
            XCTAssertEqual((index["native_size_preparation"] as? [String: Any])?["target"] as? String, target)
            for entry in try XCTUnwrap(index["samples"] as? [[String: Any]]) {
                let folder = prepared.appendingPathComponent(try XCTUnwrap(entry["path"] as? String))
                let metadata = try object(folder.appendingPathComponent("sample.json"))
                XCTAssertEqual(Set((metadata["maps"] as? [String: String] ?? [:]).keys), ["input", target])
                XCTAssertEqual(metadata["available_targets"] as? [String], [target])
                XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasSuffix(".png") }.count, 2)
                XCTAssertNil(metadata["source_files"])
            }
            XCTAssertFalse(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: target).isEmpty)
            let absent = ["height", "roughness", "normal"].first { $0 != target }!
            XCTAssertTrue(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: absent).isEmpty)
            _ = try await output(["cleanup-size", "--dataset", prepared.path])
        }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dataset.path).sorted(), ["height.png", "input.png"])
    }

    func testColorProfilesStayInOriginalPNGAndPreparedMetadataRemainsCompact() async throws {
        let dataset = try temporary(); defer { try? FileManager.default.removeItem(at: dataset) }
        let input = dataset.appendingPathComponent("BaseColor.png"), height = dataset.appendingPathComponent("Height.png")
        let profile = Data(repeating: 79, count: 2 * 1024 * 1024)
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data(repeating: 97, count: 512 * 512 * 3), colorChunks: [("iCCP", profile)])
        let bytes = try rgb.encoded(); try bytes.write(to: input)
        let summary = try NativePNG.sourceMetadata(bytes)
        XCTAssertNil(summary["color_chunks"])
        XCTAssertLessThan(try JSONSerialization.data(withJSONObject: summary).count, 512)
        try fixturePNG(width: 512, height: 512, bits: 16, channels: 1).write(to: height)
        let read = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let sample = try XCTUnwrap((read["materials"] as? [[String: Any]])?.first?["samples"] as? [[String: Any]]).first
        let id = try XCTUnwrap(sample?["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", id, "--review-size", "256", "--status", "approved", "--note", "Keep profile"])
        let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let descriptor = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").first)
        XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.inputURL)).colorChunks.first?.1, profile)
        let metadata = try Data(contentsOf: descriptor.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json"))
        XCTAssertLessThan(metadata.count, 16 * 1024)
        XCTAssertFalse(String(decoding: metadata, as: UTF8.self).contains("base64"))
        XCTAssertEqual(try Data(contentsOf: input), bytes)
        let reopened = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let reviewed = ((reopened["materials"] as? [[String: Any]])?.first?["samples"] as? [[String: Any]])?.first
        XCTAssertEqual(reviewed?["status"] as? String, "approved"); XCTAssertEqual(reviewed?["note"] as? String, "Keep profile")
    }

    func testPreparationKeepsByteIdenticalDiffuseVariantsAtDistinctPaths() async throws {
        let dataset = try temporary(); defer { try? FileManager.default.removeItem(at: dataset) }
        let canonical = dataset.appendingPathComponent("brick_diff_1k.png")
        let alternate = dataset.appendingPathComponent("brick_diff2_1k.png")
        let height = dataset.appendingPathComponent("brick_disp_1k.png")
        let inputBytes = try fixturePNG(width: 512, height: 512, bits: 8, channels: 3)
        let heightBytes = try fixturePNG(width: 512, height: 512, bits: 16, channels: 1)
        try inputBytes.write(to: canonical); try inputBytes.write(to: alternate); try heightBytes.write(to: height)
        let originalSources = try sourceSnapshot(dataset)
        let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        defer { _ = try? NativeMaterialDatasetService.cleanup(prepared) }
        let index = try object(prepared.appendingPathComponent("dataset.json")), records = try XCTUnwrap(index["samples"] as? [[String: Any]])
        for record in records {
            let folder = prepared.appendingPathComponent(try XCTUnwrap(record["path"] as? String))
            let sample = try object(folder.appendingPathComponent("sample.json"))
            let variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
            let names = variants.compactMap { $0["filename"] as? String }
            XCTAssertEqual(Set(names).count, 2)
            XCTAssertEqual(Set(variants.compactMap { $0["variant_id"] as? String }), ["color_default", "color_2"])
            let maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual((maps["input"]?["source"] as? [String: Any])?["path"] as? String, canonical.path)
            XCTAssertEqual((sample["maps"] as? [String: String])?["input"], "diffuse.png")
            let expected = try NativePNG.decode(inputBytes).crop([128, 128, 256, 256]).pixels
            for name in names { XCTAssertEqual(try NativePNG.decode(Data(contentsOf: folder.appendingPathComponent(name))).pixels, expected) }
        }
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, records.count * 2)
        XCTAssertEqual(Set(descriptors.map { $0.inputURL.path }).count, records.count * 2)
        XCTAssertEqual(try sourceSnapshot(dataset), originalSources)
    }

    func testFailedSourceVerificationQuarantinesInvalidPairAndPreservesValidOriginals() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 3, dimension: 512)
        let original = dataset.appendingPathComponent("Material 0/Height.png")
        let truncated = try Data(contentsOf: original).prefix(33)
        try truncated.write(to: original)
        let originals = try sourceSnapshot(dataset)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 3)
        let events = PreparationEventRecorder()
        let result = try await preparationOutput(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"],
                                                policy: policy, events: events)
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertEqual(samples.count, 2, "One invalid pair must not prevent the valid corpus from training")
        XCTAssertFalse(samples.contains { $0.augmentationSourceTargetURL == original })
        XCTAssertEqual(Set(samples.compactMap { $0.augmentationSourceTargetURL?.deletingLastPathComponent().lastPathComponent }), ["Material 1", "Material 2"])
        for sample in samples {
            let source = try XCTUnwrap(sample.augmentationSourceTargetURL)
            let crop = try XCTUnwrap(sample.cropRectangle)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels,
                           try NativePNG.decode(Data(contentsOf: source)).crop(crop).pixels)
        }
        let event = try XCTUnwrap(events.objects().first { $0["source_path"] as? String == original.path && $0["issue"] as? String != nil })
        XCTAssertFalse(try XCTUnwrap(event["issue"] as? String).isEmpty)
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let issue = try XCTUnwrap((inspected["source_inventory"] as? [[String: Any]])?.first { $0["path"] as? String == original.path })
        XCTAssertEqual(issue["state"] as? String, "header_error")
        XCTAssertFalse(try XCTUnwrap(issue["issue"] as? String).isEmpty)
        XCTAssertEqual(try sourceSnapshot(dataset), originals)
        _ = try await output(["cleanup-size", "--dataset", prepared.path])
        let cache = try NativeMaterialDatasetService.trainingCacheRoot(datasetURL: dataset)
        XCTAssertEqual((try? FileManager.default.contentsOfDirectory(atPath: cache.path)) ?? [], [])
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--training-size", "512"])
    }

    func testProviderTierIdentityAcceptsOriginalAspectRatioAndKeepsCropEligibilitySeparate() throws {
        for (dimensions, resolution) in [([4096, 3893], "4k"), ([4096, 4005], "4k"), ([2048, 2538], "2k"), ([4096, 5076], "4k")] {
            XCTAssertTrue(NativeMaterialDatasetService.isProviderResolutionField(dimensions: dimensions, resolution: resolution))
            XCTAssertEqual(NativeMaterialDatasetService.cropLayout(dimensions, size: 1024).count, 1)
        }
        XCTAssertTrue(NativeMaterialDatasetService.isProviderResolutionField(dimensions: [3893, 4096], resolution: "4k"))
        XCTAssertTrue(NativeMaterialDatasetService.isProviderResolutionField(dimensions: [1024, 973], resolution: "1k"))
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([1024, 973], size: 1024).isEmpty,
                      "A genuine nominal tier can be too short for this crop; its original pixels must not be resized")
        XCTAssertTrue(NativeMaterialDatasetService.isProviderResolutionField(dimensions: [1024, 1024], resolution: "1k"))
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([1024, 1024], size: 2048).isEmpty,
                      "An existing1k field still identifies its provider for acquiring usable2k/4k/8k tiers")
        for dimensions in [[256, 256], [512, 512], [0, 1024], [1024, 0], [-1, 1024]] {
            XCTAssertFalse(NativeMaterialDatasetService.isProviderResolutionField(dimensions: dimensions, resolution: "1k"))
        }
        XCTAssertFalse(NativeMaterialDatasetService.isProviderResolutionField(dimensions: [2048, 2538], resolution: "4k"))
        XCTAssertFalse(NativeMaterialDatasetService.isProviderResolutionField(dimensions: [1024, 1024], resolution: "999999999999999k"))
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let input = root.appendingPathComponent("wool_boucle_diff_2k.png")
        try constantPNG(width: 2048, height: 2538, bits: 8, channels: 3).write(to: input)
        let requests = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 1024, target: "height", tierCount: 3)
        XCTAssertEqual(Set(requests.map { $0.resolution + "|" + $0.target }), ["1k|input", "1k|height", "2k|height", "4k|input", "4k|height"])
        let paired = try XCTUnwrap(requests.first { $0.resolution == "2k" })
        XCTAssertEqual(paired.width, 2048); XCTAssertEqual(paired.height, 2538); XCTAssertTrue(paired.preserveOriginalDimensions)
        let largerGrid = NativeMaterialDatasetService.croppingRecoveryRequests(files: [input], size: 2048, target: "height", tierCount: 3)
        XCTAssertEqual(Set(largerGrid.map { $0.resolution }), ["2k", "4k", "8k"])
    }

    func testBalancedCropLayoutUsesOneSampleAtEveryResolutionTier() throws {
        func crop(_ dimensions: [Int], seed: UInt64 = 0, source: String = "stone") throws -> (String, [Int]) {
            let selected = NativeMaterialDatasetService.cropLayout(dimensions, size: 1024, subject: "/sources/stone", sourceSet: source, seed: seed)
            XCTAssertEqual(selected.count, 1, "A larger source must not multiply this material's training weight")
            return try XCTUnwrap(selected.first)
        }
        XCTAssertEqual(try crop([1024, 1024]).0, "full")
        XCTAssertEqual(try crop([1024, 1024], seed: 17).1, [0, 0, 1024, 1024])
        XCTAssertEqual(try crop([2048, 2048], seed: 1).1, [512, 512, 1024, 1024])
        XCTAssertEqual(try crop([2048, 2048], seed: 99).0, "center")
        let anchors = Set(["0,0", "3072,0", "0,3072", "3072,3072", "1536,1536"])
        var chosen = Set<String>()
        for seed in UInt64(0)..<64 {
            let anchor = try crop([4096, 4096], seed: seed)
            XCTAssertEqual(anchor.0, "selected")
            let position = "\(anchor.1[0]),\(anchor.1[1])"
            XCTAssertTrue(anchors.contains(position)); chosen.insert(position)
            XCTAssertEqual(anchor.1, try crop([4096, 4096], seed: seed, source: "other-color-or-target").1,
                           "The five-position choice belongs to the folder, not a variant or target")
        }
        XCTAssertEqual(chosen, anchors)
        let restricted = NativeMaterialDatasetService.cropLayout([4096, 4096], size: 1024, subject: "/sources/stone", sourceSet: "stone", seed: 7, excludedAnchors: [0, 1, 2, 3])
        XCTAssertEqual(restricted.first?.1, [1536, 1536, 1024, 1024], "A reserved diagnostic position must remain separate from training")
        let random = try crop([8192, 8192], seed: 17)
        XCTAssertEqual(random.0, "random")
        XCTAssertEqual(random.1, try crop([8192, 8192], seed: 17).1)
        XCTAssertNotEqual(random.1, try crop([8192, 8192], seed: 18).1)
        XCTAssertTrue((0...7168).contains(random.1[0])); XCTAssertTrue((0...7168).contains(random.1[1]))
        XCTAssertEqual(Array(random.1.suffix(2)), [1024, 1024])
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([512, 512], size: 1024).isEmpty)
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([1024, 1024], size: 0).isEmpty)
    }

    func testRectangularNativeFieldsChooseOneSeededSquareWithoutChangingMaterialWeight() throws {
        for dimensions in [[2048, 4096], [4096, 2048]] {
            let portrait = dimensions[0] < dimensions[1]
            let possible = portrait ? [[0, 0, 2048, 2048], [0, 2048, 2048, 2048]] : [[0, 0, 2048, 2048], [2048, 0, 2048, 2048]]
            var chosen = Set<[Int]>()
            for seed in UInt64(0)..<64 {
                let layout = NativeMaterialDatasetService.cropLayout(dimensions, size: 2048, subject: "bark", sourceSet: "bark-native", seed: seed)
                XCTAssertEqual(layout.count, 1, "A rectangular tier contributes one spatial sample, just like a square tier")
                let rectangle = try XCTUnwrap(layout.first?.1)
                XCTAssertTrue(possible.contains(rectangle)); chosen.insert(rectangle)
                XCTAssertEqual(layout.first?.0, "selected")
                XCTAssertEqual(rectangle, NativeMaterialDatasetService.cropLayout(dimensions, size: 2048, subject: "bark", sourceSet: "other-color", seed: seed).first?.1)
                XCTAssertEqual(rectangle, NativeMaterialDatasetService.cropLayout(dimensions, size: 2048, subject: "bark", sourceSet: "bark-native", seed: seed).first?.1)
            }
            XCTAssertEqual(chosen, Set(possible), "Both long-axis regions must be reachable")
            XCTAssertEqual(NativeMaterialDatasetService.cropLayout(dimensions, size: 2048, subject: "bark", seed: 1, excludedAnchors: [0]).first?.1, possible[1])
            XCTAssertTrue(NativeMaterialDatasetService.cropLayout(dimensions, size: 2048, subject: "bark", excludedAnchors: [0, 1]).isEmpty)
        }
        let centers = Set([[512, 512, 1024, 1024], [512, 2560, 1024, 1024]])
        let selected = Set((UInt64(0)..<64).compactMap { NativeMaterialDatasetService.cropLayout([2048, 4096], size: 1024, subject: "bark", seed: $0).first?.1 })
        XCTAssertEqual(selected, centers, "Center crops come from a selected native square rather than the middle seam")
        let ends = Set((UInt64(0)..<64).compactMap { NativeMaterialDatasetService.cropLayout([2048, 5120], size: 2048, subject: "bark", seed: $0).first?.1 })
        XCTAssertEqual(ends, Set([[0, 0, 2048, 2048], [0, 3072, 2048, 2048]]), "A partial trailing strip is covered by another exact native square")
        XCTAssertTrue(NativeMaterialDatasetService.cropLayout([1024, 2048], size: 2048).isEmpty, "The long edge or filename cannot make an undersized short edge eligible")
        for seed in UInt64(0)..<32 {
            let crop = try XCTUnwrap(NativeMaterialDatasetService.cropLayout([8192, 16384], size: 1024, subject: "bark", seed: seed).first)
            XCTAssertEqual(crop.0, "random")
            XCTAssertTrue((0...7168).contains(crop.1[0]))
            XCTAssertTrue((0...7168).contains(crop.1[1]) || (8192...15360).contains(crop.1[1]))
        }
    }

    func testRectangularPreparationPreservesEveryPairedTargetAndColorAndSkipsActualUndersizedTier() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), folder = dataset.appendingPathComponent("bark")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Rectangular bark", "--training-size", "256"])
        var originals: [URL: Data] = [:]
        for (edge, label) in [(128, "1k"), (512, "2k"), (1024, "4k")] {
            for (suffix, bits, channels) in [("diff", 16, 3), ("diff2", 16, 3), ("disp", 16, 1), ("rough", 16, 1), ("nor_dx", 16, 3)] {
                let url = folder.appendingPathComponent("bark_\(suffix)_\(label).png")
                let bytes = try fixturePNG(width: edge, height: edge * 2, bits: bits, channels: channels)
                try bytes.write(to: url); originals[url] = bytes
            }
        }
        let sourceBytes = try sourceSnapshot(dataset)
        var geometry: [String: [Int]] = [:]
        for target in ["height", "roughness", "normal"] {
            let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: target, cropSeed: 17, automaticValidation: false)
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: target)
            XCTAssertEqual(samples.count, 4, "Each of two adequate native tiers retains both original colors")
            for sample in samples {
                let input = try XCTUnwrap(sample.augmentationSourceInputURL), output = try XCTUnwrap(sample.augmentationSourceTargetURL)
                XCTAssertFalse(input.lastPathComponent.contains("_1k"), "Actual dimensions override nominal tier labels")
                let rectangle = try XCTUnwrap(sample.cropRectangle), identity = String(sample.id.split(separator: "|")[0])
                if let prior = geometry[identity] { XCTAssertEqual(rectangle, prior) } else { geometry[identity] = rectangle }
                let inputBytes = try XCTUnwrap(originals[input]), outputBytes = try XCTUnwrap(originals[output])
                XCTAssertEqual(sample.augmentationSourceInputSHA256, NativeMaterialTrainer.checksum(inputBytes))
                XCTAssertEqual(sample.augmentationSourceTargetSHA256, NativeMaterialTrainer.checksum(outputBytes))
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.inputURL)).pixels, try NativePNG.decode(inputBytes).crop(rectangle).pixels)
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels, try NativePNG.decode(outputBytes).crop(rectangle, flipGreen: target == "normal").pixels)
                let metadata = try object(sample.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json"))
                let square = try XCTUnwrap(metadata["source_square_rectangle_top_left_xywh"] as? [Int])
                let edge = try NativePNG.inspect(input).width
                XCTAssertEqual(square, [0, rectangle[1] < edge ? 0 : edge, edge, edge])
                let isCenter = rectangle == [square[0] + (edge - 256) / 2, square[1] + (edge - 256) / 2, 256, 256]
                XCTAssertEqual(sample.cropKind, isCenter ? "center" : "anchor")
            }
        }
        XCTAssertEqual(geometry.count, 2)
        XCTAssertEqual(try sourceSnapshot(dataset), sourceBytes)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    func testRectangularFullSquareChoiceKeepsDiagnosticPixelsFixedAcrossTrainingSeeds() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), folder = dataset.appendingPathComponent("bark")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Two native squares", "--training-size", "256", "--validation-percent", "100"])
        for (suffix, bits, channels) in [("diff", 16, 3), ("disp", 16, 1)] {
            try fixturePNG(width: 256, height: 512, bits: bits, channels: channels).write(to: folder.appendingPathComponent("bark_\(suffix)_2k.png"))
        }
        var diagnosticID: String?, diagnosticRectangle: [Int]?
        for seed in [UInt64(0), 1, 17, 64] {
            let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: "height", cropSeed: seed)
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
            let train = try XCTUnwrap(samples.first { $0.split == "train" }), check = try XCTUnwrap(samples.first { $0.split == "validation" })
            XCTAssertNotEqual(train.cropRectangle, check.cropRectangle, "Refreshing a native-square choice must not reuse its reserved diagnostic pixels")
            if let diagnosticID { XCTAssertEqual(check.id, diagnosticID); XCTAssertEqual(check.cropRectangle, diagnosticRectangle) }
            else { diagnosticID = check.id; diagnosticRectangle = check.cropRectangle }
        }
    }

    func testSeededPreparationKeepsVariantAndTargetGeometryPairedAndReusesOnlyMatchingSelection() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), folder = dataset.appendingPathComponent("stone")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Balanced", "--training-size", "256", "--validation-percent", "100"])
        var originals: [URL: Data] = [:]
        for (dimension, label) in [(1024, "1k"), (2048, "2k")] {
            for (suffix, bits, channels) in [("diff", 8, 3), ("diff2", 8, 3), ("disp", 16, 1), ("rough", 8, 1), ("nor_dx", 8, 3)] {
                let url = folder.appendingPathComponent("stone_\(suffix)_\(label).png")
                let bytes = try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                try bytes.write(to: url); originals[url] = bytes
            }
        }
        let sourceBytes = try sourceSnapshot(dataset)
        var geometry: [String: [Int]] = [:]
        var firstHeightURL: URL?
        for target in ["height", "roughness", "normal"] {
            let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target, "--crop-seed", "17"])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
            if target == "height" { firstHeightURL = preparedURL }
            let preparedIndex = try object(preparedURL.appendingPathComponent("dataset.json"))
            let entries = try XCTUnwrap(preparedIndex["samples"] as? [[String: Any]])
            XCTAssertEqual(entries.count, 2, "One spatial sample must come from each registered resolution")
            let binding = try XCTUnwrap(preparedIndex["native_size_preparation"] as? [String: Any])
            let selection = try XCTUnwrap(binding["crop_selection"] as? [String: Any])
            XCTAssertEqual(selection["policy"] as? String, NativeMaterialDatasetService.cropSelectionPolicy)
            XCTAssertEqual(selection["seed"] as? String, "17")
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(samples.count, 4, "Each registered diffuse variant contributes once at its source resolution")
            for sample in samples {
                let sourceInput = try XCTUnwrap(sample.augmentationSourceInputURL)
                let sourceTarget = try XCTUnwrap(sample.augmentationSourceTargetURL)
                XCTAssertEqual(sample.augmentationSourceInputSHA256, SHA256.hash(data: try XCTUnwrap(originals[sourceInput])).map { String(format: "%02x", $0) }.joined())
                XCTAssertEqual(sample.augmentationSourceTargetSHA256, SHA256.hash(data: try XCTUnwrap(originals[sourceTarget])).map { String(format: "%02x", $0) }.joined())
                XCTAssertEqual(sample.subjectID, folder.path)
                let rectangle = try XCTUnwrap(sample.cropRectangle), identity = String(sample.id.split(separator: "|")[0])
                let sourceSize = sourceInput.lastPathComponent.contains("_1k") ? 1024 : 2048
                let isCenter = rectangle == [(sourceSize - 256) / 2, (sourceSize - 256) / 2, 256, 256]
                XCTAssertEqual(sample.cropKind, isCenter ? "center" : sourceSize == 1024 ? "anchor" : "random")
                if let recorded = geometry[identity] { XCTAssertEqual(rectangle, recorded) }
                else { geometry[identity] = rectangle }
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.inputURL)).pixels,
                               try NativePNG.decode(Data(contentsOf: sourceInput)).crop(rectangle).pixels)
                let expectedTarget = try NativePNG.decode(Data(contentsOf: sourceTarget)).crop(rectangle, flipGreen: target == "normal")
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels, expectedTarget.pixels)
                XCTAssertEqual(sample.targetConvention, "opengl")
                XCTAssertEqual(sample.augmentationSourceTargetConvention, target == "normal" ? "directx" : "opengl")
            }
        }
        let originalPrepared = try XCTUnwrap(firstHeightURL)
        let reused = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--crop-seed", "17"])
        XCTAssertEqual(reused["dataset_path"] as? String, originalPrepared.path)
        XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true)
        let refreshed = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: originalPrepared, size: 256, target: "height", cropSeed: 18, material: "stone_1k", automaticValidation: false)
        XCTAssertNotEqual(refreshed, originalPrepared, "A refreshed choice must not overwrite another run's immutable crops")
        let refreshedSamples = try NativeMaterialDatasetService.trainingSamples(datasetURL: refreshed, size: 256, target: "height")
        XCTAssertEqual(refreshedSamples.count, 4, "Selecting a source folder must retain its asset's other native resolution sets")
        XCTAssertTrue(refreshedSamples.contains { $0.cropRectangle != geometry[String($0.id.split(separator: "|")[0])] })
        let selectedInfo = try await output(["dataset", "--dataset", refreshed.path])
        let selectedDataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: String(decoding: JSONSerialization.data(withJSONObject: selectedInfo), as: UTF8.self))
        XCTAssertEqual(selectedDataset.automaticValidation?.quickFitMaterialId, "stone_1k")
        XCTAssertTrue(selectedDataset.readyForTraining(size: 256, material: "stone_1k", target: "height"))
        XCTAssertFalse(selectedDataset.readyForTraining(size: 256, material: nil, target: "height"))
        let centerRectangle = [384, 384, 256, 256]
        let centerSeed = try XCTUnwrap((UInt64(0)..<256).first {
            NativeMaterialDatasetService.cropLayout([1024, 1024], size: 256, subject: folder.path, sourceSet: "stone_1024x1024", seed: $0).first?.1 == centerRectangle
        })
        let centerURL = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: "height", cropSeed: centerSeed, material: "stone_1k", automaticValidation: false)
        let centerSamples = try NativeMaterialDatasetService.trainingSamples(datasetURL: centerURL, size: 256, target: "height")
            .filter { $0.augmentationSourceInputURL?.lastPathComponent.contains("_1k") == true }
        XCTAssertEqual(centerSamples.count, 2)
        for sample in centerSamples {
            XCTAssertEqual(sample.cropRectangle, centerRectangle)
            XCTAssertEqual(sample.cropKind, "center", "The center among the five anchor positions must be eligible for a center spin")
            XCTAssertTrue(sample.id.contains("_selected"), "Spin eligibility must preserve the reviewed spatial slot identity")
            let metadata = try object(sample.inputURL.deletingLastPathComponent().appendingPathComponent("sample.json"))
            XCTAssertEqual(metadata["source_region_id"] as? String, "selected")
            XCTAssertEqual(metadata["crop_kind"] as? String, "center")
        }
        var diagnostic: [(String, [Int])] = []
        for seed in [UInt64(17), 18] {
            let url = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: dataset, size: 256, target: "height", cropSeed: seed)
            let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: url, size: 256, target: "height")
            let check = try XCTUnwrap(descriptors.first { $0.split == "validation" })
            diagnostic.append((check.id, try XCTUnwrap(check.cropRectangle)))
            XCTAssertFalse(descriptors.contains { $0.split == "train" && $0.augmentationSourceTargetURL == check.augmentationSourceTargetURL && $0.cropRectangle == check.cropRectangle })
        }
        XCTAssertEqual(diagnostic[0].0, diagnostic[1].0, "Run refresh must preserve diagnostic source membership")
        XCTAssertEqual(diagnostic[0].1, diagnostic[1].1, "Diagnostic pixels must remain stable when training refreshes")
        XCTAssertEqual(try sourceSnapshot(dataset), sourceBytes)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        do {
            _ = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--crop-seed", "-1"])
            XCTFail("Invalid selection seeds must not publish crops")
        } catch { XCTAssertTrue(error.localizedDescription.contains("seed")) }
    }

    func testSourceScanWorkersRespectCPUAndMappedFileMemory() {
        let gib = MachineResources.gibibyte
        let resources = MachineResources(physicalBytes: 16 * gib, availableProcessorCount: 12)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 1, resources: resources), 8)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 3, largestFileBytes: 1, resources: resources), 3)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 256 * 1_048_576, resources: resources), 5)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: UInt64.max, resources: resources), 1)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 0, largestFileBytes: 0, resources: resources), 0)
        let singleCPU = MachineResources(physicalBytes: 16 * gib, availableProcessorCount: 1)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 1, resources: singleCPU), 1)
    }

    func testSourceScanReadsConcurrentlyPreservesOrderAndIsolatesCorruptFiles() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        var paths: [URL] = [], originals: [Data] = []
        for number in 0..<6 {
            let path = root.appendingPathComponent("map-\(number).png")
            let png = NativePNG(header: .init(width: 32, height: 32, bits: 8, channels: 3, color: 2, interlace: 0),
                                pixels: Data(repeating: UInt8(number), count: 32 * 32 * 3), colorChunks: [])
            var bytes = try png.encoded(); if number == 3 { bytes[29] ^= 1 }
            try bytes.write(to: path); paths.append(path); originals.append(bytes)
        }
        let inputPaths = paths, probe = SourceScanProbe()
        let results = try await Task.detached {
            try NativeMaterialDatasetService.scanSourceFiles(inputPaths, maximumWorkers: 2) { path in
                try probe.begin(requireConcurrentReaders: 2)
                defer { probe.end() }
                let bytes = try Data(contentsOf: path)
                _ = try NativePNG.sourceMetadata(bytes)
                return bytes
            }
        }.value
        XCTAssertEqual(probe.peakReaders, 2, "Two files must actually be inspected at the same time")
        XCTAssertEqual(probe.activeReaders, 0)
        XCTAssertEqual(probe.totalReaders, paths.count)
        XCTAssertEqual(results.count, paths.count)
        for (offset, result) in results.enumerated() {
            if offset == 3 { XCTAssertThrowsError(try result.get()) }
            else { XCTAssertEqual(try result.get(), originals[offset], "Completion order must not change file identity") }
            XCTAssertEqual(try Data(contentsOf: paths[offset]), originals[offset])
        }
    }

    func testCancellingSourceScanStopsAndJoinsReadersBeforeReturning() async throws {
        let paths = (0..<20).map { URL(fileURLWithPath: "/source-\($0).png") }, probe = SourceScanProbe()
        let task = Task.detached {
            try NativeMaterialDatasetService.scanSourceFiles(paths, maximumWorkers: 2) { _ in
                try probe.begin(requireConcurrentReaders: 2)
                defer { probe.end() }
                while true { try Task.checkCancellation(); Thread.sleep(forTimeInterval: 0.001) }
            }
        }
        XCTAssertTrue(probe.waitForReaders(2))
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancelled scans must not return a partial inventory") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(probe.totalReaders, 2, "Cancellation must stop scheduling the remaining files")
        XCTAssertEqual(probe.activeReaders, 0, "The command must join every reader before returning")
    }

    func testPreparationWorkersRespectCPUsAndWorkingMemory() throws {
        let gib = MachineResources.gibibyte
        let policy = NativeMaterialDatasetService.PreparationPolicy(resources: .init(physicalBytes: 8 * gib), processorCount: 12)
        XCTAssertEqual(policy.memoryBudgetBytes, gib)
        XCTAssertEqual(try policy.workerCount(jobCount: 20, estimatedPeakBytes: gib / 3), 3)
        XCTAssertEqual(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib / 3), 2)
        XCTAssertThrowsError(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib + 1))
        let cores = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 8 * gib, maximumWorkers: 2)
        XCTAssertEqual(try cores.workerCount(jobCount: 8, estimatedPeakBytes: gib), 2)
        XCTAssertEqual(try policy.limitingWorkers(to: 2).workerCount(jobCount: 20, estimatedPeakBytes: gib / 3), 2)
        XCTAssertEqual(try policy.limitingWorkers(to: 100).maximumWorkers, 12)
        XCTAssertThrowsError(try policy.limitingWorkers(to: 0))
        let lowMemory = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 32 * 1_048_576, maximumWorkers: 12)
        XCTAssertThrowsError(try lowMemory.workerCount(jobCount: 1, estimatedPeakBytes: 64 * 1_048_576))
        XCTAssertThrowsError(try lowMemory.limitingWorkers(to: 1).workerCount(jobCount: 1, estimatedPeakBytes: 64 * 1_048_576),
                             "Choosing one worker cannot admit a crop that exceeds the entire memory budget")
    }

    func testPreparationCommandRespectsUserWorkerLimit() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let result = try await preparationOutput(["prepare-size", "--dataset", dataset.path, "--size", "256",
                                                  "--target", "height", "--preparation-workers", "2"], policy: policy)
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let index = try object(prepared.appendingPathComponent("dataset.json"))
        let binding = try XCTUnwrap(index["native_size_preparation"] as? [String: Any])
        XCTAssertEqual(binding["preparation_workers"] as? Int, 2)
    }

    func testPreparedReuseSkipsUnchangedFileChecksumsAndDetectsChangedBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for dimension in [256, 512] {
            let fixtureRoot = root.appendingPathComponent("Grid-\(dimension)")
            try FileManager.default.createDirectory(at: fixtureRoot, withIntermediateDirectories: false)
            let dataset = try await preparationFixture(fixtureRoot, materials: 1, dimension: dimension)
            let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
            let prepared = try await output(arguments), preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let index = try object(preparedURL.appendingPathComponent("dataset.json"))
            let binding = try XCTUnwrap(index["native_size_preparation"] as? [String: Any]), proof = try XCTUnwrap(binding["source_snapshot"] as? [String: Any])
            let descriptor = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height").first)
            let originalInput = try Data(contentsOf: descriptor.inputURL)
            var reads: [URL] = []
            func checksum(_ file: URL) throws -> String {
                reads.append(file)
                return SHA256.hash(data: try Data(contentsOf: file, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
            }
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "Both exact original references and generated crops reuse their recorded file state")
            let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
            let sampleURL = preparedURL.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
            var legacy = try object(sampleURL), maps = try XCTUnwrap(legacy["map_metadata"] as? [String: [String: Any]])
            var input = try XCTUnwrap(maps["input"])
            if dimension == 256 {
                var source = try XCTUnwrap(input["source"] as? [String: Any]); source.removeValue(forKey: "source_stat"); input["source"] = source
            } else { input.removeValue(forKey: "prepared_stat") }
            maps["input"] = input; legacy["map_metadata"] = maps
            try JSONSerialization.data(withJSONObject: legacy).write(to: sampleURL)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "Older references or crops without stored state receive one full checksum check")
            reads.removeAll()
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "The missing state is restored after verification")
            let attributes = try FileManager.default.attributesOfItem(atPath: descriptor.inputURL.path)
            let modified = try XCTUnwrap(attributes[.modificationDate] as? Date)
            try FileManager.default.setAttributes([.modificationDate: modified.addingTimeInterval(2)], ofItemAtPath: descriptor.inputURL.path)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "A changed stat triggers one checksum, even for the duplicated canonical input variant")
            reads.removeAll()
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "Verified timestamps are refreshed so subsequent reuse remains cheap")
            var damaged = originalInput; damaged[damaged.count - 13] ^= 1
            try damaged.write(to: descriptor.inputURL)
            reads.removeAll()
            XCTAssertFalse(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "Corruption after the IHDR must still fail checksum validation")
            try originalInput.write(to: descriptor.inputURL)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof))
            let rebound = try await output(arguments)
            let reboundURL = URL(fileURLWithPath: try XCTUnwrap(rebound["dataset_path"] as? String))
            let reboundIndex = try object(reboundURL.appendingPathComponent("dataset.json"))
            let reboundBinding = try XCTUnwrap(reboundIndex["native_size_preparation"] as? [String: Any])
            let reboundProof = try XCTUnwrap(reboundBinding["source_snapshot"] as? [String: Any])
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(reboundURL, size: 256, proof: reboundProof))
            let reboundSample = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: reboundURL, size: 256, target: "height").first)
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: reboundSample.inputURL)).pixels, try NativePNG.decode(originalInput).pixels)
            let reused = try await output(arguments)
            XCTAssertEqual(reused["dataset_path"] as? String, reboundURL.path)
            XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true,
                           "Unchanged rebound source evidence must reuse its immutable maps")
            _ = try await output(["cleanup-size", "--dataset", reboundURL.path])
        }
    }

    func testPreparedReuseAndTrainingRejectLegacyResampledMapsAndRebuildExactCrops() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 1, dimension: 512)
        let inputBytes = try Data(contentsOf: dataset.appendingPathComponent("Material 0/BaseColor.png"))
        let heightBytes = try Data(contentsOf: dataset.appendingPathComponent("Material 0/Height.png"))
        let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
        let prepared = try await output(arguments)
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let indexURL = preparedURL.appendingPathComponent("dataset.json"), index = try object(indexURL)
        let binding = try XCTUnwrap(index["native_size_preparation"] as? [String: Any])
        let proof = try XCTUnwrap(binding["source_snapshot"] as? [String: Any])
        let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sampleURL = preparedURL.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
        let originalSample = try object(sampleURL)
        for mutation in 0..<4 {
            var sample = originalSample
            var maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            var height = try XCTUnwrap(maps["height"])
            switch mutation {
            case 0:
                height["transforms"] = [["type": "native_resize", "algorithm": "bilinear"]]
            case 1:
                height["transforms"] = [["type": "native_crop", "algorithm": "exact_native_integer_codes",
                                         "range_normalization": false, "gamma_applied": true]]
            case 2:
                height["exact_source_crop"] = false
            default:
                var variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
                variants[0]["scaled"] = true; sample["input_variants"] = variants
            }
            maps["height"] = height; sample["map_metadata"] = maps
            try JSONSerialization.data(withJSONObject: sample).write(to: sampleURL)
            XCTAssertFalse(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof),
                           "Correct dimensions and checksums cannot authorize scaled or altered pixel values")
            XCTAssertThrowsError(try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height"))
        }
        try JSONSerialization.data(withJSONObject: originalSample).write(to: sampleURL)
        var scaledIndex = index, scaledBinding = binding
        scaledBinding["target_resized"] = true; scaledIndex["native_size_preparation"] = scaledBinding
        try JSONSerialization.data(withJSONObject: scaledIndex).write(to: indexURL)
        XCTAssertFalse(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof))
        XCTAssertThrowsError(try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height"))
        let refreshed = try await output(arguments)
        let refreshedURL = URL(fileURLWithPath: try XCTUnwrap(refreshed["dataset_path"] as? String))
        XCTAssertEqual((refreshed["preparation"] as? [String: Any])?["reused"] as? Bool, false)
        let refreshedIndex = try object(refreshedURL.appendingPathComponent("dataset.json"))
        let refreshedBinding = try XCTUnwrap(refreshedIndex["native_size_preparation"] as? [String: Any])
        let refreshedProof = try XCTUnwrap(refreshedBinding["source_snapshot"] as? [String: Any])
        XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(refreshedURL, size: 256, proof: refreshedProof))
        XCTAssertFalse(try NativeMaterialDatasetService.trainingSamples(datasetURL: refreshedURL, size: 256, target: "height").isEmpty)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("Material 0/BaseColor.png")), inputBytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("Material 0/Height.png")), heightBytes)
    }

    func testLargeSourcePreparationBudgetIncludesMappedInputAndAllRetainedCrops() throws {
        let gib = MachineResources.gibibyte
        let source: [String: Any] = ["width": 8192, "height": 8192, "channels": 3, "sample_bits": 16, "file_bytes": 363_824_405]
        let peak = try NativeMaterialDatasetService.preparationPeakBytes(sources: [source], size: 2048, cropCount: 3)
        // Covers an actual 8K RGB16 map plus three 2K crops and encoder buffers.
        XCTAssertGreaterThan(peak, 363_824_405 + 3 * 2048 * 2048 * 6)
        XCTAssertLessThan(peak, 700 * 1024 * 1024)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: gib, maximumWorkers: 128)
        XCTAssertEqual(try policy.workerCount(jobCount: 1000, estimatedPeakBytes: peak), 1)
        let tiny = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 128 * 1024 * 1024, maximumWorkers: 128)
        XCTAssertThrowsError(try tiny.workerCount(jobCount: 1000, estimatedPeakBytes: peak))
        XCTAssertThrowsError(try NativeMaterialDatasetService.preparationPeakBytes(sources: [["width": -1]], size: 2048, cropCount: 3))
    }

    func testParallelPreparationKeepsOrderedMetadataAndExactCropBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        let originalSources = try sourceSnapshot(dataset)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 1)
        let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
        let serial = try await preparationOutput(arguments, policy: policy)
        let serialRoot = URL(fileURLWithPath: try XCTUnwrap(serial["dataset_path"] as? String))
        let firstInventory = try preparedPNGBytes(serialRoot)
        let firstContracts = try preparedContracts(serialRoot)
        let firstIndex = try object(serialRoot.appendingPathComponent("dataset.json"))
        XCTAssertEqual((firstIndex["native_size_preparation"] as? [String: Any])?["preparation_workers"] as? Int, 1)
        _ = try await output(["cleanup-size", "--dataset", serialRoot.path])
        // Restore identical review inputs; cleanup deliberately persists the
        // generated crop review snapshot as a new per-size sidecar.
        let reviews = try NativeMaterialDatasetService.trainingCacheRoot(datasetURL: dataset).deletingLastPathComponent().appendingPathComponent(".material-size-reviews.json")
        if FileManager.default.fileExists(atPath: reviews.path) { try FileManager.default.removeItem(at: reviews) }
        let concurrent = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let events = PreparationEventRecorder()
        let parallel = try await preparationOutput(arguments, policy: concurrent, events: events)
        let parallelRoot = URL(fileURLWithPath: try XCTUnwrap(parallel["dataset_path"] as? String))
        let secondIndex = try object(parallelRoot.appendingPathComponent("dataset.json"))
        let binding = try XCTUnwrap(secondIndex["native_size_preparation"] as? [String: Any])
        XCTAssertEqual(binding["preparation_workers"] as? Int, 4)
        let peak = try XCTUnwrap(binding["estimated_worker_peak_bytes"] as? NSNumber).uint64Value
        XCTAssertLessThanOrEqual(peak * 4, concurrent.memoryBudgetBytes)
        XCTAssertEqual(try preparedPNGBytes(parallelRoot), firstInventory, "Worker count cannot change lossless map bytes")
        XCTAssertEqual(try preparedContracts(parallelRoot), firstContracts, "Worker count cannot change source hashes, map roles, native geometry or exact crop transforms")
        XCTAssertEqual(try JSONSerialization.data(withJSONObject: firstIndex["samples"]!, options: [.sortedKeys]),
                       try JSONSerialization.data(withJSONObject: secondIndex["samples"]!, options: [.sortedKeys]))
        XCTAssertEqual(try sourceSnapshot(dataset), originalSources)
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
        let originals = try sourceSnapshot(dataset)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let operation = Task {
            try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path, "--size", "512", "--target", "height"], preparationPolicy: policy)
        }
        let staging = try NativeMaterialDatasetService.trainingCacheRoot(datasetURL: dataset)
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
        XCTAssertEqual(try sourceSnapshot(dataset), originals)
        // Once the cancelled worker join returns, no background writer can
        // recreate files or keep the source dataset locked.
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--description", "After cancellation"])
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), [])
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

    func testNativePNGAcceptsSupportedTransparencyAndRejectsAmbiguousHeaders() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 0, count: 6), colorChunks: [("tRNS", Data(repeating: 0, count: 6))])
        let decoded = try NativePNG.decode(source.encoded())
        XCTAssertEqual(decoded.pixels, source.pixels)
        XCTAssertEqual(decoded.colorChunks.first?.1, source.colorChunks.first?.1)
        XCTAssertEqual(try decoded.modelFloatSamples(role: "input", encoding: "linear_rgb"), [Float](repeating: 0, count: 6))
        let malformed = NativePNG(header: source.header, pixels: source.pixels, colorChunks: [("tRNS", Data(repeating: 0, count: 2))])
        XCTAssertThrowsError(try malformed.encoded())
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

    func testPreparedCleanupAcceptsLateFinderMetadataAndPreservesUnrelatedData() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let prepared = root.appendingPathComponent("prepared"), child = prepared.appendingPathComponent("samples/crop")
        try FileManager.default.createDirectory(at: child, withIntermediateDirectories: true)
        try Data("owned manifest".utf8).write(to: prepared.appendingPathComponent("dataset.json"))
        try Data("owned crop".utf8).write(to: child.appendingPathComponent("height.png"))
        let finder = Data([0, 0, 0, 1, 0x42, 0x75, 0x64, 0x31] + [UInt8](repeating: 0, count: 16))
        try finder.write(to: child.appendingPathComponent(".DS_Store"))
        try NativeMaterialDatasetService.removePreparedTrainingDirectory(prepared,
            ownedFiles: ["dataset.json", "samples/crop/height.png"], ownedDirectories: ["samples", "samples/crop"],
            beforeFinalRemoval: { try finder.write(to: prepared.appendingPathComponent(".DS_Store")) })
        XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.path))
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
        let unrelated = Data("User notes".utf8)
        XCTAssertThrowsError(try NativeMaterialDatasetService.removePreparedTrainingDirectory(prepared, ownedFiles: [], ownedDirectories: [],
            beforeFinalRemoval: { try unrelated.write(to: prepared.appendingPathComponent(".DS_Store")) }))
        XCTAssertEqual(try Data(contentsOf: prepared.appendingPathComponent(".DS_Store")), unrelated)
    }

    func testCleanupResumesOwnedReceiptButDoesNotRemoveReusableDatasetCaches() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let original = root.appendingPathComponent("dataset"), prepared = original.appendingPathComponent(".training-data/256-recoverable")
        let cache = original.appendingPathComponent(".training-data/depthpro-inputs")
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let bytes = Data("Retained numeric cache".utf8), cached = cache.appendingPathComponent("fixture.safetensors")
        try bytes.write(to: cached)
        let receipt: [String: Any] = ["schema": "texture-studio-native-crop-cleanup-v2", "source_dataset_path": original.path,
            "prepared_dataset_path": prepared.path, "prepared_index_sha256": String(repeating: "a", count: 64),
            "owned_files_sha256": ["dataset.json": String(repeating: "a", count: 64),
                ".material-workbench.lock": NativeMaterialTrainer.checksum(Data())], "owned_directories": [String]()]
        try JSONSerialization.data(withJSONObject: receipt).write(to: prepared.appendingPathComponent(".native-crop-cleanup.json"))
        let finder = Data([0, 0, 0, 1, 0x42, 0x75, 0x64, 0x31] + [UInt8](repeating: 0, count: 16))
        try finder.write(to: prepared.appendingPathComponent(".DS_Store"))
        let result = try await output(["cleanup-size", "--dataset", prepared.path])
        XCTAssertEqual(result["removed"] as? Bool, true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.path))
        XCTAssertEqual(try Data(contentsOf: cached), bytes)
        let unowned = original.appendingPathComponent(".training-data/unowned")
        try FileManager.default.createDirectory(at: unowned, withIntermediateDirectories: true)
        var wrongReceipt = receipt; wrongReceipt["prepared_dataset_path"] = prepared.path
        try JSONSerialization.data(withJSONObject: wrongReceipt).write(to: unowned.appendingPathComponent(".native-crop-cleanup.json"))
        do { _ = try await output(["cleanup-size", "--dataset", unowned.path]); XCTFail("Cleanup receipt must match its exact path") }
        catch { XCTAssertTrue(FileManager.default.fileExists(atPath: unowned.path)) }
    }

    func testPartialChildCleanupResumesWithoutSampleManifestAndRejectsChangedOrForeignData() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for failure in ["foreign", "permission"] {
            let original = root.appendingPathComponent("sources-" + failure), sources = original
            try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
            let input = sources.appendingPathComponent("diffuse.png"), height = sources.appendingPathComponent("height.png")
            let inputBytes = try fixturePNG(width: 512, height: 512, bits: 8, channels: 3)
            let heightBytes = try fixturePNG(width: 512, height: 512, bits: 16, channels: 1)
            try inputBytes.write(to: input); try heightBytes.write(to: height)
            _ = try await output(["edit-dataset", "--dataset", original.path, "--training-size", "256"])
            _ = try await output(["add-material", "--dataset", original.path, "--name", "wall", "--input", input.path, "--height", height.path])
            let preparedInfo = try await output(["prepare-size", "--dataset", original.path, "--size", "256", "--target", "height"])
            let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
            let sample = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").first)
            let child = sample.targetURL.deletingLastPathComponent(), foreign = child.appendingPathComponent("user-notes.txt")
            let foreignBytes = Data("Keep these user notes".utf8)
            let indexURL = prepared.appendingPathComponent("dataset.json"), indexBytes = try Data(contentsOf: indexURL)
            do {
                _ = try NativeMaterialDatasetService.cleanup(prepared, beforeDirectoryRemoval: { directory in
                    guard directory == child else { return }
                    if failure == "foreign" { try foreignBytes.write(to: foreign) }
                    else { throw POSIXError(.EACCES) }
                })
                XCTFail("Injected child cleanup failure must be reported")
            } catch {
                if failure == "foreign" { XCTAssertTrue(error.localizedDescription.contains("unrelated files")) }
                else { XCTAssertEqual((error as? POSIXError)?.code, .EACCES) }
            }
            XCTAssertTrue(FileManager.default.fileExists(atPath: indexURL.path))
            XCTAssertFalse(FileManager.default.fileExists(atPath: child.appendingPathComponent("sample.json").path),
                           "Recovery must succeed after a child sample manifest has already been deleted")
            XCTAssertTrue(FileManager.default.fileExists(atPath: prepared.appendingPathComponent(".native-crop-cleanup.json").path))
            if failure == "foreign" {
                XCTAssertEqual(try Data(contentsOf: foreign), foreignBytes)
                do { _ = try await output(["cleanup-size", "--dataset", prepared.path]); XCTFail("Foreign contents must prevent recovery removal") }
                catch { XCTAssertEqual(try Data(contentsOf: foreign), foreignBytes) }
                try FileManager.default.removeItem(at: foreign)
                try Data("Changed index".utf8).write(to: indexURL)
                do { _ = try await output(["cleanup-size", "--dataset", prepared.path]); XCTFail("Remaining owned files must match their receipt hashes") }
                catch { XCTAssertTrue(error.localizedDescription.contains("contents changed")) }
                try indexBytes.write(to: indexURL)
                let link = child.appendingPathComponent("unrelated-link")
                try FileManager.default.createSymbolicLink(at: link, withDestinationURL: input)
                do { _ = try await output(["cleanup-size", "--dataset", prepared.path]); XCTFail("Recovery must retain symbolic-link protection") }
                catch { XCTAssertTrue(error.localizedDescription.contains("symbolic link")) }
                try FileManager.default.removeItem(at: link)
            }
            let recovered = try await output(["cleanup-size", "--dataset", prepared.path])
            XCTAssertEqual(recovered["removed"] as? Bool, true)
            XCTAssertFalse(FileManager.default.fileExists(atPath: prepared.path))
            XCTAssertEqual(try Data(contentsOf: input), inputBytes)
            XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        }
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

    func testFolderProjectionIdentifiesMissingAndUnsupportedTargetSources() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Sources"), sources = dataset
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Useful preview", "--training-size", "256"])
        let fixtures = [("complete", 16), ("low_precision", 8), ("missing_height", 0)]
        var heightPaths: [String: String] = [:], inputPaths: [String: String] = [:]
        for (name, heightBits) in fixtures {
            let folder = sources.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            let input = folder.appendingPathComponent(name + "_diff_1k.png")
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 3).write(to: input)
            inputPaths[name] = input.path
            if heightBits > 0 {
                let map = folder.appendingPathComponent(name + "_disp_1k.png")
                try fixturePNG(width: 512, height: 512, bits: heightBits, channels: 1).write(to: map)
                heightPaths[name] = map.path
            }
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 1).write(to: folder.appendingPathComponent(name + "_rough_1k.png"))
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 3).write(to: folder.appendingPathComponent(name + "_nor_gl_1k.png"))
        }
        let preview = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 3)
        XCTAssertEqual(preview["warnings"] as? [String], [], "Nominal filename resolutions must not become per-map warning spam")
        let targets = try XCTUnwrap(preview["training_plans"] as? [String: [String: Any]])
        let plan = try XCTUnwrap(targets["height"]), issues = try XCTUnwrap(plan["source_issues"] as? [[String: Any]])
        let missing = try XCTUnwrap(issues.first { $0["material_id"] as? String == "missing_height_1k" && $0["code"] as? String == "missing_target" })
        XCTAssertEqual(missing["source_path"] as? String, inputPaths["missing_height"])
        XCTAssertEqual(missing["target"] as? String, "height"); XCTAssertEqual(missing["crop_count"] as? Int, 1)
        XCTAssertFalse(try XCTUnwrap(missing["reason"] as? String).isEmpty)
        let lowPrecision = try XCTUnwrap(issues.first { $0["material_id"] as? String == "low_precision_1k" && $0["code"] as? String == "unsupported_precision" })
        XCTAssertEqual(lowPrecision["source_path"] as? String, heightPaths["low_precision"])
        XCTAssertEqual(lowPrecision["target"] as? String, "height"); XCTAssertEqual(lowPrecision["crop_count"] as? Int, 1)
        let precisionReason = try XCTUnwrap(lowPrecision["reason"] as? String)
        XCTAssertTrue(precisionReason.contains("8") && precisionReason.contains("16"))
        XCTAssertEqual(plan["train_count"] as? Int, 1)
        XCTAssertEqual(plan["unavailable_target_count"] as? Int, 2)
        for (target, expectedIDs) in [("height", ["complete_1k_center"]),
                                      ("roughness", ["complete_1k_center", "low_precision_1k_center", "missing_height_1k_center"]),
                                      ("normal", ["complete_1k_center", "low_precision_1k_center", "missing_height_1k_center"])] {
            let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(Set(descriptors.map(\.id)), Set(expectedIDs), "Only the affected target may exclude a source set")
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        }
    }

    func testMixedResolutionFolderProjectionKeepsTrueSizeDiagnostics() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Sources"), sources = dataset
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Mixed resolutions", "--training-size", "256"])
        var originals: [URL: Data] = [:]
        let fixtures = [("Rock Face 04", "rock_face_04", "1k", 128),
                        ("Rock Face 04", "rock_face_04", "2k", 256),
                        ("Rock Face 04", "rock_face_04", "4k", 512),
                        ("Small only", "small_only", "1k", 128),
                        ("Small only", "unrelated", "2k", 256)]
        for (directory, family, label, dimension) in fixtures {
            let folder = sources.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (suffix, bits, channels) in [("diff", 8, 3), ("disp", 16, 1), ("rough", 8, 1), ("nor_gl", 8, 3)] {
                let path = folder.appendingPathComponent("\(family)_\(suffix)_\(label).png")
                let bytes = try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let preview = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 5)
        XCTAssertEqual(preview["warnings"] as? [String], [])
        let plans = try XCTUnwrap(preview["training_plans"] as? [String: [String: Any]])
        for target in ["height", "roughness", "normal"] {
            let plan = try XCTUnwrap(plans[target])
            XCTAssertEqual(plan["train_count"] as? Int, 3)
            XCTAssertEqual(plan["undersized_source_set_count"] as? Int, 2, "Both smaller originals remain represented in the inventory")
            XCTAssertEqual(plan["smaller_alternate_source_set_count"] as? Int, 1)
            let issues = try XCTUnwrap(plan["source_issues"] as? [[String: Any]])
            XCTAssertEqual(issues.count, 1, "A usable sibling prevents a misleading small-alternate diagnostic")
            let issue = try XCTUnwrap(issues.first)
            XCTAssertEqual(issue["material_id"] as? String, "small_only_1k", "An unrelated family in the same folder cannot supply this material")
            XCTAssertEqual(issue["code"] as? String, "undersized")
            XCTAssertTrue(try XCTUnwrap(issue["reason"] as? String).contains("128 × 128"))
        }
        let entries = try NativeMaterialDatasetService.sourceFolderRecords(dataset).map(\.entry)
        XCTAssertEqual(Set(entries.compactMap { $0["material_id"] as? String }),
                       ["rock_face_04_1k", "rock_face_04_2k", "rock_face_04_4k", "small_only_1k", "unrelated_2k"])
        for target in ["height", "roughness", "normal"] {
            let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(Set(descriptors.map(\.id)), ["rock_face_04_2k_full", "rock_face_04_4k_center", "unrelated_2k_full"])
            for descriptor in descriptors {
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.inputURL)).header.width, 256)
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.targetURL)).header.height, 256)
            }
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, "Folder reads and preparation must preserve original pixels") }
    }

    func testLargerSiblingMissingTargetRemainsExcludedForThatTarget() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Rock Face 04"), sources = dataset
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Target availability", "--training-size", "256"])
        for (label, dimension, suffix, bits, channels) in [("1k", 128, "diff", 8, 3), ("1k", 128, "disp", 16, 1),
                                                          ("2k", 256, "diff", 8, 3), ("2k", 256, "rough", 8, 1)] {
            try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                .write(to: sources.appendingPathComponent("rock_face_04_\(suffix)_\(label).png"))
        }
        let preview = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let plans = try XCTUnwrap(preview["training_plans"] as? [String: [String: Any]])
        let height = try XCTUnwrap(plans["height"]), roughness = try XCTUnwrap(plans["roughness"])
        XCTAssertEqual(height["train_count"] as? Int, 0, "Larger pixels do not supply a missing training target")
        XCTAssertEqual(height["unavailable_target_count"] as? Int, 1)
        XCTAssertEqual(height["smaller_alternate_source_set_count"] as? Int, 1)
        let issues = try XCTUnwrap(height["source_issues"] as? [[String: Any]])
        XCTAssertEqual(issues.count, 1)
        XCTAssertEqual(issues.first?["material_id"] as? String, "rock_face_04_2k")
        XCTAssertEqual(issues.first?["code"] as? String, "missing_target")
        XCTAssertEqual(roughness["train_count"] as? Int, 1)
        XCTAssertEqual((roughness["source_issues"] as? [[String: Any]])?.count, 0)
    }

    func testValidationCropCountsPersistZeroAndValuesBeyondFormerUICaps() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        let created = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Validation preferences",
            "--validation-enabled", "no", "--validation-percent", "0", "--validation-max-crops", "0", "--validation-quick-count", "0"])
        let disabled = try XCTUnwrap(created["validation"] as? [String: Any])
        XCTAssertEqual(disabled["enabled"] as? Bool, false)
        XCTAssertEqual(disabled["percent"] as? Double, 0)
        XCTAssertEqual(disabled["max_crops"] as? Int, 0)
        XCTAssertEqual(disabled["quick_count"] as? Int, 0)

        for count in [100_001, Int.max, 0] {
            _ = try await output(["edit-dataset", "--dataset", dataset.path, "--validation-enabled", "yes",
                "--validation-max-crops", String(count), "--validation-quick-count", String(count)])
            let reopened = try await output(["dataset", "--dataset", dataset.path])
            let settings = try XCTUnwrap(reopened["validation"] as? [String: Any])
            XCTAssertEqual(settings["max_crops"] as? Int, count)
            XCTAssertEqual(settings["quick_count"] as? Int, count)
        }
        let accepted = try await output(["dataset", "--dataset", dataset.path])
        let acceptedBytes = try JSONSerialization.data(withJSONObject: accepted["validation"]!, options: [.sortedKeys])
        for field in ["--validation-max-crops", "--validation-quick-count"] {
            do {
                _ = try await output(["edit-dataset", "--dataset", dataset.path, field, "-1"])
                XCTFail("Negative crop counts are outside the action's range.")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("nonnegative crop counts"))
            }
            let reopened = try await output(["dataset", "--dataset", dataset.path])
            XCTAssertEqual(try JSONSerialization.data(withJSONObject: reopened["validation"]!, options: [.sortedKeys]), acceptedBytes)
        }
    }

    func testFolderPrepareCleanupAndPhysicalDeletionLifecycleKeepsOriginalCodes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Walls"), dataset = sources
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("diffuse.png"), height = sources.appendingPathComponent("height.png")
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 127, count: 512 * 512 * 3), colorChunks: [])
        let rawHeight = Data((0..<512 * 512 * 2).map { UInt8(truncatingIfNeeded: $0 * 97) })
        let numeric = NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0), pixels: rawHeight, colorChunks: [])
        let inputBytes = try rgb.encoded(), heightBytes = try numeric.encoded()
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Native", "--training-size", "256"])
        let imported = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let originalSources = try sourceSnapshot(dataset)
        let materials = try XCTUnwrap(imported["materials"] as? [[String: Any]])
        let sample = try XCTUnwrap((materials[0]["samples"] as? [[String: Any]])?.first)
        let sampleID = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--status", "approved", "--note", "Keep relief", "--expected-review-sha256", try XCTUnwrap(imported["review_sha256"] as? String)])
        XCTAssertEqual(try sourceSnapshot(dataset), originalSources, "Crop reviews do not modify original image bytes")
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        let preparedPath = try XCTUnwrap(prepared["dataset_path"] as? String), preparedURL = URL(fileURLWithPath: preparedPath)
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        let descriptor = try XCTUnwrap(descriptors.first)
        let preparedHeight = try NativePNG.decode(Data(contentsOf: descriptor.targetURL))
        XCTAssertEqual(preparedHeight.pixels, try numeric.crop([128, 128, 256, 256]).pixels)
        XCTAssertEqual(try preparedHeight.modelFloatSamples(role: "height").count, 256 * 256)
        let cropBytes = try Data(contentsOf: descriptor.targetURL)
        let reviewURL = try NativeMaterialDatasetService.trainingCacheRoot(datasetURL: dataset).deletingLastPathComponent().appendingPathComponent(".material-size-reviews.json")
        let reviewBytes = try Data(contentsOf: reviewURL)
        try Data([1, 2, 3]).write(to: descriptor.targetURL)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Modified crops need inspection before cleanup") }
        catch { XCTAssertTrue(error.localizedDescription.contains("crop contents changed")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: reviewURL), reviewBytes)
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
        let finderMetadata = preparedURL.appendingPathComponent(".DS_Store")
        let unrelatedBytes = Data("Unrelated user content, even with a Finder filename".utf8)
        try unrelatedBytes.write(to: finderMetadata)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("The Finder filename alone must not authorize unrelated data removal") }
        catch { XCTAssertTrue(error.localizedDescription.contains("unrelated files (.DS_Store)")) }
        XCTAssertEqual(try Data(contentsOf: finderMetadata), unrelatedBytes)
        try FileManager.default.removeItem(at: finderMetadata)
        try FileManager.default.createSymbolicLink(at: finderMetadata, withDestinationURL: input)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Finder metadata must not bypass symbolic-link protection") }
        catch { XCTAssertTrue(error.localizedDescription.contains("unexpected symbolic link")) }
        try FileManager.default.removeItem(at: finderMetadata)
        // This exact version/Bud1 signature is written by macOS Finder. The
        // disposable fixture is permitted at both the cache and sample level.
        let finderBytes = Data([0, 0, 0, 1, 0x42, 0x75, 0x64, 0x31, 0, 0, 0x10, 0, 0, 0, 8, 0])
        try finderBytes.write(to: finderMetadata)
        try finderBytes.write(to: descriptor.targetURL.deletingLastPathComponent().appendingPathComponent(".DS_Store"))
        let reused = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true)
        _ = try await output(["cleanup-size", "--dataset", preparedPath])
        XCTAssertFalse(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try sourceSnapshot(dataset), originalSources)
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        let inspectedMaterials = inspected["materials"] as? [[String: Any]]
        let inspectedSamples = inspectedMaterials?.first?["samples"] as? [[String: Any]]
        XCTAssertEqual(inspectedSamples?.first?["note"] as? String, "Keep relief")
        try FileManager.default.removeItem(at: height)
        let removed = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertFalse((removed["source_inventory"] as? [[String: Any]] ?? []).contains { $0["path"] as? String == height.path })
        XCTAssertEqual(removed["source_set_count"] as? Int, 0, "Deleted maps leave folder membership immediately")
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
    }

    func testRecognizedPNGMapsWithWrongExtensionsPrepareWithoutChangingOriginals() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Sources"), sources = dataset
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Misnamed PNG maps", "--training-size", "256"])
        let rgb = try fixturePNG(width: 512, height: 512, bits: 8, channels: 3)
        let heightPNG = try fixturePNG(width: 512, height: 512, bits: 16, channels: 1)
        let roughnessPNG = try fixturePNG(width: 512, height: 512, bits: 8, channels: 1)
        let layouts = [
            ("PolyHaven", "granite_tile_diff_1k.png", "granite_tile_disp_1k.txt", "granite_tile_rough_1k.txt"),
            ("AmbientCG", "Tiles014_1K-PNG_Color.txt", "Tiles014_1K-PNG_Displacement.txt", "Tiles014_1K-PNG_Roughness.png"),
            ("Generic", "diffuse.txt", "height.txt", "roughness.png")
        ]
        var originals: [URL: Data] = [:]
        for (directory, inputName, heightName, roughnessName) in layouts {
            let folder = sources.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            for (filename, bytes) in [(inputName, rgb), (heightName, heightPNG), (roughnessName, roughnessPNG)] {
                let file = folder.appendingPathComponent(filename); try bytes.write(to: file); originals[file] = bytes
            }
        }
        let canonicalHeight = sources.appendingPathComponent("PolyHaven/granite_tile_disp_1k.png")
        try heightPNG.write(to: canonicalHeight); originals[canonicalHeight] = heightPNG
        let preview = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 3)
        let notices = try XCTUnwrap(preview["warnings"] as? [String])
        for file in originals.keys where file.pathExtension == "txt" {
            XCTAssertTrue(notices.contains { ($0.contains(file.path) || $0.contains(file.resolvingSymlinksInPath().path)) && $0.contains(".txt extension") && $0.contains("detected and read as PNG") },
                          "Every misnamed PNG must identify its exact path and actual format, including identical duplicates")
        }
        let records = try NativeMaterialDatasetService.sourceFolderRecords(dataset)
        let materials = records.map(\.sample)
        XCTAssertEqual(materials.count, 3)
        for material in materials {
            let maps = try XCTUnwrap(material["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual(Set(maps.keys), ["input", "height", "roughness"])
            let heightPath = try XCTUnwrap(maps["height"]?["filename"] as? String)
            if material["material_id"] as? String == "granite_tile_1k" {
                XCTAssertEqual(URL(fileURLWithPath: heightPath).resolvingSymlinksInPath(), canonicalHeight.resolvingSymlinksInPath(),
                               "Identical target duplicates must keep one canonical source")
            } else { XCTAssertTrue(heightPath.hasSuffix(".txt"), "The original path must retain its actual extension") }
            XCTAssertEqual(maps["height"]?["sample_bits"] as? Int, 16)
        }

        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 3)
        let expected = try NativePNG.decode(heightPNG).crop([128, 128, 256, 256]).pixels
        for descriptor in descriptors { XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.targetURL)).pixels, expected) }
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
    }


    func testRecognizedNonPNGExtensionsStillRequireValidPNGContentAndChunkChecksums() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Sources"), sources = dataset
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Invalid originals", "--training-size", "256"])
        let input = try fixturePNG(width: 256, height: 256, bits: 8, channels: 3)
        var damagedPNG = try fixturePNG(width: 256, height: 256, bits: 16, channels: 1)
        damagedPNG[29] ^= 1
        let invalidSources = [("text", Data("This is text, not a PNG image.".utf8)), ("crc", damagedPNG)]
        var originals: [URL: Data] = [:]
        for (name, bytes) in invalidSources {
            let folder = sources.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            try input.write(to: folder.appendingPathComponent(name + "_diff_1k.png"))
            let file = folder.appendingPathComponent(name + "_disp_1k.txt")
            try bytes.write(to: file); originals[file] = bytes
        }
        let preview = try await output(["dataset", "--dataset", dataset.path, "--review-size", "256"])
        XCTAssertEqual(preview["source_set_count"] as? Int, 0)
        let notices = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertTrue(notices.contains { $0.contains(sources.appendingPathComponent("crc/crc_disp_1k.txt").path) },
                      "A PNG signature alone must not bypass full chunk validation")
        XCTAssertTrue(notices.contains { $0.contains(sources.appendingPathComponent("text/text_disp_1k.txt").path) },
                      "Invalid content must identify the exact original file that was rejected")
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
    }


    private func preparationFixture(_ root: URL, materials: Int, dimension: Int) async throws -> URL {
        let dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: dataset, withIntermediateDirectories: false)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--training-size", "256", "--validation-enabled", "no"])
        for number in 0..<materials {
            let folder = dataset.appendingPathComponent("Material \(number)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            let input = folder.appendingPathComponent("BaseColor.png"), height = folder.appendingPathComponent("Height.png")
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
            for (file, role) in [(input, "input"), (height, "height")] {
                try NativeSourceFileTags.update(file, role: role, classLabel: "Stone", classifierIdentity: "user")
            }
        }
        return dataset
    }
    private func sourceSnapshot(_ root: URL) throws -> [String: Data] {
        let files = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]))
        var snapshot: [String: Data] = [:]
        for case let file as URL in files where try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
            snapshot[file.path] = try Data(contentsOf: file)
        }
        return snapshot
    }

    private func preparationOutput(_ arguments: [String], policy: NativeMaterialDatasetService.PreparationPolicy, events: PreparationEventRecorder? = nil) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, preparationPolicy: policy, onEvent: { events?.append($0) })
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    private func preparedPNGBytes(_ root: URL) throws -> [String: Data] {
        let prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        var result: [String: Data] = [:]
        for case let file as URL in FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])! {
            guard file.pathExtension.lowercased() == "png", try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let relative = String(file.resolvingSymlinksInPath().standardizedFileURL.path.dropFirst(prefix.count))
            result[relative] = try Data(contentsOf: file)
        }
        return result
    }

    private func preparedContracts(_ root: URL) throws -> [String: Data] {
        let prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        var result: [String: Data] = [:]
        let geometry: Set<String> = ["sample_id", "material_id", "source_family_id", "source_set_id", "source_directory",
            "sample_pixel_dimensions", "source_pixel_dimensions", "source_square_rectangle_top_left_xywh",
            "crop_rectangle_top_left_xywh", "source_region_id", "crop_kind", "available_targets", "maps"]
        let mapFields: Set<String> = ["filename", "storage", "sample_sha256", "sample_bits", "channels", "encoding",
            "transforms", "exact_source_crop", "variant_id"]
        let sourceFields: Set<String> = ["path", "file_sha256", "width", "height", "sample_bits", "channels",
            "suffix", "resolution_label", "source_family_id", "variant_id"]
        func mapContract(_ map: [String: Any]) throws -> [String: Any] {
            var contract = map.filter { mapFields.contains($0.key) }
            let source = try XCTUnwrap(map["source"] as? [String: Any])
            XCTAssertEqual((source["file_sha256"] as? String)?.count, 64, "Each serial and parallel map remains bound to verified original bytes")
            contract["source"] = source.filter { sourceFields.contains($0.key) }
            return contract
        }
        for case let file as URL in FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])! {
            guard file.lastPathComponent == "sample.json" else { continue }
            let sample = try object(file)
            var contract = sample.filter { geometry.contains($0.key) }
            let maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual(Set(maps.keys), ["input", "height"])
            contract["map_metadata"] = try maps.mapValues(mapContract)
            let variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
            contract["input_variants"] = try variants.map(mapContract)
            let relative = String(file.resolvingSymlinksInPath().standardizedFileURL.path.dropFirst(prefix.count))
            // File verification timestamps and cleanup's review snapshots are
            // mutable bookkeeping. Compare the actual training contract separately.
            result[relative] = try JSONSerialization.data(withJSONObject: contract, options: [.sortedKeys])
        }
        return result
    }

    private func constantPNG(width: Int, height: Int, bits: Int, channels: Int) throws -> Data {
        try NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                   color: channels == 1 ? 0 : 2, interlace: 0),
                      pixels: Data(repeating: 97, count: width * height * channels * bits / 8), colorChunks: []).encoded()
    }

    private func fixturePNG(width: Int, height: Int, bits: Int, channels: Int) throws -> Data {
        let pixels = Data((0..<width * height * channels * bits / 8).map { UInt8(truncatingIfNeeded: $0 * 97) })
        return try NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                          color: channels == 1 ? 0 : 2, interlace: 0), pixels: pixels, colorChunks: []).encoded()
    }

    private func output(_ arguments: [String], sourceRecovery: NativeMaterialSourceRecovery? = nil) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, sourceRecovery: sourceRecovery)
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

private actor DatasetRecoveryMetadataProbe {
    private var requested: [String] = []
    func record(_ url: URL) { requested.append(url.absoluteString) }
    func urls() -> [String] { requested }
}

private final class SourceScanProbe: @unchecked Sendable {
    private let condition = NSCondition()
    private var active = 0, peak = 0, total = 0
    func begin(requireConcurrentReaders: Int) throws {
        condition.lock(); defer { condition.unlock() }
        active += 1; total += 1; peak = max(peak, active); condition.broadcast()
        let deadline = Date().addingTimeInterval(5)
        while total < requireConcurrentReaders {
            if !condition.wait(until: deadline) { active -= 1; throw StudioError("Source scan did not inspect files concurrently.") }
        }
    }
    func end() { condition.lock(); active -= 1; condition.broadcast(); condition.unlock() }
    func waitForReaders(_ minimum: Int) -> Bool {
        condition.lock(); defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(5)
        while total < minimum { if !condition.wait(until: deadline) { return false } }
        return true
    }
    var activeReaders: Int { condition.lock(); defer { condition.unlock() }; return active }
    var peakReaders: Int { condition.lock(); defer { condition.unlock() }; return peak }
    var totalReaders: Int { condition.lock(); defer { condition.unlock() }; return total }
}
