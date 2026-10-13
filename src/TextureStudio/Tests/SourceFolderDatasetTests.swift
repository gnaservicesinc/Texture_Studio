import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class SourceFolderDatasetTests: XCTestCase {
    func testArbitraryImageRenamesKeepNamedPairsAndTagsWithoutKeepingRemovedFiles() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let template = try writeMaterial(root, name: "Template")
        let source = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: false)
        for family in ["brick_wall_001", "brick_wall_002"] {
            try FileManager.default.copyItem(at: template.appendingPathComponent("BaseColor.png"),
                to: source.appendingPathComponent(family + "_diff_1k.png"))
            try FileManager.default.copyItem(at: template.appendingPathComponent("Height.png"),
                to: source.appendingPathComponent(family + "_disp_1k.png"))
        }
        let diffuse = source.appendingPathComponent("brick_wall_001_diff_1k.png")
        let height = source.appendingPathComponent("brick_wall_001_disp_1k.png")
        try NativeSourceFileTags.update(diffuse, classLabel: "Bricks", classifierIdentity: "user")
        let before = try await read(source, size: 256)
        XCTAssertEqual((before["original_materials"] as? [Any])?.count, 2)
        let diffuseBytes = try Data(contentsOf: diffuse), heightBytes = try Data(contentsOf: height)
        let renamedDiffuse = source.appendingPathComponent("My renamed photo.png")
        let renamedHeight = source.appendingPathComponent("My renamed depth.png")
        try FileManager.default.moveItem(at: diffuse, to: renamedDiffuse)
        try FileManager.default.moveItem(at: height, to: renamedHeight)
        let renamed = try await read(source, size: 256)
        XCTAssertEqual((renamed["original_materials"] as? [Any])?.count, 2,
            "Two equal-sized families keep their own pairs after arbitrary image renames")
        XCTAssertEqual((renamed["source_inventory"] as? [Any])?.count, 4)
        XCTAssertEqual(NativeSourceFileTags.read(renamedDiffuse)?.classLabel, "Bricks")
        XCTAssertEqual(NativeSourceFileTags.read(renamedHeight)?.role, "height")
        XCTAssertEqual(try Data(contentsOf: renamedDiffuse), diffuseBytes)
        XCTAssertEqual(try Data(contentsOf: renamedHeight), heightBytes)
        try FileManager.default.moveItem(at: renamedDiffuse, to: root.appendingPathComponent("Moved photo.png"))
        let removed = try await read(source, size: 256)
        XCTAssertEqual((removed["original_materials"] as? [Any])?.count, 1)
        XCTAssertEqual((removed["source_inventory"] as? [Any])?.count, 3,
            "Remembered pairing names cannot create an absent physical member")
    }

    func testObsoleteManifestCannotHideAddedMaterialsOrKeepRemovedLastMaterial() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let original = try writeMaterial(sources, name: "Physical Material")
        let manifest = sources.appendingPathComponent("dataset.json")
        let legacy: [String: Any] = ["schema_version": 2, "generator": "ipde-material-dataset-v2",
            "dataset_management": ["schema": "texture-studio-dataset-management-v1", "owns_directory": true],
            "storage_policy": "original-source-references-v1", "samples": [["material_id": "ghost", "sample_id": "ghost_full", "path": "samples/ghost_full"]]]
        let bytes = try JSONSerialization.data(withJSONObject: legacy)
        try bytes.write(to: manifest)
        let added = try await read(sources, size: 256)
        XCTAssertEqual((added["original_materials"] as? [Any])?.count, 1, "A saved listing cannot omit physical folders")
        try FileManager.default.moveItem(at: original, to: root.appendingPathComponent("Moved Out"))
        let removed = try await read(sources, size: 256)
        XCTAssertEqual((removed["original_materials"] as? [Any])?.count, 0, "The last removed material cannot survive as a manifest ghost")
        XCTAssertEqual((removed["source_inventory"] as? [Any])?.count, 0)
        XCTAssertEqual(try Data(contentsOf: manifest), bytes, "Folder inspection does not rewrite unrelated metadata")
    }

    func testFolderIsDatasetAndMembershipFollowsAddedAndMovedDirectoriesWithoutMetadata() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let first = try await read(sources)
        XCTAssertEqual(first["is_source_folder"] as? Bool, true)
        XCTAssertEqual(first["training_size"] as? Int, 2048)
        XCTAssertEqual((first["original_materials"] as? [Any])?.count, 0)
        let material = try writeMaterial(sources, name: "Carpet")
        let added = try await read(sources)
        XCTAssertEqual((added["original_materials"] as? [Any])?.count, 1)
        XCTAssertEqual((added["source_inventory"] as? [Any])?.count, 3)
        XCTAssertNotEqual(added["index_sha256"] as? String, first["index_sha256"] as? String)
        XCTAssertEqual((added["materials"] as? [Any])?.count, 0, "Small originals remain visible, never mislabeled as 2048 crops")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: sources.path), ["Carpet"])
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: material.path)), ["BaseColor.png", "Height.png", "AO.png"])
        try FileManager.default.moveItem(at: material, to: root.appendingPathComponent("MovedOut"))
        let removed = try await read(sources)
        XCTAssertEqual((removed["original_materials"] as? [Any])?.count, 0)
        XCTAssertEqual((removed["source_inventory"] as? [Any])?.count, 0)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: sources.path), [])
    }

    func testCreatingSourceFolderCreatesOnlyAnEmptyDirectory() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("New Folder")
        let result = try await command(["create-dataset", "--dataset", folder.path, "--name", "New Folder", "--source-folder", "true"])
        XCTAssertEqual(result["is_source_folder"] as? Bool, true)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), [])
    }

    func testCreatedFolderKeepsRequestedGridAndValidationPreferencesOnNextRead() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("Configured Folder")
        _ = try await command(["create-dataset", "--dataset", folder.path, "--source-folder", "true",
            "--training-size", "1024", "--validation-enabled", "no"])
        let reopened = try await read(folder)
        XCTAssertEqual(reopened["training_size"] as? Int, 1024)
        XCTAssertEqual((reopened["validation"] as? [String: Any])?["enabled"] as? Bool, false)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path), [])
    }

    func testUnownedProjectionMarkersNeverRestoreSavedMaterialLists() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let prepared = root.appendingPathComponent("Unowned Cache")
        try FileManager.default.createDirectory(at: prepared, withIntermediateDirectories: false)
        let marker: [String: Any] = ["native_size_preparation": ["schema": "texture-studio-native-crops-v1",
            "source_dataset_path": root.path, "target_resized": false],
            "samples": [["material_id": "ghost", "path": "samples/ghost"]]]
        try JSONSerialization.data(withJSONObject: marker).write(to: prepared.appendingPathComponent("dataset.json"))
        _ = try writeMaterial(prepared, name: "Physical Material")
        let physical = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: prepared, size: 256)
        XCTAssertEqual((physical["original_materials"] as? [Any])?.count, 1, "Loose native-cache metadata cannot replace the ordinary physical folder")
        let ownedLocation = root.appendingPathComponent(".training-data/Invalid Cache")
        try FileManager.default.createDirectory(at: ownedLocation, withIntermediateDirectories: true)
        var invalid = marker
        invalid["native_size_preparation"] = ["schema": "texture-studio-native-crops-v1", "source_dataset_path": root.appendingPathComponent("Unrelated").path, "target_resized": false]
        try JSONSerialization.data(withJSONObject: invalid).write(to: ownedLocation.appendingPathComponent("dataset.json"))
        XCTAssertThrowsError(try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: ownedLocation, size: 256))
        let workspace = root.appendingPathComponent("Workspace"), category = workspace.appendingPathComponent("datasets/wood")
        try FileManager.default.createDirectory(at: category, withIntermediateDirectories: true)
        try JSONSerialization.data(withJSONObject: ["samples": [["material_id": "ghost", "path": "samples/ghost"]]])
            .write(to: category.appendingPathComponent("dataset.json"))
        try JSONSerialization.data(withJSONObject: ["schema": "texture-studio-material-specialists-v1",
            "specialist_id": "wood", "workspace_path": workspace.path])
            .write(to: category.appendingPathComponent(".specialist-binding.json"))
        XCTAssertThrowsError(try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: category, size: 256))
    }

    func testPreparedMapsFollowCurrentTargetRolesReviewAndPhysicalMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let material = try writeMaterial(folder, name: "Current")
        let target = material.appendingPathComponent("Height.png")
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: folder, size: 256,
            target: "height", automaticValidation: false)
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 1)
        try NativeSourceFileTags.update(target, role: "roughness")
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 0)
        try NativeSourceFileTags.update(target, role: "height", reviewStatus: "excluded")
        let excludedTarget = try await read(folder, size: 256)
        let plans = try XCTUnwrap(excludedTarget["training_plans"] as? [String: [String: Any]])
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 0)
        XCTAssertEqual((excludedTarget["source_inventory"] as? [Any])?.count, 3, "Excluded target maps remain physically visible")
        XCTAssertThrowsError(try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: folder, size: 256,
            target: "height", automaticValidation: false))
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 0)
        try NativeSourceFileTags.update(target, reviewStatus: "approved")
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 1)
        try FileManager.default.moveItem(at: material, to: root.appendingPathComponent("Moved Out"))
        XCTAssertEqual(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").count, 0)
        let current = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: prepared, size: 256)
        XCTAssertEqual((current["materials"] as? [Any])?.count, 0)
        XCTAssertEqual((current["original_materials"] as? [Any])?.count, 0)
    }

    func testRewritingLargerOriginalAtSamePathCannotTrainStalePreparedPixels() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let material = try writeMaterial(folder, name: "Replaceable", width: 512, height: 512)
        let target = material.appendingPathComponent("Height.png")
        let prepared = try NativeMaterialDatasetService.prepareTrainingDataset(datasetURL: folder, size: 256,
            target: "height", automaticValidation: false)
        XCTAssertFalse(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height").isEmpty)
        let original = try NativePNG.decode(Data(contentsOf: target))
        var changed = original.pixels; changed[0] ^= 255
        try NativePNG(header: original.header, pixels: changed, colorChunks: original.colorChunks).encoded().write(to: target)
        XCTAssertThrowsError(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")) { error in
            XCTAssertTrue(error.localizedDescription.contains("Prepare fresh training crops"))
        }
        let current = try NativeMaterialDatasetService.readOnlyDatasetInfo(datasetURL: folder, size: 256)
        XCTAssertEqual((current["original_materials"] as? [Any])?.count, 1)
    }

    func testNamedTargetTagsDetermineInventoryRoleAndManualCollisionsRemainVisible() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let folder = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let material = try writeMaterial(folder, name: "Named")
        let input = material.appendingPathComponent("wood_diff_1k.png"), target = material.appendingPathComponent("wood_disp_1k.png")
        try FileManager.default.moveItem(at: material.appendingPathComponent("BaseColor.png"), to: input)
        try FileManager.default.moveItem(at: material.appendingPathComponent("Height.png"), to: target)
        try NativeSourceFileTags.update(target, role: "roughness")
        let tagged = try await read(folder, size: 256)
        let catalog = try XCTUnwrap(tagged["source_inventory"] as? [[String: Any]])
        XCTAssertEqual(catalog.first { $0["path"] as? String == target.path }?["role"] as? String, "roughness")
        let extra = material.appendingPathComponent("AO.png")
        _ = try await command(["add-material", "--dataset", folder.path, "--name", "Named",
            "--input", input.path, "--height", target.path, "--roughness", extra.path])
        try NativeSourceFileTags.update(extra, role: "height")
        let ambiguous = try await read(folder, size: 256)
        XCTAssertEqual((ambiguous["source_inventory"] as? [Any])?.count, 3, "Ambiguous teaching cannot hide originals")
        XCTAssertEqual((ambiguous["original_materials"] as? [Any])?.count, 0, "An ambiguous target cannot be selected by dictionary order")
        XCTAssertTrue((ambiguous["warnings"] as? [String] ?? []).contains { $0.contains("multiple maps to height") })
    }

    func testSixteenBitDepthAndOddEdgesSurviveTrainingPreparationOutsideSourceFolder() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let material = try writeMaterial(sources, name: "depth", width: 260, height: 258)
        let input = material.appendingPathComponent("BaseColor.png"), height = material.appendingPathComponent("Height.png")
        let originalInput = try Data(contentsOf: input), originalHeight = try Data(contentsOf: height)
        let opened = try await read(sources, size: 256)
        let id = try XCTUnwrap((opened["original_materials"] as? [[String: Any]])?.first?["material_id"] as? String)
        let output = try await command(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height", "--material", id])
        XCTAssertEqual(output["is_source_folder"] as? Bool, true)
        let preparation = try XCTUnwrap(output["preparation"] as? [String: Any])
        XCTAssertEqual(preparation["source_dataset_path"] as? String, sources.path)
        let prepared = URL(fileURLWithPath: try XCTUnwrap(output["dataset_path"] as? String))
        XCTAssertFalse(prepared.path.hasPrefix(sources.path + "/"))
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(samples.isEmpty)
        for sample in samples {
            let png = try NativePNG.decode(Data(contentsOf: sample.targetURL))
            XCTAssertEqual(png.header.bits, 16)
            XCTAssertEqual(png.header.width, 256); XCTAssertEqual(png.header.height, 256)
            let region = try XCTUnwrap(sample.cropRectangle)
            let expected = UInt16((region[0] + region[1] * 260) % 65536)
            XCTAssertEqual(png.pixels[0], UInt8(expected >> 8)); XCTAssertEqual(png.pixels[1], UInt8(expected & 255))
            let tensor = try png.modelFloatSamples(role: "height")
            XCTAssertEqual(tensor[0], Float(expected) / 65535, accuracy: 0.0000001)
        }
        XCTAssertEqual(try Data(contentsOf: input), originalInput)
        XCTAssertEqual(try Data(contentsOf: height), originalHeight)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: sources.path), ["depth"])
        let originals = try XCTUnwrap(output["original_materials"] as? [[String: Any]])
        XCTAssertEqual(originals.count, 1, "Preparing a target must retain the original browser")
        XCTAssertEqual((output["source_inventory"] as? [Any])?.count, 3, "Supplemental AO remains available after preparation")
        _ = try NativeMaterialDatasetService.cleanup(prepared)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: sources.path), ["depth"])
    }

    func testDirectoryObserverReportsMembershipChangeWithoutWritingFiles() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let changed = expectation(description: "Folder membership event")
        changed.assertForOverFulfill = false
        let observer = SourceFolderObserver(root: root) { changed.fulfill() }
        try await Task.sleep(for: .milliseconds(150))
        try FileManager.default.createDirectory(at: root.appendingPathComponent("Added"), withIntermediateDirectories: false)
        await fulfillment(of: [changed], timeout: 3)
        withExtendedLifetime(observer) {}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["Added"])
    }

    func testExcludingOneCropKeepsItsOriginalAndOtherSourcesTrainableAcrossPreparationAndCleanup() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let first = try writeMaterial(sources, name: "First", width: 260, height: 258)
        _ = try writeMaterial(sources, name: "Second", width: 260, height: 258)
        let opened = try await read(sources, size: 256)
        let crops = (opened["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        let selected = try XCTUnwrap(crops.first)
        let id = try XCTUnwrap(selected["sample_id"] as? String)
        let reviewed = try await command(["curate", "--dataset", sources.path, "--sample", id, "--review-size", "256", "--status", "excluded", "--note", "Only this crop", "--expected-review-sha256", try XCTUnwrap(opened["review_sha256"] as? String)])
        let originals = (reviewed["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(originals.count, 2)
        XCTAssertTrue(originals.allSatisfy { $0["status"] as? String == "unreviewed" })
        XCTAssertNil(NativeSourceFileTags.read(first.appendingPathComponent("BaseColor.png"))?.reviewStatus)
        let reviewedCrops = (reviewed["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(reviewedCrops.first { $0["sample_id"] as? String == id }?["status"] as? String, "excluded")
        XCTAssertEqual(reviewedCrops.first { $0["sample_id"] as? String == id }?["note"] as? String, "Only this crop")
        XCTAssertEqual(reviewedCrops.filter { $0["status"] as? String == "excluded" }.count, 1)
        let preparedInfo = try await command(["prepare-size", "--dataset", sources.path, "--size", "256", "--target", "height"])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(preparedInfo["dataset_path"] as? String))
        let training = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(training.isEmpty)
        XCTAssertFalse(training.contains { $0.id == id })
        _ = try NativeMaterialDatasetService.cleanup(prepared)
        let reopened = try await read(sources, size: 256)
        let persisted = (reopened["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(persisted.first { $0["sample_id"] as? String == id }?["status"] as? String, "excluded")
        let reopenedOriginals = (reopened["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(reopenedOriginals.allSatisfy { $0["status"] as? String == "unreviewed" })
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: sources.path)), ["First", "Second"])
    }

    func testFolderValidationPreferenceIsValidatedAndCanReturnToAutomatic() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let material = try writeMaterial(sources, name: "Wide", width: 768, height: 768)
        let disabled = try await command(["edit-dataset", "--dataset", sources.path, "--training-size", "256", "--validation-percent", "100", "--validation-subject", material.path, "--subject-validation", "disabled"])
        let disabledSettings = try XCTUnwrap(disabled["validation"] as? [String: Any])
        XCTAssertEqual((disabledSettings["folders"] as? [String: Bool])?[material.path], false)
        let enabled = try await command(["edit-dataset", "--dataset", sources.path, "--validation-subject", material.path, "--subject-validation", "enabled"])
        XCTAssertEqual(((enabled["validation"] as? [String: Any])?["folders"] as? [String: Bool])?[material.path], true)
        XCTAssertEqual((enabled["subjects"] as? [[String: Any]])?.first?["selected"] as? Bool, true)
        let automatic = try await command(["edit-dataset", "--dataset", sources.path, "--validation-subject", material.path, "--subject-validation", "automatic"])
        XCTAssertNil(((automatic["validation"] as? [String: Any])?["folders"] as? [String: Bool])?[material.path])
        do {
            _ = try await command(["edit-dataset", "--dataset", sources.path, "--validation-subject", root.appendingPathComponent("Missing").path, "--subject-validation", "enabled"])
            XCTFail("A disappeared folder cannot be configured for validation")
        } catch { XCTAssertTrue(error.localizedDescription.contains("no longer")) }
    }

    func testCropSplitCanBeManuallyAssignedAndResetWithoutChangingOriginalReview() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try writeMaterial(sources, name: "First")
        _ = try writeMaterial(sources, name: "Second")
        let opened = try await read(sources, size: 256)
        let crops = (opened["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        let id = try XCTUnwrap(crops.first?["sample_id"] as? String)
        let held = try await command(["curate", "--dataset", sources.path, "--sample", id, "--review-size", "256", "--status", "approved", "--split", "validation"])
        let heldCrops = (held["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(heldCrops.count, 2)
        XCTAssertEqual(heldCrops.first { $0["sample_id"] as? String == id }?["split"] as? String, "validation")
        XCTAssertEqual(heldCrops.first { $0["sample_id"] as? String == id }?["split_assignment"] as? String, "manual")
        let originals = (held["original_materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(originals.allSatisfy { $0["status"] as? String == "unreviewed" })
        let automatic = try await command(["curate", "--dataset", sources.path, "--sample", id, "--review-size", "256", "--status", "approved", "--split", "automatic"])
        let reset = (automatic["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertEqual(reset.first { $0["sample_id"] as? String == id }?["split"] as? String, "train")
        XCTAssertEqual(reset.first { $0["sample_id"] as? String == id }?["status"] as? String, "approved")
    }

    func testDeveloperCorpusOpensAsSourceFolderUnderOneSecondWithoutWritingMetadata() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_SOURCE_FOLDER_PROOF"] == "1" else { throw XCTSkip("Read-only real source-folder proof") }
        let source = URL(fileURLWithPath: "/opt/ipde/sources_mats")
        let before = try treePaths(source)
        let start = ProcessInfo.processInfo.systemUptime
        let info = try await read(source)
        let seconds = ProcessInfo.processInfo.systemUptime - start
        let originals = try XCTUnwrap(info["original_materials"] as? [[String: Any]])
        let catalog = try XCTUnwrap(info["source_inventory"] as? [[String: Any]])
        XCTAssertGreaterThanOrEqual(originals.count, 459)
        let pngCount = before.filter { $0.lowercased().hasSuffix(".png") }.count
        XCTAssertGreaterThanOrEqual(pngCount, 1885)
        XCTAssertEqual(catalog.filter { ($0["path"] as? String)?.lowercased().hasSuffix(".png") == true }.count, pngCount)
        XCTAssertEqual(try treePaths(source), before)
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("dataset.json").path))
        let crops = (info["materials"] as? [[String: Any]] ?? []).flatMap { $0["samples"] as? [[String: Any]] ?? [] }
        XCTAssertTrue(crops.allSatisfy { $0["width"] as? Int == 2048 && $0["height"] as? Int == 2048 })
        try JSONSerialization.data(withJSONObject: ["source_folder": source.path, "seconds": seconds,
            "original_sets": originals.count, "accounted_files": catalog.count, "exact_2048_crop_views": crops.count,
            "source_tree_unchanged": true, "required_source_metadata": false], options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: "/tmp/texture-source-folder-proof.json"))
        XCTAssertLessThan(seconds, 1)
    }

    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("source-folder-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        return root
    }
    private func writeMaterial(_ root: URL, name: String, width: Int = 256, height: Int = 256) throws -> URL {
        let folder = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
        let input = NativePNG(header: .init(width: width, height: height, bits: 8, channels: 3, color: 2, interlace: 0),
            pixels: Data(repeating: 93, count: width * height * 3), colorChunks: [])
        var codes = Data(count: width * height * 2)
        for i in 0..<(width * height) { let code = UInt16(i % 65536); codes[i * 2] = UInt8(code >> 8); codes[i * 2 + 1] = UInt8(code & 255) }
        let depth = NativePNG(header: .init(width: width, height: height, bits: 16, channels: 1, color: 0, interlace: 0), pixels: codes, colorChunks: [])
        try input.encoded().write(to: folder.appendingPathComponent("BaseColor.png"))
        try depth.encoded().write(to: folder.appendingPathComponent("Height.png"))
        try NativePNG(header: .init(width: width, height: height, bits: 8, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: 255, count: width * height), colorChunks: [])
            .encoded().write(to: folder.appendingPathComponent("AO.png"))
        return folder
    }
    private func command(_ arguments: [String]) async throws -> [String: Any] {
        let result = try await NativeMaterialDatasetService.run(arguments: arguments + ["--skip-source-recovery"])
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(result).utf8)) as? [String: Any])
    }
    private func read(_ root: URL, size: Int = 2048) async throws -> [String: Any] { try await command(["dataset", "--dataset", root.path, "--review-size", String(size)]) }
    private func treePaths(_ root: URL) throws -> Set<String> {
        let files = try XCTUnwrap(FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil))
        return Set(files.compactMap { ($0 as? URL)?.path })
    }
}
