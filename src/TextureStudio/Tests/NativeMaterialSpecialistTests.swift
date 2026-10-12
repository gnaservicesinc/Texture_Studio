import CryptoKit
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialSpecialistTests: XCTestCase {
    func testFileTagsReusePredictionsAndManualTeachingWinsUntilTheFileChanges() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("cedar", size: 256, value: 31, to: dataset, under: root)
        let input = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "input.png" }?.0)
        let probe = PredictionProbe()
        let classifier = fake(labels: ["Wood", "Paper"], probe: probe, choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: root.appendingPathComponent("FirstWorkspace"), classifier: classifier)
        XCTAssertEqual(probe.paths, [input.path])
        XCTAssertEqual(try specialist("Wood", initial).materialIds.count, 1)
        let automatic = try XCTUnwrap(NativeSourceFileTags.read(input))
        XCTAssertEqual(automatic.classLabel, "Wood")
        XCTAssertEqual(automatic.classifierIdentity, classifier.sha256)
        XCTAssertEqual(automatic.verifiedSHA256?.count, 64)

        // A fresh workspace proves reuse comes from the file, independently
        // of any prior specialist assignment or loaded Core ML prediction.
        let secondWorkspace = root.appendingPathComponent("SecondWorkspace")
        let tagged = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1)
        XCTAssertEqual(try specialist("Wood", tagged).materialIds.count, 1)
        try NativeSourceFileTags.update(input, classLabel: "Paper", classifierIdentity: "user")
        let taught = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "Teaching a file must override cached classifier assignments without another prediction")
        XCTAssertEqual(try specialist("Paper", taught).assignments.first?.source, "user_file_tag")
        XCTAssertEqual(try specialist("Paper", taught).materialIds.count, 1)
        XCTAssertEqual(NativeSourceFileTags.read(input)?.classifierIdentity, "user")
        XCTAssertEqual(NativeSourceFileTags.read(input)?.classLabel, "Paper")
        XCTAssertEqual(try Data(contentsOf: input), try XCTUnwrap(originals.first { $0.0 == input }?.1))
        try NativeSourceFileTags.update(input, classLabel: "Handmade Finish", classifierIdentity: "user")
        let custom = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: root.appendingPathComponent("CustomTaughtWorkspace"), classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1)
        XCTAssertEqual(try specialist("Handmade Finish", custom).materialIds.count, 1,
                       "User teaching may define a class beyond the classifier's original taxonomy")

        try png(size: 256, value: 103).write(to: input)
        XCTAssertNil(NativeSourceFileTags.read(input), "Changed original bytes invalidate file teaching and cached classifier output")
        _ = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset,
            workspaceURL: secondWorkspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "An altered registered original must fail its source binding before any prediction")
    }

    func testBareSourceFolderClassificationChoosesTwoKOriginalAndPreservesSixteenBitTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("SourceFolders"), woodFolder = source.appendingPathComponent("Wood")
        let smallFolder = source.appendingPathComponent("SmallOnly")
        try FileManager.default.createDirectory(at: woodFolder, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: smallFolder, withIntermediateDirectories: true)
        var originals: [URL: Data] = [:]
        for (folder, family, dimension, label) in [(woodFolder, "wood", 1024, "1k"),
                                                  (woodFolder, "wood", 2048, "2k"),
                                                  (smallFolder, "small", 1024, "1k")] {
            for (suffix, bits, channels) in [("diff", 8, 3), ("disp", 16, 1)] {
                let path = folder.appendingPathComponent("\(family)_\(suffix)_\(label).png")
                let bytes = try png(size: dimension, value: suffix == "disp" ? 171 : 83, bits: bits, channels: channels)
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let beforeFolders = Set(try FileManager.default.contentsOfDirectory(atPath: source.path))
        let probe = PredictionProbe(), workspace = root.appendingPathComponent("Specialists")
        let classifier = fake(labels: ["Wood", "Paper"], probe: probe, choose: { _ in "Wood" })
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths, [woodFolder.appendingPathComponent("wood_diff_2k.png").path],
                       "A new source-folder dataset must not classify one-K originals when the classifier grid is two K")
        let woodSpecialist = try specialist("Wood", registry)
        XCTAssertEqual(Set(woodSpecialist.materialIds), ["wood_1k", "wood_2k"])
        XCTAssertEqual(woodSpecialist.assignments.first?.inputPath, probe.paths.first)
        XCTAssertTrue(registry.warnings.contains { $0.contains("small") && $0.contains("2048") })
        let specialistDataset = URL(fileURLWithPath: woodSpecialist.datasetPath)
        let index = try object(specialistDataset.appendingPathComponent("dataset.json"))
        let entries = try XCTUnwrap(index["samples"] as? [[String: Any]])
        XCTAssertEqual(entries.count, 2)
        for entry in entries {
            let sample = try object(specialistDataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json"))
            let sourceMetadata = try XCTUnwrap((sample["map_metadata"] as? [String: [String: Any]])?["height"]?["source"] as? [String: Any])
            XCTAssertEqual(sourceMetadata["sample_bits"] as? Int, 16)
            let originalURL = URL(fileURLWithPath: try XCTUnwrap(sourceMetadata["path"] as? String))
            XCTAssertEqual(try NativePNG.inspect(originalURL).bits, 16)
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("dataset.json").path))
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath: source.path)), beforeFolders)
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes) }
        let oneK = woodFolder.appendingPathComponent("wood_diff_1k.png")
        try NativeSourceFileTags.update(oneK, classLabel: "Paper", classifierIdentity: "user")
        let taughtFamily = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "A user's label on another native tier overrides the cached family prediction")
        XCTAssertEqual(Set(try specialist("Paper", taughtFamily).materialIds), ["wood_1k", "wood_2k"])
        try NativeSourceFileTags.update(oneK, classLabel: "", classifierIdentity: "user")
        let unlabeled = try await NativeMaterialSpecialistService.refresh(datasetURL: source, workspaceURL: workspace,
            classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "Explicitly clearing a label must not trigger another automatic prediction")
        XCTAssertTrue(unlabeled.specialists.flatMap(\.assignments).isEmpty)
        XCTAssertTrue(unlabeled.warnings.contains { $0.contains("unlabeled") })
    }

    func testRealCompactSaveRegistersApprovesAndRoutesOnlyTheIntendedSpecialist() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("wood_planks", size: 256, value: 83, to: dataset, under: root)
        let paperOriginals = try await add("paper", size: 256, value: 48, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists")
        let classifier = fake(labels: ["Wood", "Paper"], choose: { $0.path.contains("/paper/") ? "Paper" : "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial)
        let membership = try XCTUnwrap(wood.targets.first { $0.target == "height" }?.membershipSha256)
        let output = root.appendingPathComponent("Trained")
        let text = try await NativeMaterialTrainer.run(arguments: ["train", "--dataset", wood.datasetPath,
            "--output", output.path, "--model-family", "compact-scalar", "--target", "height", "--size", "256",
            "--compact-width", "16", "--updates-per-map", "1", "--randomize-crops", "never", "--seed", "19"])
        let trained = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        let updates = try XCTUnwrap(trained["completed_updates"] as? Int)
        XCTAssertEqual(updates, 1); XCTAssertEqual(trained["training_performed"] as? Bool, true)
        let checkpoint = URL(fileURLWithPath: try XCTUnwrap(trained["checkpoint_path"] as? String))
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 256,
            completedUpdates: updates, expectedMembershipSHA256: membership)
        let saved = try XCTUnwrap(specialist("Wood", recorded).models.first)
        XCTAssertFalse(saved.approved)
        let prediction = NativeDiffuseTextureClassifier.Prediction(label: "Wood", confidence: 0.94,
            probabilities: ["Wood": 0.94, "Paper": 0.06], modelSHA256: classifier.sha256)
        let awaiting = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: recorded,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertTrue(awaiting.checkpoints.isEmpty)
        _ = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: saved.id)
        let unrelatedTarget = try XCTUnwrap(paperOriginals.first { $0.0.lastPathComponent == "height.png" }?.0)
        try png(size: 256, value: 118, bits: 16, channels: 1).write(to: unrelatedTarget)
        let live = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood")
        XCTAssertThrowsError(try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Paper"))
        let route = try NativeMaterialSpecialistRouter.resolve(prediction: prediction, registry: live,
            workspaceURL: workspace, modelDirectory: root.appendingPathComponent("UnusedBase").path)
        XCTAssertEqual(route.checkpoints["height"]?.checkpointPath, checkpoint.path)
        XCTAssertEqual(route.checkpoints["height"]?.sha256, saved.checkpointSha256)
        XCTAssertNil(route.checkpoints["normal"]); XCTAssertNil(route.checkpoints["roughness"])
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        // A changed original target invalidates live routing even if the saved
        // registry membership has not yet refreshed.
        let target = try XCTUnwrap(originals.first { $0.0.lastPathComponent == "height.png" }?.0)
        try png(size: 256, value: 91, bits: 16, channels: 1).write(to: target)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace, classLabel: "Wood"))
    }
    func testDynamicFamilyClassificationCreatesOriginalReferenceDatasetsWithoutScaling() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        let originals = try await add("wood_planks", size: 32, value: 51, to: dataset, under: root)
            + add("wood_planks", size: 64, value: 73, to: dataset, under: root)
        let probe = PredictionProbe()
        let classifier = fake(labels: ["Wood", "Paper", "Leather"], probe: probe) { _ in "Wood" }
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(registry.materialCount, 2); XCTAssertEqual(registry.familyCount, 1)
        XCTAssertEqual(probe.paths.count, 1, "Resolution tiers share one class prediction to avoid family leakage")
        XCTAssertEqual(try NativePNG.inspect(URL(fileURLWithPath: XCTUnwrap(probe.paths.first))).width, 32)
        let wood = try specialist("Wood", registry)
        XCTAssertEqual(wood.materialIds.count, 2); XCTAssertEqual(wood.assignments.count, 1)
        for target in ["height", "roughness", "normal"] {
            XCTAssertEqual(wood.targets.first { $0.target == target }?.materialCount, 2)
            XCTAssertEqual(wood.targets.first { $0.target == target }?.status, "ready")
            let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: URL(fileURLWithPath: wood.datasetPath), size: 32, target: target)
            XCTAssertEqual(samples.count, 1); XCTAssertEqual(samples.first?.width, 32)
            XCTAssertTrue(try XCTUnwrap(samples.first?.inputURL.path).hasPrefix(root.path + "/originals/"))
            XCTAssertTrue(try XCTUnwrap(samples.first?.targetURL.path).hasPrefix(root.path + "/originals/"))
        }
        XCTAssertEqual(try specialist("Paper", registry).targets.first?.materialCount, 0)
        XCTAssertEqual(try specialist("Paper", registry).targets.first?.status, "no_targets")
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(at: URL(fileURLWithPath: wood.datasetPath), includingPropertiesForKeys: nil)
            .allSatisfy { $0.pathExtension.lowercased() != "png" })
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        let cached = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1); XCTAssertEqual(try specialist("Wood", cached).materialIds, wood.materialIds)
        XCTAssertEqual(try NativeMaterialSpecialistService.load(workspaceURL: workspace).classifierLabels, ["Wood", "Paper", "Leather"])
    }

    func testCuratedClassAndClassifierOnlyPhotosRemainSeparateFromDepthTargets() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root), originals = try await add("cardboard", size: 32, value: 45, to: dataset, under: root)
        let curated = root.appendingPathComponent("Curated")
        for label in ["Paper", "Leather", "Other Diffuse Textures"] {
            try FileManager.default.createDirectory(at: curated.appendingPathComponent(label), withIntermediateDirectories: true)
        }
        let copied = curated.appendingPathComponent("Paper/cardboard_color.png")
        try XCTUnwrap(originals.first { $0.0.lastPathComponent == "input.png" }?.1).write(to: copied)
        let userFile = curated.appendingPathComponent("Leather/my-leather.png"), userBytes = try png(size: 29, value: 201)
        try userBytes.write(to: userFile)
        let workspace = root.appendingPathComponent("Specialists")
        let registry = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, classifierDatasetURL: curated,
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" }))
        XCTAssertEqual(registry.classifierOnlyCount, 1); XCTAssertEqual(registry.classifierOnlyAssets.first?.classLabel, "Leather")
        XCTAssertEqual(registry.classifierOnlyAssets.first?.status, "classifier_only_no_targets")
        XCTAssertEqual(registry.classifierOnlyAssets.first?.width, 29)
        let paper = try specialist("Paper", registry), leather = try specialist("Leather", registry)
        XCTAssertEqual(paper.materialIds.count, 1); XCTAssertEqual(paper.assignments.first?.source, "curated_classifier_dataset")
        XCTAssertEqual(paper.assignments.first?.predictedLabel, "Wood")
        XCTAssertTrue(leather.materialIds.isEmpty); XCTAssertTrue(leather.targets.allSatisfy { $0.materialCount == 0 })
        let another = curated.appendingPathComponent("Paper/user-paper.png"), anotherBytes = try png(size: 27, value: 169)
        try anotherBytes.write(to: another)
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" }))
        XCTAssertEqual(refreshed.classifierOnlyCount, 2); XCTAssertEqual(refreshed.materialCount, 1)
        XCTAssertEqual(try Data(contentsOf: userFile), userBytes); XCTAssertEqual(try Data(contentsOf: another), anotherBytes)
        for (url, bytes) in originals { XCTAssertEqual(try Data(contentsOf: url), bytes) }
    }

    func testRefreshPreservesManualExclusionRemovalAndAddedPairedMaterials() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 50, to: dataset, under: root)
        _ = try await add("wood_boards", size: 32, value: 60, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", first), specialistDataset = URL(fileURLWithPath: wood.datasetPath)
        let removedID = try XCTUnwrap(wood.materialIds.first { $0.hasPrefix("wood_boards") })
        _ = try await NativeMaterialDatasetService.run(arguments: ["remove-material", "--dataset", specialistDataset.path, "--material", removedID])
        let planksID = try XCTUnwrap(wood.materialIds.first { $0.hasPrefix("wood_planks") })
        _ = try await NativeMaterialDatasetService.run(arguments: ["curate", "--dataset", specialistDataset.path,
            "--sample", planksID + "_full", "--status", "excluded", "--note", "User reviewed this pair"])
        let extras = try await add("manual_wood", size: 32, value: 80, to: specialistDataset, under: root)
        let updated = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let newWood = try specialist("Wood", updated)
        XCTAssertFalse(newWood.materialIds.contains(removedID)); XCTAssertTrue(newWood.materialIds.contains("manual_wood_32x32"))
        XCTAssertEqual(newWood.targets.first { $0.target == "height" }?.materialCount, 1, "Excluded originals stay excluded after refresh")
        let sample = try object(specialistDataset.appendingPathComponent("samples/" + planksID + "_full/sample.json"))
        XCTAssertEqual(sample["status"] as? String, "excluded"); XCTAssertEqual(sample["curation_note"] as? String, "User reviewed this pair")
        for (url, bytes) in extras { XCTAssertEqual(try Data(contentsOf: url), bytes) }
        let moved = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Paper"], hash: String(repeating: "b", count: 64), choose: { _ in "Paper" }))
        XCTAssertTrue(try specialist("Wood", moved).materialIds.contains(planksID), "Curated references survive classifier changes")
        XCTAssertEqual(try specialist("Paper", moved).materialIds.count, 2)
        XCTAssertFalse(moved.warnings.isEmpty)
    }

    func testCustomSubtypeAndAdditionalSourceInventoryUseOriginalTargetsWithoutMutatingMainMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 43, to: dataset, under: root)
        _ = try await add("wood_boards", size: 32, value: 66, to: dataset, under: root)
        let additional = root.appendingPathComponent("Additional/Paper003")
        try FileManager.default.createDirectory(at: additional, withIntermediateDirectories: true)
        let paperBytes = try png(size: 31, value: 191), heightBytes = try png(size: 31, value: 99, bits: 16, channels: 1)
        try paperBytes.write(to: additional.appendingPathComponent("input.png")); try heightBytes.write(to: additional.appendingPathComponent("height.png"))
        let workspace = root.appendingPathComponent("Specialists")
        let first = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, sourceFolderURL: additional.deletingLastPathComponent(),
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Paper"], choose: { $0.path.contains("Paper003") ? "Paper" : "Wood" }))
        XCTAssertEqual(first.materialCount, 3); XCTAssertEqual(try specialist("Paper", first).materialIds.count, 1)
        let main = try object(dataset.appendingPathComponent("dataset.json"))
        XCTAssertEqual((main["samples"] as? [[String: Any]])?.count, 2, "Additional source pairs never enter the main database during classification")
        let result = try await NativeMaterialSpecialistService.addSpecialist(workspaceURL: workspace,
            name: "Wood Planks", classLabel: "Wood", subtype: "planks")
        let custom = try XCTUnwrap(result.specialists.first { $0.isCustom })
        XCTAssertEqual(custom.materialIds, ["wood_planks_32x32"]); XCTAssertEqual(custom.subtype, "planks")
        XCTAssertEqual(custom.targets.first { $0.target == "height" }?.materialCount, 1)
        XCTAssertEqual(try Data(contentsOf: additional.appendingPathComponent("input.png")), paperBytes)
        XCTAssertEqual(try Data(contentsOf: additional.appendingPathComponent("height.png")), heightBytes)
    }

    func testSavedModelsRequireTrainedSpecialistProvenanceReviewAndCurrentMembership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 54, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists"), classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial), native = URL(fileURLWithPath: wood.datasetPath)
        let membership = try XCTUnwrap(wood.targets.first { $0.target == "height" }?.membershipSha256)
        let checkpoint = try savedCheckpoint(root, dataset: native, step: 2)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64, completedUpdates: 0))
        let foreign = try savedCheckpoint(root.appendingPathComponent("foreign"), dataset: dataset, step: 2)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: foreign, trainingSize: 64, completedUpdates: 2))
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64,
            completedUpdates: 2, expectedMembershipSHA256: membership)
        let model = try XCTUnwrap(specialist("Wood", recorded).models.first)
        XCTAssertFalse(model.approved); XCTAssertEqual(try specialist("Wood", recorded).targets.first?.status, "awaiting_review")
        let approved = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id)
        XCTAssertTrue(try XCTUnwrap(specialist("Wood", approved).models.first?.approved))
        XCTAssertEqual(try specialist("Wood", approved).targets.first?.status, "approved")
        _ = try await add("manual_wood", size: 32, value: 89, to: native, under: root)
        let live = try NativeMaterialSpecialistService.routingRegistry(workspaceURL: workspace)
        XCTAssertEqual(try specialist("Wood", live).targets.first?.status, "needs_update")
        XCTAssertNotEqual(try specialist("Wood", live).targets.first?.membershipSha256, model.membershipSha256)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id))
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(try specialist("Wood", refreshed).models.first?.checkpointSha256, model.checkpointSha256)
        XCTAssertEqual(try specialist("Wood", refreshed).targets.first?.status, "needs_update")
        try Data("changed checkpoint".utf8).write(to: checkpoint)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id))
        let revoked = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id, approved: false)
        XCTAssertFalse(try XCTUnwrap(specialist("Wood", revoked).models.first?.approved))
    }

    func testUnownedWorkspaceAndSpecialistFolderArePreserved() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 44, to: dataset, under: root)
        let workspace = root.appendingPathComponent("Specialists")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let userFile = workspace.appendingPathComponent("my-record.txt"), bytes = Data("Keep this".utf8)
        try bytes.write(to: userFile)
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
                classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
            XCTFail("Unowned workspace must be rejected")
        } catch { XCTAssertEqual(try Data(contentsOf: userFile), bytes) }
    }

    func testLegacyWorkspaceMigratesToItsActualSourceFolderAndPreservesModelHistoryAndCuration() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let legacy = try await dataset(root), sources = root.appendingPathComponent("Actual Sources")
        let folder = sources.appendingPathComponent("Wood")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let input = folder.appendingPathComponent("wood_diff_2k.png"), height = folder.appendingPathComponent("wood_disp_2k.png")
        try png(size: 2048, value: 62).write(to: input)
        try png(size: 2048, value: 91, bits: 16, channels: 1).write(to: height)
        _ = try await NativeMaterialDatasetService.run(arguments: ["add-material", "--dataset", legacy.path,
            "--name", "wood", "--input", input.path, "--height", height.path])
        let workspace = root.appendingPathComponent("Loom Room"), curation = root.appendingPathComponent("Classifier Curation")
        try FileManager.default.createDirectory(at: curation.appendingPathComponent("Paper"), withIntermediateDirectories: true)
        let classifier = fake(labels: ["Wood", "Paper"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: legacy, classifierDatasetURL: curation,
            workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial), native = URL(fileURLWithPath: wood.datasetPath)
        let checkpoint = try savedCheckpoint(root.appendingPathComponent("Models"), dataset: native, step: 2)
        let recorded = try NativeMaterialSpecialistService.recordCompletedModel(workspaceURL: workspace,
            specialistID: wood.id, target: "height", checkpointURL: checkpoint, trainingSize: 64, completedUpdates: 2)
        let model = try XCTUnwrap(specialist("Wood", recorded).models.first)
        let approved = try NativeMaterialSpecialistService.approveModel(workspaceURL: workspace, specialistID: wood.id, modelID: model.id)
        let approvedModel = try XCTUnwrap(specialist("Wood", approved).models.first)
        let custom = try await NativeMaterialSpecialistService.addSpecialist(workspaceURL: workspace,
            name: "My Wood", classLabel: "Wood", subtype: "wood")
        let customID = try XCTUnwrap(custom.specialists.first { $0.isCustom }?.id)
        let materialID = try XCTUnwrap(wood.materialIds.first)
        _ = try await NativeMaterialDatasetService.run(arguments: ["curate", "--dataset", native.path,
            "--sample", materialID + "_full", "--status", "excluded", "--note", "Keep my review"])
        let sampleURL = native.appendingPathComponent("samples/" + materialID + "_full/sample.json")
        let priorSample = try Data(contentsOf: sampleURL), checkpointBytes = try Data(contentsOf: checkpoint)
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: sources))
        let migrated = try await NativeMaterialSpecialistService.refresh(datasetURL: sources,
            workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(migrated.sourceDatasetPath, sources.path)
        XCTAssertEqual(migrated.classifierDatasetPath, curation.path)
        XCTAssertEqual(try specialist("Wood", migrated).id, wood.id)
        XCTAssertTrue(migrated.specialists.contains { $0.id == customID && $0.isCustom })
        let preserved = try XCTUnwrap(specialist("Wood", migrated).models.first)
        XCTAssertEqual(preserved.id, approvedModel.id)
        XCTAssertEqual(preserved.checkpointPath, approvedModel.checkpointPath)
        XCTAssertEqual(preserved.checkpointSha256, approvedModel.checkpointSha256)
        XCTAssertEqual(preserved.membershipSha256, approvedModel.membershipSha256)
        XCTAssertEqual(preserved.approved, true); XCTAssertEqual(preserved.approvedAt, approvedModel.approvedAt)
        XCTAssertEqual(try Data(contentsOf: checkpoint), checkpointBytes)
        XCTAssertEqual(try Data(contentsOf: sampleURL), priorSample, "Migration must preserve specialist curation in place")
        XCTAssertEqual(try object(workspace.appendingPathComponent(".specialist-workspace.json"))["source_dataset_path"] as? String, sources.path)
        XCTAssertEqual(try NativeMaterialSpecialistService.load(workspaceURL: workspace).sourceDatasetPath, sources.path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: sources.appendingPathComponent("dataset.json").path))
    }

    func testWorkspaceMigrationRejectsPartialContainmentAndConflictingRegistryWithoutChangingOwnership() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let legacy = try await dataset(root)
        _ = try await add("wood", size: 32, value: 48, to: legacy, under: root)
        _ = try await add("paper", size: 32, value: 62, to: legacy, under: root)
        let workspace = root.appendingPathComponent("Loom Room"), classifier = fake(labels: ["Wood"], choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: legacy, workspaceURL: workspace, classifier: classifier)
        let markerURL = workspace.appendingPathComponent(".specialist-workspace.json")
        let registryURL = workspace.appendingPathComponent(NativeMaterialSpecialistService.registryFilename)
        let markerBytes = try Data(contentsOf: markerURL), registryBytes = try Data(contentsOf: registryURL)
        let onlyOneFamily = root.appendingPathComponent("originals/wood")
        XCTAssertFalse(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: onlyOneFamily))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: onlyOneFamily, workspaceURL: workspace, classifier: classifier)
            XCTFail("One overlapping family cannot claim a workspace for another corpus")
        } catch { XCTAssertTrue(error.localizedDescription.contains("another source")) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes); XCTAssertEqual(try Data(contentsOf: registryURL), registryBytes)
        let allOriginals = root.appendingPathComponent("originals")
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        let owned = try specialist("Wood", initial)
        let bindingURL = URL(fileURLWithPath: owned.datasetPath).appendingPathComponent(".specialist-binding.json")
        let bindingBytes = try Data(contentsOf: bindingURL)
        var colliding = try object(bindingURL); colliding["workspace_path"] = root.appendingPathComponent("Unrelated Workspace").path
        try JSONSerialization.data(withJSONObject: colliding).write(to: bindingURL, options: .atomic)
        XCTAssertThrowsError(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: allOriginals, workspaceURL: workspace, classifier: classifier)
            XCTFail("A colliding category folder must retain its existing binding")
        } catch { XCTAssertTrue(error.localizedDescription.contains("unrelated dataset")) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes); XCTAssertEqual(try Data(contentsOf: registryURL), registryBytes)
        try bindingBytes.write(to: bindingURL, options: .atomic)
        var mismatched = try object(registryURL)
        mismatched["source_dataset_path"] = root.appendingPathComponent("Another Legacy Dataset").path
        try JSONSerialization.data(withJSONObject: mismatched, options: [.sortedKeys]).write(to: registryURL, options: .atomic)
        let conflictingBytes = try Data(contentsOf: registryURL)
        XCTAssertFalse(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: workspace, datasetURL: allOriginals))
        do {
            _ = try await NativeMaterialSpecialistService.refresh(datasetURL: allOriginals, workspaceURL: workspace, classifier: classifier)
            XCTFail("Conflicting ownership cannot be repaired by claiming it")
        } catch { XCTAssertTrue(error.localizedDescription.contains("another source")) }
        XCTAssertEqual(try Data(contentsOf: markerURL), markerBytes)
        XCTAssertEqual(try Data(contentsOf: registryURL), conflictingBytes)
        let partial = root.appendingPathComponent("Partial Workspace")
        try FileManager.default.createDirectory(at: partial, withIntermediateDirectories: true)
        var partialMarker = try object(markerURL); partialMarker["workspace_path"] = partial.path
        try JSONSerialization.data(withJSONObject: partialMarker).write(to: partial.appendingPathComponent(".specialist-workspace.json"))
        XCTAssertFalse(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: partial, datasetURL: allOriginals))
        let absent = root.appendingPathComponent("New Workspace")
        XCTAssertTrue(try NativeMaterialSpecialistService.canUseWorkspace(workspaceURL: absent, datasetURL: allOriginals))
        XCTAssertFalse(FileManager.default.fileExists(atPath: absent.path), "Eligibility must remain read-only")
    }

    func testMigrationRefreshesRenamedManagedSourcePathsWhileKeepingOriginalAndCropReviews() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let legacy = try await dataset(root), sources = root.appendingPathComponent("Actual Sources")
        let oldFolder = sources.appendingPathComponent("TypoWood"), newFolder = sources.appendingPathComponent("Wood")
        try FileManager.default.createDirectory(at: oldFolder, withIntermediateDirectories: true)
        let input = oldFolder.appendingPathComponent("wood_diff_2k.png"), height = oldFolder.appendingPathComponent("wood_disp_2k.png")
        let inputBytes = try png(size: 2048, value: 62), heightBytes = try png(size: 2048, value: 93, bits: 16, channels: 1)
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        _ = try await NativeMaterialDatasetService.run(arguments: ["add-material", "--dataset", legacy.path,
            "--name", "wood", "--input", input.path, "--height", height.path])
        let workspace = root.appendingPathComponent("Loom Room"), probe = PredictionProbe()
        let classifier = fake(labels: ["Wood"], probe: probe, choose: { _ in "Wood" })
        let initial = try await NativeMaterialSpecialistService.refresh(datasetURL: legacy,
            workspaceURL: workspace, classifier: classifier)
        let wood = try specialist("Wood", initial), native = URL(fileURLWithPath: wood.datasetPath)
        let materialID = try XCTUnwrap(wood.materialIds.first), originalID = materialID + "_full"
        _ = try await NativeMaterialDatasetService.run(arguments: ["curate", "--dataset", native.path,
            "--sample", originalID, "--original-source", "true", "--status", "approved", "--note", "Keep original review"])
        let browserText = try await NativeMaterialDatasetService.run(arguments: ["dataset", "--dataset", native.path, "--review-size", "256"])
        let browser = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(browserText).utf8)) as? [String: Any])
        let crop = try XCTUnwrap((browser["materials"] as? [[String: Any]])?.first?["samples"] as? [[String: Any]]).first
        let cropID = try XCTUnwrap(crop?["sample_id"] as? String)
        _ = try await NativeMaterialDatasetService.run(arguments: ["curate", "--dataset", native.path, "--sample", cropID,
            "--review-size", "256", "--status", "approved", "--split", "train", "--note", "Keep crop review"])
        var nativeIndex = try object(native.appendingPathComponent("dataset.json"))
        var validation = try XCTUnwrap(nativeIndex["validation"] as? [String: Any]); validation["folders"] = [oldFolder.path: false]
        nativeIndex["validation"] = validation
        try JSONSerialization.data(withJSONObject: nativeIndex).write(to: native.appendingPathComponent("dataset.json"), options: .atomic)
        let managedSample = native.appendingPathComponent("samples/" + originalID + "/sample.json")
        let sampleBefore = try object(managedSample)
        try FileManager.default.moveItem(at: oldFolder, to: newFolder)
        func rebase(_ value: Any) -> Any {
            if let value = value as? String, value == oldFolder.path || value.hasPrefix(oldFolder.path + "/") { return newFolder.path + value.dropFirst(oldFolder.path.count) }
            if let values = value as? [Any] { return values.map(rebase) }
            if let object = value as? [String: Any] { return object.mapValues(rebase) }
            return value
        }
        let legacySample = legacy.appendingPathComponent("samples/" + originalID + "/sample.json")
        try JSONSerialization.data(withJSONObject: rebase(try object(legacySample))).write(to: legacySample, options: .atomic)
        let refreshed = try await NativeMaterialSpecialistService.refresh(datasetURL: sources,
            workspaceURL: workspace, classifier: classifier)
        XCTAssertEqual(probe.paths.count, 1, "Renaming a folder must reuse the valid classifier file tag")
        XCTAssertEqual(try specialist("Wood", refreshed).assignments.first?.inputPath, newFolder.appendingPathComponent(input.lastPathComponent).path)
        let sample = try object(managedSample)
        for key in ["status", "review_status", "curation_note", "split", "sample_id", "material_id"] {
            XCTAssertEqual(sample[key] as? String, sampleBefore[key] as? String)
        }
        XCTAssertEqual(sample["source_directory"] as? String, newFolder.path)
        let target = try XCTUnwrap((sample["map_metadata"] as? [String: [String: Any]])?["height"]?["source"] as? [String: Any])
        XCTAssertEqual(target["path"] as? String, newFolder.appendingPathComponent(height.lastPathComponent).path)
        XCTAssertNil(target["file_sha256"], "An unverified new target path must not inherit the previous stored checksum")
        XCTAssertEqual(target["source_verification"] as? String, "header_inspected")
        let updatedSettings = try XCTUnwrap(try object(native.appendingPathComponent("dataset.json"))["validation"] as? [String: Any])
        XCTAssertEqual((updatedSettings["folders"] as? [String: Bool])?[newFolder.path], false)
        XCTAssertNil((updatedSettings["folders"] as? [String: Bool])?[oldFolder.path])
        let reviews = try object(native.appendingPathComponent(".material-size-reviews.json"))
        let review = try XCTUnwrap(reviews["256:" + cropID] as? [String: Any])
        XCTAssertEqual(review["status"] as? String, "approved"); XCTAssertEqual(review["curation_note"] as? String, "Keep crop review")
        XCTAssertEqual(review["split"] as? String, "train"); XCTAssertEqual(review["split_assignment"] as? String, "manual")
        XCTAssertEqual(review["source_binding_sha256"] as? String, NativeMaterialDatasetService.sourceBinding(sample))
        let outputText = try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", native.path, "--size", "256", "--target", "height", "--skip-source-recovery"])
        let output = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(outputText).utf8)) as? [String: Any])
        let prepared = URL(fileURLWithPath: try XCTUnwrap(output["dataset_path"] as? String))
        let training = try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: "height")
        XCTAssertFalse(training.isEmpty)
        XCTAssertTrue(training.allSatisfy { $0.augmentationSourceInputURL?.path.hasPrefix(newFolder.path + "/") == true && $0.augmentationSourceTargetURL?.path.hasPrefix(newFolder.path + "/") == true })
        _ = try NativeMaterialDatasetService.cleanup(prepared)
        XCTAssertEqual(try Data(contentsOf: newFolder.appendingPathComponent(input.lastPathComponent)), inputBytes)
        XCTAssertEqual(try Data(contentsOf: newFolder.appendingPathComponent(height.lastPathComponent)), heightBytes)
    }

    func testCurrentDirectoriesReplaceStaleManifestTaxonomyAndRetireOnlyEmptyAutomaticRecords() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await dataset(root)
        _ = try await add("wood_planks", size: 32, value: 34, to: dataset, under: root)
        let curated = root.appendingPathComponent("Curated"), workspace = root.appendingPathComponent("Specialists")
        try FileManager.default.createDirectory(at: curated.appendingPathComponent("Paper"), withIntermediateDirectories: true)
        let obsolete = NativeDiffuseClassifierDatasetService.TextureClass(id: "obsolete", name: "Obsolete Finish", description: "Old class")
        let manifest = NativeDiffuseClassifierDatasetService.Manifest(schema: NativeDiffuseClassifierDatasetService.schema,
            updatedAt: "fixture", datasetPath: curated.path, selectionPolicy: "fixture", classes: [obsolete], entries: [])
        try JSONEncoder().encode(manifest).write(to: curated.appendingPathComponent(".classifier-manifest.json"))
        let old = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, classifierDatasetURL: curated,
            workspaceURL: workspace, classifier: fake(labels: ["Wood", "Obsolete Finish"], choose: { _ in "Wood" }))
        XCTAssertEqual(old.specialists.count, 3)
        let obsoleteRecord = try specialist("Obsolete Finish", old)
        let next = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        XCTAssertEqual(Set(next.specialists.map(\.classLabel)), ["Wood", "Paper"])
        XCTAssertTrue(FileManager.default.fileExists(atPath: obsoleteRecord.datasetPath), "Retiring an empty registry entry never deletes its folder")
        // User-added paired data prevents retirement even though no model or
        // current classifier label mentions this old class.
        let returned = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood", "Obsolete Finish"], choose: { _ in "Wood" }))
        let restored = try specialist("Obsolete Finish", returned)
        _ = try await add("manual_finish", size: 32, value: 94, to: URL(fileURLWithPath: restored.datasetPath), under: root)
        let preserved = try await NativeMaterialSpecialistService.refresh(datasetURL: dataset, workspaceURL: workspace,
            classifier: fake(labels: ["Wood"], choose: { _ in "Wood" }))
        XCTAssertEqual(try specialist("Obsolete Finish", preserved).materialIds, ["manual_finish_32x32"])
    }

    private func fake(labels: [String], hash: String = String(repeating: "a", count: 64), probe: PredictionProbe? = nil,
                      choose: @escaping @Sendable (URL) -> String) -> NativeMaterialSpecialistService.Classifier {
        .init(path: "/fixture.mlmodel", sha256: hash, labels: labels) { url in
            probe?.append(url.path); let label = choose(url)
            return .init(label: label, confidence: 0.94, probabilities: Dictionary(uniqueKeysWithValues: labels.map { ($0, $0 == label ? 0.94 : 0.01) }), modelSHA256: hash)
        }
    }
    private func specialist(_ label: String, _ registry: NativeMaterialSpecialistService.Registry) throws -> NativeMaterialSpecialistService.Specialist {
        try XCTUnwrap(registry.specialists.first { !$0.isCustom && $0.classLabel == label })
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("specialist-tests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root.resolvingSymlinksInPath().standardizedFileURL
    }
    private func dataset(_ root: URL) async throws -> URL {
        let url = root.appendingPathComponent("Main")
        _ = try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", url.path,
            "--name", "Specialist fixture", "--training-size", "256", "--subject-validation", "disabled"])
        return url
    }
    private func add(_ family: String, size: Int, value: UInt8, to dataset: URL, under root: URL) async throws -> [(URL, Data)] {
        let folder = root.appendingPathComponent("originals/" + family + "/" + String(size))
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        var arguments = ["add-material", "--dataset", dataset.path, "--name", family], originals: [(URL, Data)] = []
        for (role, bits, channels) in [("input", 8, 3), ("height", 16, 1), ("roughness", 8, 1), ("normal", 8, 3)] {
            let file = folder.appendingPathComponent(role + ".png"), bytes = try png(size: size, value: value, bits: bits, channels: channels)
            try bytes.write(to: file); originals.append((file, bytes)); arguments += ["--" + role, file.path]
        }
        _ = try await NativeMaterialDatasetService.run(arguments: arguments)
        return originals
    }
    private func png(size: Int, value: UInt8, bits: Int = 8, channels: Int = 3) throws -> Data {
        try NativePNG(header: .init(width: size, height: size, bits: bits, channels: channels, color: channels == 1 ? 0 : 2, interlace: 0),
            pixels: Data(repeating: value, count: size * size * channels * bits / 8), colorChunks: []).encoded()
    }
    private func savedCheckpoint(_ root: URL, dataset: URL, step: Int) throws -> URL {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let model = try NativeCompactMaterialModel(target: "height", width: 16, seed: 13)
        var configuration = try model.checkpointConfiguration(size: 64, step: step)
        configuration["training_configuration"] = ["training_cache_root": dataset.appendingPathComponent(".training-data").path,
            "initial_step": 0, "dataset_manifest_sha256": String(repeating: "f", count: 64)]
        let file = root.appendingPathComponent("trained.safetensors")
        try NativeSafetensors.write(tensors: model.weights, metadata: ["configuration": NativeMaterialTransfer.json(configuration)], to: file)
        return file
    }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
}

private final class PredictionProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: [String] = []
    var paths: [String] { lock.lock(); defer { lock.unlock() }; return stored }
    func append(_ path: String) { lock.lock(); stored.append(path); lock.unlock() }
}
